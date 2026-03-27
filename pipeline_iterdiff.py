from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from diffusers.utils.deprecation_utils import deprecate
from einops import rearrange
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor

from attn_ctrl import AttentionControl
from utils.metric import _clip_score_update


@dataclass
class Tensors:
    tensors: dict[str, torch.Tensor] = field(repr=False)
    size: int = field(init=False, repr=False)

    def __post_init__(self):
        if not self.tensors:
            raise ValueError("Empty tensors are not allowed.")

        sample_tensor = next(iter(self.tensors.values()))
        self.size = int(sample_tensor.shape[1] ** 0.5) if sample_tensor.ndim == 3 else 0

        for k, v in self.tensors.items():
            self.tensors[k] = v.cpu()

    def __iadd__(self, other: "Tensors"):
        for k in self.tensors.keys():
            self.tensors[k] += other.tensors[k].to(self.tensors[k].device)
        return self

    def __itruediv__(self, other: int):
        for k in self.tensors.keys():
            self.tensors[k] /= other
        return self

    def numel(self):
        return sum(t.numel() for t in self.tensors.values())


@dataclass
class StoreEntry(Tensors):
    is_cross: bool = False
    name: str = ""


type Store[T] = dict[int, T]


@dataclass
class MemoryBankEntry(Tensors):
    timestep: int
    nth_edit: int
    mask: torch.Tensor | None = field(default=None, repr=False)


class MemoryBank:
    def __init__(self, max_size=0):
        self.bank: dict[int, MemoryBankEntry] = {}
        self.max_size = max_size

    def __len__(self):
        return len(self.bank)

    def push(self, item: MemoryBankEntry):
        if self.max_size == 0:
            return

        self.bank[item.timestep] = item

    def pop(self, timestep: int):
        return self.bank.pop(timestep, None)

    def find_by_timestep(self, timestep: int):
        return self.bank.get(timestep, None)

    def merge(self, other: "MemoryBank"):
        for k, v in other.bank.items():
            self.bank[k] = v

    def numel(self):
        return sum(v.numel() for v in self.bank.values())


def merge_edit_store(target: Store[MemoryBank], source: Store[MemoryBank]):
    for attn_layer in target.keys():
        target[attn_layer].merge(source[attn_layer])


def merge_step_store(target: Store[StoreEntry], source: Store[StoreEntry]):
    for attn_layer in target.keys():
        target[attn_layer] += source[attn_layer]


class AttentionStore(AttentionControl):
    def __init__(
        self,
        size=64,
        mb_size: int = 30,
        clip_model_id: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14-336",
            "openai/clip-vit-large-patch14",
        ] = "openai/clip-vit-large-patch14",
        mb_save_topk: int = 20,
    ):
        """
        Args:
            size: Height or width of the input image (the both should be the same).
            mb_size: The size of the memory bank.
            clip_model_id: The model id of the CLIP model.
            mb_use_weight: The weight of the usage of the memory bank entry. 1.0 for replace the current KV with the
                memory bank entry, 0.0 for not using the memory bank entry.
            mb_save_topk: The number of top-k memory bank entries to save.
        """
        super().__init__()
        self.size = size

        # {attn_layer, Store}
        self.attn_step_store: Store[StoreEntry] = self.get_empty_store()
        self.kv_step_store: Store[StoreEntry] = self.get_empty_store()

        self.mb_size = mb_size
        self.mb_save_topk = mb_save_topk

        self.guidance_factor = 1.0
        self.is_kv_retrieved = False

        if self.mb_size < self.mb_save_topk:
            self.mb_save_topk = self.mb_size
            print(
                f"mb_size ({self.mb_size}) should be greater than or equal to mb_save_topk ({self.mb_save_topk}). "
                "Setting mb_save_topk to mb_size."
            )

        # {attn_layer, bank}
        self.kv_edit_store: Store[MemoryBank] = self.get_empty_edit_store(self.mb_size)
        self.kv_prev_edit_store: Store[MemoryBank] = self.get_empty_edit_store(
            self.mb_size
        )

        self.clip_model, self.clip_processor = _get_clip_model_and_processor(
            clip_model_id
        )

        self.image_prompt_sim_list = []
        self.image_prompt_sim_list_history = []
        self.pred_images = []
        self.pred_images_history = []
        self.t_save_history = []

    @staticmethod
    def get_empty_store() -> Store[StoreEntry]:
        return {}

    @staticmethod
    def get_empty_edit_store(memory_bank_size=10) -> Store[MemoryBank]:
        return defaultdict(lambda: MemoryBank(memory_bank_size))

    def reset(self):
        super().reset()
        # {attn_layer, Store}
        self.attn_step_store = self.get_empty_store()
        self.kv_step_store = self.get_empty_store()

    def full_reset(self):
        self.reset()
        self.kv_edit_store = self.get_empty_edit_store(self.mb_size)
        self.kv_prev_edit_store = self.get_empty_edit_store(self.mb_size)
        self.image_prompt_sim_list = []
        self.guidance_factor = 1.0

    def clip_similarity(self, text: str, image: torch.Tensor) -> torch.Tensor:
        score, _ = _clip_score_update(
            image, text, self.clip_model.to(image.device), self.clip_processor
        )
        # score, _ = _clip_score_update(
        #     image,
        #     f'A human face edited with prompt: "{text}"',
        #     self.clip_model.to(image.device),
        #     self.clip_processor,
        # )
        return score

    @torch.inference_mode()
    def step_callback(
        self,
        orig_x: torch.Tensor,
        pred_x_0: torch.Tensor,
        text: str,
        t: int,
    ):
        cur_step = self.cur_step - 1  # self.cur_step was already updated after the unet

        self.image_prompt_sim_list.insert(
            cur_step,
            self.clip_similarity(text, pred_x_0).item(),
        )
        self.pred_images.insert(cur_step, pred_x_0.cpu())

    def between_steps(self):
        return

    @torch.inference_mode()
    def between_edits(self):
        if self.mb_size == 100:  # For TFFP
            self.image_prompt_sim_list = []
            merge_edit_store(self.kv_prev_edit_store, self.kv_edit_store)
            self.kv_edit_store = self.get_empty_edit_store(self.mb_size)

        elif self.mb_size > 0:
            t_save = self.image_prompt_sim_list[: self.mb_size]
            t_save = torch.topk(
                torch.as_tensor(t_save), self.mb_save_topk, sorted=False
            ).indices.tolist()

            # self.t_save_history.append(t_save)
            self.image_prompt_sim_list_history.append(self.image_prompt_sim_list)
            self.image_prompt_sim_list = []
            self.pred_images_history.append(self.pred_images)
            self.pred_images = []

            for attn_layer, bank in self.kv_edit_store.items():
                t_to_pop = [t for t in bank.bank.keys() if t not in t_save]
                for t in t_to_pop:
                    bank.pop(t)

            merge_edit_store(self.kv_prev_edit_store, self.kv_edit_store)
            self.kv_edit_store = self.get_empty_edit_store(self.mb_size)

            cur_mb_size = len(next(iter(self.kv_prev_edit_store.values())))
            self.guidance_factor = 100.0 / (100.0 - cur_mb_size)

        super().between_edits()

    def forward(
        self,
        tensors: dict[str, torch.Tensor],
        is_cross: bool,
        attn_processor_name: str,
    ):
        if "attn" in tensors.keys():
            return self.attn_map_forward(
                attn=tensors["attn"],
                is_cross=is_cross,
                attn_processor_name=attn_processor_name,
            )
        elif "key" in tensors.keys() and "value" in tensors.keys():
            return self.kv_forward(
                key=tensors["key"],
                value=tensors["value"],
                is_cross=is_cross,
                attn_processor_name=attn_processor_name,
            )
        else:
            raise ValueError(
                "Invalid input. Please provide either {`attn_map`} or {`key` and `value`}."
            )

    @torch.inference_mode()
    def attn_map_forward(
        self,
        attn: torch.Tensor,
        is_cross: bool,
        attn_processor_name: str,
    ):
        attn_map_size = int(attn.shape[1] ** 0.5)

        # [32, 16, 8]
        if not (attn_map_size <= self.size // 2):
            return attn

        # [batch_size * attention_head_dim (8) * 3, hw, token_length (77)]
        attn_text, attn_image, attn_uncond = attn.chunk(3, dim=0)

        step_store = self.attn_step_store

        # s-cfg needs both cross and self attention of text
        # store the attention map to step store
        step_store[self.cur_attn_layer] = StoreEntry(
            tensors={"attn": attn_text},
            is_cross=is_cross,
            name=attn_processor_name,
        )
        return attn

    @torch.inference_mode()
    def kv_forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        is_cross: bool,
        attn_processor_name: str,
    ):
        attn_map_size = int(key.shape[1] ** 0.5)

        # [32, 16, 8]
        if not (attn_map_size <= self.size // 2) or is_cross or self.mb_size == 0:
            return key, value

        # [batch_size * attention_head_dim (8) * 3, hw, dim]
        # key_text, key_image, key_uncond = key.chunk(3, dim=0)
        # val_text, val_image, val_uncond = value.chunk(3, dim=0)

        edit_store = self.kv_edit_store
        prev_edit_store = self.kv_prev_edit_store
        prev_store = prev_edit_store[self.cur_attn_layer]

        # update current KV by mb's KV
        if item_to_insert := prev_store.find_by_timestep(self.cur_step):
            tensors_to_insert = item_to_insert.tensors
            key = tensors_to_insert["key"].to(key.device)
            value = tensors_to_insert["value"].to(value.device)

        self.is_kv_retrieved = item_to_insert is not None
        # print(f"KV retrieved: {self.is_kv_retrieved}")

        if 0 <= self.cur_step < self.mb_size:
            item = MemoryBankEntry(
                tensors={
                    "key": key,
                    "value": value,
                },
                timestep=self.cur_step,
                nth_edit=self.cur_edit,
            )
            edit_store[self.cur_attn_layer].push(item)

        return key, value

    @torch.inference_mode()
    def get_mask(self, r: int = 4, device: torch.device | str = torch.device("cuda")):
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        curr_r = r

        def spatial_normalize(x: torch.Tensor, eps=1e-8):
            return x / (x.mean(dim=[2, 3], keepdim=True) + eps)

        r_r = 1
        new_ca = 0
        new_fore = 0
        a_n = 0
        attention_maps = self.attn_step_store.values()

        while curr_r <= 8:
            attn_stores = [s for s in attention_maps if curr_r == self.size // s.size]

            sa = torch.stack(
                [s.tensors["attn"].to(device) for s in attn_stores if not s.is_cross],
                dim=1,
            )
            sa = rearrange(sa, "(b h) n s t -> b h n s t", h=8).mean(1)

            ca = torch.stack(
                [s.tensors["attn"].to(device) for s in attn_stores if s.is_cross],
                dim=1,
            )
            ca = rearrange(ca, "(b h) n s t -> b h n s t", h=8).mean(1)

            attn_num = sa.size(1)
            sa = rearrange(sa, "b n s t -> (b n) s t")
            ca = rearrange(ca, "b n s t -> (b n) s t")

            R = 4
            # b hw c
            ca = torch.stack(
                [torch.matrix_power(sa, i) @ ca for i in range(1, R + 1)], dim=0
            ).mean(0)

            h = w = int(ca.size(1) ** 0.5)

            ca = rearrange(ca, "bn (h w) t -> bn t h w", h=h, w=w)
            if r_r > 1:
                ca = F.interpolate(ca, scale_factor=r_r, mode="bilinear")

            ca = TF.gaussian_blur(ca.float(), kernel_size=[3, 3], sigma=[0.5, 0.5])
            new_ca += rearrange(
                spatial_normalize(ca), "(b n) t h w -> b n t h w", n=attn_num
            ).sum(1)

            # start token & others
            fore_ca = torch.stack([ca[:, 0], ca[:, 1:].sum(dim=1)], dim=1)
            new_fore += rearrange(
                spatial_normalize(fore_ca), "(b n) t h w -> b n t h w", n=attn_num
            ).sum(1)
            a_n += attn_num

            curr_r = int(curr_r * 2)
            r_r *= 2

        new_ca = new_ca / a_n  # b t h w
        ca_mask = torch.zeros_like(new_ca).scatter(
            dim=1,
            index=new_ca.argmax(dim=1, keepdim=True),
            value=1.0,
        )
        fore_mask = 1.0 - ca_mask[:, :1]
        return ca_mask, fore_mask


class IterDiffPipeline(StableDiffusionInstructPix2PixPipeline):
    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        image: PipelineImageInput | None = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        negative_prompt: str | list[str] | None = None,
        num_images_per_prompt: int | None = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        ip_adapter_image: PipelineImageInput | None = None,
        ip_adapter_image_embeds: list[torch.Tensor] | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        cross_attention_kwargs: dict[str, Any] | None = None,
        attn_ctrl: AttentionStore | None = None,
        use_scfg=True,
        use_factor=False,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.Tensor` `np.ndarray`, `PIL.Image.Image`, `list[torch.Tensor]`, `list[PIL.Image.Image]`, or `list[np.ndarray]`):
                `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept
                image latents as `image`, but if passing latents directly it is not encoded again.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                Push the generated image towards the initial `image`. Image guidance scale is enabled by setting
                `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
                linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
                value of at least `1`.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInstructPix2PixPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

        >>> image = download_image(img_url).resize((512, 512))

        >>> pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        ...     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "make the mountains snowy"
        >>> image = pipe(prompt=prompt, image=image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Check inputs
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale

        device = self._execution_device

        if image is None:
            raise ValueError("`image` input cannot be undefined.")

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare Image latents
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            self.do_classifier_free_guidance,
        )

        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents + num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds} if ip_adapter_image is not None else None
        )

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = (
                    torch.cat([latents] * 3)
                    if self.do_classifier_free_guidance
                    else latents
                )

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                scaled_latent_model_input = torch.cat(
                    [scaled_latent_model_input, image_latents], dim=1
                )

                # predict the noise residual
                noise_pred = self.unet(
                    scaled_latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = (
                        noise_pred.chunk(3)
                    )

                    # S-CFG
                    if attn_ctrl is not None and use_scfg:
                        R = 4
                        ca_mask, fore_mask = attn_ctrl.get_mask(r=R, device=device)
                        mask_t = F.interpolate(ca_mask, scale_factor=R, mode="nearest")
                        mask_fore = F.interpolate(
                            fore_mask, scale_factor=R, mode="nearest"
                        )

                        model_delta = noise_pred_text - noise_pred_image
                        model_delta_norm = model_delta.norm(dim=1, keepdim=True)

                        delta_mask_norms = (model_delta_norm * mask_t).sum([2, 3]) / (
                            mask_t.sum([2, 3]) + 1e-8
                        )
                        upnormmax = delta_mask_norms.max(dim=1)[0]  # b
                        upnormmax = upnormmax.unsqueeze(-1)

                        fore_norms = (model_delta_norm * mask_fore).sum([2, 3]) / (
                            mask_fore.sum([2, 3]) + 1e-8
                        )

                        up = fore_norms
                        down = delta_mask_norms

                        tmp_mask = (mask_t.sum([2, 3]) > 0).float()
                        rate = up * (tmp_mask) / (down + 1e-8)
                        rate = (rate.unsqueeze(-1).unsqueeze(-1) * mask_t).sum(
                            dim=1, keepdim=True
                        )

                        rate = torch.clamp(
                            rate, min=0.8, max=min(3.0, 15.0 / guidance_scale)
                        )
                        rate = TF.gaussian_blur(
                            rate, kernel_size=[3, 3], sigma=[0.5, 0.5]
                        )

                        rate = rate.to(noise_pred_text.dtype)
                    else:
                        rate = 1.0

                    if (
                        use_factor
                        and attn_ctrl is not None
                        and not attn_ctrl.is_kv_retrieved
                    ):
                        guidance_factor = attn_ctrl.guidance_factor
                    else:
                        guidance_factor = 1

                    noise_pred = (
                        noise_pred_uncond
                        + (
                            self.guidance_scale
                            * guidance_factor
                            * rate
                            * (noise_pred_text - noise_pred_image)
                        )
                        + (
                            self.image_guidance_scale
                            * (noise_pred_image - noise_pred_uncond)
                        )
                    )

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(
                #     noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                # )[0]
                output = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=True
                )
                latents, pred_latents = output.prev_sample, output.pred_original_sample

                if attn_ctrl is not None:
                    pred_images = self.latents2images(
                        pred_latents, device, prompt_embeds.dtype
                    )
                    attn_ctrl.step_callback(
                        orig_x=image,
                        pred_x_0=pred_images,
                        text=prompt,
                        t=t,
                    )

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    image_latents = callback_outputs.pop("image_latents", image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    def latents2images(self, latents, device, dtype):
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, dtype)

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(
            image, output_type="pt", do_denormalize=do_denormalize
        )

        return cast(torch.Tensor, image)
