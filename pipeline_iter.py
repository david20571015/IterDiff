from collections.abc import Callable
from typing import Any, cast

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    StableDiffusionInstructPix2PixPipeline,
)
from tqdm.auto import tqdm

from attn_ctrl import AttentionControl


class IterEditPipeline:
    def __init__(self, ip2p_pipeline: StableDiffusionInstructPix2PixPipeline):
        self.pipe = ip2p_pipeline

    def to(self, *args, **kwargs):
        self.pipe.to(*args, **kwargs)
        return self

    @torch.inference_mode()
    def __call__(
        self,
        prompt: list[list[str]],
        image: torch.Tensor | list[torch.Tensor],
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        cross_attention_kwargs: dict[str, Any] | None = None,
        attn_ctrl: AttentionControl | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            prompt:
                The prompts to guide image generation. The size should be `num_edits` * `num_images`.
            image:
                The images to be edited. If it's a tensor, its shape should be `(num_images, 3, h, w)`.
                If it's a list of tensors, its length should be equal to `num_images` and each tensor should have
                shape `(3, h, w)`.

        Returns:
            A tensor of shape `(num_images, num_edits, 3, h, w)` representing the edited images.

        """

        # check inputs
        num_prompts_each_edit = list(map(len, prompt))
        if len(set(num_prompts_each_edit)) != 1:
            raise ValueError(
                f"Numbers of prompt of each edits should be equal, but got {num_prompts_each_edit}."
            )

        if not isinstance(image, (list, torch.Tensor)):
            raise ValueError(
                f"`image` must be either a list of tensors or a tensor, but got {type(image)}."
            )
        num_images = len(image) if isinstance(image, list) else image.shape[0]

        if num_images != num_prompts_each_edit[0]:
            raise ValueError(
                f"Length of `image` ({num_images}) must be equal to `num_image` ({num_prompts_each_edit[0]})."
            )

        results: list[torch.Tensor] = []

        self.pipe.set_progress_bar_config(leave=False)
        for p in tqdm(prompt, leave=False):
            image = cast(
                torch.Tensor,
                self.pipe(
                    prompt=p,
                    image=image,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    num_images_per_prompt=1,
                    eta=eta,
                    generator=generator,
                    output_type="pt",
                    return_dict=False,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attn_ctrl=attn_ctrl,
                    **kwargs,
                )[0],
            )
            results.append(image.detach().clone())

            if attn_ctrl is not None:
                attn_ctrl.between_edits()

        return torch.stack(results, dim=1)
