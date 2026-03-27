import argparse
import json
import os
from pprint import pprint
from typing import Literal

import torch
import torchvision.transforms.v2 as T
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import logging
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm.auto import tqdm

from attn_ctrl import register_attention_controller
from pipeline_emilie import EmiliePipeline
from pipeline_iter import IterEditPipeline
from pipeline_iterdiff import AttentionStore, IterDiffPipeline

logging.set_verbosity_error()
torch.enable_grad(False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "timbrooks/instruct-pix2pix"

UNDERLYING_PIPE_TYPES = Literal["ip2p", "scfg", "iterdiff", "emilie"]
UNDERLYING_PIPES = {
    "ip2p": StableDiffusionInstructPix2PixPipeline,
    "scfg": IterDiffPipeline,
    "iterdiff": IterDiffPipeline,
    "emilie": EmiliePipeline,
}


class IterEditDataset(Dataset):
    def __init__(self, instruction_file: str, image_dir: str, transform=None):
        with open(instruction_file) as f:
            self.instructions = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        image_path, insts = self.instructions[idx].values()
        image = read_image(os.path.join(self.image_dir, image_path))
        if self.transform:
            image = self.transform(image)
        return image, insts


def get_dataloader(instruction_file: str, image_dir: str):
    dataset = IterEditDataset(
        instruction_file=instruction_file,
        image_dir=image_dir,
        transform=T.Compose(
            [
                T.Resize(512, antialias=True),
                T.ToDtype(torch.float32, scale=True),
            ]
        ),
    )

    def collate_fn(batch):
        images, insts = zip(*batch)
        images = torch.stack(images)
        insts = [list(i) for i in zip(*insts)]
        return images, insts

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return dataloader


def get_iterpipe(underlying_pipe_type: UNDERLYING_PIPE_TYPES) -> IterEditPipeline:
    underlying_pipe_cls = UNDERLYING_PIPES[underlying_pipe_type]

    return IterEditPipeline(
        underlying_pipe_cls.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, safety_checker=None
        )
    ).to(DEVICE)


def save_result(
    results_dir: str,
    images: torch.Tensor,
    results: torch.Tensor,
    title: str = "",
    filename: str = "",
):
    save_dir = os.path.join(results_dir, title, filename)
    os.makedirs(save_dir, exist_ok=True)

    for orig_image, edited_images in zip(images.unbind(dim=0), results.unbind(dim=0)):
        save_image(orig_image, os.path.join(save_dir, "0.png"))

        for i, img in enumerate(edited_images, start=1):
            save_image(img, os.path.join(save_dir, f"{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=UNDERLYING_PIPES.keys(), required=True)

    parser.add_argument("--mb_size", type=int)
    parser.add_argument("--mb_save_topk", type=int, default=20)
    parser.add_argument("--use_factor", action="store_true")

    parser.add_argument("--exp_title", type=str, required=True)
    parser.add_argument(
        "--iter_edit_bench",
        type=str,
        required=True,
        help="Path to the iter edit bench file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to save the results",
    )
    args = parser.parse_args()

    print("Running with the following arguments:")
    pprint(args)

    dataloader = get_dataloader(
        args.iter_edit_bench,
        os.path.join("datasets", "ffhq", "images1024x1024"),
    )

    pipe = get_iterpipe(args.type)

    if args.type == "scfg":
        controller = AttentionStore(
            size=pipe.pipe.unet.sample_size,
            mb_size=0,
            mb_save_topk=0,
        )
        register_attention_controller(pipe.pipe.unet, controller)

        pipe_kwargs = {
            "attn_ctrl": controller,
            "use_scfg": True,
            "use_factor": False,
        }
    elif args.type == "iterdiff":
        if args.mb_size is None:
            raise ValueError("`--mb_size` is required when `--type iterdiff`.")

        controller = AttentionStore(
            size=pipe.pipe.unet.sample_size,
            mb_size=args.mb_size,
            mb_save_topk=args.mb_save_topk,
        )
        register_attention_controller(pipe.pipe.unet, controller)

        pipe_kwargs = {
            "attn_ctrl": controller,
            "use_scfg": True,
            "use_factor": args.use_factor,
        }
    else:  # args.type in ["ip2p", "emilie"]
        pipe_kwargs = {}

    for i, (image, insts) in enumerate(tqdm(dataloader)):
        image = image.to(DEVICE)

        if "attn_ctrl" in pipe_kwargs:
            pipe_kwargs["attn_ctrl"].full_reset()

        results = pipe(
            prompt=insts,
            image=image,
            num_inference_steps=100,
            guidance_scale=7.5,
            image_guidance_scale=1.5,
            generator=torch.Generator().manual_seed(123),
            **pipe_kwargs,
        )

        if args.type == "emilie":
            pipe.pipe.clear_cache()

        save_result(args.results_dir, image, results, args.exp_title, str(i))
