import argparse
import json
import os
from itertools import pairwise

import ImageReward as RM
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image.lpips import _lpips_update, _NoTrainLpips
from torchmetrics.functional.multimodal.clip_score import _get_clip_model_and_processor
from torchvision.io import ImageReadMode, read_image
from tqdm.auto import tqdm

torch.enable_grad(False)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IterEditDataset(Dataset):
    def __init__(self, instruction_file: str, result_dir: str, transform=None):
        with open(instruction_file) as f:
            self.instructions: list[dict[str, str]] = json.load(f)

        self.image_dir = result_dir
        self.transform = transform

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx: int):
        _, insts = self.instructions[idx].values()
        image_dir = os.path.join(self.image_dir, str(idx))
        image_filenames = sorted(
            os.listdir(image_dir), key=lambda filename: int(os.path.splitext(filename)[0])
        )
        image_paths = [
            os.path.join(image_dir, image_filename)
            for image_filename in image_filenames
        ]

        images = torch.stack(
            [read_image(image_path, ImageReadMode.RGB) for image_path in image_paths],
            dim=0,
        )
        if self.transform:
            images = self.transform(images)
        return images, insts, image_paths


def collate_fn(batch):
    images, insts, paths = zip(*batch)
    images = torch.stack(images)
    return images, insts, paths


def gen_sample(results_dir: str, exp_name: str, dataloader: DataLoader):
    target_dir = os.path.join(results_dir, exp_name, "samples")
    os.makedirs(target_dir, exist_ok=True)

    for i, (images, insts, _) in enumerate(tqdm(dataloader)):
        images = images[0] / 255.0
        insts = ["original"] + insts[0]

        fig, ax = plt.subplots(1, len(images), figsize=(20, 10))
        for j, (image, inst) in enumerate(zip(images, insts)):
            ax[j].imshow(image.permute(1, 2, 0).cpu().numpy())
            ax[j].set_title(inst)
            ax[j].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(target_dir, f"{i}.png"), bbox_inches="tight")
        plt.close(fig)


@torch.inference_mode()
def clip_i_metric(dataloader: DataLoader):
    model, processor = _get_clip_model_and_processor("openai/clip-vit-large-patch14")
    model = model.to(DEVICE)

    clip_score = [0.0 for _ in range(5)]

    for images, _, _ in tqdm(dataloader):
        images = images[0].to(DEVICE)

        for i, (prev_image, edited_image) in enumerate(pairwise(images)):
            processed_input = processor(
                images=[prev_image.cpu(), edited_image.cpu()],
                return_tensors="pt",
                padding=True,
            )
            img_features = model.get_image_features(
                processed_input["pixel_values"].to(DEVICE)
            )
            prev_features, edit_features = img_features.unbind()
            score = F.cosine_similarity(prev_features, edit_features, dim=-1).mean()
            clip_score[i] += score.item()

    return list(map(lambda x: x / len(dataloader), clip_score))


@torch.inference_mode()
def lpips_metric(dataloader: DataLoader):
    model = _NoTrainLpips(net="vgg")
    model = model.to(DEVICE)

    lpips_score = [0.0 for _ in range(5)]

    for images, _, _ in tqdm(dataloader):
        images = images[0].to(DEVICE)
        images = images / 255.0

        for i, (prev_image, edited_image) in enumerate(pairwise(images)):
            loss, _ = _lpips_update(
                prev_image.unsqueeze(0),
                edited_image.unsqueeze(0),
                net=model,
                normalize=True,
            )
            lpips_score[i] += loss.item()

    return list(map(lambda x: x / len(dataloader), lpips_score))


@torch.inference_mode()
def image_reward_metric(data_loader: DataLoader):
    model = RM.load("ImageReward-v1.0", device=DEVICE)

    ir_score = [0.0 for _ in range(6)]

    for _, insts, paths in tqdm(data_loader):
        insts = ["A human face."] + [
            f'A human face edited with prompt: "{inst}"' for inst in insts[0]
        ]
        paths = paths[0]

        for i, (inst, path) in enumerate(zip(insts, paths)):
            ir_score[i] += model.score(inst, [path])

    return list(map(lambda x: x / len(data_loader), ir_score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iter_edit_bench",
        type=str,
        help="Path to the iter edit bench file",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Path to save the results",
    )
    parser.add_argument("--exp", "-e", type=str)
    args = parser.parse_args()

    dataset = IterEditDataset(
        instruction_file=args.iter_edit_bench,
        result_dir=os.path.join(args.results_dir, args.exp),
        transform=T.Compose(
            [
                T.Resize(512, antialias=True),
                T.ToDtype(torch.float32, scale=False),
            ]
        ),
    )
    # image in [0.0, 255.0]

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    gen_sample(args.results_dir, args.exp, dataloader)

    metrics = {
        # large is better
        "clip_i": clip_i_metric(dataloader),
        # small is better
        "lpips": lpips_metric(dataloader),
        # large is better
        "image_reward": image_reward_metric(dataloader),
    }

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if os.path.exists(os.path.join(args.results_dir, "metrics.json")):
        with open(os.path.join(args.results_dir, "metrics.json"), "r") as f:
            data: dict = json.load(f)
    else:
        data = {}

    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        data[args.exp] = metrics
        json.dump(data, f, indent=2)
