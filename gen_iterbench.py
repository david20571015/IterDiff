import argparse
import json
import os
import random
from functools import partial

INSTRUCTIONS = {
    "Make the face look {}.": [  # age
        "older",
        "younger",
        "more mature",
        "childlike",
    ],
    "Change the gender to {}.": [  # gender
        "male",
        "female",
    ],
    "Make the person's skin {}.": [  # skin tone
        "darker",
        "lighter",
        "paler",
        "tanned",
    ],
    "Give {} appearance.": [  # ethnicity/race
        "an Asian",
        "an Indian",
        "a Middle Eastern",
        "a Western European",
        "an African",
    ],
    "Change the hair color to {}.": [  # hair color
        "brown",
        "blonde",
        "black",
        "red",
        "gray",
    ],
    "Make the hair {}.": [  # hair style
        "shorter",
        "longer",
        "curlier",
        "straighter",
    ],
    "Put on {}.": [  # accessories
        "glasses",
        "sunglasses",
        "a cap",
        "a scarf",
    ],
    "Add {} to the face.": [  # facial hair
        "a beard",
        "a mustache",
        "a light stubble",
        "a goatee",
    ],
}


def sample_editing_instructions(num_instructions=1):
    instructions: list[str] = []
    picked_templates = []
    for _ in range(num_instructions):
        while (
            template := random.choice(list(INSTRUCTIONS.keys()))
        ) in picked_templates:
            pass
        picked_templates.append(template)

        attribute = random.choice(INSTRUCTIONS[template])
        instructions.append(template.format(attribute))
    return instructions


def generate_samples(image_paths: list[str], num_samples=20, num_instructions=5):
    samples = [
        {
            "image_path": image_path,
            "instructions": sample_editing_instructions(num_instructions),
        }
        for image_path in random.sample(image_paths, num_samples)
    ]
    return samples


def list_images(image_dir: str):
    def is_image(file: str):
        return file.endswith((".jpg", ".jpeg", ".png"))

    images = []
    for root, _, files in os.walk(image_dir):
        base = root.removeprefix(image_dir + os.sep)
        add_base_dir = partial(os.path.join, base)
        images.extend(map(add_base_dir, filter(is_image, files)))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="iterbench",
        help="Path to save the generated instructions, which will be saved as a json file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=os.path.join("datasets", "ffhq", "images1024x1024"),
        help="Path to the directory containing the images",
    )
    parser.add_argument(
        "-i",
        "--num-instructions",
        type=int,
        default=5,
        help="Number of editing instructions to generate",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for reproducibility",
    )
    args = parser.parse_args()

    image_paths = list_images(args.image_dir)

    random.seed(args.seed)
    instructions = generate_samples(
        image_paths, args.num_samples, args.num_instructions
    )

    with open(f"{args.output}.json", "w") as f:
        json.dump(instructions, f, indent=4)


if __name__ == "__main__":
    main()
