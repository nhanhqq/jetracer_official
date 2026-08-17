"""Create a reproducible synthetic YOLO detection dataset for JetRacer.

The four transparent traffic-sign PNGs are composited onto real road-following
frames. Red/green traffic lights are rendered as small, filled circles. Objects
are deliberately restricted to the upper half of each frame.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parent
DEFAULT_ROADS = ROOT / "old_codes" / "road_following_A" / "apex"
DEFAULT_SIGNS = ROOT / "bien_bao"
DEFAULT_OUTPUT = ROOT / "traffic_sign_dataset"
CLASSES = ("bien_cam", "di_thang", "re_phai", "re_trai", "den_do", "den_xanh")
SIGN_FILES = tuple(f"{name}.png" for name in CLASSES[:4])
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fit_canvas(image: Image.Image, size: int) -> Image.Image:
    """Resize with cover semantics, then center-crop to a square."""
    scale = max(size / image.width, size / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size)).convert("RGB")


def augment_background(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.72, 1.25))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.80, 1.20))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.78, 1.20))
    if rng.random() < 0.18:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
    return image


def make_light(diameter: int, color: str, rng: random.Random) -> Image.Image:
    pad = max(2, diameter // 10)
    tile = Image.new("RGBA", (diameter + 2 * pad, diameter + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    # A dark rim makes the colored disc remain visible on bright road scenes.
    draw.ellipse((1, 1, tile.width - 2, tile.height - 2), fill=(25, 25, 25, 230))
    rgb = (235, 25, 25, 255) if color == "red" else (20, 220, 45, 255)
    inset = pad
    draw.ellipse((inset, inset, tile.width - inset - 1, tile.height - inset - 1), fill=rgb)
    if rng.random() < 0.55:
        hi = max(2, diameter // 6)
        draw.ellipse((inset + 2, inset + 2, inset + hi, inset + hi), fill=(255, 255, 255, 105))
    return tile


def paste_object(
    canvas: Image.Image, obj: Image.Image, class_id: int, rng: random.Random
) -> tuple[int, float, float, float, float]:
    width, height = canvas.size
    # Keep objects fully inside the upper half with a small safety margin.
    max_x = max(0, width - obj.width)
    max_y = max(0, min(height // 2 - obj.height, int(height * 0.46) - obj.height))
    x = rng.randint(0, max_x)
    y = rng.randint(max(0, int(height * 0.02)), max(max(0, int(height * 0.02)), max_y))
    canvas.paste(obj, (x, y), obj)
    xc = (x + obj.width / 2) / width
    yc = (y + obj.height / 2) / height
    return class_id, xc, yc, obj.width / width, obj.height / height


def generate(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    road_paths = sorted(p for p in args.roads.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not road_paths:
        raise FileNotFoundError(f"No road images found in {args.roads}")
    sign_sources = [Image.open(args.signs / name).convert("RGBA") for name in SIGN_FILES]

    if args.output.exists():
        shutil.rmtree(args.output)
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True)
        (args.output / "labels" / split).mkdir(parents=True)

    class_counts = [0] * len(CLASSES)
    total = args.train_images + args.val_images
    for index in range(total):
        split = "train" if index < args.train_images else "val"
        background = fit_canvas(Image.open(rng.choice(road_paths)), args.imgsz)
        background = augment_background(background, rng)
        labels = []

        object_count = rng.randint(1, args.max_objects)
        chosen = rng.sample(range(len(CLASSES)), k=min(object_count, len(CLASSES)))
        for class_id in chosen:
            diameter = rng.randint(round(args.imgsz * 0.07), round(args.imgsz * 0.19))
            if class_id < 4:
                source = sign_sources[class_id]
                ratio = source.height / source.width
                obj = source.resize((diameter, max(8, round(diameter * ratio))), Image.Resampling.LANCZOS)
                angle = rng.uniform(-9, 9)
                obj = obj.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
            else:
                obj = make_light(diameter, "red" if class_id == 4 else "green", rng)
            label = paste_object(background, obj, class_id, rng)
            labels.append(label)
            class_counts[class_id] += 1

        stem = f"synthetic_{index:05d}"
        background.save(args.output / "images" / split / f"{stem}.jpg", quality=92)
        text = "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels)
        (args.output / "labels" / split / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")

    yaml_text = (
        f"path: {args.output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )
    (args.output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Created {args.train_images} train + {args.val_images} val images at {args.output}")
    print("Class instances:", dict(zip(CLASSES, class_counts)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROADS)
    parser.add_argument("--signs", type=Path, default=DEFAULT_SIGNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-images", type=int, default=552)
    parser.add_argument("--val-images", type=int, default=138)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2608)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
