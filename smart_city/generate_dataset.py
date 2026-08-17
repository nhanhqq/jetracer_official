#!/usr/bin/env python3
"""Generate a reproducible Smart City detector dataset.

It uses the supplied sign assets and road frames, and adds explicit
crosswalk/stop-line boxes so the runtime can recognize an intersection rather
than treating every white lane boundary as one.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


ROOT = Path(__file__).resolve().parent
CLASSES = ("bien_cam", "di_thang", "re_phai", "re_trai", "den_do", "den_xanh",
           "crosswalk", "stop_line")
SIGN_CLASSES = CLASSES[:4]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def canvas(image: Image.Image, size: int, rng: random.Random) -> Image.Image:
    scale = max(size / image.width, size / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left, top = (image.width - size) // 2, (image.height - size) // 2
    image = image.crop((left, top, left + size, top + size)).convert("RGB")
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(.75, 1.2))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(.82, 1.18))
    if rng.random() < .15:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.2, .7)))
    return image


def paste(canvas_image: Image.Image, obj: Image.Image, cls: int, rng: random.Random,
          upper: bool = True):
    size = canvas_image.width
    max_x = max(0, size - obj.width)
    max_y = max(0, (size // 2 if upper else size - obj.height) - obj.height)
    y_min = 5 if upper else int(size * .52)
    y = rng.randint(y_min, max(y_min, max_y))
    x = rng.randint(0, max_x)
    canvas_image.paste(obj, (x, y), obj if obj.mode == "RGBA" else None)
    return cls, (x + obj.width / 2) / size, (y + obj.height / 2) / size, obj.width / size, obj.height / size


def marking(size: int, cls: int, rng: random.Random):
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if cls == 6:
        y0 = rng.randint(int(size * .55), int(size * .72))
        stripe_h = rng.randint(4, 8)
        for i in range(rng.randint(4, 7)):
            y = y0 + i * (stripe_h + rng.randint(3, 7))
            draw.rectangle((int(size * .08), y, int(size * .92), y + stripe_h), fill=(245, 245, 245, 220))
        return image, (int(size * .50), y0 + (i + 1) * (stripe_h + 5) / 2, int(size * .84), (i + 1) * (stripe_h + 5))
    y = rng.randint(int(size * .58), int(size * .78))
    height = rng.randint(4, 8)
    draw.rectangle((int(size * .08), y, int(size * .92), y + height), fill=(250, 250, 250, 230))
    return image, (int(size * .50), y + height / 2, int(size * .84), height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roads", type=Path, default=ROOT.parent / "notebook3/old_codes/road_following_A/apex")
    parser.add_argument("--signs", type=Path, default=ROOT.parent / "notebook3/bien_bao")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/traffic")
    parser.add_argument("--train-images", type=int, default=552)
    parser.add_argument("--val-images", type=int, default=138)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2608)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    roads = sorted(p for p in args.roads.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not roads:
        raise SystemExit("No road frames: %s" % args.roads)
    sign_sources = [Image.open(args.signs / (name + ".png")).convert("RGBA") for name in SIGN_CLASSES]
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
        for old in (args.output / "images" / split).glob("*.jpg"):
            old.unlink()
        for old in (args.output / "labels" / split).glob("*.txt"):
            old.unlink()
    counts = [0] * len(CLASSES)
    total = args.train_images + args.val_images
    for index in range(total):
        split = "train" if index < args.train_images else "val"
        image = canvas(Image.open(rng.choice(roads)), args.imgsz, rng)
        labels = []
        choices = rng.sample(range(len(CLASSES)), k=rng.randint(1, 3))
        for cls in choices:
            if cls < 4:
                diameter = rng.randint(round(args.imgsz * .07), round(args.imgsz * .18))
                source = sign_sources[cls]
                obj = source.resize((diameter, max(8, round(diameter * source.height / source.width))), Image.Resampling.LANCZOS)
                labels.append(paste(image, obj, cls, rng, upper=True))
            elif cls < 6:
                diameter = rng.randint(round(args.imgsz * .07), round(args.imgsz * .15))
                obj = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
                d = ImageDraw.Draw(obj); d.ellipse((1, 1, diameter - 2, diameter - 2), fill=(25, 25, 25, 230))
                colour = (235, 25, 25, 255) if cls == 4 else (20, 220, 45, 255)
                d.ellipse((diameter // 5, diameter // 5, diameter * 4 // 5, diameter * 4 // 5), fill=colour)
                labels.append(paste(image, obj, cls, rng, upper=True))
            else:
                obj, box = marking(args.imgsz, cls, rng)
                image.alpha_composite(obj) if image.mode == "RGBA" else image.paste(obj, (0, 0), obj)
                cx, cy, bw, bh = box
                labels.append((cls, cx / args.imgsz, cy / args.imgsz, bw / args.imgsz, bh / args.imgsz))
            counts[cls] += 1
        stem = "smart_city_%05d" % index
        image.save(args.output / "images" / split / (stem + ".jpg"), quality=92)
        (args.output / "labels" / split / (stem + ".txt")).write_text(
            "\n".join("%d %.6f %.6f %.6f %.6f" % row for row in labels) + "\n", encoding="utf-8")
    (args.output / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n" % args.output.resolve() +
        "".join("  %d: %s\n" % (i, name) for i, name in enumerate(CLASSES)), encoding="utf-8")
    print("generated=%d train=%d val=%d classes=%s" % (total, args.train_images, args.val_images, dict(zip(CLASSES, counts))))


if __name__ == "__main__":
    main()

