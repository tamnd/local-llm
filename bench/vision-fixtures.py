#!/usr/bin/env python3
"""Generate the deterministic images the vision arm of llmbench sweeps.

The vision cost that matters on this box is set by resolution, not by subject:
the projector turns an image into a fixed number of tokens per tile, so a 2048px
image costs roughly 4x a 1024px one regardless of what it depicts. These
fixtures therefore vary only in size, and are drawn from a fixed seed so a rerun
on another machine produces byte-identical files and comparable prompt_tokens.

Each image also carries a word and a shape count, which gives the run a cheap
correctness check: a server that loaded the projector reads them back, and one
that silently dropped the image guesses. That distinction is the whole point of
the control in benchVisionCell, and it is worth being able to confirm by eye.

Usage:  python3 bench/vision-fixtures.py [outdir]   (default bench/fixtures)
"""

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZES = [512, 1024, 2048]
WORD = "GLIMMER"
SHAPES = 12
SEED = 2065  # the spec number, so the choice is not arbitrary


def draw(size: int) -> Image.Image:
    rng = random.Random(SEED)
    img = Image.new("RGB", (size, size), (18, 20, 28))
    d = ImageDraw.Draw(img)

    # A grid, so a downscaling projector has high-frequency detail to lose.
    step = max(size // 16, 8)
    for x in range(0, size, step):
        d.line([(x, 0), (x, size)], fill=(32, 36, 48), width=1)
        d.line([(0, x), (size, x)], fill=(32, 36, 48), width=1)

    # A fixed count of shapes: colours and positions come from the seeded rng, so
    # they are stable across runs and across resolutions.
    for _ in range(SHAPES):
        cx, cy = rng.randrange(size), rng.randrange(size)
        r = rng.randrange(size // 20, size // 8)
        colour = (rng.randrange(80, 255), rng.randrange(80, 255), rng.randrange(80, 255))
        if rng.random() < 0.5:
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)
        else:
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=colour)

    # The readback word, drawn large enough to survive the projector's downscale
    # at every size in SIZES.
    box = size // 6
    d.rectangle([size // 2 - box * 2, size // 2 - box // 2, size // 2 + box * 2, size // 2 + box // 2], fill=(250, 250, 250))
    d.text((size // 2 - box * 2 + box // 4, size // 2 - box // 4), WORD, fill=(10, 10, 10))
    return img


def main() -> int:
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/fixtures")
    outdir.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        path = outdir / f"vision-{size}.png"
        draw(size).save(path, optimize=True)
        print(f"{path}  {path.stat().st_size // 1024} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
