"""Download one Hugging Face model revision into a target directory.

Called by 07-models.ps1 from the Tabby venv (it has huggingface_hub). Kept tiny
and dependency-light on purpose: one repo, one revision, one destination. It is
idempotent because snapshot_download reuses the local cache and only fetches
missing or changed files, and we skip entirely when the destination already holds
a config.json (a populated EXL2 dir always has one).

Usage:
    python hf-download.py --repo turboderp/Qwen3-32B-exl2 --revision 4.0bpw --dest C:/models/exl2/Qwen3-32B-exl2-4.0bpw
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download


def main() -> int:
    ap = argparse.ArgumentParser(description="Download an HF model revision to a directory.")
    ap.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. turboderp/Qwen3-32B-exl2")
    ap.add_argument("--revision", default=None, help="branch, tag, or commit (EXL2 quants use the bpw branch)")
    ap.add_argument("--dest", required=True, help="directory to materialize the snapshot into")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.dest, "config.json")):
        print(f"already present: {args.dest}")
        return 0

    os.makedirs(args.dest, exist_ok=True)
    print(f"downloading {args.repo}@{args.revision or 'main'} -> {args.dest}")
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=args.dest,
        # local_dir_use_symlinks defaults sensibly on recent hub versions; weights
        # land as real files so TabbyAPI can mmap them without chasing the cache.
    )
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
