from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from app.adapters.qwen_vl import MODEL_ID as QWEN_MODEL_ID
from app.settings import Settings

IMAGE_MODEL_ID = "sentence-transformers/clip-ViT-B-32"
TEXT_MODEL_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="显式下载 Seek 的本地向量模型")
    parser.add_argument("--model", choices=("qwen", "clip"), default="qwen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_environment()
    settings.qwen_model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.model == "qwen":
        print(f"下载本地多模态向量模型：{QWEN_MODEL_ID}")
        snapshot_download(repo_id=QWEN_MODEL_ID, local_dir=settings.qwen_model_path)
        print(f"模型已保存到：{settings.qwen_model_path}")
    else:
        print(f"下载本地图像模型：{IMAGE_MODEL_ID}")
        snapshot_download(repo_id=IMAGE_MODEL_ID, local_dir=settings.image_model_path)
        print(f"下载本地多语言文本模型：{TEXT_MODEL_ID}")
        snapshot_download(repo_id=TEXT_MODEL_ID, local_dir=settings.text_model_path)
        print(f"模型已保存到：{settings.image_model_path.parent}")


if __name__ == "__main__":
    main()
