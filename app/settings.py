from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    catalog_path: Path
    cover_dir: Path
    index_dir: Path
    image_model_path: Path
    text_model_path: Path
    qwen_model_path: Path
    database_path: Path
    data_source: str
    embedding_model: str
    embedding_dimension: int
    qwen_max_pixels: int
    device: str

    @classmethod
    def from_environment(cls) -> Settings:
        project_root = Path(os.environ.get("SEEK_PROJECT_ROOT", PROJECT_ROOT)).resolve()
        testdata_dir = Path(
            os.environ.get("SEEK_TESTDATA_DIR", project_root / "testdata")
        ).resolve()
        model_dir = Path(os.environ.get("SEEK_MODEL_DIR", project_root / "models")).resolve()
        data_dir = Path(os.environ.get("SEEK_DATA_DIR", project_root / "data")).resolve()
        data_source = os.environ.get("SEEK_DATA_SOURCE", "sqlite")
        default_cover_dir = data_dir / "covers" if data_source == "sqlite" else testdata_dir / "covers"
        return cls(
            project_root=project_root,
            catalog_path=Path(
                os.environ.get("SEEK_CATALOG_PATH", testdata_dir / "catalog.json")
            ).resolve(),
            cover_dir=Path(os.environ.get("SEEK_COVER_DIR", default_cover_dir)).resolve(),
            index_dir=Path(os.environ.get("SEEK_INDEX_DIR", project_root / "indexes")).resolve(),
            image_model_path=Path(
                os.environ.get("SEEK_IMAGE_MODEL_PATH", model_dir / "clip-vit-b-32")
            ).resolve(),
            text_model_path=Path(
                os.environ.get(
                    "SEEK_TEXT_MODEL_PATH", model_dir / "clip-vit-b-32-multilingual-v1"
                )
            ).resolve(),
            qwen_model_path=Path(
                os.environ.get(
                    "SEEK_QWEN_MODEL_PATH", model_dir / "qwen3-vl-embedding-2b"
                )
            ).resolve(),
            database_path=Path(
                os.environ.get("SEEK_DATABASE_PATH", data_dir / "seek.sqlite3")
            ).resolve(),
            data_source=data_source,
            embedding_model=os.environ.get("SEEK_EMBEDDING_MODEL", "qwen"),
            embedding_dimension=int(os.environ.get("SEEK_EMBEDDING_DIMENSION", "2048")),
            qwen_max_pixels=int(os.environ.get("SEEK_QWEN_MAX_PIXELS", str(512 * 512))),
            device=os.environ.get("SEEK_DEVICE", "auto"),
        )
