"""Local-cache-first model loading (same shape as the drug assistant's
model_manager.py, which does this for its spaCy pipeline).

The semantic search's embedding model (RETRIEVAL_EMBEDDING_ROUTE "hf:<repo>")
is a Hugging Face Hub repo. It is downloaded once into
RETRIEVAL_EMBEDDING_MODEL_DIR/<repo slug>/ and loaded from that local path
afterwards, so a restart makes no network call (sentence-transformers would
otherwise query the Hub on every load, and fail slowly when offline).

Completion is recorded by a marker file, written only after the download
finished. The marker is trusted only together with a live check that the
model files are really there: a marker left behind without its files (a
fresh container reusing an old volume, a half-deleted folder, the directory
accidentally committed to git) would otherwise skip a download that is
still needed.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Presence of this marker is what "already downloaded" means -- see
# is_model_present for why it is re-checked against the files themselves.
_DOWNLOAD_COMPLETE_MARKER = ".download_complete"
_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
# Hub repos often carry the same weights in several formats (the default
# all-MiniLM-L6-v2 repo is about 1 GB with its ONNX / OpenVINO / TF / Rust
# copies); sentence-transformers only loads the PyTorch ones (about 90 MB).
_SKIP_PATTERNS = ["onnx/*", "openvino/*", "coreml/*", "*.onnx", "*.h5", "*.ot", "*.msgpack", "*.tflite",
                  "*.mlmodel", "*.xml"]


def model_slug(repo_id: str) -> str:
    """'sentence-transformers/all-MiniLM-L6-v2' -> 'sentence-transformers--all-minilm-l6-v2'."""
    return "--".join(re.sub(r"[^a-zA-Z0-9.]+", "-", part).strip("-") for part in repo_id.split("/")).lower()


def is_model_present(model_dir: str | Path) -> bool:
    path = Path(model_dir)
    return ((path / _DOWNLOAD_COMPLETE_MARKER).is_file() and (path / "config.json").is_file()
            and any((path / w).is_file() for w in _WEIGHT_FILES))


def ensure_hf_model_downloaded(repo_id: str, model_dir: str | Path) -> Path:
    """Return the local folder holding `repo_id`, downloading it only if it
    isn't there yet. Raises if the download fails (no network, bad repo id)."""
    path = Path(model_dir)
    if is_model_present(path):
        return path

    from huggingface_hub import list_repo_files, snapshot_download

    logger.info("Embedding model '%s' not found in %s, downloading...", repo_id, path)
    path.mkdir(parents=True, exist_ok=True)
    skip = list(_SKIP_PATTERNS)
    if "model.safetensors" in list_repo_files(repo_id):
        skip.append("pytorch_model.bin")          # the same weights twice
    snapshot_download(repo_id=repo_id, local_dir=str(path), ignore_patterns=skip)
    (path / _DOWNLOAD_COMPLETE_MARKER).touch()
    logger.info("Embedding model '%s' downloaded to %s", repo_id, path)
    return path
