"""Local-cache-first embedding model download: the marker counts only
together with the model files, and the download runs only when needed."""

import sys
import types

import pytest

import model_manager
from llm.embeddings import hf_model_dir


@pytest.fixture
def fake_hub(monkeypatch):
    calls = []

    def snapshot_download(repo_id, local_dir, ignore_patterns=()):
        calls.append((repo_id, local_dir, list(ignore_patterns)))
        from pathlib import Path
        Path(local_dir, "config.json").write_text("{}")
        Path(local_dir, "model.safetensors").write_bytes(b"weights")

    def list_repo_files(repo_id):
        return ["config.json", "model.safetensors", "pytorch_model.bin", "onnx/model.onnx"]

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=snapshot_download, list_repo_files=list_repo_files))
    return calls


def test_downloads_once_then_uses_the_local_folder(tmp_path, fake_hub):
    folder = tmp_path / "m"
    assert model_manager.ensure_hf_model_downloaded("org/model", folder) == folder
    assert model_manager.is_model_present(folder) and len(fake_hub) == 1
    model_manager.ensure_hf_model_downloaded("org/model", folder)
    assert len(fake_hub) == 1                           # no second download
    skipped = fake_hub[0][2]
    assert "onnx/*" in skipped and "pytorch_model.bin" in skipped      # PyTorch safetensors only


def test_marker_without_files_is_not_trusted(tmp_path, fake_hub):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / ".download_complete").touch()             # e.g. a stale volume or a committed marker
    assert not model_manager.is_model_present(folder)
    model_manager.ensure_hf_model_downloaded("org/model", folder)
    assert len(fake_hub) == 1 and model_manager.is_model_present(folder)


def test_files_without_marker_are_not_trusted(tmp_path, fake_hub):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "config.json").write_text("{}")          # an interrupted download
    assert not model_manager.is_model_present(folder)


def test_failed_download_leaves_no_marker(tmp_path, monkeypatch):
    def broken(repo_id, local_dir, ignore_patterns=()):
        raise OSError("no network")

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=broken, list_repo_files=lambda repo_id: []))
    folder = tmp_path / "m"
    with pytest.raises(OSError):
        model_manager.ensure_hf_model_downloaded("org/model", folder)
    assert not (folder / ".download_complete").exists()


def test_model_folder_is_under_the_configured_dir():
    path = hf_model_dir("sentence-transformers/all-MiniLM-L6-v2")
    assert path.name == "sentence-transformers--all-minilm-l6-v2" and path.parent.name == "embeddings"
