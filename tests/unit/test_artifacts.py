"""The content-addressed artifact store."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalloop.store import LocalArtifactStore, S3ArtifactStore, parse_uri

PAYLOAD = b"RIFF....fake wav bytes"


def test_identical_content_is_stored_once(artifact_root: Path) -> None:
    """Content addressing means dedup is free: the same recording referenced by
    a thousand traces occupies one file."""
    store = LocalArtifactStore(artifact_root)
    a = store.put_bytes(PAYLOAD)
    b = store.put_bytes(PAYLOAD)
    assert a == b
    assert sum(1 for _ in artifact_root.rglob("*") if _.is_file()) == 1


def test_different_content_gets_a_different_uri(artifact_root: Path) -> None:
    store = LocalArtifactStore(artifact_root)
    assert store.put_bytes(b"one") != store.put_bytes(b"two")


def test_round_trip(artifact_root: Path) -> None:
    store = LocalArtifactStore(artifact_root)
    uri = store.put_bytes(PAYLOAD)
    assert store.get_bytes(uri) == PAYLOAD
    assert store.exists(uri)


def test_put_file_matches_put_bytes(artifact_root: Path, tmp_path: Path) -> None:
    """A URI is a claim about content, so the same bytes must address the same
    way whether they arrived in memory or on disk."""
    source = tmp_path / "call.wav"
    source.write_bytes(PAYLOAD)
    store = LocalArtifactStore(artifact_root)
    assert store.put_file(source) == store.put_bytes(PAYLOAD)


def test_path_of_streams_without_loading(artifact_root: Path) -> None:
    """Parquet readers want a path, not bytes - a results table can be larger
    than memory."""
    store = LocalArtifactStore(artifact_root)
    uri = store.put_bytes(PAYLOAD)
    assert store.path_of(uri).read_bytes() == PAYLOAD


def test_hash_prefix_fan_out(artifact_root: Path) -> None:
    """A flat directory with a million entries is slow to list everywhere."""
    store = LocalArtifactStore(artifact_root)
    digest = parse_uri(store.put_bytes(PAYLOAD))
    assert (artifact_root / digest[:2] / digest[2:4] / digest).exists()


def test_missing_artifact_raises(artifact_root: Path) -> None:
    store = LocalArtifactStore(artifact_root)
    with pytest.raises(FileNotFoundError):
        store.get_bytes(f"cas://{'0' * 64}")


def test_no_partial_files_left_behind(artifact_root: Path) -> None:
    """Written to a temp name then renamed, so a crash mid-write cannot leave a
    truncated file sitting at the address of its complete content."""
    store = LocalArtifactStore(artifact_root)
    store.put_bytes(PAYLOAD)
    assert list(artifact_root.rglob("*.partial")) == []


@pytest.mark.parametrize(
    "uri",
    ["s3://bucket/key", "cas://short", f"cas://{'z' * 64}", "cas://", "", f"{'a' * 64}"],
)
def test_malformed_uri_rejected(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_uri(uri)


def test_s3_backend_is_declared_but_not_implemented() -> None:
    """Declared in P0 so nothing downstream binds to LocalArtifactStore
    concretely and has to be untangled when remote runs land."""
    with pytest.raises(NotImplementedError):
        S3ArtifactStore("bucket").put_bytes(b"x")


def test_path_of_missing_artifact_raises(artifact_root: Path) -> None:
    store = LocalArtifactStore(artifact_root)
    with pytest.raises(FileNotFoundError):
        store.path_of(f"cas://{'1' * 64}")


def test_put_file_is_idempotent(artifact_root: Path, tmp_path: Path) -> None:
    """Re-uploading an unchanged recording must not rewrite the file - the store
    holds thousands and a needless copy is pure IO."""
    source = tmp_path / "call.wav"
    source.write_bytes(PAYLOAD)
    store = LocalArtifactStore(artifact_root)
    uri = store.put_file(source)
    before = store.path_of(uri).stat().st_mtime_ns
    assert store.put_file(source) == uri
    assert store.path_of(uri).stat().st_mtime_ns == before


@pytest.mark.parametrize("method", ["put_file", "get_bytes", "exists"])
def test_every_s3_method_is_explicitly_unimplemented(method: str, tmp_path: Path) -> None:
    """An accidentally-working stub is worse than a raise: it would silently
    return nothing and look like an empty store."""
    store = S3ArtifactStore("bucket", prefix="p/")
    arg = tmp_path if method == "put_file" else f"cas://{'a' * 64}"
    with pytest.raises(NotImplementedError):
        getattr(store, method)(arg)
