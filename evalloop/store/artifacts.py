"""Content-addressed artifact store.

Large binaries do not belong in Postgres: audio recordings, Parquet tables,
model adapters. They go here, named by the SHA-256 of their own content, which
buys three things for free - identical content is stored exactly once, a URI is
proof of what it points at, and nothing can be modified in place without
becoming a different artifact.
"""

from __future__ import annotations

import hashlib
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

__all__ = ["URI_SCHEME", "ArtifactStore", "LocalArtifactStore", "S3ArtifactStore", "parse_uri"]

URI_SCHEME = "cas"
_CHUNK = 1 << 20


def parse_uri(uri: str) -> str:
    """Extract the content hash from a `cas://<sha256>` URI."""
    prefix = f"{URI_SCHEME}://"
    if not uri.startswith(prefix):
        raise ValueError(f"not an artifact URI: {uri!r} (expected {prefix}<sha256>)")
    digest = uri[len(prefix) :]
    if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
        raise ValueError(f"malformed content hash in {uri!r}")
    return digest


class ArtifactStore(ABC):
    """Put bytes in, get a URI back. Fetch by URI."""

    @abstractmethod
    def put_bytes(self, data: bytes) -> str: ...

    @abstractmethod
    def put_file(self, path: Path) -> str: ...

    @abstractmethod
    def get_bytes(self, uri: str) -> bytes: ...

    @abstractmethod
    def exists(self, uri: str) -> bool: ...


class LocalArtifactStore(ArtifactStore):
    """Local directory backend.

    Files are fanned out two levels by hash prefix (`ab/cd/abcd...`). A single
    flat directory with a million entries is slow to list on every filesystem
    that matters.
    """

    def __init__(self, root: Path | str) -> None:
        # Deliberately no mkdir. Constructing a store is not a write, and a
        # dry run that creates an empty artifacts/ directory has already
        # touched the filesystem it promised not to. Writes create the tree.
        self.root = Path(root)

    def _path_for(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:4] / digest

    def put_bytes(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        target = self._path_for(digest)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp name then rename: a crash mid-write must not leave
            # a truncated file sitting at the address of its full content.
            tmp = target.with_suffix(".partial")
            tmp.write_bytes(data)
            tmp.replace(target)
        return f"{URI_SCHEME}://{digest}"

    def put_file(self, path: Path) -> str:
        digest = self._hash_file(path)
        target = self._path_for(digest)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".partial")
            shutil.copyfile(path, tmp)
            tmp.replace(target)
        return f"{URI_SCHEME}://{digest}"

    def get_bytes(self, uri: str) -> bytes:
        target = self._path_for(parse_uri(uri))
        if not target.exists():
            raise FileNotFoundError(f"artifact not in store: {uri}")
        return target.read_bytes()

    def path_of(self, uri: str) -> Path:
        """Filesystem path for an artifact, for readers that stream (Parquet)."""
        target = self._path_for(parse_uri(uri))
        if not target.exists():
            raise FileNotFoundError(f"artifact not in store: {uri}")
        return target

    def exists(self, uri: str) -> bool:
        return self._path_for(parse_uri(uri)).exists()

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK):
                digest.update(chunk)
        return digest.hexdigest()


class S3ArtifactStore(ArtifactStore):
    """S3 backend. Interface reserved in P0; implemented alongside remote runs.

    Declared now so nothing downstream is written against `LocalArtifactStore`
    concretely and has to be untangled later.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix

    def put_bytes(self, data: bytes) -> str:
        raise NotImplementedError("S3 artifact store is not implemented yet")

    def put_file(self, path: Path) -> str:
        raise NotImplementedError("S3 artifact store is not implemented yet")

    def get_bytes(self, uri: str) -> bytes:
        raise NotImplementedError("S3 artifact store is not implemented yet")

    def exists(self, uri: str) -> bool:
        raise NotImplementedError("S3 artifact store is not implemented yet")
