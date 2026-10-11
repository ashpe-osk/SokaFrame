import asyncio
import hashlib
import io
from pathlib import Path

import pytest
from fastapi import UploadFile

from app.services.video_storage import (
    UPLOAD_CHUNK_SIZE_BYTES,
    InvalidVideoFile,
    UploadTooLarge,
    store_video_upload,
)


def test_video_upload_is_chunked_and_checksum_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"\x00\x00\x00\x10ftypisom" + b"\x00" * UPLOAD_CHUNK_SIZE_BYTES
    upload = UploadFile(filename="match.mp4", file=io.BytesIO(content))
    read_sizes: list[int] = []
    original_read = UploadFile.read

    async def tracked_read(self: UploadFile, size: int = -1) -> bytes:
        if self is upload:
            read_sizes.append(size)
        return await original_read(self, size)

    monkeypatch.setattr(UploadFile, "read", tracked_read)
    stored = asyncio.run(store_video_upload(upload, tmp_path, len(content) + 1))

    assert len(content) > UPLOAD_CHUNK_SIZE_BYTES
    assert read_sizes and all(size == UPLOAD_CHUNK_SIZE_BYTES for size in read_sizes)
    assert stored.absolute_path.read_bytes() == content
    assert stored.sha256 == hashlib.sha256(content).hexdigest()
    assert stored.media_type == "video/mp4"
    assert stored.storage_path.startswith("uploads/")
    assert stored.absolute_path.parent == tmp_path / "uploads"


def test_invalid_video_header_removes_temporary_upload(tmp_path: Path) -> None:
    upload = UploadFile(filename="match.mp4", file=io.BytesIO(b"not a video"))

    with pytest.raises(InvalidVideoFile):
        asyncio.run(store_video_upload(upload, tmp_path, 1024))

    temporary_directory = tmp_path / "uploads" / ".tmp"
    assert list(temporary_directory.iterdir()) == []
    assert list((tmp_path / "uploads").glob("*.mp4")) == []


def test_actual_stream_size_limit_cleans_up_temporary_upload(tmp_path: Path) -> None:
    upload = UploadFile(filename="match.mp4", file=io.BytesIO(b"\x00\x00\x00\x10ftypisom"))
    upload.size = 1

    with pytest.raises(UploadTooLarge):
        asyncio.run(store_video_upload(upload, tmp_path, 8))

    temporary_directory = tmp_path / "uploads" / ".tmp"
    assert list(temporary_directory.iterdir()) == []
    assert list((tmp_path / "uploads").glob("*.mp4")) == []


def test_video_signatures_cover_supported_formats(tmp_path: Path) -> None:
    samples = (
        ("match.mp4", b"\x00\x00\x00\x10ftypisom" + b"\x00" * 4, "video/mp4"),
        ("match.mov", b"\x00\x00\x00\x10ftypqt  " + b"\x00" * 4, "video/quicktime"),
        ("match.avi", b"RIFF\x04\x00\x00\x00AVI ", "video/x-msvideo"),
    )

    for filename, content, media_type in samples:
        upload = UploadFile(filename=filename, file=io.BytesIO(content))
        stored = asyncio.run(store_video_upload(upload, tmp_path, 1024))
        assert stored.media_type == media_type
