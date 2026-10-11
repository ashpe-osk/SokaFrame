import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import UploadFile

UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024
_ALLOWED_EXTENSIONS = {".avi", ".mov", ".mp4"}
_MP4_MEDIA_TYPE = "video/mp4"
_MEDIA_TYPES = {
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mp4": _MP4_MEDIA_TYPE,
}


class UnsupportedVideoFormat(Exception):
    pass


class InvalidVideoFile(Exception):
    pass


class UploadTooLarge(Exception):
    pass


@dataclass(frozen=True)
class StoredVideo:
    original_filename: str
    media_type: str
    storage_path: str
    file_size_bytes: int
    sha256: str
    absolute_path: Path

    def remove(self) -> None:
        self.absolute_path.unlink(missing_ok=True)


def _safe_original_filename(filename: str | None) -> str:
    if not filename:
        raise InvalidVideoFile
    basename = PurePosixPath(PureWindowsPath(filename).name).name
    if not basename or basename in {".", ".."} or len(basename) > 255:
        raise InvalidVideoFile
    return basename


def _detect_video_type(extension: str, header: bytes) -> str:
    if extension == ".avi":
        if header[:4] == b"RIFF" and header[8:12] == b"AVI ":
            return _MEDIA_TYPES[extension]
        raise InvalidVideoFile

    if len(header) < 12 or header[4:8] != b"ftyp":
        raise InvalidVideoFile
    brand = header[8:12]
    if extension == ".mov" and brand == b"qt  ":
        return _MEDIA_TYPES[extension]
    if extension == ".mp4" and brand != b"qt  ":
        return _MP4_MEDIA_TYPE
    raise InvalidVideoFile


def _validate_container_length(
    extension: str, header: bytes, file_size_bytes: int
) -> None:
    if extension == ".avi":
        declared_size = int.from_bytes(header[4:8], byteorder="little")
        if declared_size != 0xFFFFFFFF and (
            declared_size < 4 or declared_size + 8 > file_size_bytes
        ):
            raise InvalidVideoFile
        return

    declared_size = int.from_bytes(header[:4], byteorder="big")
    if declared_size < 16 or declared_size == 1 or declared_size > file_size_bytes:
        raise InvalidVideoFile


async def store_video_upload(
    upload: UploadFile, data_dir: Path, max_size_bytes: int
) -> StoredVideo:
    original_filename = _safe_original_filename(upload.filename)
    extension = Path(original_filename).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise UnsupportedVideoFormat
    if upload.size is not None and upload.size > max_size_bytes:
        raise UploadTooLarge

    uploads_dir = data_dir / "uploads"
    temporary_dir = uploads_dir / ".tmp"
    temporary_path: Path | None = None
    try:
        temporary_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = temporary_dir / f"{uuid.uuid4().hex}.part"
        storage_name = f"{uuid.uuid4().hex}{extension}"
        storage_path = (Path("uploads") / storage_name).as_posix()
        final_path = data_dir / Path(storage_path)

        digest = hashlib.sha256()
        file_size = 0
        prefix = bytearray()
        detected_type: str | None = None

        with temporary_path.open("xb") as destination:
            while chunk := await upload.read(UPLOAD_CHUNK_SIZE_BYTES):
                file_size += len(chunk)
                if file_size > max_size_bytes:
                    raise UploadTooLarge

                if detected_type is None:
                    prefix.extend(chunk[: max(0, 12 - len(prefix))])
                    if len(prefix) >= 12:
                        detected_type = _detect_video_type(extension, bytes(prefix))

                destination.write(chunk)
                digest.update(chunk)

            if file_size == 0 or detected_type is None:
                raise InvalidVideoFile
            _validate_container_length(extension, bytes(prefix), file_size)

            destination.flush()
            os.fsync(destination.fileno())

        os.replace(temporary_path, final_path)
        return StoredVideo(
            original_filename=original_filename,
            media_type=detected_type,
            storage_path=storage_path,
            file_size_bytes=file_size,
            sha256=digest.hexdigest(),
            absolute_path=final_path,
        )
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
