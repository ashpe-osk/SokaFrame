"""Atomic storage for trusted local CV pipeline checkpoints."""

import os
import pickle
import tempfile
import time
from pathlib import Path


CHECKPOINT_VERSION = 1
_REPLACE_RETRY_SECONDS = 0.1
_REPLACE_RETRY_ATTEMPTS = 10
_EXECUTION_FINGERPRINT_KEYS = frozenset({"backend", "device"})


def _checkpoint_fingerprint_matches(saved, expected):
    if not isinstance(saved, dict) or not isinstance(expected, dict):
        return saved == expected
    saved_identity = {
        key: value
        for key, value in saved.items()
        if key not in _EXECUTION_FINGERPRINT_KEYS
    }
    expected_identity = {
        key: value
        for key, value in expected.items()
        if key not in _EXECUTION_FINGERPRINT_KEYS
    }
    return saved_identity == expected_identity


def save_checkpoint(path, fingerprint, frame_index, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "frame_index": int(frame_index),
        "state": state,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(_REPLACE_RETRY_ATTEMPTS):
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError:
                if attempt + 1 == _REPLACE_RETRY_ATTEMPTS:
                    raise
                time.sleep(_REPLACE_RETRY_SECONDS)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint(path, expected_fingerprint):
    path = Path(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported pipeline checkpoint: {path}")
    if not _checkpoint_fingerprint_matches(
        payload.get("fingerprint"), expected_fingerprint
    ):
        raise ValueError(
            "Pipeline checkpoint does not match the video, model, or tracking settings."
        )
    frame_index = payload.get("frame_index")
    state = payload.get("state")
    if (
        not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
        or frame_index <= 0
        or not isinstance(state, dict)
    ):
        raise ValueError(f"Pipeline checkpoint has invalid state: {path}")
    return frame_index, state


def restore_checkpointed_file(handle, offset=None):
    """Discard uncommitted bytes and position a checkpointed file for writing."""
    if offset is None:
        handle.truncate(0)
        handle.seek(0)
        return
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("Checkpoint file offset must be a non-negative integer.")
    handle.truncate(offset)
    handle.seek(0, os.SEEK_END)
