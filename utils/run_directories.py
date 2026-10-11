import re
from pathlib import Path


_RUN_DIRECTORY_PATTERN = re.compile(r"run_(\d+)")


def create_next_run_directory(output_root):
    """Atomically create the next sequential run directory under output_root."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    existing_numbers = [
        int(match.group(1))
        for path in output_root.iterdir()
        if path.is_dir()
        and (match := _RUN_DIRECTORY_PATTERN.fullmatch(path.name)) is not None
    ]
    next_number = max(existing_numbers, default=0) + 1

    while True:
        run_directory = output_root / f"run_{next_number}"
        try:
            run_directory.mkdir()
        except FileExistsError:
            next_number += 1
            continue
        return run_directory
