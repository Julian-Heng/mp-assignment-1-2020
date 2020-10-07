#!/usr/bin/env python3

import os

from contextlib import contextmanager
from pathlib import Path


@contextmanager
def use_dir(_path):
    cwd = os.getcwd()
    _path = Path(_path)

    if not _path.exists():
        _path.mkdir(parents=True, exist_ok=True)

    os.chdir(str(_path))

    try:
        yield
    finally:
        os.chdir(cwd)
