"""
This file contains database utilities.
"""
import itertools
import json
import multiprocessing
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List


def _load_and_append(json_filename: Path, shared_list: List):
    shared_list.append(json.load(json_filename.open(mode="r", encoding="utf-8")))


def collate_json(data_dir: PathLike, output_filename: PathLike) -> List[Dict[str, Any]]:
    """
    Loads all JSON files from a directory into a single list.
    :param data_dir: a directory of JSON files.
    :return: a list of dictionaries of all json objects in the directory
    """
    path = data_dir
    out = output_filename
    if not isinstance(path, Path):
        path = Path(path)
    if not isinstance(out, Path):
        out = Path(out)

    json_filenames = list(path.glob("*.json"))  # all json file paths

    with multiprocessing.Manager() as manager:
        l = manager.list()
        # arguments for multiprocessing
        args = list(itertools.zip_longest(json_filenames, [l], fillvalue=l))
        with multiprocessing.Pool() as pool:
            pool.starmap(_load_and_append, args)
        json.dump(list(l), out.open(mode="w", encoding="utf-8"))


if __name__ == "__main__":
    collate_json(
        data_dir=Path.cwd() / "reddit/data/PushShiftAndRedditAPICrawler-output",
        output_filename=Path.cwd() / "test.json",
    )
