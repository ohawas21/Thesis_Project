"Utilities for handling data formats."
import json
from typing import List


def read_json(path: str):
    "Reads a JSON file."
    with open(path, "r") as f:
        j = json.load(f)
    return j


def write_string(s: str, path: str):
    "Writes a string into a path."
    with open(path, "w") as f:
        f.write(s)


def most_frequent(lst: List):
    "Returns the most frequent element in a list."
    return max(set(lst), key=lst.count)
