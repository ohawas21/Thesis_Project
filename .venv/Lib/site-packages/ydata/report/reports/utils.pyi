from typing import TypeVar

T = float | int
U = TypeVar('U')

def score_to_label_leq(p: [T], mapping: dict[[T], U]) -> U | None: ...
