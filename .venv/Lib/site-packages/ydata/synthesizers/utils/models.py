"""Synthesizer model enum definition."""
from enum import Enum

from ydata.__models._cartmodel import SeqCartHierarchical, CartHierarchical


class RegularSynthesizerModel(Enum):
    CART = CartHierarchical

    def __call__(self, *args, **kwargs):
        return self.value(*args)


class TimeSeriesSynthesizerModel(Enum):
    CART = SeqCartHierarchical

    def __call__(self, *args, **kwargs):
        return self.value(*args)
