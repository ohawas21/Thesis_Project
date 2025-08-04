from enum import Enum
from ydata.__models._cartmodel import CartHierarchical, SeqCartHierarchical

class RegularSynthesizerModel(Enum):
    CART = CartHierarchical
    def __call__(self, *args, **kwargs): ...

class TimeSeriesSynthesizerModel(Enum):
    CART = SeqCartHierarchical
    def __call__(self, *args, **kwargs): ...
