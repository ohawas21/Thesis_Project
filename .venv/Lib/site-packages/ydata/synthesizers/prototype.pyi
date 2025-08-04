from dataclasses import dataclass
from typing import Any
from ydata.preprocessors.base import Preprocessor
from ydata.synthesizers.base_synthesizer import BaseSynthesizer
from ydata.synthesizers.utils.segmentation import SegmentInfo
from ydata.synthesizers.utils.slicing import SliceInfo
from ydata.utils.acceleration_device import Device

@dataclass
class DataBlock:
    """Concrete representation of a block.

    Each block has its own Synthesizer instance.
    """
    synthesizer: BaseSynthesizer
    info: SliceInfo
    preprocessor: Preprocessor = ...
    def __init__(self, synthesizer, info, preprocessor=...) -> None: ...

@dataclass
class DataSegment:
    """Concrete representation of a segment.

    Each segment has a preprocessor and a dictionary of blocks. The
    blocks are constructed by the slicing strategy selected by the main
    Synthesizer class.
    """
    info: SegmentInfo
    preprocessor: Preprocessor = ...
    blocks: dict[Any, DataBlock] = ...
    def __init__(self, info, preprocessor=..., blocks=...) -> None: ...

@dataclass
class PipelinePrototype:
    """A Pipeline Prototype is an abstract description on how train a
    synthesizer and how to generate sample.

    A synthesizer has a block strategy, itself composed of a Segmentation and a Slice strategy.
    The result of a block strategy is a way to decompose an input dataset into blocks.

    The Pipeline Prototype describes how to preprocess the data at each step of the decomposition:
        - Globally (before any segmentation or slicing)
        - Segment
        - Block

    The BlockScope also describes the class of Synthesizer model to use, e.g. CartHierarchical.

    The Pipeline Prototype also holds contextual information such as the device on which the synthesizer
    instance should be trained, e.g. CPU, GPU.
    """
    @dataclass()
    class DatasetScope:
        preprocessor: type[Preprocessor] = ...
        preprocessor_params: dict = ...
        def __init__(self, preprocessor=..., preprocessor_params=...) -> None: ...
    @dataclass()
    class SegmentScope:
        preprocessor: type[Preprocessor] = ...
        preprocessor_params: dict = ...
        def __init__(self, preprocessor=..., preprocessor_params=...) -> None: ...
    @dataclass()
    class BlockScope:
        synthesizer: type[BaseSynthesizer]
        synthesizer_params: dict = ...
        preprocessor: type[Preprocessor] = ...
        preprocessor_params: dict = ...
        def __init__(self, synthesizer, synthesizer_params=..., preprocessor=..., preprocessor_params=...) -> None: ...
    block: BlockScope
    segment: SegmentScope
    dataset: DatasetScope
    device: Device
    def __init__(self, block, segment, dataset=..., device=...) -> None: ...
