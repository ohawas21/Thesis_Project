from dataclasses import dataclass, field
from typing import Any, Dict, Type

from ydata.preprocessors.base import Preprocessor
from ydata.preprocessors.identity import Identity as IdentityPreprocessor
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
    preprocessor: Preprocessor = IdentityPreprocessor()


@dataclass
class DataSegment:
    """Concrete representation of a segment.

    Each segment has a preprocessor and a dictionary of blocks. The
    blocks are constructed by the slicing strategy selected by the main
    Synthesizer class.
    """

    info: SegmentInfo
    preprocessor: Preprocessor = IdentityPreprocessor()
    blocks: Dict[Any, DataBlock] = field(default_factory=dict)


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
        preprocessor: Type[Preprocessor] = IdentityPreprocessor
        preprocessor_params: Dict = field(default_factory=dict)

    @dataclass()
    class SegmentScope:
        preprocessor: Type[Preprocessor] = IdentityPreprocessor
        preprocessor_params: Dict = field(default_factory=dict)

    @dataclass()
    class BlockScope:
        synthesizer: Type[BaseSynthesizer]
        synthesizer_params: Dict = field(default_factory=dict)
        preprocessor: Type[Preprocessor] = IdentityPreprocessor
        preprocessor_params: Dict = field(default_factory=dict)

    block: BlockScope
    segment: SegmentScope
    dataset: DatasetScope = field(default_factory=DatasetScope())
    device: Device = Device.CPU
