from _typeshed import Incomplete
from ydata.dataset import Dataset
from ydata.metadata import Metadata as Metadata
from ydata.metadata.column import Column as Column

metrics_logger: Incomplete
logger: Incomplete

class FakerSynthesizer:
    '''
    A synthesizer for generating synthetic data based on user-defined configurations.

    The `FakerSynthesizer` allows users to create synthetic tabular data **without needing
    an existing dataset**. Instead, it generates data based on user-provided **metadata**
    or manually defined column configurations. This approach is useful for:

    - Creating **mock datasets** for testing and development.
    - Generating **data prototypes** before real data is available.
    - Ensuring **privacy-preserving synthetic data** without reference to actual records.

    ## Key Features:
    - **Metadata-Driven Generation**: Generates synthetic data based on predefined Metadata.
    - **Customizable Column Types**: Supports user-defined column structures.
    - **Multi-Language Support**: Uses locale settings to generate realistic names, addresses, etc.

    ## Example Usage:
    ```python
    from ydata.synthesizers import FakerSynthesizer

    # Initialize the synthesizer with a specific locale
    faker_synth = FakerSynthesizer(locale="en")  # English data generation

    faker_synth.fit(metadata)

    # Generate synthetic data
    synthetic_data = faker_synth.sample(n_samples=1000)
    ```
    '''
    metadata: Incomplete
    columns: Incomplete
    locale: Incomplete
    def __init__(self, locale: str = 'en') -> None: ...
    domains: Incomplete
    nunique: Incomplete
    missings: Incomplete
    nrows: Incomplete
    extra_data: Incomplete
    value_counts: Incomplete
    def fit(self, metadata: Metadata):
        """
        Configure the `FakerSynthesizer` using provided metadata.

        This method sets up the synthesizer by defining the structure of the synthetic
        dataset based on the given `Metadata`. The metadata can either be:

        - **Computed**: Automatically extracted from an existing dataset.
        - **User-Defined**: Manually constructed to specify custom column types and distributions.

        Once `fit()` is called, the synthesizer will use this metadata to generate
        structured synthetic data that adheres to the defined schema.

        Args:
            metadata: A metadata object describing the structure of the synthetic dataset, including:
                      - Column names and data types.
                      - Faker-based data generators (e.g., names, addresses, emails).
                      - Value constraints (e.g., numeric ranges, categorical options).
        """
    def sample(self, sample_size: int = 1000) -> Dataset:
        """
        Generate a synthetic dataset based on the configured metadata.

        This method produces synthetic data according to the schema defined in the
        `fit()` step. The generated data adheres to the column types, constraints,
        and distributions specified in the provided `Metadata`.

        Args:
            sample_size (int): The number of synthetic records/rows to generate. Defaults to `1000`.

        Returns:
            dataset (Dataset): A Dataset object with the generated synthetic records/rows.

        """
    def save(self, path) -> None:
        """Saves the SYNTHESIZER and the model fitted per variable."""
    @classmethod
    def load(cls, path: str): ...
