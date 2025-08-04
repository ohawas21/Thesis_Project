import copy
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from dacite import from_dict

from pandas import DataFrame as pdDataFrame
from tqdm.auto import tqdm
from visions import VisionsTypeset

from ydata_profiling import ProfileReport as _ProfileReport
from ydata_profiling.config import Settings
from ydata_profiling.report.presentation.core import Root
from ydata_profiling.report.presentation.flavours.html.templates import create_html_assets
from ydata.profiling.model.describe import describe as describe_df

from ydata.dataset import Dataset
from ydata.profiling.model import BaseDescription
from ydata.profiling.model.pandas.outliers_pandas import describe_outliers
from ydata.profiling.report.structure.report import get_report_structure
from ydata.profiling.typeset import ProfilingTypeSet

from ydata_profiling.model.summarizer import BaseSummarizer
from ydata.profiling.model.summarizer import YDataProfilingSummarizer

warnings.simplefilter(action='ignore', category=FutureWarning)

from ydata._licensing import profiling_report_check, profiling_report_charge
from ydata.utils.logger import SDKLogger

metrics_logger = SDKLogger(name="Metrics logger")

class ProfileReport(_ProfileReport):
    __default_config = os.path.join(
        os.path.dirname(__file__),
        "config_default.yaml"
    )

    __default_comparison_config = os.path.join(
        os.path.dirname(__file__),
        "config_comparison.yaml"
    )

    __default_assets_path = os.path.join(
        os.path.dirname(__file__),
        "assets"
    )

    def __init__(
        self,
        dataset: Dataset | pdDataFrame | None = None,
        minimal: bool = False,
        explorative: bool = False,
        sensitive: bool = False,
        tsmode: bool = False,
        sortby: str | None = None,
        sample: dict | None = None,
        config_file: Optional[Union[Path, str]] = None,
        lazy: bool = True,
        typeset: VisionsTypeset | None = None,
        summarizer: BaseSummarizer | None = None,
        config: Settings | None = None,
        outlier: bool = False,
        **kwargs
    ):
        if isinstance(dataset, Dataset):
            dataset = dataset.to_pandas()
        self._create_ydata_colormaps()
        self._outlier = outlier

        if isinstance(dataset, Dataset) or isinstance(dataset, pdDataFrame):
            metrics_logger.info(dataset=dataset,
                                datatype="timeseries" if tsmode else "tabular",
                                method='profiling')

        if dataset is not None:
            if dataset.index.duplicated().any():
                warnings.warn(
                    "Duplicate index detected in the input DataFrame. The index has been automatically reset to ensure uniqueness during analysis. "
                            "If your index is meaningful and should not be modified, consider disabling features that require unique indexing, such as "
                            "duplicate detection and correlation matrices before running your report.",
                    UserWarning
                )
                dataset = dataset.copy()
                dataset = dataset.reset_index(drop=True)

        super().__init__(
            df=dataset,
            minimal=minimal,
            tsmode=tsmode,
            sortby=sortby,
            sensitive=sensitive,
            explorative=explorative,
            sample=sample,
            config_file=config_file,
            lazy=lazy,
            typeset=typeset,
            summarizer=summarizer,
            config=config,
            **kwargs
        )

    @property
    def summarizer(self) -> BaseSummarizer:
        if self._summarizer is None:
            self._summarizer = YDataProfilingSummarizer(self.typeset)
        return self._summarizer

    @property
    def typeset(self) -> Optional[VisionsTypeset]:
        if self._typeset is None:
            self._typeset = ProfilingTypeSet(self.config, self._type_schema)
        return self._typeset

    @property
    def description_set(self) -> BaseDescription:
        if self._description_set is None:
            self._description_set = describe_df(
                self.config,
                self.df,
                self.summarizer,
                self.typeset,
                self._sample,
            )

            profiling_report_check(self)
            """
            self._description_set = from_dict(
                BaseDescription, asdict(super().description_set)
            )
            """

            if getattr(self, "_outlier", False) and self.description_set.outliers is None:
                self.description_set.outliers = describe_outliers(
                    self.df, self.config)
            profiling_report_charge(self)

        return self._description_set

    @property
    def report(self) -> Root:
        if self._report is None:
            self._report = get_report_structure(
                self.config, self.description_set)
        return self._report

    def _create_missing_section_html(self, section: str):
        path = os.path.join(
            self.__default_assets_path,
            "empty_section_template.html"
        )
        with open(path, mode="r") as f:
            doc = "".join(f.readlines())
        path = os.path.join(
            self.__default_assets_path,
            "section_template_style.html"
        )
        with open(path, mode="r") as f:
            style = "".join(f.readlines())
        return doc.format(style, section, section)

    def _render_breakdown_html(self) -> Dict[str, str]:
        from ydata_profiling.report.presentation.flavours import HTMLReport

        report = self.report
        sections = ["Variables", "Interactions",
                    "Correlations", "Missing values",
                    "Outliers"
                    ]
        with tqdm(
            total=1, desc="Render HTML", disable=not self.config.progress_bar
        ) as pbar:
            base_report = HTMLReport(copy.deepcopy(report))
            html = {}
            for section in base_report.content["body"].content["items"]:
                if section.name not in sections:
                    continue
                _report = copy.deepcopy(base_report)
                _report.content["body"].content["items"] = [section]
                _report.content["footer"].content["html"] = ""

                html[section.name] = _report.render(
                    nav=False,
                    offline=self.config.html.use_local_assets,
                    inline=self.config.html.inline,
                    assets_prefix=self.config.html.assets_prefix,
                    primary_color=self.config.html.style.primary_color,
                    logo=self.config.html.style.logo,
                    theme=self.config.html.style.theme,
                    title=self.description_set.analysis.title,
                    date=self.description_set.analysis.date_start,
                    version=self.description_set.package["ydata_profiling_version"],
                )

            if self.config.html.minify_html:
                from htmlmin.main import minify
                html = {
                    name: minify(rendered, remove_all_empty_space=True,
                                 remove_comments=True)
                    for name, rendered in html.items()
                }

            for section in ["Interactions", "Correlations", "Outliers"]:
                if section not in html:
                    html[section] = self._create_missing_section_html(section)

            pbar.update()

        return html

    def _remove_reproduction(self, report: str) -> str:
        """Workaround while the Overview section is not implemented from the
        Metadata."""
        start = report.find(
            "<li role=presentation><a href=#overview-reproduction"
        )
        if start == -1:
            return report
        end = report[start:].find("</li>")
        report = report.replace(report[start:start + end + 5], "")
        return report

    def to_file(self, output_file: Union[str, Path], silent: bool = True, html_breakdown=False) -> None:
        """Write the report to a file.

        By default a name is generated.

        Args:
            output_file: The name or the path of the file to generate including the extension (.html, .json).
            silent: if False, opens the file in the default browser or download it in a Google Colab environment
        """
        if not isinstance(output_file, Path):
            output_file = Path(str(output_file))

        if output_file.suffix == ".json":
            data = self.to_json()
        else:
            if not self.config.html.inline:
                self.config.html.assets_path = str(output_file.parent)
                if self.config.html.assets_prefix is None:
                    self.config.html.assets_prefix = str(
                        output_file.stem) + "_assets"
                create_html_assets(self.config, output_file)

            if html_breakdown:
                data = self._render_breakdown_html()
            else:
                data = self.to_html()

            if output_file.suffix != ".html":
                suffix = output_file.suffix
                output_file = output_file.with_suffix(".html")
                warnings.warn(
                    f"Extension {suffix} not supported. For now we assume .html was intended. "
                    f"To remove this warning, please use .html or .json."
                )

        disable_progress_bar = not self.config.progress_bar
        with tqdm(
            total=1, desc="Export report to file", disable=disable_progress_bar
        ) as pbar:
            if html_breakdown:
                for name, html in data.items():
                    file_name = output_file.name[:-5] + "_" + name + ".html"
                    file_name = file_name.replace(" ", "_").lower()
                    file = Path(str(output_file.parent) + "/" + file_name)
                    file.write_text(html, encoding="utf-8")
            else:
                data = self._remove_reproduction(data)
                output_file.write_text(data, encoding="utf-8")

            pbar.update()

        if not silent:
            try:
                from google.colab import files  # noqa: F401

                files.download(output_file.absolute().as_uri())
            except ModuleNotFoundError:
                import webbrowser

                webbrowser.open_new_tab(output_file.absolute().as_uri())

    @staticmethod
    def _create_custom_colormap(name: str, values: List[float], colors: List[str], N: int = 256):
        if len(values) != len(colors):
            warnings.warn(
                f"Unable to create [{name}] colormap, values and and colors length mismatch."
            )
            return

        if name not in plt.colormaps:
            cmap = LinearSegmentedColormap.from_list(
                name,
                list(zip(values, colors)),
                N=256
            )
            plt.colormaps.register(name=name, cmap=cmap)

        if f"{name}_r" not in plt.colormaps:
            cmap = LinearSegmentedColormap.from_list(
                f"{name}_r",
                list(zip(values, reversed(colors))),
                N=256
            )
            plt.colormaps.register(name=f"{name}_r", cmap=cmap)

    @staticmethod
    def _create_ydata_colormaps():
        ProfileReport._create_custom_colormap(
            name="ydata",
            values=[0, 0.5, 1],
            colors=["#BD0F06", "#F6F6F6", "#0052CC"],
        )
        ProfileReport._create_custom_colormap(
            name="ydata2",
            values=[0, 1],
            colors=["#BD0F06", "#0052CC"],
        )


    def compare(
        self, other: "ProfileReport", config: Optional[Settings] = None
    ) -> "ProfileReport":
        """Compare this report with another ProfileReport
        Alias for:
        ```
        ydata_profiling.compare([report1, report2], config=config)
        ```
        See `ydata_profiling.compare` for details.

        Args:
            other: the ProfileReport to compare to
            config: the settings object for the merged ProfileReport. If `None`, uses the caller's config

        Returns:
            Comparison ProfileReport
        """
        from ydata.profiling.compare_reports import compare

        return compare([self, other], config if config is not None else self.config)

    def dumps(self, include_schema: bool = False) -> bytes:
        """Serialize ProfileReport and return bytes for reproducing
        ProfileReport or Caching.

        Args:
            include_schema (bool): True if the dataframe schema should be saved
        Returns:
            Bytes which contains hash of DataFrame, config, _description_set and _report
        """
        import pickle

        # Note: _description_set and _report may are None if they haven't been computed
        return pickle.dumps(
            [
                self.df_hash,
                self.df.head(1).copy() if (
                    include_schema and self.df is not None and not self.df.empty) else None,
                self.config,
                asdict(self._description_set) if self._description_set else None,
                self._report,
            ]
        )

    @staticmethod
    def loads(data: bytes) -> "ProfileReport":
        """Deserialize the serialized report.

        Args:
            data: The bytes of a serialize ProfileReport object.
        Raises:
            ValueError: if ignore_config is set to False and the configs do not match.
        Returns:
            self
        """
        import pickle

        profile = ProfileReport()

        try:
            (
                df_hash,
                df,
                loaded_config,
                loaded_description_set,
                loaded_report,
            ) = pickle.loads(data)
        except Exception as e:
            raise ValueError("Failed to load data") from e
        if loaded_description_set:
            loaded_description_set = from_dict(
                BaseDescription, loaded_description_set)

        if not all(
            (
                df_hash is None or isinstance(df_hash, str),
                isinstance(loaded_config, Settings),
                loaded_description_set is None
                or isinstance(loaded_description_set, BaseDescription),
                loaded_report is None or isinstance(loaded_report, Root),
            )
        ):
            raise ValueError(
                "Failed to load data: file may be damaged or from an incompatible version"
            )
        profile.df = df
        profile._description_set = loaded_description_set
        profile._report = loaded_report
        profile.config = loaded_config
        profile._df_hash = df_hash
        return profile

    def save(self, output_file: Union[Path, str], include_schema: bool = False):
        """Save the report to a file.

        Args:
            output_file: Path where to save the ProfileReport
            include_schema (bool): True if the dataframe schema should be saved
        """
        if not isinstance(output_file, Path):
            output_file = Path(str(output_file))
        output_file.write_bytes(self.dumps(include_schema=include_schema))

    @staticmethod
    def load(path: Path, dataset: Optional[pdDataFrame] = None) -> "ProfileReport":
        """Load a ProfileReport from a file.

        Args:
            path: Path where to load the ProfileReport
            dataset (Optional[pd.DataFrame]): dataset to re-assign to the ProfileReport
        """
        if not isinstance(path, Path):
            path = Path(str(path))
        report = ProfileReport.loads(path.read_bytes())
        report.__class__ = ProfileReport
        report.config.vars.cat.redact = True
        if dataset:
            report.df = dataset
        return report
