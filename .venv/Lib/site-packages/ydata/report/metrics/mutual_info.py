import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from seaborn import heatmap
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder

from ydata.report.metrics import MetricType
from ydata.report.metrics.base_metric import BaseMetric
from ydata.report.metrics.utils import get_categorical_vars, get_numerical_vars
from ydata.report.style_guide import FIG_SIZE, TITLE_FONT_NOTEBOOK, YDATA_HEATMAP_CMAP
from ydata.report.styles import StyleHTML

def reduce_dimensionality(df: pd.DataFrame, k=50):
    if df.shape[1] <= k:
        return df

    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Downsample to speed up
    df_sampled = df.sample(n=min(5000, len(df)), random_state=42)

    # Use f_classif instead of mutual_info_classif for speed
    y_dummy = np.random.randint(0, 2, len(df_sampled))
    selector = SelectKBest(score_func=f_classif, k=min(k, df.shape[1]))
    selector.fit(df_sampled.fillna(0), y_dummy)
    return df[df.columns[selector.get_support()]]


class MutualInformationMatrix(BaseMetric):
    def __init__(self, formatter=StyleHTML, include_plots: bool = True) -> None:
        super().__init__(formatter)
        self._include_plots = include_plots
        self._NUMBER_COLUMNS_TO_KEEP = 5

    @property
    def name(self) -> str:
        return "Mutual Information"

    @property
    def type(self) -> MetricType:
        return MetricType.VISUAL

    @staticmethod
    def _get_description(formatter):
        return f"{formatter.bold('MUTUAL INFORMATION (MI)')} measures how much information " \
               f"can be obtained about one feature by observing another. This metric calculates " \
               f"the similarity between real and synthetic MI values for each pair of features. " \
               f"It returns values between [0, 1], where closer to 1 is desirable (i.e., equal MI)."

    @staticmethod
    def _get_top_k_columns(values, k):
        seen = set()
        top_k = []
        for col in values:
            if col not in seen:
                seen.add(col)
                top_k.append(col)
                if len(top_k) == k:
                    break
        return top_k

    @staticmethod
    def _ensure_data_has_same_shape(source: pd.DataFrame, synthetic: pd.DataFrame):
        min_rows = min(len(source), len(synthetic))
        return source.sample(min_rows), synthetic.sample(min_rows)

    @staticmethod
    def _calculate_mi(col1_data, col2_data, col1_type, col2_type):
        """Helper to select the appropriate MI function based on feature types"""
        try:
            if col1_type == "cat" and col2_type == "cat":
                return mutual_info_classif(col1_data.reshape(-1, 1), col2_data, discrete_features=True)
            elif col1_type == "num" and col2_type == "cat":
                return mutual_info_classif(col1_data.reshape(-1, 1), col2_data, discrete_features=False)
            elif col1_type == "cat" and col2_type == "num":
                return mutual_info_classif(col2_data.reshape(-1, 1), col1_data, discrete_features=False)
            else:
                return mutual_info_regression(col1_data.reshape(-1, 1), col2_data, discrete_features=False)
        except ValueError:
            return [0.0]

    def _calculate_mi_matrix(self, source, synthetic, numerical_vars, categorical_vars, metadata=None):
        # Combine both datasets for feature selection
        combined = pd.concat([source[numerical_vars + categorical_vars], synthetic[numerical_vars + categorical_vars]],
                             axis=0)
        reduced = reduce_dimensionality(combined, k=50)
        columns = list(reduced.columns)

        # Adjust variable types based on what survived
        numerical_vars = [col for col in numerical_vars if col in columns]
        categorical_vars = [col for col in categorical_vars if col in columns]

        col_types = {col: "num" for col in numerical_vars}
        col_types.update({col: "cat" for col in categorical_vars})

        source = source[columns]
        synthetic = synthetic[columns]

        mi_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)

        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if j > i:
                    continue  # Symmetry

                # High-cardinality check for categoricals
                if (
                    (col1 in categorical_vars and source[col1].nunique() == len(source)) or
                    (col2 in categorical_vars and source[col2].nunique() == len(source))
                ):
                    val = 0.0
                else:
                    real_mi = self._calculate_mi(source[col1].values, source[col2].values, col_types[col1],
                                                 col_types[col2])
                    synth_mi = self._calculate_mi(synthetic[col1].values, synthetic[col2].values, col_types[col1],
                                                  col_types[col2])
                    real_score = (expit(real_mi[0]) - 0.5) / 0.5
                    synth_score = (expit(synth_mi[0]) - 0.5) / 0.5
                    val = abs(real_score - synth_score)

                mi_matrix.loc[col1, col2] = val
                mi_matrix.loc[col2, col1] = val  # Ensure symmetry

        return 1.0 - mi_matrix

    @staticmethod
    def get_heatmap(df, title=None):
        mask = np.triu(np.ones_like(df, dtype=bool), k=1)
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        heatmap(
            data=df,
            vmin=0,
            vmax=1,
            annot=True,
            mask=mask,
            fmt=".2f",
            cmap=YDATA_HEATMAP_CMAP,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            annot_kws={"size": 20},
            ax=ax
        )
        if title:
            ax.set_title(title, **TITLE_FONT_NOTEBOOK)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
        return ax

    def _evaluate(self, source: pd.DataFrame, synthetic: pd.DataFrame, **kwargs):

        metadata = kwargs.get("metadata", {})
        categorical_vars = get_categorical_vars(source, metadata)
        numerical_vars = get_numerical_vars(source, metadata)

        source_proc, synthetic_proc = self._ensure_data_has_same_shape(source, synthetic)
        mi_matrix = self._calculate_mi_matrix(source_proc, synthetic_proc, numerical_vars, categorical_vars)

        # Flatten and sort to get top/bottom pairs
        ordered_cols = [col for pair in mi_matrix.stack().sort_values(ascending=False).index for col in pair]
        top_k = self._get_top_k_columns(ordered_cols, self._NUMBER_COLUMNS_TO_KEEP)
        bottom_k = self._get_top_k_columns(reversed(ordered_cols), self._NUMBER_COLUMNS_TO_KEEP)

        def filter_matrix(cols):
            return mi_matrix.loc[cols, cols].sort_index().sort_index(axis=1)

        results = {
            "mean": mi_matrix.stack().mean(),
            "matrix_top_k_cols": filter_matrix(top_k),
            "matrix_bottom_k_cols": filter_matrix(bottom_k),
        }

        if self._include_plots:
            results.update({
                "chart_top_k_cols": self.get_heatmap(results["matrix_top_k_cols"], title="Mutual Information - Highest Values"),
                "chart_bottom_k_cols": self.get_heatmap(results["matrix_bottom_k_cols"], title="Mutual Information - Lowest Values"),
                "title": self.name,
                "description": self._description
            })

        return results
