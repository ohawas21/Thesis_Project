"""Implementation of dimension reduction methods."""
import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean as np_mean
from numpy import ones_like
from numpy import sum as np_sum
from numpy import triu
from pandas import DataFrame as pdDataFrame
from seaborn import heatmap, scatterplot, set_palette
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE

from ydata.report.metrics import MetricType
from ydata.report.metrics.base_metric import BaseMetric
from ydata.report.style_guide import (FIG_SIZE, NOTEBOOK_FIG_SIZE, TITLE_FONT_NOTEBOOK, TITLE_FONT_UNI,
                                      YDATA_HEATMAP_CMAP, YDATA_PCA_COLORS)
from ydata.report.styles import StyleHTML

set_palette(YDATA_PCA_COLORS)


def compute_umap(df_real: pdDataFrame, df_synth: pdDataFrame):
    from umap.umap_ import UMAP

    """
    Compute UMAP mappings and embeddings for real and synth dataset
    """
    reducer = UMAP()
    data_real = df_real.to_numpy()
    data_synth = df_synth.to_numpy()
    mapper_real = reducer.fit(data_real)
    mapper_synth = reducer.fit(data_synth)
    embeddings_real = reducer.transform(data_real)
    embedding_synth = reducer.transform(data_synth)
    return mapper_real, mapper_synth, embeddings_real, embedding_synth


def compute_tsne(df_real: pdDataFrame, df_synth: pdDataFrame, n_components: int = 2):
    """Compute TSNE embedding for real and synth dataset."""
    tsne = TSNE(n_components=n_components)
    real_embedded = tsne.fit_transform(df_real.to_numpy())
    synth_embedded = tsne.fit_transform(df_synth.to_numpy())
    return real_embedded, synth_embedded


def compute_pca(df_real: pdDataFrame, df_synth: pdDataFrame, n_components: int = 2):
    pca = IncrementalPCA(n_components=n_components)
    explained = []
    real_embedded = pca.fit_transform(df_real)
    explained.append(np_sum(pca.explained_variance_ratio_))
    synth_embedded = pca.fit_transform(df_synth)
    explained.append(np_sum(pca.explained_variance_ratio_))

    return real_embedded, synth_embedded, np_mean(explained)


def get_umap_plots(mapper_real, mapper_synth):
    from umap.plot import connectivity, diagnostic, points

    """Get UMAP plots using mapping: points, connectivity, diagnostic"""
    plots = {"real": {}, "synth": {}}
    plots["real"]["points"] = points(mapper_real)
    plots["synth"]["points"] = points(mapper_synth)
    plots["real"]["connectivity"] = connectivity(mapper_real, show_points=True)
    plots["synth"]["connectivity"] = connectivity(
        mapper_synth, show_points=True)
    plots["real"]["diagnostic"] = diagnostic(
        mapper_real, diagnostic_type="pca")
    plots["synth"]["diagnostic"] = diagnostic(
        mapper_synth, diagnostic_type="pca")
    return plots


def get_embedding_plot(real_embedded, synth_embedded, title=""):
    """Get embeddings plots."""
    fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
    df_real = pdDataFrame(data=real_embedded, columns=["x", "y"])
    df_real["type"] = "real"
    df_synth = pdDataFrame(data=synth_embedded, columns=["x", "y"])
    df_synth["type"] = "synth"
    df = pd.concat([df_real, df_synth], axis=0)
    ax = scatterplot(
        data=df,
        hue="type",
        x="x",
        y="y",
        ax=ax1,
        legend="full",
        alpha=0.8,
    )

    # remove x, y labels
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(fontsize=20)

    # remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig1.axes[0]


class DimensionReduction(BaseMetric):
    def __init__(self, formatter=StyleHTML) -> None:
        super().__init__(formatter)

    @staticmethod
    def _get_description(formatter):
        description = "The dimensionality reduction visualization plots show how \
            closely the distribution of the synthetic data resembles that of the \
            original data on a two-dimensional graph. Principal Component Analysis \
            (PCA) algorithm used to reduce the datasets dimensionality. \
            <br/>PCA captures any fundamental difference in the distributions of the \
            datasets. The scatterplots depict this difference visually. \
            <br/>The two first main Eigenvectors together \
            explain {}% of the total variance of the dataset."

        return formatter.paragraph(description)

    def _evaluate(self, source, synthetic, **kwargs):
        source_emb, synth_emb, variance_ratio = compute_pca(source, synthetic)
        plots = {
            "pca_chart": get_embedding_plot(source_emb, synth_emb, "PCA"),
            "title": self.name,
            "pca_description": "PCA: principal component analysis",
            "dimension_reduction_description": self._description.format(
                round(variance_ratio * 100, 2)
            ),
        }
        return plots

    @property
    def name(self) -> str:
        return "Dimensionality Reduction"

    @property
    def type(self) -> MetricType:
        return MetricType.VISUAL


def get_heatmap(df, title=""):
    """Get seabort heatmap."""
    mask = triu(ones_like(df, dtype=bool))

    fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
    heatmap(
        data=df,
        annot=True,
        fmt=".3f",
        mask=mask,
        cmap=YDATA_HEATMAP_CMAP,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax1,
    )
    ax1.set_title(title, **TITLE_FONT_NOTEBOOK)
    return fig1.axes[0]


def compute_corr(df):
    return df.corr()


def get_notebook_heatmaps(df_dict: dict, title="Correlation matrices comparision"):
    fig, ax = plt.subplots(1, 3, figsize=NOTEBOOK_FIG_SIZE)
    for i, items in enumerate(df_dict.items()):
        name, df = items
        mask = triu(ones_like(df, dtype=bool))
        heatmap(
            data=df,
            annot=True,
            fmt=".2f",
            mask=mask,
            cmap=YDATA_HEATMAP_CMAP,
            vmax=0.3,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
            ax=ax[i],
        )
        ax[i].set_title(name, **TITLE_FONT_NOTEBOOK)

    fig.suptitle(title, fontsize=20, weight="bold")
    return fig.axes[0]


def get_notebook_embeddings(df_dict: dict, title="Datasets embeddings - PCA"):
    fig, ax = plt.subplots(1, len(df_dict.items()), figsize=NOTEBOOK_FIG_SIZE)
    for i, items in enumerate(df_dict.items()):
        name, datasets = items
        df_real = pdDataFrame(data=datasets["Real"], columns=["x", "y"])
        df_real["type"] = "real"
        df_synth = pdDataFrame(data=datasets["Synth"], columns=["x", "y"])
        df_synth["type"] = "synth"
        df = pd.concat([df_real, df_synth], axis=0)
        if len(df_dict.items()) > 1:
            current_ax = ax[i]
        else:
            current_ax = ax
        scatterplot(
            data=df,
            hue="type",
            x="x",
            y="y",
            ax=current_ax,
            legend="full",
            alpha=0.8,
        )
        current_ax.set_title(name, **TITLE_FONT_UNI)

    fig.suptitle(title, fontsize=20, weight="bold")
    return fig.axes[0]


def get_corr_heatmap(df):
    """Get correlation heatmap."""
    corr = df.corr()
    return get_heatmap(corr), corr
