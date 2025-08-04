"""Style Guide Python variables."""
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# COLOR CONVERSION
# hex(#E32212) > hsl(5, 85, 48)
# hex(#040404) > hsl(0, 0, 2)

FIG_SIZE = (16, 10)
NOTEBOOK_FIG_SIZE = (24, 7)
TITLE_FONT_UNI = {'size': '15', 'weight': 'bold'}
TITLE_FONT_NOTEBOOK = {'size': '15'}
YDATA_COLORS = ["#E32212", "#040404", "#474747", "#7A7A7A"]
YDATA_CMAP_COLORS = ["#70413d", "#a62419", "#E32212", "#fff5f5"]

YDATA_CMAP = ListedColormap(YDATA_COLORS)
YDATA_HEATMAP_CMAP = LinearSegmentedColormap.from_list("", YDATA_CMAP_COLORS[::-2])
