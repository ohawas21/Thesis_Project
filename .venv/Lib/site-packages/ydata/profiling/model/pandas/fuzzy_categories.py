"""
    File that contains the logic to compute and find potential dirty categories for variables with high-cardinality
"""
import pandas as pd
from itertools import combinations
from fuzzywuzzy import fuzz
import re

def get_fuzzy_categories(series: pd.Series, threshold=0.60):
    """
    Identifies fuzzy/dirty categories efficiently and calculates risk based on fuzzy ratio.
    Ensures numeric variations are ignored, and unique values are counted post-normalization.

    Args:
        series: a pandas series with the column values
        threshold: the fuzzy ratio to identify potential fuzziness
        risk_threshold: defines the level of risk that it might bring to the data quality

    Returns: the ratio of fuzziness identified as well as whether poses a risk
    """
    def remove_numeric_variations(value):
        """Removes numeric variations while keeping textual components intact."""
        return re.sub(r'\d+', '', value).strip()

    raw_values = series.dropna().unique() #check if this is already done before
    normalized_values = {val: remove_numeric_variations(val) for val in raw_values} #check if this is done
    unique_normalized_values = list(set(normalized_values.values()))  # Get unique values after normalization

    fuzzy_clusters = {}  # Dictionary to store category clusters
    standard_categories = {}  # Store the most common suggestion for each cluster

    # Step 1: Cluster similar categories based on fuzzy matching
    for val1, val2 in combinations(unique_normalized_values, 2):
        if fuzz.partial_ratio(val1, val2) >= threshold * 100:
            if val1 in fuzzy_clusters:
                fuzzy_clusters[val1].add(val2)
            elif val2 in fuzzy_clusters:
                fuzzy_clusters[val2].add(val1)
            else:
                fuzzy_clusters[val1] = {val1, val2}

    # Step 2: Determine the most common standardized category for each cluster
    for cluster in fuzzy_clusters.values():
        cluster_list = list(cluster)
        most_common = max(cluster_list, key=lambda x: series.str.contains(x, regex=False, case=False).sum())
        standard_categories[most_common] = cluster_list

    # Step 3: Construct the structured table of fuzzy categories
    fuzzy_category_table = pd.DataFrame([
        {"Cluster Representative": std_cat, "Detected Variants": ", ".join(variants)}
        for std_cat, variants in standard_categories.items()
    ])

    n_dirty_categories = len(fuzzy_clusters)
    p_dirty_categories = n_dirty_categories / max(1, len(unique_normalized_values))  # Avoid division by zero

    return {'n_fuzzy_vals': n_dirty_categories,
            'p_fuzzy_vals': p_dirty_categories,
            'dirty_categories_values': fuzzy_category_table if n_dirty_categories>0 else None}

