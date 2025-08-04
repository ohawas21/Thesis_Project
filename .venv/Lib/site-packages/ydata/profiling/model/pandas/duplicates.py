"""
    Metrics that identify near duplicate records using fuzzy matching
"""
import time
from typing import Any, Sequence, Dict, Tuple, Optional

import pandas as pd
import re
import unicodedata
import numpy as np
from datasketch import MinHash, MinHashLSH


from ydata_profiling.config import Settings

def normalize_text(text):
    """Improved normalization function for both text and numeric data."""

    if isinstance(text, (int, float)):  # Preserve numeric types
        return str(text)

    text = str(text)  # Convert to string for processing
    text = unicodedata.normalize("NFKC", text)  # Unicode normalization
    text = text.lower().strip()  # Convert to lowercase and trim spaces

    # Preserve numeric formatting (keep decimals)
    if re.match(r"^[0-9\.\-]+$", text):
        return text

        # For text, remove punctuation but preserve numeric values
    text = re.sub(r"[^\w\s.]", "", text)
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces

    return text

def get_potential_duplicates(
    config: Settings,
    df: pd.DataFrame,
    supported_columns: Sequence) -> Optional[pd.DataFrame]:
    """
    Detects near duplicate rows using fuzzy matching and similarity scores
    Args:
        df: The original dataframe to be searched/validated

    Returns: a pandas Dataframe with the near duplicate records and respective percentages
    """
    # **MinHash Function (With Token Filtering)**
    def create_minhash(row):
        """Create MinHash signature for a single row with token filtering."""
        minhash = MinHash(num_perm=num_perm)
        words = set(" ".join(row).split())  # Remove duplicate words
        for word in words:  # Unique tokens only
            minhash.update(word.encode('utf8'))
        return minhash

    #get the number of duplocates to be shown
    n_head = config.duplicates.head

    if n_head > 0:
        if supported_columns and len(df) > 0:
            #supported_columns and len(df) > 0:
            # Apply text normalization to all fields
            df = df[list(supported_columns)]
            df = df.applymap(normalize_text)

            # **Optimized MinHash Parameters**
            num_perm =128  # More permutations for finer similarity distinction
            lsh_threshold = 0.8  # Stricter threshold to avoid clustering everything

            # **Initialize LSH**
            lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)

            # **Batch Compute MinHashes**
            df["minhash"] = df.apply(create_minhash, axis=1)

            # **Batch Insert MinHashes into LSH**
            index_list = df.index.astype(str).tolist()
            minhash_list = df["minhash"].tolist()

            minhash_dict = {}
            for idx, minhash in zip(index_list, minhash_list):
                lsh.insert(idx, minhash)  # **Batch insert**
                minhash_dict[idx] = minhash

            # **Batch Query LSH & Compute Jaccard Similarity**
            similar_pairs = []
            batch_size = 10000  # Query 10K records at a time for efficiency

            for batch_start in range(0, len(index_list), batch_size):
                batch_indexes = index_list[batch_start:batch_start + batch_size]
                batch_minhashes = minhash_list[batch_start:batch_start + batch_size]

                for idx, minhash in zip(batch_indexes, batch_minhashes):
                    candidates = lsh.query(minhash)

                    for candidate in candidates:
                        if candidate != idx:
                            # **Actually compute Jaccard similarity**
                            sim_score = minhash.jaccard(minhash_dict[candidate])

                            # **Only store valid near-duplicates**
                            if sim_score >= lsh_threshold:
                                similar_pairs.append((idx, candidate, sim_score))

            df_similar = pd.DataFrame(similar_pairs, columns=["record_index","duplicate_index","similarity_score"])

            if df_similar.empty:
                return None
            else:
                # filter duplicate records by the duplicates with the highest similarity in case there are multiple duplicate candidates
                # per record
                idx = df_similar.groupby('record_index')['similarity_score'].idxmax().values
                df_similar = df_similar.iloc[idx]

                # **Convert Index Columns to Integers**
                df_similar["record_index"] = df_similar["record_index"].astype(int)
                df_similar["duplicate_index"] = df_similar["duplicate_index"].astype(int)

                # **Join to Original Data**
                near_duplicates = df_similar.merge(df, left_on="record_index", right_index=True).drop('minhash', axis=1)
                near_duplicates = near_duplicates.round({'similarity_score': 2})

                return near_duplicates[supported_columns+['record_index', 'duplicate_index', 'similarity_score']].set_index('record_index')
        else:
            return None
    else:
       return None

def get_duplicates(
    config: Settings,
    df: pd.DataFrame,
    len_df: int) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:

    metrics = {}
    n_head=config.duplicates.head
    if n_head > 0:
        if df is not None:
            duplicates=df[df['similarity_score']==1.0].drop(['similarity_score'], axis=1)
            metrics["n_duplicates"] = len(duplicates)
            metrics['p_duplicates'] = metrics["n_duplicates"]/len_df
            return (metrics, duplicates.head(n_head))
        else:
            metrics["n_duplicates"] = 0
            metrics["p_duplicates"] = 0.0
            return metrics, None

    else:
        return metrics, None

def get_near_duplicates(
    config: Settings,
    df: pd.DataFrame,
    len_df: int) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    metrics = {}
    n_head = config.duplicates.head
    if n_head > 0:
        if df is not None:
            near_duplicates = df[df['similarity_score'] < 1.0]
            metrics["n_near_dups"] = len(near_duplicates)
            metrics['p_near_dups'] = metrics["n_near_dups"] / len_df
            return (metrics, near_duplicates.sort_values('similarity_score', ascending=False).head(n_head)
                                                                    if not near_duplicates.empty else None)
        else:
            metrics["n_near_dups"] = 0
            metrics["p_near_dups"] = 0.0
            return metrics, None
    else:
        return metrics, None


