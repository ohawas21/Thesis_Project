import random

import numpy as np
import pandas as pd

from ydata.utils.data_types import _DT_TO_VT


def generate_dummy_dataset(n: int, n_numerical: int = 1, n_categorical: int = 1):
    df = pd.DataFrame()

    if n_numerical == 1:
        df["numerical"] = 100 * np.random.normal(0, 0.1, n)
    else:
        for i in range(1, n_numerical + 1):
            df[f"numerical_{i}"] = 100 * np.random.normal(0, 0.1, n)

    if n_categorical == 1:
        df["categorical"] = np.random.choice(
            a=["CAT_A", "CAT_B", "CAT_C"], size=n, p=[0.5, 0.3, 0.2]
        )
    else:
        for i in range(1, n_categorical + 1):
            df[f"categorical_{i}"] = np.random.choice(
                a=["CAT_A", "CAT_B", "CAT_C"], size=n, p=[0.5, 0.3, 0.2]
            )
    return df


def generate_dummy_dataset_with_all_datatypes(n):
    def gen_id_str(n):
        A, Z = np.array(["A", "Z"]).view("int32")
        LEN = 3
        s = pd.Series(
            np.random.randint(low=A, high=Z, size=n * LEN, dtype="int32").view(
                f"U{LEN}"
            )
        )
        return s.convert_dtypes()

    def gen_id_int(n):
        a = np.arange(n)
        np.random.shuffle(a)
        return a

    def random_datetimes_or_dates(start, end, n, out_format="datetime"):
        (divide_by, unit) = (
            (10**9, "s")
            if out_format == "datetime"
            else (24 * 60 * 60 * 10**9, "D")
        )
        start_u = start.value // divide_by
        end_u = end.value // divide_by
        return pd.Series(
            pd.to_datetime(np.random.randint(start_u, end_u, n), unit=unit)
        )

    def gen_date(n):
        start = pd.to_datetime("2015-01-01")
        end = pd.to_datetime("2018-01-01")
        return random_datetimes_or_dates(start, end, n, out_format="datetime")

    all_columns = []
    for dt, vts in _DT_TO_VT.items():
        all_columns.extend(list(zip([dt] * len(vts), vts)))

    raw_data = [
        np.random.randint(0, 20, n),
        100 * np.random.normal(0, 0.1, n),
        np.random.choice(a=["CAT_A", "CAT_B", "CAT_C"],
                         size=n, p=[0.5, 0.3, 0.2]),
        np.random.choice(a=[1, 2, 5, 10], size=n, p=[0.4, 0.3, 0.2, 0.1]),
        np.random.choice(a=[True, False], size=n, p=[0.4, 0.6]),
        np.random.choice(a=[1.1, 2.4, 4.5, 10], size=n,
                         p=[0.2, 0.5, 0.1, 0.2]),
        gen_date(n),
        gen_date(n),
        gen_date(n),
        gen_date(n),
        # gen_id_int(n),
        # gen_id_str(n),
    ]

    return pd.DataFrame(
        {
            f"{k[0].name.lower()}_{k[1].name.lower()}": raw_data[i]
            for i, k in enumerate(all_columns)
        }
    )


def insert_missing_at_random(df, col, miss, ma_val=None):
    n = miss
    if isinstance(miss, float):
        if miss < 0.0 or miss > 1.0:
            raise ValueError("When a float, 'miss' should belong to )0,1].")
        n = int(miss * df.shape[0])

    locs = random.sample(range(df.shape[0]), n)
    # mv = np.random.choice(a=list(_NULL_VALUES), size=n).tolist()
    # mv = ['?'] * n
    mv = [ma_val] * n
    for i, l in enumerate(locs):
        df.at[l, col] = mv[i]
    return df
