from pandas import DataFrame as pdDataFrame

from ydata.utils.associations.metrics import cramers_v as cv


def test_compute_chi_squared_based_values():
    data = {
        "favourite_snack": ["Pastel de Nata", "Crackling", "Hula Hoops"],
        "satisfaction": ["high", "high", "medium"],
        "avg_weight_consumer": [70, 115, 65],
    }

    df = pdDataFrame(data)
    cross_t_shape, chi2, sample_size = cv.compute_chi_squared_based_values(
        df["favourite_snack"], df["satisfaction"]
    )
    assert cross_t_shape == (3, 2)
    assert chi2 == 3
    assert sample_size == 3


def test_compute_adjusted_cramersv_is_zero():
    cross_t_shape = (3, 2)
    chi2 = 3
    sample_size = 3
    assert cv.compute_adjusted_cramers_v(cross_t_shape, chi2, sample_size) == 0


def test_compute_adjusted_cramersv():
    data = {
        "favourite_snack": [
            "Pastel de Nata",
            "Crackling",
            "Hula Hoops",
            "KFC",
            "chicken",
            "wine gums",
            "a",
            "b",
            "c",
            "d",
            "Pastel de Nata",
            "Crackling",
            "Hula Hoops",
        ],
        "satisfaction": [
            "high",
            "high",
            "medium",
            "low",
            "Medium",
            "low",
            "low",
            "low",
            "low",
            "d",
            "high",
            "high",
            "medium",
        ],
        "avg_weight_consumer": [70, 115, 65, 120, 85, 20, 120, 85, 20, 22, 85, 20, 22],
    }
    df = pdDataFrame(data)
    cross_t_shape, chi2, sample_size = cv.compute_chi_squared_based_values(
        df["favourite_snack"], df["satisfaction"]
    )
    return cv.compute_adjusted_cramers_v(cross_t_shape, chi2, sample_size)
