from pandas import DataFrame

def random_walk_upsample(data: DataFrame, n_upsample: int = 5) -> DataFrame:
    """
    Function to upsample the data with a random walk. What the function does is to calculate
    the difference between the 2 dates to upsample (17.00: 357, 18.00: 400, diff = 43) and calculate a
    random movement from that position.

    Args:
        data: pd.DataFrame. The dataframe to upsample.
        n_upsample: int. Number of additioned value wanted.
                         (i.e. if you have hourly data and want 10minutes window.
                         17.00 to 18.00 there are 5 values that you want to add:
                         [17.10, 17.20, 17.30, 17.40, 17.50])

    Returns: pd.DataFrame. The dataframe with upsampled data.

    """
def linear_interpolation_upsample(data: DataFrame, time_to_resample: str = '10T', add_noise: bool = False, noise_reduction: float = 0.25):
    """Function to upsample data and linearly interpolate the resulting NaN
    values. In case of add_noise == True noise from a gaussian is added to the
    linear interpolation.

    Args:
        data: pd.DataFrame. The data to be upsampled. NB. Needs to have a DateTimeIndex.
        time_to_resample: str. Frequency string to use. The frequencies can be found here:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        add_noise: bool. True if you want to add noise to the linear interpolation.
        noise_reduction: float. Number between 0 and 1 to reduce the level of noise you want.
                                the closer to 0 the less noise you are going to input in the linear interpolation.

    Returns: pd.DataFrame. DataFrame resampled and linearly interpolated w/ or w/o noise.
    """
def distorted_mean_upsample(data: DataFrame, time_to_resample: str = '10T', linear_interpolated: bool = False):
    """Function to upsample data the resulting NaN values. In case of
    linear_interpolated == True a linear interpolation is done and the results
    are averaged.

    Args:
        data: pd.DataFrame. The data to be upsampled. NB. Needs to have a DateTimeIndex.
        time_to_resample: str. Frequency string to use. The frequencies can be found here:
                https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        linear_interpolated: bool. If you want or not a linear interpolation.

    Returns: pd.DataFrame. Upsampled DataFrame
    """
