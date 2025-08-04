from numpy import array
from numpy.random import normal, randint, random, uniform
from pandas import DataFrame
from pandas.core.dtypes.common import is_datetime_or_timedelta_dtype


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
    new_dataframe = DataFrame()
    data = data.copy()
    for column in data:
        # Save the max and min values to set constraints
        # Could add a tune value for the level of noise
        max_value = data[column].max()
        min_value = data[column].min()
        upsampled_array = []
        series = array(data[column])
        for i in range(len(series) - 1):
            upsampled_array.append(series[i])
            difference = abs(series[i] - series[i + 1])
            for y in range(n_upsample):
                movement = (
                    -(difference / randint(1, 5))
                    if random() < 0.5
                    else (difference / randint(1, 5))
                )
                value = series[i] + round(movement, 3)
                # If they go out of boundaries recalculate the movement
                while value < min_value or value > max_value:
                    movement = (
                        -(difference / randint(1, 5))
                        if random() < 0.5
                        else (difference / randint(1, 5))
                    )
                    value = series[i] + round(movement, 3)
                upsampled_array.append(value)
        upsampled_array.append(series[-1])
        new_dataframe[column] = upsampled_array

    return new_dataframe


def linear_interpolation_upsample(
    data: DataFrame,
    time_to_resample: str = "10T",
    add_noise: bool = False,
    noise_reduction: float = 0.25,
):
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

    if not is_datetime_or_timedelta_dtype(data.index):
        raise Exception(
            "The index provided is not supported. Please input a dataset with a DateTime index."
        )

    data = data.copy()

    if not add_noise:

        data = data.resample(time_to_resample).mean()

        return data.interpolate("time")

    else:
        data = data.resample(time_to_resample).mean()
        for col in data:
            # Save the indexes of null data
            null_indexes = data[col].isnull()
            # Interpolate
            data_interpolated = data[col].interpolate("time")
            # Create some noise for the interpolated values
            noise = normal(
                data_interpolated.mean(),
                data_interpolated.std(),
                len(data_interpolated.loc[null_indexes]),
            )
            # Do the mean of the noise
            data[col].loc[null_indexes] = (
                data_interpolated.loc[null_indexes] + (noise * noise_reduction)
            ) / 2

        return data


def distorted_mean_upsample(
    data: DataFrame, time_to_resample: str = "10T", linear_interpolated: bool = False
):
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

    if not is_datetime_or_timedelta_dtype(data.index):
        raise Exception(
            "The index provided is not supported. Please input a dataset with a DateTime index."
        )

    resampled_dataframe = data.resample(time_to_resample).mean().copy()

    if linear_interpolated:
        interpolated = resampled_dataframe.interpolate("linear")

    for column in data:
        curr_data_column = DataFrame(data[column]).copy()
        # Calculate the mean of the values we need to upsample
        curr_data_column["means"] = (
            (curr_data_column + curr_data_column.shift(1)) / 2
        ).bfill()
        # Resample the data
        curr_data_column = curr_data_column.resample(time_to_resample).mean()
        # Save index of null
        null_indexes = curr_data_column[column].isnull()
        # Input noise on those
        curr_data_column["means"] = curr_data_column["means"].bfill()
        curr_data_column["means"] = (
            uniform(0.85, 1.15, len(
                curr_data_column["means"].loc[null_indexes]))
            * curr_data_column["means"].loc[null_indexes]
        )
        # fill the nans with these
        resampled_dataframe[column] = curr_data_column[column].fillna(
            curr_data_column["means"]
        )

    if linear_interpolated:
        for col in data:
            resampled_dataframe[col] = (interpolated[col] * 0.7) + (
                resampled_dataframe[col] * 0.3
            )

    return resampled_dataframe
