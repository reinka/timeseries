# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import numpy as np


def replace_every_last_nth(curr_frame, replace, max_steps, n_ts):
    """Replace every last nth value of the current frame.

    For a given number ``max_steps``, ``curr_frame`` and ``replace`` are being
    traversed backwards during which for each step i the (i * n_ts)-th
    value of ``curr_frame`` gets replaced by the i-th value of ``replace``.

    Because of the backwards traversal an offset relative to the end of
    ``curr_frame`` and ``replace`` is being used.

    I.e.
    curr_frame = [1, 2, 3, 4, 5]
    replace = [66, 99]
    max_steps = 2
    n = 2
    would result in [1, 66, 3, 99, 5].

    Parameters
    -----------
    curr_frame : np.array
        Array whose values need to be replaced

    replace : list
        List that contains new values to be inserted

    max_steps : int
        Maximum number of steps to traverse backwards

    n_ts: int
        Number of non-lagged, unique time series in the data.

    Returns
    -------
    curr_frame : np.array
        Current frame with replaced values

    replaced : list
        List containing the replaced values
    """
    replaced = []
    for i in range(1, max_steps + 1):
        # compute n-th value to be replaced
        last_nth = -i * n_ts
        # for sanity check
        replaced.append(curr_frame[last_nth])
        # finally replace actual value
        curr_frame[last_nth] = replace[-i]
    return curr_frame, replaced


def predict_multi_step(model, data, window_size, n_ts=1, keras=True):
    """Predict multiple steps ahead.

    Replace true values by predicted values, before doing the next prediction.
    At some point, new predictions will be based on past predicted values only.

    Parameters
    ----------

    model : model object
        e.g. Keras or scikit-learn

    data : np.array
        data set on which to predict

    n_ts : int, default 1
        number of unique, non-lagged time series the data set contains.
        This is used for backwards traversing and replacing true values:
        the n-th value to be replaced will becomputed via:
            n_ts * min(window_size, predicted_values)
        The min is calculated since in the beginning the number of predicted
        values will be less then the (lagged) ``window_size``.
        For a data set consisting only of one time series and its lagged values
        this variable is 1. In fact this variable is needed when there is more
        than one time series in the data set + their corresponding lags of size
        ``window_size``: in this case, we want to replace the values of the
        predicted time series only, so we need to step over the other time
        series when doing backward traversing.

    window_size : int
        window size of past time stamps the prediction should be based on

    keras : bool
        True if working with Keras else False

    Returns
    -------
    predicted: list
        predictions of the model

    curr_frame : np.array
        Most recent subset of the data set that was predicted on

    replaced : list
        List containing the replaced values
    """
    curr_frame = data[0, :]
    predicted = []
    # for sanity check
    replaced = {}
    replaced_frames = []
    for i in range(len(data)):
        if keras:
            p = model.predict(curr_frame[np.newaxis, :])[0]
        else:
            p = model.predict(curr_frame)
        predicted.append(p)
        if (i + 1) == len(data):
            break

        # update current frame: replace by next one in the sequence
        curr_frame = data[i + 1, :]

        # replace values in time series with predicted one
        curr_frame, replaced[i] = replace_every_last_nth(
            curr_frame, predicted, max_steps=min(window_size, len(predicted)),
            n_ts=n_ts)
        replaced_frames.append(curr_frame)

    return predicted, curr_frame, replaced, replaced_frames


def forecast_multi_step(model, data, forecast_len, keras=1):
    """Multi step ahead forecasting.

    This is a simpler version of the ``predict_multi_step`` function for the
    case of doing multi step ahead forecasting on a one 1-dim time series.
    After each prediction, shift the time series by 1 time stamp and append
    the newly predicted value to its end. Use this newly created time series
    for the next prediction. Repeat this process ``forecast_len`` times.

    Parameters
    ----------
    model : object
        Keras or scikit learn model object

    data : np.array
        1 dimensional time series data

    forecast_len : int
        length of the forecast

    keras : int, default 1
        if 1, Keras is being used, so we need adjust the dimension of the
        time series data accordingly.

    Returns
    -------
    result : dict
        dictionary containing the predictions of size ``forecast_len``,
        the last used frame ``curr_frame`` and all ``replaced_frames`` to allow
        sanity checking.

    """
    curr_frame = data
    predicted = []
    # sanity check
    replaced_frames = []
    for i in range(forecast_len):
        replaced_frames.append(curr_frame)

        if keras:
            if curr_frame.ndim == 1:
                curr_frame = curr_frame[np.newaxis, :]
            p = model.predict(curr_frame)[0]
        else:
            p = model.predict(curr_frame)

        predicted.append(p)

        # shift current frame by 1 time stamp and append
        # newly predicted value to the end of the array
        if curr_frame.ndim == 2:  # keras shape
            curr_frame = np.append(curr_frame[0][1:], p)[np.newaxis, :]
        else:
            curr_frame = np.append(curr_frame[1:], p)

    result = {
        'predictions': predicted,
        'last_frame': curr_frame,
        'replaced_frames': replaced_frames
    }
    return result
