# -*- coding: utf-8 -*-
# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import numpy as np


# from IPython.core.debugger import set_trace


def replace_every_last_nth(curr_frame, replace, max_steps, n):
    """Replace every last nth value of the current frame.

    For a given ``max_steps``, ``curr_frame`` is being traversed backwards
    during which each ``n``th value gets replaced by a value of ``replace``.

    ``replace`` will be also traversed backwards.

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

    n: int
        Number of values to replace

    Returns
    -------

    curr_frame : np.array
        Current frame with replaced values

    replaced : list
        List containing the replaced values
    """
    replaced = []
    for i in range(1, max_steps + 1):
        replaced.append(curr_frame[i * (-n)])
        curr_frame[i * (-n)] = replace[-i]
    return curr_frame, replaced


def predict_sequence_full(model, data, n_cols, window_size, keras=True):
    """Predict on a full sequence of predicted values.

    Replace true values by predicted values, before doing the next prediction.
    At some point, new predictions will be based on past predicted values only.

    Parameters
    ----------

    model : model object
        e.g. Keras or scikit-learn

    data : np.array
        data set on which to predict

    n_cols : int
        number of columns the data set contains

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
    replaced = {}
    for i in range(len(data)):
        if keras:
            p = model.predict(curr_frame[np.newaxis, :])[0]
        else:
            p = model.predict(curr_frame)
        predicted.append(p)
        if (i + 1) == len(data):
            break

        # update current frame to next one in the sequence
        curr_frame = data[i + 1, :]

        # replace values in time series with predicted one
        curr_frame, replaced[i] = replace_every_last_nth(curr_frame, predicted,
                                                         max_steps=min(
                                                             window_size,
                                                             len(predicted)),
                                                         n=n_cols)
    return predicted, curr_frame, replaced
