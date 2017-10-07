# -*- coding: utf-8 -*-
# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import numpy as np


def crosscorr(datax, datay, lag=0, window=None):
    """Compute lag-n cross correlation.

    Taken from: https://stackoverflow.com/questions/33171413/
                cross-correlation-time-lag-correlation-with-pandas

    datax and datay should be of equal length.

    Parameters
    -----------
    datax, datay : pandas.Series

    lag : int, optional, default 0.

    window :  int, optional, default None.
        For a given window size n, the cross-correlation of datax and datay
        will be calculated over the range [-n, n] with datay always being
        shifted by 1, starting with lag -n until n is reached.

    Returns
    -------
    crosscorr : float or list(floats)
        the cross correlation of both series If ``window`` is set, a list of
        floats with corresponding cross correlation is returned.
    """
    if window:
        win = range(-window, window)
        result = [crosscorr(datax, datay, lag=i) for i in win]
        return [win, result], -window + np.argmax(np.abs(result))

    return datax.corr(datay.shift(lag))


def percent_change(df, lag=1):
    """Compute the percent change from one time stamp to another.

    Parameters
    ----------
    df : pandas.Series or pandas.Dataframe
    lag : int

    Returns
    -------
    percent_change : pd.Series or pd.Dataframe
        The percent change between time stamp t and t - lag.
    """
    return df / (df.shift(lag)) - 1
