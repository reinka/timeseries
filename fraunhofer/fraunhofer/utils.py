# -*- coding: utf-8 -*-
# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import numpy as np


def crosscorr(datax, datay, lag=0, n=None):
    """Compute lag-n cross correlation.

    Taken from: https://stackoverflow.com/questions/33171413/
                cross-correlation-time-lag-correlation-with-pandas

    datax and datay should be of equal length.

    Parameters
    -----------
    datax, datay : pandas.Series

    lag : int, optional, default 0.

    window :  int, optional, default None.
        For a given window size n, crosscorr of datax and datay will be
        calculated with datay always being shifted by 1 until n is reached.

    Returns
    -------
    crosscorr : float or list(floats)
        the cross correlation of both series If ``window`` is set, a list of
        floats with corresponding cross correlation is returned.
    """
    if n:
        window = range(-n, n)
        result = [crosscorr(datax, datay, lag=i) for i in window]
        return [window, result], -n + np.argmax(np.abs(result))

    return datax.corr(datay.shift(lag))