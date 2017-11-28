# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import numpy as np
import pandas as pd


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


def percent_change(df, lag=1, rsuffix='_perch'):
    """Compute the percent change from one time stamp to another.

    Parameters
    ----------
    df : pandas.Series or pandas.Dataframe

    lag : int

    rsuffix : str, optional, default '_perch'
        Suffix which will be appended to the current feature name.

    Returns
    -------
    percent_change : pd.Series or pd.Dataframe
        The percent change between time stamp t and t - lag.
    """
    result = df / (df.shift(lag)) - 1

    if df.ndim > 1:
        result.columns = df.columns.values + rsuffix
    else:
        result.name = df.name + rsuffix

    return result


def compute_fft(ts, sampling_rate=1):
    """Compute Fast Fourier Transform for given time series.

    Parameters
    ----------
    ts : pandas.Series
        Time series for which to compute the FFT.

    sampling_rate : float, default 1
        Number of samples per second. It is the reciprocal of
        the sampling time, i.e. 1/T, also called the sampling frequency.

    Returns
    -------
    yf : np.array
        FFT result computed via numpy.fft

    freq : np.array
        Corresponding Discrete Fourier Transform sample frequencies
    """
    N = ts.size

    # sampling time / sample space
    T = 1.0 / sampling_rate

    yf = np.abs(np.fft.fft(ts))
    freq = np.fft.fftfreq(N, T)
    nyquist_freq = 0.5 * sampling_rate

    return yf, freq, nyquist_freq


def create_date_range(pivot, periods, future, freq):
    """Helper function to create a date range.

    Give a pivot date, construct a pd.Datetime range of length `periods` that
    starts / ends at the given pivot.

    Parameters
    ----------
    pivot : str
        Datetime format string relative to which the date range should be
        computed

    periods : int
        Date range expressed in number of periods

    future : bool
        If True:    computes future dates
        If False:   compute past dates

    freq : str
        Frequency string, can have multiples, e.g. '15min'

    Returns
    -------
    date_range : DatetimeIndex

    Examples
    --------
    >>> create_date_range(pivot='2016-07-11 23:15:00', periods=12,
                          future=False, freq='5h')
    DatetimeIndex(['2016-07-09 16:15:00', '2016-07-09 21:15:00',
               '2016-07-10 02:15:00', '2016-07-10 07:15:00',
               '2016-07-10 12:15:00', '2016-07-10 17:15:00',
               '2016-07-10 22:15:00', '2016-07-11 03:15:00',
               '2016-07-11 08:15:00', '2016-07-11 13:15:00',
               '2016-07-11 18:15:00', '2016-07-11 23:15:00'],
              dtype='datetime64[ns]', freq='5H')

    >>> create_date_range(pivot='2016-07-11 23:15:00', periods=5,
                          future=True,  freq='15min')
    DatetimeIndex(['2016-07-11 23:15:00', '2016-07-11 23:30:00',
               '2016-07-11 23:45:00', '2016-07-12 00:00:00',
               '2016-07-12 00:15:00'],
              dtype='datetime64[ns]', freq='15T')
    """
    if future:
        return pd.date_range(start=pivot, periods=periods,
                             freq=freq)

    return pd.date_range(end=pivot, periods=periods,
                         freq=freq)
