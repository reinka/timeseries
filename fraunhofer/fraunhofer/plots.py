# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import compute_fft, crosscorr

plt.style.use('default')


def plot_corrmat(df, figsize=(10, 10), print_max=True):
    """Plot correlation matrix as heat map.

    Optional: prints max correlation value for each of the feature.

    Parameters
    ----------
    df : pandas.dataframe
        data set consisting of multiple timeseries

    figsize : tuple of int, optional, default (10, 10)
        matplotlib figure size option

    print_max : bool, optional, default True.
        If the max corr values should be printed out.
    """
    corrmat = df.corr()
    if print_max:
        # set diagonal to 0
        # Get max correlation
        corrmat.values[[np.arange(corrmat.shape[0])] * 2] = 0
        print(corrmat.max())

    f, ax = plt.subplots(figsize=figsize)
    colormap = plt.cm.inferno
    plt.title('Pearson correlation of continuous features', y=1.05, size=15)
    sns.heatmap(corrmat, linewidths=0.1, vmax=1.0, square=True, cmap=colormap,
                linecolor='white', annot=True)

    plt.show()


def plot_raw_random_samples(df, n=5, seq_len=200, start=0, end=None):
    """Plot random samples of raw dataframe.

    If ``df`` contains several columns each column will be plotted separately.

    Parameters
    ----------
    df : pandas.dataframe

    n : int
        number of plots

    seq_len : int
        number of data points each plot should contain

    start : int
        lowest index of the df to start plotting from

    end : int
        highest index of the df to plot up to
    """
    if not end:
        end = df.shape[0] - seq_len
    try:
        _ = df.columns
        for index in np.random.randint(start, end - seq_len, n):
            df[index: index + seq_len].plot(subplots=True, figsize=(15, 15))
    except AttributeError:
        for index in np.random.randint(start, end - seq_len, n):
            df[index: index + seq_len].plot(figsize=(15, 1))
            plt.show()


def plot_against_each_other(df, col1, col2, title, alpha=.3, figsize=(12,3)):
    """Plot two features of a multiple time series against each other.

    df : pandas.dataframe
        dataframe consisting of multiple timeseries with its index
        being the time variable

    col1 : str
        name of the first column

    col2 : str
        name of the second column
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    ax2 = ax1.twinx()
    ax1.plot(df.index, df[col1], 'k-')
    ax2.plot(df.index, df[col2], 'r-', alpha=alpha)

    ax1.set_xlabel('Date')
    ax1.set_ylabel(col1, color='k')
    ax2.set_ylabel(col2, color='r', alpha=alpha)
    plt.title(title)
    plt.show()


def plot_random_samples(preds, obs, date_index, n, start, end, seq_len=50,
                        preds_lag=None, obs_lag=None, train=False):
    title = 'Predictions on '
    title += 'test set' if not train else 'train set'
    fig = plt.figure(figsize=(13, 4))
    if not train:
        plt.plot(date_index, obs, '+-', label='observed', color='blue')
    else:
        plt.plot(date_index, obs, '+-', label='observed', color='black')

    plt.plot(date_index, preds, '+-', color='red', label="predicted")
    plt.title(title, fontsize=20)
    plt.legend()
    plt.show()

    try:
        assert (end - seq_len > 0)
    except AssertionError:
        print('Value Error: end - seq_len leads to ', end - seq_len)
        return

    for index in np.random.randint(start, end - seq_len, n):
        fig, ax = plt.subplots(1, figsize=(13, 4))
        date = date_index[index: index + seq_len]
        if not train:
            ax.plot(date, obs[index: index + seq_len], '+-',
                    label='observed', color='blue')
        else:
            ax.plot(date, obs[index: index + seq_len], '+-',
                    label='observed', color='black')

        ax.plot(date, preds[index: index + seq_len], '+-',
                color='red', label="predicted")

        if preds_lag:
            lagged_index = index + preds_lag
            ax.plot(date, preds[lagged_index:lagged_index + seq_len],
                    '+-', color='.75', label="predicted + shift left by 1")
        if obs_lag:
            lagged_index = index - obs_lag
            ax.plot(date, obs[lagged_index:lagged_index + seq_len],
                    '+-', color='.9', label="observed + shift right by 1")

        plt.xlabel('Time', fontsize=18)
        plt.title(
            'Predictions for the period: {0} to {1}'.format(date[0], date[-1]),
            fontsize=20
        )
        plt.legend()
        plt.show()


def shift_correlate_and_plot(df, predictor, covariate, n=500, plot=True):
    past = []
    for i in range(1, n + 1):
        past.append(pd.concat(
            [df[covariate].shift(i), df[predictor]],
            axis=1)[n:].corr()[predictor][0])
    # use future prod_total values
    future = []
    for i in range(1, n + 1):
        future.append(
            pd.concat(
                [df[covariate].shift(-i), df[predictor]],
                axis=1)[n:].corr()[predictor][0])
    if plot:
        plt.figure(figsize=(13, 4))
        plt.plot(past, label='past')
        plt.plot(future, label='future')
        plt.xlabel('Timesteps into the past / future')
        plt.ylabel('Correlation')
        plt.title(predictor + ' VS. ' + covariate)
        plt.legend()
        plt.show()
    return past, future


def plot_crosscorr(df, relevant_cols, n_shift=150, figsize=(13, 2),
                   autocorr=False):
    """Plot cross correlation for the relevant cols of a multi time series.

    df : pandas.dataframe
        data set consisting of multiple time series

    relevant_cols : list of str
        name of relevant columns

    n_shift : int
        max shift / lag value. Cross-correlation will be computed over the
        range [-n_shift, n_shift]

    figsize : tuple of int
        matplotlib figsize option

    autocorr : bool
        if False autocorrelation will be excluded
    """
    # build cartesian product of relevant columns
    for col1, col2 in ((x, y) for x in relevant_cols for y in relevant_cols):
        if col1 == col2 and not autocorr:
            continue
        plt.figure(figsize=figsize)
        xcorr, index = crosscorr(df[col1], df[col2], n=n_shift)
        print('Lag with highest cross correlation is', index)
        plt.plot(xcorr[0], xcorr[1])
        plt.axvline(x=index, color='red')
        plt.title('Cross-correlation of %s and %s' % (col1, col2))
        plt.show()


def plot_fft(ts, sampling_rate=1, period=True, start=0, end=-1,
             figsize=(15, 7)):
    """Plot Fast Fourier Transform for given time series.

    http://www.cbcity.de/die-fft-mit-python-einfach-erklaert

    Parameters
    ----------
    ts : pandas.Series
        Time series with its index being the datetime time stamps.

    sampling_rate : float, default 1
        Number of samples per second. It is the reciprocal of
        the sampling time, i.e. 1/T, also called the sampling frequency.

    period : boolean, default True
        Indicating if frequency should be converted to hourly period.

    start : int
        Starting index for plotting.

    end : int
        Ending index for plotting.
    """
    N = ts.size
    yf, freq, nyquist_freq = compute_fft(ts, sampling_rate)

    plt.figure(figsize=figsize)
    plt.plot(freq[start:end], 2.0 / N * yf[start:end])
    plt.axvline(nyquist_freq, color='red', label='Nyquist frequency')
    plt.grid()
    plt.title("Fast Fourier Transform for {0}\nDate range: {1} -- {2}".format(
        ts.name, ts.index[0], ts.index[-1]))
    plt.xlabel('Frequency ($Hz$)')
    plt.legend()
    plt.show()

    # convert freq to period
    if period:
        period = 1.0 / freq  # in seconds
        periodh = period / (60.0 * 60.0)  # in hours

        plt.figure(figsize=(15, 6))
        plt.plot(periodh[start:end], 2.0 / N * yf[start:end])
        # plt.xticks([12, 24, 72, 144])
        # plt.xlim(0, 180)
        # plt.ylim(0, 50)
        plt.xlabel('Period ($h$)')
        plt.show()
