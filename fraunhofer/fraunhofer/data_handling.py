# -*- coding: utf-8 -*-
# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from IPython.core.debugger import set_trace


def load_csv(filename, sep=',', timeformat='%Y-%m-%d %H:%M:%S',
             timecol='Zeit', droptime=True, resample_window=None,
             resample_func='mean',
             select_dtypes=None, verbose=False
             ):
    df = pd.read_csv(filename, sep=sep)

    # convert index to datetime
    if timeformat:
        df.index = pd.to_datetime(df[timecol], format=timeformat)
    if droptime:
        df = df.drop(timecol, axis=1)

    if resample_window:
        if resample_func == 'mean':
            df = df.resample(resample_window).mean()
        else:
            df = df.resample(resample_window).sum()

    if select_dtypes:
        df = df.select_dtypes(select_dtypes)
    if verbose:
        df.info()

    return df


def create_windows(df, n_windows, exclude=[],
                   exclude_original_timeseries=False,
                   drop_na=True, predictor=None):
    """Create new DF, where each row contains current values of the timestamp
    + values from the previous ``n_windows`` timestamps.
    """
    if exclude:
        df = df.drop(exclude, axis=1)

    tmp = df.copy()
    for i in range(1, n_windows + 1):
        shifted = df.copy().shift(i)

        # Rename columns
        try:
            shifted.columns = shifted.columns + '_t-%s' % i
        except AttributeError:  # dealing with a series
            shifted.name = shifted.name + '_t-%s' % i

        tmp = pd.concat([shifted, tmp], axis=1)

    if exclude_original_timeseries:
        try:
            tmp = tmp.drop(df.columns, axis=1)
        except AttributeError:  # dealing with a series
            tmp = tmp.drop(df.name, axis=1)

    try:
        if predictor.any():
            tmp[predictor.name] = predictor
    except AttributeError:
        pass

    return tmp[n_windows:] if drop_na else tmp


def train_test_split(df, predictor_colname, exclude_cols=None, scale=True,
                     return_datasets=False,
                     train_ratio=.7, test_ratio=.5, scale_on_train=True):
    """Split Pandas data frame into train test and validation set.

    Parameters
    ----------
    df : pandas.Dataframe

    predictor_colname : str
        name of the predictor column

    exclude_cols : int, optional
        columns to exclude, starting from index 0

    scale: bool, optional
        standardize data before splitting

    return_datasets : bool, optional
        return raw train, test and val data frames

    train_ratio : float, optional
        ratio of train and test set

    test_ratio : float, optional
        ratio of test and val set. After splitting the df into train and test
        set, the test set will be then used to split it again into a test and
        valset.

    scale_on_train : bool, optional
        if scaler should be calibrated based on the train set only.
        if false: scaler will be calibrated on the whole dataset before
        splitting it into train, test and val set

    Returns
    -------
    trX, trY, teX, teY, valX, valY : np.array
        train, test and val covariates and predictor

    scalerX, scalerY : StandardScaler object

    trainset, testset, valset : pd.Dataframe
        raw train, test and validation set
    """
    scalerX = None
    scalerY = None

    if exclude_cols:
        df = df[df.columns[exclude_cols:]]

    N = df.shape[0]

    index = int(train_ratio * N)

    trainset = df[:index]
    test = df[index:]
    index = int(test_ratio * test.shape[0])
    testset, valset = test[:index], test[index:]
    print(trainset.shape, testset.shape, valset.shape)

    trX = trainset.drop(predictor_colname, axis=1).values
    trY = trainset[predictor_colname].values.reshape(-1, 1)

    teX = testset.drop(predictor_colname, axis=1).values
    teY = testset[predictor_colname].values.reshape(-1, 1)

    valX = valset.drop(predictor_colname, axis=1).values
    valY = valset[predictor_colname].values.reshape(-1, 1)

    result = trX, trY, teX, teY, valX, valY

    # standardize with respect to train set
    if scale:
        scalerX = StandardScaler()
        scalerY = StandardScaler()

        if scale_on_train:
            scalerX = scalerX.fit(trX)
            scalerY = scalerY.fit(trY)
        else:
            scalerX = scalerX.fit(df.drop(predictor_colname, axis=1))
            scalerY = scalerY.fit(df[predictor_colname].values.reshape(-1, 1))

        trX = scalerX.transform(trX)
        trY = scalerY.transform(trY)

        teX = scalerX.transform(teX)
        teY = scalerY.transform(teY)

        valX = scalerX.transform(valX)
        valY = scalerY.transform(valY)

        result = trX, trY, teX, teY, valX, valY, scalerX, scalerY

    if return_datasets:
        return result, trainset, testset, valset

    return result


def add_volatility(data, window, rsuffix='_volat'):
    """Compute the volatility for given data and window.

    For a given ``window``, its volatility will be computed by computing the
    standard deviation over that window.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame

    window : int
        backwards range on which to compute the volatility

    rsuffix : str, optional, default ``_volat``
        Suffix that should be appended to the the name of the feature for which
        the gradient is being computed.

    Returns
    -------
    volatility : pd.Dataframe
        DataFrame that contains both the old series / dataframe and the newly
        computed volatility features
    """
    data = data.copy()
    rsuffix += str(window)
    if data.ndim == 1:
        volat = data.rolling(window).std()
        volat.name = data.name + rsuffix
        return pd.concat([data, volat], axis=1)

    for col in data.columns:
        data[col + rsuffix] = data.rolling(window).std()

    return data
