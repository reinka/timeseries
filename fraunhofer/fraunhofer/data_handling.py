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


def add_gradient(df, cols=None, rsuffix='_grad'):
    """Add gradients of features to a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame for whose features the gradient should be computed

    cols : str or list of str, optional
        Subset of features for which the gradient should be computed.

    rsuffix : str, optional, default ``_grad``
        Suffix that should be appended to the the name of the feature for which
        the gradient is being computed.

    Returns
    -------
    gradient : pd.DataFrame
        New DataFrame which contains both old features and newly computed
        gradient features.
    """
    df = df.copy()
    relevant_cols = cols
    # TODO Series handling e.g. df.ndim > 1
    if not relevant_cols:
        relevant_cols = df.columns

    for col in relevant_cols:
        df[col + rsuffix] = np.gradient(df[col])

    return df


def add_lag(df, lag, exclude=[], exclude_original_timeseries=False,
            drop_na=True, predictor=None):
    """Append previous ``lag`` timestamps to each ``df`` row.

    Creates a new DF, where each row contains the values of the current
    timestamp + values from the previous ``lag`` timestamps.

    Examples
    --------
    1-dim case, pandas.Series
    >>> df1 = pd.Series(np.arange(10), name='t')
    >>> add_lag(df1, 5, drop_na=False)
    	t_t-5	t_t-4	t_t-3	t_t-2	t_t-1	t
    0	NaN	NaN	NaN	NaN	NaN	0
    1	NaN	NaN	NaN	NaN	0.0	1
    2	NaN	NaN	NaN	0.0	1.0	2
    3	NaN	NaN	0.0	1.0	2.0	3
    4	NaN	0.0	1.0	2.0	3.0	4
    5	0.0	1.0	2.0	3.0	4.0	5
    6	1.0	2.0	3.0	4.0	5.0	6
    7	2.0	3.0	4.0	5.0	6.0	7
    8	3.0	4.0	5.0	6.0	7.0	8
    9	4.0	5.0	6.0	7.0	8.0	9

    1-dim case, pandas.DataFrame
    >>> df2 = pd.DataFrame(np.arange(10), columns=['t'])
    >>> add_lag(df2, 5)
    	t_t-5	t_t-4	t_t-3	t_t-2	t_t-1	t
    5	0.0	1.0	2.0	3.0	4.0	5
    6	1.0	2.0	3.0	4.0	5.0	6
    7	2.0	3.0	4.0	5.0	6.0	7
    8	3.0	4.0	5.0	6.0	7.0	8
    9	4.0	5.0	6.0	7.0	8.0	9

    Multi-dim case, pandas.DataFrame
    >>> df3 = pd.DataFrame(np.arange(20).reshape(10,2), columns=['A', 'B'])
    >>> add_lag(df3, 5)
    	A_t-5	B_t-5	A_t-4	B_t-4	A_t-3	B_t-3	A_t-2	B_t-2	A_t-1	B_t-1	A	B
    5	0.0	1.0	2.0	3.0	4.0	5.0	6.0	7.0	8.0	9.0	10	11
    6	2.0	3.0	4.0	5.0	6.0	7.0	8.0	9.0	10.0	11.0	12	13
    7	4.0	5.0	6.0	7.0	8.0	9.0	10.0	11.0	12.0	13.0	14	15
    8	6.0	7.0	8.0	9.0	10.0	11.0	12.0	13.0	14.0	15.0	16	17
    9	8.0	9.0	10.0	11.0	12.0	13.0	14.0	15.0	16.0	17.0	18	19

    Extra parameters
    >>> add_lag(df3, 5, exclude=['B'], exclude_original_timeseries=True)
    	A_t-5	A_t-4	A_t-3	A_t-2	A_t-1
    5	0.0	2.0	4.0	6.0	8.0
    6	2.0	4.0	6.0	8.0	10.0
    7	4.0	6.0	8.0	10.0	12.0
    8	6.0	8.0	10.0	12.0	14.0
    9	8.0	10.0	12.0	14.0	16.0

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series

    lag : int
        Number of lags / range to look back.

    exclude : list of str, optional
        List of column names that can be provided if some columns should be
        excluded.

    exclude_original_timeseries : bool, optional
        Set to true if the original time series should be removed, so that only
        past ``lag`` values will be contained in each row.

    drop_na : bool, optional
        Shifting the time series will lead to NAs for the first ``lag`` rows
        (since previous values are not available for those data points).
        Set true, if those rows should be excluded.

    predictor : pandas.Series
        Predictor, that should be appended to the end of the newly resulting
        data frame.

    Returns
    -------
    DataFrame, where each row has been extended to also contain the previous
    ``lag`` values. New columns are named by the follwing pattern:
        column_t-1, column_t-2, ..., column_t-``lag``
    """
    if exclude:
        df = df.drop(exclude, axis=1)

    tmp = df.copy()
    for i in range(1, lag + 1):
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

    return tmp[lag:] if drop_na else tmp


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
