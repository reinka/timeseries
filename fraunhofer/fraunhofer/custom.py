# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd

from .predictions import forecast_multi_step
from .utils import create_date_range


def forecast(models, column, df, forecast_len, window_size,
             start_date=None, freq='15min', plot=1):
    """Multi-step ahead forecast.

    TODO complete doc-string

    Parameters
    ----------
    models
    column
    df
    forecast_len
    window_size
    start_date
    freq
    plot

    Returns
    -------

    """
    # get model
    m = models.get(column).get('model')
    # get scalers
    sx = models.get(column).get('scalerX')
    sy = models.get(column).get('scalerY')

    if not start_date:
        # get last `window_size` data points
        data = df[column][-window_size:]
    else:
        # construct start and end date of prediction data
        # Exclude start date from the date range since this is supposed to be
        # the starting date --> shift date ranges by 1 into the past
        date_range = create_date_range(pivot=start_date,
                                       periods=window_size + 1,
                                       future=False, freq=freq)[:-1]
        data = df[column].loc[date_range]

    # compute future time stamps
    last_timestamp = data.index[-1]
    f_index = pd.date_range(last_timestamp,
                            last_timestamp + timedelta(forecast_len),
                            freq='15min')[:forecast_len]

    # scale data
    data = sx.transform(data.values.reshape(1, -1))

    # forecast
    f = forecast_multi_step(m, data, forecast_len)
    f['predictions'] = sy.inverse_transform(f.get('predictions'))
    f['data_date'] = date_range
    f['prediction_date'] = create_date_range(pivot=start_date, freq='15min',
                                             periods=forecast_len, future=True)

    if plot:
        # rescale forecast and plot
        plt.plot(f_index, f.get('predictions'),
                 label='one step ahead')
        plt.title('Forecast for %s' % (column))
        plt.show()

    return f


def recursive_pred_interval():
    raise NotImplementedError('Function not implemented yet.')
