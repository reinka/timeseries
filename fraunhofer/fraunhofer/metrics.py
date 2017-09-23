import numpy as np

def rmsle(y_true, y_pred):
    """Compute Root Mean Squared Logarithmic Error.

    Parameters
    ----------
    y_true : np.array-like
    y_pred : np.array-like

    Returns
    -------
    rmsle : float
    """
    p = np.log(y_pred + 1)
    r = np.log(y_true + 1)
    s = np.sum(np.square((p - r)))

    return np.sqrt(s/len(y_pred))