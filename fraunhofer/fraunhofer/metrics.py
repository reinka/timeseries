# License AGPL-3.0 or later (http://www.gnu.org/licenses/agpl.html).
import keras.backend as K
import numpy as np

from IPython.core.debugger import set_trace


def rmsle(y_true, y_pred):
    """Compute Root Mean Squared Logarithmic Error.

    Parameters
    ----------
    y_true, y_pred : tensor

    Returns
    -------
    rmsle : float
    """
    p = np.log(y_pred + 1)
    r = np.log(y_true + 1)
    s = np.sum(np.square((p - r)))

    return np.sqrt(s / len(y_pred))


def mda(y_true, y_pred):
    """Compute Mean Directional Accuracy.

    https://en.wikipedia.org/wiki/Mean_Directional_Accuracy_(MDA)

    Parameters
    ----------
    y_true, y_pred : tensor

    Returns
    -------
    mda : tensor
    """
    s = K.equal(K.sign(y_true[1:] - y_true[:-1]),
                K.sign(y_pred[1:] - y_pred[:-1]))
    return K.mean(K.cast(s, K.floatx()))

    return K.sum(K.cast(K.equal(K.sign(y_true[1:] - y_true[:-1]),
                                K.sign(y_pred[1:] - y_pred[:-1])
                                ), K.floatx()
                        )
                 ) / (y_true._keras_shape[0] - 1.0)


def lar(y_true, y_pred):
    """Compute Log Accuracy Ratio.

    https://poseidon01.ssrn.com/delivery.php?ID=391084087124086000077097111124
    064119041027020035083029074065030002112116093006004101106062010057062109039
    083000005008105091095025060001076032095082119072088009005065062055004120113
    068009007087104024123002009077113030004093107000027090113000065121123&EXT
    =pdf

    Parameters
    ----------
    y_true, y_pred : tensor

    Returns
    -------
    lar : float
    """
    return K.log(y_pred / K.clip(y_true, K.epsilon(), None))
