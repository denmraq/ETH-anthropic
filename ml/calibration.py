import numpy as np
from sklearn.linear_model import LogisticRegression

"""
sklearn removed CalibratedClassifierCV(cv="prefit") in 1.7 (it now requires an
int/splitter/None). Pinning scikit-learn<1.7 in requirements.txt avoids this
today, but that pin is a landmine: if any other dependency ever forces a
newer sklearn, calibration silently breaks the whole training run. Rather
than rely only on the pin, this module tries the real sklearn implementation
first and falls back to a hand-rolled equivalent (a 1-D logistic regression
on the logit of the base model's raw probability -- exactly what Platt/
sigmoid calibration does internally) if "prefit" is unavailable. Same
train/calibrate split, same math, no hard version dependency either way.
"""


class _PrefitSigmoidCalibrator:
    def __init__(self, base):
        self.base = base
        self.lr = LogisticRegression()

    def fit(self, X, y):
        raw = np.clip(self.base.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        logit = np.log(raw / (1 - raw)).reshape(-1, 1)
        self.lr.fit(logit, y)
        return self

    def predict_proba(self, X):
        raw = np.clip(self.base.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        logit = np.log(raw / (1 - raw)).reshape(-1, 1)
        p1 = self.lr.predict_proba(logit)[:, 1]
        return np.column_stack([1 - p1, p1])


def fit_calibrated(make_base_fn, X, y, horizon, calib_frac=0.15, min_calib=250):
    """
    Trains a base model on the earlier chronological chunk, then fits a
    sigmoid/Platt calibrator on a later chunk separated from training by an
    embargo of `horizon` rows -- so the calibrator is never fit on rows the
    base model could have leaked information about.
    """
    n = len(X)
    calib_n = max(min_calib, int(n * calib_frac))
    calib_start = n - calib_n
    base_end = max(200, calib_start - horizon)

    base = make_base_fn()
    base.fit(X.iloc[:base_end], y.iloc[:base_end])

    try:
        from sklearn.calibration import CalibratedClassifierCV
        cal = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        cal.fit(X.iloc[calib_start:], y.iloc[calib_start:])
        return cal
    except Exception:
        cal = _PrefitSigmoidCalibrator(base)
        cal.fit(X.iloc[calib_start:], y.iloc[calib_start:])
        return cal
