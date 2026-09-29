import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, lightgbm as lgb
from common import *

tr = load("data/train_test.csv"); va = load("data/validation.csv")
dmi = daily_market_index(tr, va)
wm = weight_medians(tr)
tr = clean_features(tr, wm, dmi)
tr["y"] = np.log(tr["posted_rate"])

def metrics(y_true, y_pred):
    e = y_pred - y_true
    return dict(MAE=np.mean(np.abs(e)), MAPE=np.mean(np.abs(e)/y_true)*100, medAPE=np.median(np.abs(e)/y_true)*100,
                RMSE=np.sqrt(np.mean(e**2)), R2=1-np.sum(e**2)/np.sum((y_true-y_true.mean())**2))

P = dict(objective="l1", learning_rate=0.05, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.9,
         bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0, num_threads=1)

def fit_predict(Xtr, ytr, Xte, rounds=600, params=P):
    m = lgb.train(params, lgb.Dataset(Xtr, ytr), rounds)
    return m.predict(Xte)

# ---- pass 1: flag label outliers with out-of-fold residuals (random 5-fold, robust L1 model) ----
X1 = add_features(tr, dmi)
oof = np.zeros(len(tr)); rng = np.random.RandomState(0); folds = rng.randint(0,5,len(tr))
for k in range(5):
    oof[folds==k] = fit_predict(X1[folds!=k], tr.y[folds!=k], X1[folds==k], rounds=300)
tr["resid"] = tr.y - oof
print(np.histogram(tr.resid, bins=[-3,-1.5,-1,-0.6,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.6,1,1.5,3]))
tr["outlier"] = (tr.resid.abs() > 0.5)
print("outliers:", tr.outlier.sum(), tr.outlier.mean())
tr.to_pickle("/tmp/tr_flagged.pkl")

# ---- time-based evaluation: train Jan-Aug, test Sep-Oct (Nov-Dec is 2 months ahead too) ----
cut = pd.Timestamp("2025-09-01")
A = tr[tr.date < cut]; B = tr[tr.date >= cut]
Ac = A[~A.outlier]
print("\nTrain rows", len(A), "clean", len(Ac), "| Test rows", len(B), "clean", (~B.outlier).sum())

def report(name, pred_log, mask_clean=True):
    out = {}
    for lab, m in (("clean", ~B.outlier.values), ("all", np.ones(len(B),bool))):
        out[lab] = metrics(B.posted_rate.values[m], np.exp(pred_log[m]))
    print(f"{name:34s} clean MAE={out['clean']['MAE']:7.1f} MAPE={out['clean']['MAPE']:5.2f}% R2={out['clean']['R2']:.4f} | all MAE={out['all']['MAE']:7.1f} MAPE={out['all']['MAPE']:5.2f}% R2={out['all']['R2']:.4f}")

# baselines
rt = (A.pickup+"|"+A.delivery)
g = Ac.assign(rt=Ac.pickup+"|"+Ac.delivery, rpm=Ac.posted_rate/Ac.distance).groupby(["rt","equipment"]).rpm.median()
Bk = list(zip(B.pickup+"|"+B.delivery, B.equipment))
rpm = np.array([g.get(k, np.nan) for k in Bk]); rpm = np.where(np.isnan(rpm), np.nanmedian(rpm), rpm)
report("baseline: route+equip median $/mi", np.log(rpm*B.distance.values))
gm = Ac.assign(rpm=Ac.posted_rate/Ac.distance).groupby("equipment").rpm.median()
report("baseline: equip median $/mi", np.log(B.equipment.map(gm).values*B.distance.values))

# linear model on log
from sklearn.linear_model import Ridge
def lin(Xa, ya, Xb):
    mu, sd = Xa.mean(), Xa.std()+1e-9
    return Ridge(alpha=1.0).fit((Xa-mu)/sd, ya).predict((Xb-mu)/sd)
XA = add_features(A, dmi); XB = add_features(B, dmi)
report("ridge (log) all rows", lin(XA, A.y, XB))
report("ridge (log) clean rows", lin(XA[~A.outlier], Ac.y, XB))

# LightGBM variants
for name, kw in [("LGBM L1 clean, no mkt/signal", dict(use_market=False, use_signal=False)),
                 ("LGBM L1 clean, +market", dict(use_market=True, use_signal=False)),
                 ("LGBM L1 clean, +market+signal", dict(use_market=True, use_signal=True)),
                 ("LGBM L1 clean, +mkt+sig+doy", dict(use_market=True, use_signal=True, use_calendar=True))]:
    XA = add_features(A, dmi, **kw); XB = add_features(B, dmi, **kw)
    report(name, fit_predict(XA[~A.outlier], Ac.y, XB))
XA = add_features(A, dmi); XB = add_features(B, dmi)
report("LGBM L1 ALL rows (no outlier removal)", fit_predict(XA, A.y, XB))
p2 = dict(P, objective="huber", alpha=0.3)
report("LGBM huber ALL rows", fit_predict(XA, A.y, XB, params=p2))
p3 = dict(P, objective="regression")
report("LGBM L2 clean", fit_predict(XA[~A.outlier], Ac.y, XB, params=p3))
report("LGBM L2 ALL rows", fit_predict(XA, A.y, XB, params=p3))
