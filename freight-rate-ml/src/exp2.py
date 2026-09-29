import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from common import *
tr=pd.read_pickle("/tmp/tr_flagged.pkl"); va=load("data/validation.csv")
dmi=daily_market_index(load("data/train_test.csv"),va)
P=dict(objective="l1",learning_rate=0.05,num_leaves=31,min_data_in_leaf=40,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=0,num_threads=1)
def gb(X,y,Xt,r=400): return lgb.train(P,lgb.Dataset(X,y),r).predict(Xt)
T0=pd.Timestamp("2025-01-01")
def tfeat(dates, kind):
    d=pd.DataFrame(index=range(len(dates)))
    m=np.log(pd.Series(dates).map(dmi).values)
    d["m"]=m
    if "sq" in kind: d["m2"]=m**2
    if "trend" in kind: d["t"]=(pd.Series(dates)-T0).dt.days.values/300
    if "dow" in kind:
        dw=pd.Series(dates).dt.dayofweek.values
        for k in range(1,7): d[f"d{k}"]=(dw==k).astype(float)
    if "lag" in kind:
        s=dmi.sort_index(); l=s.rolling(14,min_periods=3).mean().shift(1)
        d["ml"]=np.log(pd.Series(dates).map(l).fillna(pd.Series(dates).map(s)).values)
    return d
def run(A,B,kind,alpha=1.0):
    Ac=A[~A.outlier]
    X=add_features(Ac,dmi,use_market=False,use_signal=False); XB=add_features(B,dmi,use_market=False,use_signal=False)
    base_B=gb(X,Ac.y,XB)
    if kind is None: return base_B
    # OOF stage-1 residuals -> daily mean
    oof=np.zeros(len(Ac)); f=np.random.RandomState(0).randint(0,5,len(Ac))
    for k in range(5): oof[f==k]=gb(X[f!=k],Ac.y[f!=k],X[f==k],250)
    res=pd.DataFrame({"date":Ac.date.values,"r":Ac.y.values-oof}).groupby("date").r.mean()
    Ft=tfeat(res.index.values,kind); mu,sd=Ft.mean(),Ft.std()+1e-9
    lm=Ridge(alpha=alpha).fit((Ft-mu)/sd,res.values)
    return base_B+lm.predict((tfeat(B.date.values,kind)-mu)/sd)
def score(B,p):
    m=~B.outlier.values; y=B.posted_rate.values; e=np.abs(np.exp(p)-y)
    return e[m].mean(), (e[m]/y[m]).mean()*100, e.mean()
folds=[("Jan-Jun->Jul-Aug",6,8),("Jan-Jul->Aug-Sep",7,9),("Jan-Aug->Sep-Oct",8,10)]
kinds=[None,"lin","sq","sq+trend","sq+dow","sq+trend+dow","sq+lag","sq+lag+dow"]
rows=[]
for name,a,b in folds:
    A=tr[tr.date.dt.month<=a]; B=tr[(tr.date.dt.month>a)&(tr.date.dt.month<=b)]
    for kd in kinds:
        mae,mape,alle=score(B,run(A,B,kd)); rows.append((name,str(kd),mae,mape,alle))
r=pd.DataFrame(rows,columns=["fold","time_component","MAE_clean","MAPE_clean","MAE_all"])
print(r.pivot(index="time_component",columns="fold",values="MAE_clean").assign(mean=lambda d:d.mean(axis=1)).round(1))
print(r.pivot(index="time_component",columns="fold",values="MAPE_clean").assign(mean=lambda d:d.mean(axis=1)).round(2))
