import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"src")
import numpy as np, pandas as pd, lightgbm as lgb
from common import *
tr=pd.read_pickle("/tmp/tr_flagged.pkl"); va=load("data/validation.csv")
dmi=daily_market_index(load("data/train_test.csv"),va)
tr["ly"]=np.log(tr.posted_rate); tr["lrpm"]=tr.ly-np.log(tr.distance)
base=dict(objective="l1",learning_rate=0.05,num_leaves=31,min_data_in_leaf=40,feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,verbose=-1,seed=0,num_threads=1)
def feats(d,variant):
    X=add_features(d,dmi,use_market=False,use_signal=False)
    if variant.get("nodow"): X=X.drop(columns="dow")
    if variant.get("hav"):
        a,b,c,e=[np.radians(d[k]) for k in ["pickup_lat","pickup_lon","delivery_lat","delivery_lon"]]
        X["hav"]=3958.8*2*np.arcsin(np.sqrt(np.sin((c-a)/2)**2+np.cos(a)*np.cos(c)*np.sin((e-b)/2)**2))
        X["dist_over_hav"]=d.distance/X["hav"]
    if variant.get("month"): X["doy"]=d.date.dt.dayofyear
    return X
def fit(A,B,variant,params=None,rounds=400,target="ly"):
    p=dict(base,**(params or {})); Ac=A[~A.outlier]
    XA,XB=feats(Ac,variant),feats(B,variant)
    m=lgb.train(p,lgb.Dataset(XA,Ac[target]),rounds); pr=m.predict(XB)
    return np.exp(pr+(np.log(B.distance.values) if target=="lrpm" else 0))
def sc(B,p):
    m=~B.outlier.values; y=B.posted_rate.values; e=np.abs(p-y); return e[m].mean(),(e[m]/y[m]).mean()*100
folds=[(7,9),(8,10)]
def cv(variant,params=None,rounds=400,target="ly"):
    out=[]
    for a,b in folds:
        A=tr[tr.date.dt.month<=a]; B=tr[(tr.date.dt.month>a)&(tr.date.dt.month<=b)]
        out.append(sc(B,fit(A,B,variant,params,rounds,target)))
    return np.mean([o[0] for o in out]),np.mean([o[1] for o in out])
new=["Charlotte","Knoxville","Laredo","Norfolk","Jackson","San Diego","Chicago","Allentown"]
# city-holdout: pretend 8 real cities are unseen. Use time-holdout Sep-Oct and drop those cities from training.
def cv_city(variant,params=None,rounds=400):
    held=["Cincinnati","Nashville","Phoenix","Richmond","Tulsa","Syracuse","Reno","Savannah"]
    A=tr[(tr.date.dt.month<=8)&~tr.pickup.isin(held)&~tr.delivery.isin(held)]
    B=tr[(tr.date.dt.month>8)&(tr.pickup.isin(held)|tr.delivery.isin(held))]
    B2=tr[(tr.date.dt.month>8)&~(tr.pickup.isin(held)|tr.delivery.isin(held))]
    return sc(B,fit(A,B,variant,params,rounds)), sc(B2,fit(A,B2,variant,params,rounds)), len(B)
print("baseline           ",cv({}))
print("target=log rpm     ",cv({},target="lrpm"))
print("no dow             ",cv({"nodow":1}))
print("+hav               ",cv({"hav":1}))
print("+doy               ",cv({"month":1}))
for nl,md,r,lr in [(15,40,600,0.05),(63,40,400,0.05),(31,100,400,0.05),(31,20,800,0.03),(127,40,600,0.03),(31,40,1200,0.03)]:
    print(f"nl={nl} mdl={md} r={r} lr={lr}",cv({},dict(num_leaves=nl,min_data_in_leaf=md,learning_rate=lr),r))
print("city-holdout (held cities | other cities, n):",cv_city({}))
