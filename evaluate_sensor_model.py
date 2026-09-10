"""Evaluate the TinyMLP on a labelled CSV with a strict held-out split.
Columns required: all FEATURES from minesafe.models.sensor_ai plus label (0/1).
"""
import argparse, json
import numpy as np, pandas as pd
from minesafe.models.sensor_ai import TinyMLP, FEATURES

def main():
    p=argparse.ArgumentParser(); p.add_argument("csv"); p.add_argument("--test-size",type=float,default=.2); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
    df=pd.read_csv(args.csv); missing=[x for x in FEATURES+["label"] if x not in df.columns]
    if missing: raise SystemExit("Missing columns: "+", ".join(missing))
    rng=np.random.default_rng(args.seed); idx=rng.permutation(len(df)); cut=max(1,int(len(df)*(1-args.test_size))); tr,te=idx[:cut],idx[cut:]
    model=TinyMLP(seed=args.seed); model.fit(df.iloc[tr][FEATURES].values,df.iloc[tr].label.values,epochs=900,lr=.025)
    prob=model.predict_proba(df.iloc[te][FEATURES].values); pred=(prob>=.5).astype(int); y=df.iloc[te].label.values.astype(int)
    tp=int(((pred==1)&(y==1)).sum()); tn=int(((pred==0)&(y==0)).sum()); fp=int(((pred==1)&(y==0)).sum()); fn=int(((pred==0)&(y==1)).sum())
    precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0; f1=2*precision*recall/(precision+recall) if precision+recall else 0
    out={"train_rows":len(tr),"test_rows":len(te),"accuracy":float((pred==y).mean()),"precision":precision,"recall":recall,"f1":f1,"confusion_matrix":{"tn":tn,"fp":fp,"fn":fn,"tp":tp},"seed":args.seed}
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()
