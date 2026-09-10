from __future__ import annotations
"""Small NumPy-only neural network for demonstration sensor anomaly scoring.

IMPORTANT: the bundled weights are trained on synthetic engineering scenarios,
not real mine data. The UI explicitly labels this as DEMO-TRAINED. A CSV with
columns matching FEATURES plus `label` (0 normal, 1 hazard) can be used to
train a project-specific model from real historical data.
"""
import os
import numpy as np
from typing import Tuple

FEATURES = ["o2","co","ch4","co2","temp","humidity","strata_vibe_g","cgr","water_level_cm"]

class TinyMLP:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.mu = np.zeros(len(FEATURES)); self.sigma = np.ones(len(FEATURES))
        self.W1 = self.rng.normal(0,.15,(len(FEATURES),12)); self.b1=np.zeros(12)
        self.W2 = self.rng.normal(0,.15,(12,1)); self.b2=np.zeros(1)
        self.trained = False

    @staticmethod
    def _sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=700, lr=.035) -> float:
        X = np.asarray(X,float); y=np.asarray(y,float).reshape(-1,1)
        self.mu=X.mean(0); self.sigma=X.std(0)+1e-6; X=(X-self.mu)/self.sigma
        for _ in range(epochs):
            z1=X@self.W1+self.b1; h=np.tanh(z1); p=self._sig(h@self.W2+self.b2)
            dz2=p-y; dW2=h.T@dz2/len(X); db2=dz2.mean(0)
            dh=dz2@self.W2.T; dz1=dh*(1-h*h); dW1=X.T@dz1/len(X); db1=dz1.mean(0)
            self.W2-=lr*dW2; self.b2-=lr*db2; self.W1-=lr*dW1; self.b1-=lr*db1
        self.trained=True
        pred=(self.predict_proba(X)>=.5).astype(int)
        return float((pred==y).mean())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trained: raise RuntimeError("Model is not trained")
        X=(np.asarray(X,float)-self.mu)/self.sigma
        return self._sig(np.tanh(X@self.W1+self.b1)@self.W2+self.b2).ravel()

    def save(self,path):
        np.savez(path,mu=self.mu,sigma=self.sigma,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2)

    @classmethod
    def load(cls,path):
        d=np.load(path); m=cls(); m.mu=d['mu'];m.sigma=d['sigma'];m.W1=d['W1'];m.b1=d['b1'];m.W2=d['W2'];m.b2=d['b2'];m.trained=True; return m


def synthetic_training_data(n=2400, seed=7):
    rng=np.random.default_rng(seed)
    half=n//2
    normal=rng.normal([20.7,8,.18,900,28,60,.08,.12,5],[.28,5,.07,220,2.5,10,.05,.12,3],(half,9))
    normal=np.clip(normal,[19.6,0,0,300,15,20,0,0,0],[22,35,.55,4200,37,90,.8,1.2,25])
    scenarios=np.array([
        [17.5,15,.40,2200,29,60,.10,.2,8],
        [19.8,85,.30,1800,31,65,.10,.1,8],
        [18.9,20,1.45,1500,30,70,.12,.3,9],
        [20.1,10,.20,1100,28,65,2.5,3.8,7],
        [19.4,35,.60,3100,32,80,.25,1.1,15],
        [20.2,10,.15,950,42,90,.08,.1,7],
        [16.8,120,1.80,6200,38,85,4.0,4.5,40],
    ],dtype=float)
    idx=rng.integers(0,len(scenarios),size=n-half)
    hazard=scenarios[idx]+rng.normal(0,[.22,8,.10,280,1.8,5,.18,.25,4],(n-half,9))
    hazard=np.clip(hazard,[16,0,0,300,-5,10,0,0,0],[22,250,3,10000,55,100,8,8,1000])
    X=np.vstack([normal,hazard]); y=np.concatenate([np.zeros(half,dtype=int),np.ones(n-half,dtype=int)])
    order=rng.permutation(n); return X[order],y[order]


def demo_model() -> TinyMLP:
    m=TinyMLP(); X,y=synthetic_training_data(); m.fit(X,y,epochs=900,lr=.025); return m
