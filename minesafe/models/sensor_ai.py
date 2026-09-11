from __future__ import annotations
import numpy as np

FEATURES = ["o2", "co", "ch4", "co2", "temp", "humidity", "strata_vibe_g"]

class TinyMLP:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.mu = np.zeros(len(FEATURES))
        self.sigma = np.ones(len(FEATURES))
        self.W1 = self.rng.normal(0, .15, (len(FEATURES), 12))
        self.b1 = np.zeros(12)
        self.W2 = self.rng.normal(0, .15, (12, 1))
        self.b2 = np.zeros(1)
        self.trained = False

    @staticmethod
    def _sig(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=700, lr=.035) -> float:
        X = np.asarray(X, float)
        y = np.asarray(y, float).reshape(-1, 1)
        self.mu = X.mean(0)
        self.sigma = X.std(0) + 1e-6
        X = (X - self.mu) / self.sigma
        for _ in range(epochs):
            z1 = X @ self.W1 + self.b1
            h = np.tanh(z1)
            p = self._sig(h @ self.W2 + self.b2)
            dz2 = p - y
            dW2 = h.T @ dz2 / len(X)
            db2 = dz2.mean(0)
            dh = dz2 @ self.W2.T
            dz1 = dh * (1 - h * h)
            dW1 = X.T @ dz1 / len(X)
            db1 = dz1.mean(0)
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
        self.trained = True
        pred = (self.predict_proba(X) >= .5).astype(int)
        return float((pred == y).mean())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model is not trained")
        X = (np.asarray(X, float) - self.mu) / self.sigma
        return self._sig(np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2).ravel()

    def save(self, path):
        np.savez(path, mu=self.mu, sigma=self.sigma, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls()
        m.mu = d['mu']
        m.sigma = d['sigma']
        m.W1 = d['W1']
        m.b1 = d['b1']
        m.W2 = d['W2']
        m.b2 = d['b2']
        m.trained = True
        return m


def synthetic_training_data(n=2400, seed=7):
    rng = np.random.default_rng(seed)
    half = n // 2
    normal = rng.normal([20.7, 8.0, 0.18, 900.0, 28.0, 60.0, 0.08], [0.28, 5.0, 0.07, 220.0, 2.5, 10.0, 0.05], (half, 7))
    normal = np.clip(normal, [19.6, 0.0, 0.0, 300.0, 15.0, 20.0, 0.0], [22.0, 35.0, 0.55, 4200.0, 37.0, 90.0, 0.8])
    
    scenarios = np.array([
        [17.5, 15.0, 0.40, 2200.0, 29.0, 60.0, 0.10],
        [19.8, 85.0, 0.30, 1800.0, 31.0, 65.0, 0.10],
        [18.9, 20.0, 1.45, 1500.0, 30.0, 70.0, 0.12],
        [20.1, 10.0, 0.20, 1100.0, 28.0, 65.0, 2.5],
        [19.4, 35.0, 0.60, 3100.0, 32.0, 80.0, 0.25],
        [20.2, 10.0, 0.15, 950.0, 42.0, 90.0, 0.08],
        [16.8, 120.0, 1.80, 6200.0, 38.0, 85.0, 4.0],
    ], dtype=float)
    
    idx = rng.integers(0, len(scenarios), size=n - half)
    hazard = scenarios[idx] + rng.normal(0, [0.22, 8.0, 0.10, 280.0, 1.8, 5.0, 0.18], (n - half, 7))
    hazard = np.clip(hazard, [16.0, 0.0, 0.0, 300.0, -5.0, 10.0, 0.0], [22.0, 250.0, 3.0, 10000.0, 55.0, 100.0, 8.0])
    
    X = np.vstack([normal, hazard])
    y = np.concatenate([np.zeros(half, dtype=int), np.ones(n - half, dtype=int)])
    order = rng.permutation(n)
    return X[order], y[order]


def demo_model() -> TinyMLP:
    m = TinyMLP()
    X, y = synthetic_training_data()
    m.fit(X, y, epochs=900, lr=.025)
    return m