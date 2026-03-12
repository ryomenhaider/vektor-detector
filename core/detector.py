import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pickle
from pyod.models.iforest import IForest
from sklearn.preprocessing import RobustScaler

class ManipulationDetector:
  def __init__(self, contamination=0.05, n_estimators=200):
    self.scaler = RobustScaler()
    self.model = IForest(
      n_estimators=n_estimators,
      contamination=contamination,
      random_state=42,
      n_jobs=-1
    )

    self.is_fitted = False

  def fit(self, X: np.ndarray):
      X_s = self.scaler.fit_transform(X)
      self.model.fit(X_s)
      train_scores = self.model.decision_function(X_s)
      self.score_min = train_scores.min()
      self.score_max = train_scores.max()
      self.is_fitted = True

  def score(self, X: np.ndarray) -> np.ndarray:
      if not self.is_fitted:
          raise RuntimeError('Call fit() first')
      X_s = self.scaler.transform(X)
      raw = self.model.decision_function(X_s)
      
      if self.score_max == self.score_min:
          return np.zeros(len(raw))
      
      normalized = (raw - self.score_min) / (self.score_max - self.score_min)
      return np.clip(normalized, 0, 1)

  def save(self, path='model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump({
            'model': self.model,
            'scaler': self.scaler,
            'score_min': self.score_min,
            'score_max': self.score_max
        }, f)

  def load(self, path='model.pkl'):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    self.model = d['model']
    self.scaler = d['scaler']
    self.score_min = d['score_min']
    self.score_max = d['score_max']
    self.is_fitted = True

