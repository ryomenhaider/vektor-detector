import numpy as np
import pickle
from pyod.models.iforest import IForest
from sklearn.preprocessing import RobustScaler

class ManipulationDetector:
  def __init__(self, contamination=0.05, n_estimators=200):
    self.scaler = RobustScaler()
    self.model = IForest()
    self.is_fitted = False

  def fit(self, X):
    pass
  def score(self, X):
    pass
  def save(self, path='model.pkl'):
    pass
  def load(self, path='model.pkl'):
    pass