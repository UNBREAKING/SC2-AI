import numpy as np
import pandas as pd
import math
import os.path
from os import path

class RewardCollector:
  def __init__(self, tableName = None):
    self.table = self.load_table(tableName) if path.exists(tableName) else pd.DataFrame(columns=['reward'], dtype=np.float64)

  def save_table(self, filepath):
    self.table.to_csv(filepath)
        
  def load_table(self, filepath):
    return pd.read_csv(filepath,  index_col = 0)

  def collectReward(self, reward):
    self.table = self.table.append({'reward': reward }, ignore_index=True)