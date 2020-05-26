import numpy as np
import pandas as pd
import math
import os.path
from os import path

class RewardCollector:
  def __init__(self, tableName = None, columnsRow= None):
    self.table = self.load_table(tableName) if path.exists(tableName) else pd.DataFrame(columns=columnsRow if columnsRow else ['reward'], dtype=np.float64)

  def save_table(self, filepath):
    self.table.to_csv(filepath)
        
  def load_table(self, filepath):
    return pd.read_csv(filepath,  index_col = 0)

  def collectReward(self, reward = None, rewardRow = None):
    rewardData = rewardRow if rewardRow else {'reward': reward}
    self.table = self.table.append(rewardData, ignore_index=True)