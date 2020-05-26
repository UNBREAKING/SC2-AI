import random
import math
import importlib
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from QLearningTable import QLearningTable
from RewardCollector import RewardCollector

# Units Ids
MINERALFIELD = 341
TERRAN_COMMAND_CENTER = 18
TERRAN_SCV = 45
SUPPLY_DEPOT = 19
BARRACKS  = 21
MARINE = 48

#Functions
FUNCTIONS = actions.FUNCTIONS
SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id

# Parameters
SCREEN = [0]

# Features
_AI_NEUTRAL = features.PlayerRelative.NEUTRAL
_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_AI_SELF = features.PlayerRelative.SELF
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_WORKER = 'selectWorker'
ACTION_GATHER = 'gather'
ACTION_SELECT_COMMAND_CENTER = 'selectCommandCenter'
ACTION_TRAIN_SCV = 'trainScv'
ACTION_SELECT_SCV = 'selectScv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildSupplyDepot'
ACTION_BUILD_BARRACKS = 'buildBarracks'
ACTION_TRAIN_MARINE = 'trainMarine'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_WORKER,
    ACTION_GATHER,
    ACTION_SELECT_COMMAND_CENTER,
    ACTION_TRAIN_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_SELECT_SCV,
    ACTION_BUILD_BARRACKS,
    ACTION_TRAIN_MARINE
]

REWARD_COLLECT_MINERAL = 0.3
REWARD_BUILD_MARINE = 0.6
REWARD_BUILD_SCV = 0.1
REWARD_BUILD_BARRACKS = 0.1
REWARD_BUILD_SUPPLY_DEPOT = 0.1
REWARD_SCV_BUSY = 0.1


def transformLocation(base_top_left, x, x_distance, y, y_distance):
      if not base_top_left:
        return [x - x_distance, y - y_distance]
        
      return [x + x_distance, y + y_distance]

def get_state(obs):
    unit_type = obs.observation.feature_screen[UNIT_TYPE]
    playerInformation = obs.observation['player']

    mineral_count = playerInformation[1]
    vispen_count = playerInformation[2]
    supply_limit = playerInformation[4]
    scv_count = playerInformation[6]
    idle_workers = playerInformation[7]
    army_count = playerInformation[8]
    army_food_taken = playerInformation[5]

    player_y, player_x = (obs.observation.feature_screen[_AI_RELATIVE] == _AI_SELF).nonzero()
    base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

    canSelectWorker = 1 if SELECT_IDLE_WORKER in obs.observation['available_actions'] else 0
    canGather = 1 if GATHER in obs.observation['available_actions'] else 0
    canBuildBarracks = 1  if ACTION_BUILD_BARRACKS in obs.observation['available_actions'] else 0
    canTrainScv = 1 if TRAIN_SCV in obs.observation['available_actions'] and supply_limit > (scv_count + army_food_taken) else 0
    canBuildSupplyDepot = 1 if BUILD_SUPPLY_DEPOT in obs.observation['available_actions'] else 0
    canTrainMarine = 1 if TRAIN_MARINE in obs.observation['available_actions'] and supply_limit > (scv_count + army_food_taken) else 0

    mineral = (unit_type == MINERALFIELD)
    supply_depot = (unit_type == SUPPLY_DEPOT)
    barracks = (unit_type == BARRACKS)

    barracks_y, barracks_x = barracks.nonzero()
    supply_depot_y, supply_depot_x = supply_depot.nonzero()
    mineral_y, mineral_x = mineral.nonzero()
    terran_center_y, terran_center_x = (unit_type == TERRAN_COMMAND_CENTER).nonzero()
    targetTerranCenter = [terran_center_x.mean(), terran_center_y.mean()]
    targetForBuild = transformLocation(base_top_left,int(terran_center_x.mean()), 20, int(terran_center_y.mean()), 0)

    supply_depot_count = round(len(supply_depot_y) / 112) if supply_depot_y.any() else 0
    barracks_count = round(len(barracks_y)) if barracks_y.any() else 0
    print(barracks_count)

    targetScv = None
    scv_y, scv_x = (unit_type == TERRAN_SCV).nonzero()
    if scv_y.any():
      i = random.randint(0, len(scv_y) - 1)
      targetScv = [scv_x[i], scv_y[i]]

    return (mineral_count, vispen_count, supply_limit, scv_count, idle_workers, supply_depot_count, barracks_count, army_count), (canSelectWorker, canGather, canBuildBarracks, canTrainScv, canBuildSupplyDepot, canTrainMarine), (mineral_y, mineral_x), targetTerranCenter, targetForBuild, targetScv

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))), load_qt= 'learningBuildMarines.csv')
        self.rewardTable = RewardCollector(tableName = 'learningBuildMarinesReward.csv', columnsRow = ['mineral_count', 'vispen_count', 'scv_count', 'reward'])

        self.previous_score = 0
        self.episodes = 0
        self.previous_mineral_count = 0
        self.previous_scv_count = 12
        self.previous_idle_workers = 12
        self.previous_army_count = 0
        self.previous_supply_depot_count = 0
        self.previous_barracks_count = 0

        self.previous_action = None
        self.previous_state = None
    
    def reset(self):
        self.qlearn.save_qtable('learningBuildMarines.csv')
        self.rewardTable.collectReward(rewardRow = self.previous_score)
        self.rewardTable.save_table('learningBuildMarinesReward.csv')
        self.previous_score = 0
        self.episodes += 1
        self.previous_mineral_count = 0
        self.previous_scv_count = 12
        self.previous_idle_workers = 12
        self.previous_army_count = 0
        self.previous_supply_depot_count = 0
        self.previous_barracks_count = 0


    def step(self, obs):
        super(SmartAgent, self).step(obs)

        state, playerInformation, mineralsPosition, targetTerranCenter, targetForBuild, targetScv = get_state(obs)
        current_state = [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7]]

        if self.previous_action is not None:
            reward = 0

            if state[0] > self.previous_mineral_count :
                  reward += REWARD_COLLECT_MINERAL

            if state[3] > self.previous_scv_count:
                reward += REWARD_BUILD_SCV

            if state[4] < self.previous_idle_workers:
                reward += REWARD_SCV_BUSY

            if state[7] < self.previous_army_count:
                reward += REWARD_BUILD_MARINE

            if state[5] < self.previous_supply_depot_count:
                reward += REWARD_BUILD_SUPPLY_DEPOT

            if state[6] < self.previous_barracks_count:
                reward += REWARD_BUILD_BARRACKS
            
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[int(action)]
        
        self.previous_score = { 'mineral_count': int(state[0]) + ((int(state[3]) - 12) * 50) + (int(state[5]) * 100) + (int(state[6]) * 75), 'vispen_count': state[1], 'scv_count': state[3], 'reward': obs.reward  }
        self.previous_mineral_count = state[0]
        self.previous_scv_count = state[3]
        self.previous_idle_workers = state[4]
        self.previous_army_count = state[7]
        self.previous_state = current_state
        self.previous_action = action
        
        
        if smart_action == ACTION_DO_NOTHING:
          return FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_WORKER:
          if playerInformation[0] == 1:
            return FUNCTIONS.select_idle_worker("select")
        
        elif smart_action == ACTION_GATHER:
          if playerInformation[1] == 1:
            return actions.FunctionCall(GATHER, [SCREEN, [mineralsPosition[1][10],  mineralsPosition[0][10]]])
            
        elif smart_action == ACTION_SELECT_COMMAND_CENTER:
          return actions.FunctionCall(SELECT_POINT, [SCREEN, targetTerranCenter])

        elif smart_action == ACTION_TRAIN_SCV:
          if  playerInformation[3] == 1:
            return actions.FunctionCall(TRAIN_SCV, [SCREEN])

        elif smart_action == ACTION_TRAIN_MARINE:
          if  playerInformation[5] == 1:
            return actions.FunctionCall(TRAIN_MARINE, [SCREEN])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
          if  playerInformation[4] == 1:
            return actions.FunctionCall(BUILD_SUPPLY_DEPOT, [SCREEN, targetForBuild])
        
        elif smart_action == ACTION_BUILD_BARRACKS:
          if  playerInformation[2] == 1:
            return actions.FunctionCall(BUILD_BARRACKS, [SCREEN, targetForBuild])
        
        elif smart_action == ACTION_SELECT_SCV:
          if targetScv and playerInformation[0] == 1:
            return actions.FunctionCall(SELECT_POINT, [SCREEN, targetScv])

        return FUNCTIONS.no_op()
