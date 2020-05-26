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
GEYSER = 342
SUPPLY_DEPOT = 19
REFINERY = 20

#Functions
FUNCTIONS = actions.FUNCTIONS
SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id

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
ACTION_BUILD_REFINERY = 'buildRefinery'
ACTION_SELECT_COMMAND_CENTER = 'selectCommandCenter'
ACTION_TRAIN_SCV = 'trainScv'
ACTION_SELECT_SCV = 'selectScv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildSupplyDepot'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_WORKER,
    ACTION_GATHER,
    ACTION_BUILD_REFINERY,
    ACTION_SELECT_COMMAND_CENTER,
    ACTION_TRAIN_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_SELECT_SCV
]

REWARD_COLLECT_MINERAL = 0.3
REWARD_COLLECT_GAS = 0.6
REWARD_BUILD_SCV = 0.1
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

    player_y, player_x = (obs.observation.feature_screen[_AI_RELATIVE] == _AI_SELF).nonzero()
    base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

    canSelectWorker = 1 if SELECT_IDLE_WORKER in obs.observation['available_actions'] else 0
    canGather = 1 if GATHER in obs.observation['available_actions'] else 0
    canBuildRefinery = 1  if BUILD_REFINERY in obs.observation['available_actions'] else 0
    canTrainScv = 1 if TRAIN_SCV in obs.observation['available_actions'] and supply_limit > scv_count else 0
    canBuildSupplyDepot = 1 if BUILD_SUPPLY_DEPOT in obs.observation['available_actions'] else 0
    hasIdleWorkers = 1 if idle_workers > 0 else 0

    mineral = (unit_type == MINERALFIELD)
    vespine = (unit_type == GEYSER)
    supply_depot = (unit_type == SUPPLY_DEPOT)
    refinery = (unit_type == REFINERY)

    refinery_y, refinery_x = refinery.nonzero()
    supply_depot_y, supply_depot_x = supply_depot.nonzero()
    mineral_y, mineral_x = mineral.nonzero()
    vespine_y, vespine_x = vespine.nonzero()
    indexOfVespine = random.randint(0, len(vespine_y) - 1)
    targetVespine = [vespine_x[indexOfVespine], vespine_y[indexOfVespine]]
    terran_center_y, terran_center_x = (unit_type == TERRAN_COMMAND_CENTER).nonzero()
    targetTerranCenter = [terran_center_x.mean(), terran_center_y.mean()]
    targetForBuild = transformLocation(base_top_left,int(terran_center_x.mean()), 20, int(terran_center_y.mean()), 0)

    supply_depot_length = round(len(supply_depot_y) / 112) if supply_depot_y.any() else 0
    refinery_length = round(len(refinery_y) / 69) if refinery_y.any() else 0

    targetScv = None
    scv_y, scv_x = (unit_type == TERRAN_SCV).nonzero()
    if scv_y.any():
      i = random.randint(0, len(scv_y) - 1)
      targetScv = [scv_x[i], scv_y[i]]

    return (mineral_count, vispen_count, supply_limit, scv_count, idle_workers, supply_depot_length, refinery_length), (canSelectWorker, canGather, canBuildRefinery, canTrainScv, canBuildSupplyDepot, hasIdleWorkers), (mineral_y, mineral_x), targetVespine, targetTerranCenter, targetForBuild, targetScv

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))), load_qt= 'learningCollectMineralsAndGasV3.csv')
        self.rewardTable = RewardCollector(tableName = 'learningCollectMineralsAndGasV3Reward.csv', columnsRow = ['mineral_count', 'vispen_count', 'scv_count'])


        self.previous_score = 0
        self.episodes = 0
        self.previous_mineral_count = 0
        self.previous_gas_count = 0
        self.previous_scv_count = 12
        self.previous_idle_workers = 12

        self.previous_action = None
        self.previous_state = None
    
    def reset(self):
        self.qlearn.save_qtable('learningCollectMineralsAndGasV3.csv')
        self.rewardTable.collectReward(rewardRow = self.previous_score)
        self.rewardTable.save_table('learningCollectMineralsAndGasV3Reward.csv')
        self.previous_score = 0
        self.episodes += 1
        self.previous_mineral_count = 0
        self.previous_gas_count = 0
        self.previous_scv_count = 12
        self.previous_idle_workers = 12

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        state, playerInformation, mineralsPosition, targetVespine, targetTerranCenter, targetForBuild, targetScv = get_state(obs)
        current_state = [state[0], state[1], state[2], state[3], state[4]]

        if self.previous_action is not None:
            reward = 0

            if state[0] > self.previous_mineral_count :
                  reward += REWARD_COLLECT_MINERAL

            if state[1] > self.previous_gas_count:
                reward += REWARD_COLLECT_GAS

            if state[3] > self.previous_scv_count:
                reward += REWARD_BUILD_SCV

            if state[4] < self.previous_idle_workers:
                reward += REWARD_SCV_BUSY
            
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[int(action)]
        
        self.previous_score = { 'mineral_count': int(state[0]) + ((int(state[3]) - 12) * 50) + (int(state[5]) * 100) + (int(state[6]) * 75), 'vispen_count': state[1], 'scv_count': state[3], 'reward': obs.reward  }
        self.previous_gas_count = state[1]
        self.previous_mineral_count = state[0]
        self.previous_scv_count = state[3]
        self.previous_idle_workers = state[4]
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

        elif smart_action == ACTION_BUILD_REFINERY:
          if playerInformation[2] == 1:
            return actions.FunctionCall(BUILD_REFINERY, [SCREEN, targetVespine])
            
        elif smart_action == ACTION_SELECT_COMMAND_CENTER:
          return actions.FunctionCall(SELECT_POINT, [SCREEN, targetTerranCenter])

        elif smart_action == ACTION_TRAIN_SCV:
          if  playerInformation[3] == 1:
            return actions.FunctionCall(TRAIN_SCV, [SCREEN])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
          if  playerInformation[4] == 1:
            return actions.FunctionCall(BUILD_SUPPLY_DEPOT, [SCREEN, targetForBuild])
        
        elif smart_action == ACTION_SELECT_SCV:
          if targetScv and playerInformation[0] == 1:
            return actions.FunctionCall(SELECT_POINT, [SCREEN, targetScv])

        return FUNCTIONS.no_op()
