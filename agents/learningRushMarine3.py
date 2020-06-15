import random
import math
import importlib
import numpy as np

from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions
from pysc2.lib import features
from absl import app

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
RALLY_WORKERS = actions.FUNCTIONS.Rally_Workers_screen.id
ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Parameters
SCREEN = [0]

# Features
_AI_NEUTRAL = features.PlayerRelative.NEUTRAL
_AI_ENEMY = features.PlayerRelative.ENEMY
_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_AI_SELF = features.PlayerRelative.SELF
UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_WORKER = 'selectWorker'
ACTION_GATHER = 'gather'
ACTION_SELECT_SCV = 'selectScv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildSupplyDepot'
ACTION_BUILD_BARRACKS = 'buildBarracks'
ACTION_TRAIN_MARINE = 'trainMarine'
ACTION_SELECT_BARRACKS = 'selectBarracks'
ACTION_SELECT_ARMY = 'selectArmy'
# ACTION_ATTACK_BASE = 'attackBase'
ACTION_ATTACK_ENEMY = 'attackEnemy'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_WORKER,
    ACTION_GATHER,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_SELECT_SCV,
    ACTION_BUILD_BARRACKS,
    ACTION_TRAIN_MARINE,
    ACTION_SELECT_BARRACKS,
    ACTION_SELECT_ARMY,
    # ACTION_ATTACK_BASE,
    ACTION_ATTACK_ENEMY
]

REWARD_BUILD_MARINE = 0.6
REWARD_BUILD_SCV = 0.6
REWARD_BUILD_BARRACKS = 0.6
REWARD_KILLED_ENEMY = 1
REWARD_SCV_BUSY = 0.3

def get_loc(mask):
  y, x = mask
  return list(zip(x, y))

def transformLocation(base_top_left, x, x_distance, y, y_distance):
      if not base_top_left:
        return [x - x_distance, y - y_distance]
        
      return [x + x_distance, y + y_distance]

def get_state(obs):
    unit_type = obs.observation.feature_screen[UNIT_TYPE]
    ai_view = obs.observation.feature_screen[_AI_RELATIVE]
    mini_map = obs.observation.feature_minimap[_AI_RELATIVE]
    playerInformation = obs.observation['player']

    mineral_count = playerInformation[1]
    vispen_count = playerInformation[2]
    supply_limit = playerInformation[4]
    scv_count = playerInformation[6]
    idle_workers = playerInformation[7]
    army_count = playerInformation[8]
    army_food_taken = playerInformation[5]

    killed_unit_score = obs.observation['score_cumulative'][5]
    killed_building_score = obs.observation['score_cumulative'][6]

    enemies_dead_count = killed_unit_score + killed_building_score

    supply_depot_count = int(round((supply_limit - 15) / 8))

    enemyX, enemyY = (mini_map == _AI_ENEMY).nonzero()
    player_y, player_x = (mini_map == _AI_SELF).nonzero()
    base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

    targetEnemyBase = [38, 44] if base_top_left else [19, 23]
    
    enemies = get_loc([enemyX, enemyY])
    targetAttackEnemy = None
    if len(enemyX) > 0:
      targetAttackEnemy = enemies[np.argmin(np.array(enemies)[:, 0])] if base_top_left == 1 else enemies[np.argmax(np.array(enemies)[:, 0])]

    canSelectWorker = 1 if SELECT_IDLE_WORKER in obs.observation['available_actions'] else 0
    canGather = 1 if GATHER in obs.observation['available_actions'] else 0
    canBuildBarracks = 1  if BUILD_BARRACKS in obs.observation['available_actions'] else 0
    canTrainScv = 1 if TRAIN_SCV in obs.observation['available_actions'] and supply_limit > (scv_count + army_food_taken) else 0
    canBuildSupplyDepot = 1 if BUILD_SUPPLY_DEPOT in obs.observation['available_actions'] else 0
    canTrainMarine = 1 if TRAIN_MARINE in obs.observation['available_actions'] and supply_limit > (scv_count + army_food_taken) else 0
    canSelectArmy = 1 if army_count > 0 else 0
    isArmyGrows = 1 if army_food_taken > army_count else 0
    hasArmy = 1 if army_count >= 10  else 0
    isEnemyVisible = 1 if len(enemyX) > 0 else 0
    canAttack = 1 if ((len(obs.observation.single_select) > 0 and obs.observation.single_select[0][0] == MARINE) or (len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0][0] == MARINE)) and ATTACK_MINIMAP in obs.observation["available_actions"] else 0

    barracks_y, barracks_x = (unit_type == BARRACKS).nonzero()
    mineral_y, mineral_x =  (unit_type == MINERALFIELD).nonzero()
    terran_center_y, terran_center_x = (unit_type == TERRAN_COMMAND_CENTER).nonzero()


    targetTerranCenter = [terran_center_x.mean(), terran_center_y.mean()] if terran_center_x.any() else None

    targetBarracks = None
    if barracks_y.any():
      i = random.randint(0, len(barracks_y) - 1)
      targetBarracks = [barracks_x[i].mean(), barracks_y[i].mean()]

    mineralTarget = None
    if mineral_y.any():
      i = random.randint(0, len(mineral_y) - 1)
      mineralTarget = [mineral_x[i], mineral_y[i]]

    x = random.randint(1, 3)

    targetForBuild = None
    if terran_center_x.any(): 
      targetForBuild = transformLocation(base_top_left, int(terran_center_x.mean()), 20 , int(terran_center_y.mean()), 0)                
    
    targetForBuildBaracks = None
    if terran_center_x.any(): 
      targetForBuildBaracks = transformLocation(base_top_left,int(terran_center_x.mean()), 0, int(terran_center_y.mean()), 20)

    barracks_count = round(len(barracks_y)/ 137) if barracks_y.any() else 0

    targetScv = None
    scv_y, scv_x = (unit_type == TERRAN_SCV).nonzero()
    if scv_y.any():
      i = random.randint(0, len(scv_y) - 1)
      targetScv = [scv_x[i], scv_y[i]]

    return (isArmyGrows, idle_workers, canAttack, enemies_dead_count, hasArmy ), (canSelectWorker, canGather, canBuildBarracks, canTrainScv, canBuildSupplyDepot, canTrainMarine, canSelectArmy), mineralTarget, targetTerranCenter, targetForBuild, targetScv, targetForBuildBaracks, targetBarracks, targetAttackEnemy, targetEnemyBase

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))), load_qt= 'learningRushMarine3.csv')
        self.rewardTable = RewardCollector(tableName = 'learningRushMarine3Reward.csv')

        self.previous_score = 0
        self.episodes = 0
        self.previous_idle_workers = 0
        self.previous_workers_count = 12
        self.previus_enemy_count = 0

        self.previous_action = None
        self.previous_state = None
    
    def reset(self):
        self.qlearn.save_qtable('learningRushMarine3.csv')
        self.rewardTable.collectReward(rewardRow = self.previous_score)
        self.rewardTable.save_table('learningRushMarine3Reward.csv')
        self.previous_score = 0
        self.episodes += 1
        self.previous_idle_workers = 0
        self.previous_workers_count = 12
        self.previus_enemy_count = 0

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        state, playerInformation, mineralTarget, targetTerranCenter, targetForBuild, targetScv, targetForBuildBaracks, targetBarracks, targetAttackEnemy, targetEnemyBase = get_state(obs)
        current_state = [state[0], state[1], state[2], state[3], state[4]]

        if self.previous_action is not None:
            reward = 0

            if smart_actions[int(self.previous_action)] == ACTION_TRAIN_MARINE and state[0] and state[4] < 1:
              reward += REWARD_BUILD_MARINE

            if smart_actions[int(self.previous_action)] == ACTION_BUILD_BARRACKS and state[1] < self.previous_idle_workers:
              reward += REWARD_BUILD_BARRACKS
            elif smart_actions[int(self.previous_action)] == ACTION_BUILD_SUPPLY_DEPOT and state[1] < self.previous_idle_workers:
              reward += REWARD_BUILD_BARRACKS
            elif state[1] < self.previous_idle_workers:
              reward += REWARD_SCV_BUSY

            # smart_actions[int(self.previous_action)] == ACTION_ATTACK_ENEMY and 
            if state[3] > self.previus_enemy_count:
              reward += REWARD_KILLED_ENEMY
            
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[int(action)]
        
        self.previous_score = { 'reward': obs.reward }
        self.previous_idle_workers = state[1]
        self.previus_enemy_count = state[3]
        self.previous_state = current_state
        self.previous_action = action
        
        
        if smart_action == ACTION_DO_NOTHING or targetTerranCenter == None:
          return FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_WORKER:
          if playerInformation[0] == 1:
            return FUNCTIONS.select_idle_worker("select")
        
        elif smart_action == ACTION_GATHER:
          if playerInformation[1] == 1 and mineralTarget:
            return actions.FunctionCall(GATHER, [SCREEN, mineralTarget])

        elif smart_action == ACTION_SELECT_BARRACKS:
          if targetBarracks:
            return FUNCTIONS.select_point("select_all_type", targetBarracks)

        elif smart_action == ACTION_TRAIN_MARINE:
          if  playerInformation[5] == 1:
            return actions.FunctionCall(TRAIN_MARINE, [SCREEN])

        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
          if  playerInformation[4] == 1:
            return actions.FunctionCall(BUILD_SUPPLY_DEPOT, [SCREEN, targetForBuild])
        
        elif smart_action == ACTION_BUILD_BARRACKS:
          if  playerInformation[2] == 1:
            return actions.FunctionCall(BUILD_BARRACKS, [SCREEN, targetForBuildBaracks])
        
        elif smart_action == ACTION_SELECT_SCV:
          if targetScv:
            return actions.FunctionCall(SELECT_POINT, [SCREEN, targetScv])
            
        elif smart_action == ACTION_SELECT_ARMY:
          if playerInformation[6] == 1:
            return FUNCTIONS.select_army("select")

        elif smart_action == ACTION_ATTACK_ENEMY:
          if state[2] == 1 and targetAttackEnemy:
            return actions.FunctionCall(ATTACK_MINIMAP, [SCREEN, targetAttackEnemy])
        
        # elif smart_action == ACTION_ATTACK_BASE:
        #   if state[2] == 1 and state[4] == 1 and targetEnemyBase:
        #     return actions.FunctionCall(ATTACK_MINIMAP, [SCREEN, targetEnemyBase])

        return FUNCTIONS.no_op()


def main(unused_argv):
  agent = SmartAgent()
  try:
    while True:
      with sc2_env.SC2Env(
        map_name ="Simple64",
        players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
        agent_interface_format=features.AgentInterfaceFormat(
          feature_dimensions=features.Dimensions(screen=84, minimap=64)),
          step_mul=8,
          game_steps_per_episode=0,
          disable_fog=True,
          visualize=True) as env:
            run_loop.run_loop([agent], env, max_episodes=1000)
  except KeyboardInterrupt:
    pass

if __name__ == "__main__":
  app.run(main)