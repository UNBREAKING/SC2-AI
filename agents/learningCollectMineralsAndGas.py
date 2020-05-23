import random
import math
import importlib
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from QLearningTable import QLearningTable

# Units Ids
MINERALFIELD = 341
TERRAN_COMMANDCENTER = 18
GEYSER = 342

#Functions
FUNCTIONS = actions.FUNCTIONS
SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

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

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_WORKER,
    ACTION_GATHER
]

MINERALS_REWARD = 1

def get_loc(mask):
  y, x = mask
  return list(zip(x, y))

def get_state(obs):
    unit_type = obs.observation.feature_screen[UNIT_TYPE]
    canSelectWorker = 0
    canGather = 0

    mineral = (unit_type == MINERALFIELD)
    mineral_y, mineral_x = mineral.nonzero()

    if SELECT_IDLE_WORKER in obs.observation['available_actions']:
      canSelectWorker = 1
    else:
      canSelectWorker = 0

    if GATHER in obs.observation['available_actions']:
      canGather = 1
    else:
      canGather = 0

    return (canSelectWorker, canGather), mineral_y, mineral_x

class SmartAgent(base_agent.BaseAgent):
    def __init__(self):
        super(SmartAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_score = 0
        self.episodes = 0

        self.previous_action = None
        self.previous_state = None
    
    def reset(self):
        self.episodes += 1

    def step(self, obs):
        super(SmartAgent, self).step(obs)

        state, mineral_y, mineral_x = get_state(obs)
        current_state = [state[0], state[1]]

        if self.previous_action is not None:
            reward = 0

            if state[0] > 0 :
                reward += MINERALS_REWARD

            self.previous_score = reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[action]

        self.previous_state = current_state
        self.previous_action = action
        
        
        if smart_action == ACTION_DO_NOTHING:
            return FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_WORKER:
            if state[0] == 1:
              return FUNCTIONS.select_idle_worker("select")
        
        elif smart_action == ACTION_GATHER:
            if state[1] == 1:
              return actions.FunctionCall(GATHER, [SCREEN, [mineral_x[10],  mineral_y[10]]])

        return FUNCTIONS.no_op()
