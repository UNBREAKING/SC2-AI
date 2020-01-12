import random
import math
import importlib
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import units

from QLearningTable import QLearningTable

FUNCTIONS = actions.FUNCTIONS

_AI_ENEMY = features.PlayerRelative.ENEMY
_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_AI_SELF = 1

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectArmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_ATTACK,
]

ROACH_KILL_REWARD = 10
MARINE_KILLED = 1

def get_loc(mask):
  y, x = mask
  return list(zip(x, y))

def get_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units 
        if unit.unit_type == unit_type]

def get_state(obs):
    ai_view = obs.observation.feature_screen[_AI_RELATIVE]
    roachesX, roachesY = (ai_view == _AI_ENEMY).nonzero()
    marinexs, marineys = (ai_view == _AI_SELF).nonzero()
    marinex, mariney = marinexs.mean(), marineys.mean()
    
    roaches = get_loc([roachesX, roachesY])

    target = roaches[np.argmax(np.array(roaches)[:, 1])]
        
    ai_selected = obs.observation.feature_screen[_AI_SELECTED]
    marine_selected = int((ai_selected == 1).any())
    
    return [marine_selected, len(marinexs), len(roachesX)], target

class LearningAgent(base_agent.BaseAgent):
    def __init__(self):
        super(LearningAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_score = 0
        self.reward = 0
        self.episodes = 0

        self.previous_action = None
        self.previous_state = None

    def reset(self):
        self.episodes += 1

    def step(self, obs):
        super(LearningAgent, self).step(obs)

        state, target = get_state(obs)

        if self.previous_action is not None:
            self.reward = 0

            if state[1] < self.previous_state[1]:
                self.reward -= MARINE_KILLED

            if state[2] < self.previous_state[2]:
                self.reward += ROACH_KILL_REWARD

            self.previous_score = self.reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, self.reward, str(state))

        action = self.qlearn.choose_action(str(state))
        smart_action = smart_actions[action]

        self.previous_state = state
        self.previous_action = action
        
        
        if smart_action == ACTION_DO_NOTHING:
            return FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_ARMY:
            return FUNCTIONS.select_army("select")

        elif state[0] and smart_action == ACTION_ATTACK:
            return FUNCTIONS.Attack_screen("now", target)

        return FUNCTIONS.no_op()
