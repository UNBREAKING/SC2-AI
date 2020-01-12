import random
import math
import importlib
import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from QLearningTable import QLearningTable

FUNCTIONS = actions.FUNCTIONS

_AI_NEUTRAL = features.PlayerRelative.NEUTRAL
_AI_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_AI_SELECTED = features.SCREEN_FEATURES.selected.index
_AI_SELF = 1

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectArmy'
ACTION_MOVE_SCREEN = 'moveScreen'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_MOVE_SCREEN,
]

BEACON_REWARD = 1

def get_beacon_loc(mask):
  y, x = mask
  return list(zip(x, y))

def get_state(obs):
    ai_view = obs.observation.feature_screen[_AI_RELATIVE]
    beaconxs, beaconys = (ai_view == _AI_NEUTRAL).nonzero()
    marinexs, marineys = (ai_view == _AI_SELF).nonzero()
    marinex, mariney = marinexs.mean(), marineys.mean()
        
    marine_on_beacon = np.min(beaconxs) <= marinex <=  np.max(beaconxs) and np.min(beaconys) <= mariney <=  np.max(beaconys)
        
    ai_selected = obs.observation.feature_screen[_AI_SELECTED]
    marine_selected = int((ai_selected == 1).any())
    
    return (marine_selected, int(marine_on_beacon)), [beaconxs, beaconys]

class LearningAgent(base_agent.BaseAgent):
    def __init__(self):
        super(LearningAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_score = 0

        self.previous_action = None
        self.previous_state = None

    def step(self, obs):
        super(LearningAgent, self).step(obs)

        state, beacon_loc = get_state(obs)
        current_state = [state[0], state[1]]

        if self.previous_action is not None:
            reward = 0

            if state[1] > 0 :
                reward += BEACON_REWARD

            self.previous_score = reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[action]

        self.previous_state = current_state
        self.previous_action = action
        
        
        if smart_action == ACTION_DO_NOTHING:
            return FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_ARMY:
            return FUNCTIONS.select_army("select")

        elif state[0] and smart_action == ACTION_MOVE_SCREEN:
            beacon_center = np.mean(get_beacon_loc(beacon_loc), axis=0).round()
            return FUNCTIONS.Move_screen("now", beacon_center)

        return FUNCTIONS.no_op()
