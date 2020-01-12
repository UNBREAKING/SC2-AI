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
_AI_SELF = features.PlayerRelative.SELF

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_ARMY = 'selectArmy'
ACTION_MOVE_SCREEN = 'moveScreen'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_ARMY,
    ACTION_MOVE_SCREEN,
]

MINERALS_REWARD = 1

def get_loc(mask):
  y, x = mask
  return list(zip(x, y))

def get_state(obs):
    ai_view = obs.observation.feature_screen[_AI_RELATIVE]
    meneralsxs, meneralsys = (ai_view == _AI_NEUTRAL).nonzero()
    marinexs, marineys = (ai_view == _AI_SELF).nonzero()
    marinex, mariney = marinexs.mean(), marineys.mean()
        
    marine_on_menerals = np.min(meneralsxs) <= marinex <=  np.max(meneralsxs) and np.min(meneralsys) <= mariney <=  np.max(meneralsys)
        
    ai_selected = obs.observation.feature_screen[_AI_SELECTED]
    marine_selected = int((ai_selected == 1).any())

    marines = get_loc([marinexs, marineys])
    minerals = get_loc([meneralsxs, meneralsys])
    marine_xy = np.mean(marines, axis=0).round()  # Average location.
    distances = np.linalg.norm(np.array(minerals) - marine_xy, axis=1)
    losest_mineral_xy = minerals[np.argmin(distances)] 
    
    return (marine_selected, int(marine_on_menerals)), losest_mineral_xy

class LearningAgent(base_agent.BaseAgent):
    def __init__(self):
        super(LearningAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_score = 0
        self.episodes = 0

        self.previous_action = None
        self.previous_state = None
    
    def reset(self):
        self.episodes += 1

    def step(self, obs):
        super(LearningAgent, self).step(obs)

        state, minerals_loc = get_state(obs)
        current_state = [state[0], state[1]]

        if self.previous_action is not None:
            reward = 0

            if state[1] > 0 :
                reward += MINERALS_REWARD

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
            return FUNCTIONS.Move_screen("now", minerals_loc)

        return FUNCTIONS.no_op()
