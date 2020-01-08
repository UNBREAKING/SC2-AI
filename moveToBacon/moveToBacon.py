from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

FUNCTIONS = actions.FUNCTIONS


def get_beacon_loc(mask):
  y, x = mask.nonzero()
  return list(zip(x, y))


class MoveToBaconScriptAgent(base_agent.BaseAgent):

  def step(self, obs):
    super(MoveToBaconScriptAgent, self).step(obs)
    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      player_relative = obs.observation.feature_screen.player_relative
      beacon = get_beacon_loc(player_relative == _PLAYER_NEUTRAL)
      if not beacon:
        return FUNCTIONS.no_op()
      beacon_center = numpy.mean(beacon, axis=0).round()
      return FUNCTIONS.Move_screen("now", beacon_center)
    else:
      return FUNCTIONS.select_army("select")
