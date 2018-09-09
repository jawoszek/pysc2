# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.agents.data.terran_state import TerranState
from pysc2.agents.data.terran_build_order import TerranBuildOrder

import random

from pysc2.lib.point import Point

FUNCTIONS = actions.FUNCTIONS


class TerranParameters(object):

    def __init__(self, build_order: TerranBuildOrder, screen_size, minimap_size):
        self.build_order = build_order
        self.screen_size = screen_size
        self.minimap_size = minimap_size

    def build_order_reached(self, obs, state: TerranState):
        return self.build_order.build_order_reached(obs, state)

    def build_order_building(self, state: TerranState, index=None):
        return self.build_order.build_order_building(state, index)

    def recruit_order_finished(self, state: TerranState):
        return self.build_order.recruit_order_finished(state)

    def recruit_order_next(self, state: TerranState):
        return self.build_order.recruit_order_next(state)

    def screen_point(self, x, y):
        if x < 0:
            print("Screen x coord with wrong value - {0}".format(x))
            x = 0
        if x >= self.screen_size:
            print("Screen x coord with wrong value - {0}".format(x))
            x = self.screen_size - 1
        if y < 0:
            print("Screen y coord with wrong value - {0}".format(y))
            y = 0
        if y >= self.screen_size:
            print("Screen y coord with wrong value - {0}".format(y))
            y = self.screen_size - 1
        return Point(x, y)

    def minimap_point(self, x, y):
        if x < 0:
            print("Minimap x coord with wrong value - {0}".format(x))
            x = 0
        if x >= self.minimap_size:
            print("Minimap x coord with wrong value - {0}".format(x))
            x = self.minimap_size - 1
        if y < 0:
            print("Minimap y coord with wrong value - {0}".format(y))
            y = 0
        if y >= self.minimap_size:
            print("Minimap y coord with wrong value - {0}".format(y))
            y = self.minimap_size - 1
        return Point(x, y)
