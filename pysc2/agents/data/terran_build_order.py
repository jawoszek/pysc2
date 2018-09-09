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

import random

from pysc2.lib.point import Point

FUNCTIONS = actions.FUNCTIONS

RECRUIT_ORDER_DEFAULT = [
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.SCV,
        units.Terran.Marine,
        units.Terran.SCV,
        units.Terran.Marine,
        units.Terran.SCV,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine,
        units.Terran.Marine
    ]

BUILD_ORDER_DEFAULT = [
    (12, units.Terran.SupplyDepot),
    (14, units.Terran.Barracks),
    (16, units.Terran.SupplyDepot),
    (17, units.Terran.Refinery),
    (18, units.Terran.Barracks),
    (19, units.Terran.SupplyDepot),
    (22, units.Terran.Barracks),
    (23, units.Terran.Refinery),
    (24, units.Terran.SupplyDepot),
    (28, units.Terran.Barracks),
    (34, units.Terran.SupplyDepot),
    (40, units.Terran.SupplyDepot),
    (48, units.Terran.SupplyDepot),
    (56, units.Terran.SupplyDepot),
    (64, units.Terran.SupplyDepot)
]


def recruit_order_pos_to_str(recruit_order):
    return map(lambda unit: str(unit.value), recruit_order)


class TerranBuildOrder(object):

    def __init__(self, recruit_order=None, build_order=None):
        if recruit_order is None:
            self.recruit_order = RECRUIT_ORDER_DEFAULT
        else:
            self.recruit_order = recruit_order

        if build_order is None:
            self.build_order = BUILD_ORDER_DEFAULT
        else:
            self.build_order = build_order

    def build_order_reached(self, obs, state: TerranState):
        return state.build_order_pos < len(self.build_order) \
               and \
               obs.observation.player.food_used >= self.build_order[state.build_order_pos][0]

    def build_order_building(self, state: TerranState, index=None):
        if index is None:
            index = state.build_order_pos

        return self.build_order[index][1] if index < len(self.build_order) else None

    def recruit_order_finished(self, state: TerranState):
        return state.recruit_order_pos >= len(self.recruit_order)

    def recruit_order_next(self, state: TerranState):
        return self.recruit_order[state.recruit_order_pos]

    def storage_format(self):
        return "{0},{1}".format(self.build_order_storage_format(), self.recruit_order_storage_format())

    def build_order_storage_format(self):
        return '[' + ','.join(["({0},{1})".format(pop, building) for pop, building in self.build_order]) + ']'

    def recruit_order_storage_format(self):
        return '[' + ','.join(recruit_order_pos_to_str(self.recruit_order)) + ']'
