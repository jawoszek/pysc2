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

import random

from pysc2.lib.point import Point

FUNCTIONS = actions.FUNCTIONS


class TerranState(object):

    def __init__(self, minimap_size):
        self.already_recruit = {
            units.Terran.SCV: 12
        }
        self.already_built = {
            units.Terran.CommandCenter: 1
        }

        self.current_loc = None
        self.current_main_cc_loc = None

        self.recruit_order_pos = 0
        self.build_order_pos = 0

        self.SCOUT_GROUP = 1
        self.CCS_GROUP = 6
        self.RACKS_GROUP = 7

        self.MINIMAP_SIZE = minimap_size

        self.currently_building = None
        self.army_selected = False

        self.current_scout_target = None
        self.current_scout_list = None

    def centered_at_cc(self):
        return self.current_loc == self.current_main_cc_loc

    def add_building(self, building, count=1):
        self.already_built[building] = self.already_built.get(building, 0) + count

    def add_unit(self, unit, count=1):
        self.already_recruit[unit] = self.already_recruit.get(unit, 0) + count
