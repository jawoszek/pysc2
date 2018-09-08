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

from pysc2.agents.stages.stage import Stage
from pysc2.agents.data.terran_state import TerranState
from pysc2.agents.data.terran_parameters import TerranParameters

from pysc2.lib.point import Point

FUNCTIONS = actions.FUNCTIONS


class TerranScoutStage(Stage):

    def __init__(self, state: TerranState, parameters: TerranParameters, stage_provider):
        super().__init__(1, state, parameters, stage_provider)
        self.scout_selected = False

    def process(self, obs):
        enemies_location_minimap = self.get_positions_of_enemy_on_minimap(obs)

        if not enemies_location_minimap:
            if obs.observation.control_groups[self.state.SCOUT_GROUP][1] < 1:
                if not self.state.centered_at_cc():
                    self.move_screen_to_cc()
                    return
                if not self.unit_type_selected(obs, units.Terran.SCV, 1):
                    self.select_units(obs, units.Terran.SCV, 1)
                    return

                self.queue.append(FUNCTIONS.select_control_group('set', self.state.SCOUT_GROUP))
                return

            self.queue.append(FUNCTIONS.select_control_group('recall', self.state.SCOUT_GROUP))
            if not self.unit_type_selected(obs, units.Terran.SCV, 1):
                self.queue.append(FUNCTIONS.select_control_group('recall', self.state.SCOUT_GROUP))
                return

            if not self.scout_selected:
                self.scout_selected = True
                return

            if self.state.current_scout_target is None:

                if not self.state.current_scout_list:
                    self.state.current_scout_list = self.sorted_expansions(obs)

                target = self.state.current_scout_list.pop()
                self.state.current_scout_target = target
                self.queue.append(FUNCTIONS.Move_minimap('now', target))
                self.remaining_actions -= 1
                return

            scout_y, scout_x = obs.observation.feature_minimap.selected.nonzero()
            scout_location = self.parameters.minimap_point(scout_x[0], scout_y[0])

            if scout_location.dist(self.state.current_scout_target) < 2:
                self.state.current_scout_list = self.filter_close_expansions(scout_location)
                if not self.state.current_scout_list:
                    self.state.current_scout_target = None
                    self.remaining_actions -= 1
                    return
                self.sort_scout_targets_by_distance(scout_location)
                target = self.state.current_scout_list.pop(0)
                self.state.current_scout_target = target
                self.queue.append(FUNCTIONS.Move_minimap('now', target))
                self.remaining_actions -= 1
                return

            self.remaining_actions -= 1
            return

        if obs.observation.control_groups[self.state.SCOUT_GROUP][1] < 1:
            self.state.current_scout_target = None
            self.state.current_scout_list = None
            self.remaining_actions -= 1
            return

        self.remaining_actions -= 1
        return

    def sort_scout_targets_by_distance(self, scout_location):
        self.state.current_scout_list.sort(key=lambda loc: loc.dist(scout_location))

    def filter_close_expansions(self, scout_location, distance=2):
        return [location for location in self.state.current_scout_list if location.dist(scout_location) > distance]
