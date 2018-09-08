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


class TerranBuildStage(Stage):

    def __init__(self, state: TerranState, parameters: TerranParameters, stage_provider):
        super().__init__(1, state, parameters, stage_provider)
        self.steps_since_ordered_building = None
        self.count_of_building_during_order = None

    def process(self, obs):
        if not self.state.centered_at_cc():
            self.move_screen_to_cc()
            return

        if self.state.currently_building:
            if self.steps_since_ordered_building is not None:
                if self.count_units_on_screen(obs, self.state.currently_building, False) \
                        > self.count_of_building_during_order:
                    print('Built successfully')
                    self.state.build_order_pos += 1
                    self.state.add_building(self.state.currently_building)
                    self.state.currently_building = None
                    self.steps_since_ordered_building = None
                    self.count_of_building_during_order = None
                    self.remaining_actions -= 1
                    return
                elif self.steps_since_ordered_building > 20:
                    print('Abandoned due to timeout')
                    self.state.currently_building = None
                    self.steps_since_ordered_building = None
                    self.count_of_building_during_order = None
                    self.remaining_actions -= 1
                    return
                else:
                    self.steps_since_ordered_building += 1
                    return
            else:
                should_pass_action = self.build(obs, self.state.currently_building)
                if should_pass_action:
                    self.remaining_actions -= 1
                return

        if self.next_build_order_reached(obs) and self.previous_building_built(obs):
            should_pass_action = self.build(obs, self.parameters.build_order_building(self.state))
            if should_pass_action:
                self.remaining_actions -= 1
            return

        self.remaining_actions -= 1
        return

    def next_build_order_reached(self, obs):
        return self.parameters.build_order_reached(obs, self.state)

    def previous_building_built(self, obs):
        if self.state.build_order_pos < 1:
            return True

        previous_building = self.parameters.build_order_building(self.state, self.state.build_order_pos - 1)
        return self.count_units_on_screen(obs, previous_building) > 0

    def build(self, obs, building):
        enough_minerals = self.building_cost(building) <= obs.observation.player.minerals
        if not enough_minerals:
            self.state.currently_building = building
            return True

        if not self.unit_type_selected(obs, units.Terran.SCV):
            self.select_units(obs, units.Terran.SCV, 1)
            return False

        building_action = self.building_action(building)
        building_possible = building_action.id in obs.observation.available_actions

        if not building_possible:
            raise EnvironmentError('Building not possible')

        self.state.currently_building = building
        location_for_building = self.location_for_building(obs, building)
        self.count_of_building_during_order = self.count_units_on_screen(obs, building, False)
        self.steps_since_ordered_building = 0
        self.queue.append(building_action('now', location_for_building))
        return False

    @staticmethod
    def building_action(building):
        return {
            units.Terran.Refinery: FUNCTIONS.Build_Refinery_screen,
            units.Terran.SupplyDepot: FUNCTIONS.Build_SupplyDepot_screen,
            units.Terran.Barracks: FUNCTIONS.Build_Barracks_screen,
            units.Terran.CommandCenter: None
        }[building]

    @staticmethod
    def building_cost(building):
        return {
            units.Terran.Refinery: 75,
            units.Terran.SupplyDepot: 100,
            units.Terran.Barracks: 150,
            units.Terran.CommandCenter: 400
        }[building]

    @staticmethod
    def location_for_building(obs, building_type):
        n = units.Neutral
        minerals_types = [n.MineralField, n.MineralField750, n.RichMineralField, n.RichMineralField750]
        geysers_types = [n.VespeneGeyser, n.RichVespeneGeyser]

        if building_type == units.Terran.Refinery:
            geysers = [unit for unit in obs.observation.feature_units if unit.unit_type in geysers_types]
            if not geysers:
                raise NotImplementedError('No place for refinery')
            chosen_geyser = random.choice(geysers)
            return Point(chosen_geyser.x, chosen_geyser.y)

        minerals = [unit for unit in obs.observation.feature_units if unit.unit_type in minerals_types]
        random_x = random.randint(5, 80)
        random_y = random.randint(5, 80)
        loc = Point(random_x, random_y)
        minerals_condition = lambda given_loc: True

        if minerals:
            sum_x = sum(map(lambda unit: unit.x, minerals))
            sum_y = sum(map(lambda unit: unit.y, minerals))
            minerals_avg_loc_x = sum_x / len(minerals)
            minerals_avg_loc_y = sum_y / len(minerals)
            minerals_avg_loc = Point(minerals_avg_loc_x, minerals_avg_loc_y)
            minerals_condition = lambda given_loc: minerals_avg_loc.dist(given_loc) > 20

        points = [Point(unit.x, unit.y) for unit in obs.observation.feature_units]
        print('loc')
        print(loc)
        print(points)
        while any([loc.dist(point) < 10 for point in points]) or not minerals_condition(loc):
            print('while')
            print(loc)
            random_x = random.randint(5, 80)
            random_y = random.randint(5, 80)
            loc = Point(random_x, random_y)

        return loc
