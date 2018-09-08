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

UNIT_TO_BUILDING = {
    units.Terran.SCV: units.Terran.CommandCenter,
    units.Terran.Marine: units.Terran.Barracks
}

UNIT_TO_COST = {
    units.Terran.SCV: 50,
    units.Terran.Marine: 50
}

UNIT_TO_ACTIONS = {
    units.Terran.SCV: FUNCTIONS.Train_SCV_quick,
    units.Terran.Marine: FUNCTIONS.Train_Marine_quick
}


class TerranRecruitStage(Stage):

    def __init__(self, state: TerranState, parameters: TerranParameters, stage_provider):
        super().__init__(1, state, parameters, stage_provider)
        self.currently_recruiting = None
        self.army_selected = False

    def process(self, obs):
        if not self.state.centered_at_cc():
            self.move_screen_to_cc()
            return

        if not self.parameters.recruit_order_finished(self.state):
            unit_to_recruit = self.parameters.recruit_order_next(self.state)
            if self.recruit(obs, unit_to_recruit):
                self.remaining_actions -= 1
            return

        if obs.observation.player.food_workers < self.state.already_recruit[units.Terran.SCV]:
            unit_to_recruit = units.Terran.SCV
            if self.recruit(obs, unit_to_recruit, replacement=True):
                self.remaining_actions -= 1
            return

        if self.get_already_recruited_army_units() > obs.observation.player.army_count:
            if self.currently_recruiting is None and not self.army_selected and self.can_select_army(obs):
                self.select_army()
                self.army_selected = True
                return

            missing_units = self.get_missing_army_units(obs)
            if missing_units:
                unit_to_recruit = list(missing_units.items())[0][0]
                if self.recruit(obs, unit_to_recruit, replacement=True):
                    self.remaining_actions -= 1
                    return
                return

        self.remaining_actions -= 1
        return

    def recruit(self, obs, unit_to_recruit, replacement=False):
        self.currently_recruiting = unit_to_recruit
        building_to_use = self.get_recruitment_building(unit_to_recruit)
        all_units_to_check_in_queue = self.get_units_recruited_at(building_to_use)
        count_of_buildings_to_use = self.count_units_on_screen(obs, building_to_use)

        if self.get_unit_cost(unit_to_recruit) > obs.observation.player.minerals:
            self.currently_recruiting = None
            return True

        if count_of_buildings_to_use < 1:
            self.currently_recruiting = None
            return True

        already_recruiting = self.get_units_in_queue(obs, all_units_to_check_in_queue)

        if already_recruiting > 3 * count_of_buildings_to_use:
            self.currently_recruiting = None
            return True

        if not self.unit_type_selected(obs, building_to_use):
            self.select_units(obs, building_to_use)
            self.army_selected = False
            return False

        action = self.get_unit_to_actions(unit_to_recruit)

        if action.id not in obs.observation.available_actions:
            print("Unit {0} can't be recruited".format(unit_to_recruit))
            self.currently_recruiting = None
            return True

        self.queue.append(action('now'))
        if not replacement:
            self.state.recruit_order_pos += 1
            self.state.add_unit(unit_to_recruit, 1)
        self.currently_recruiting = None
        return False

    def get_already_recruited_army_units(self):
        return sum([count for unit, count in self.state.already_recruit.items() if unit is not units.Terran.SCV])

    def get_missing_army_units(self, obs):
        selected_units = TerranRecruitStage.get_selected_units_count(obs)
        missing_units = {}
        army_units = {unit: count for unit, count in self.state.already_recruit.items() if unit is not units.Terran.SCV}
        for unit, count in army_units.items():
            missing_count = count - selected_units.get(unit, 0)
            if missing_count > 0:
                missing_units[unit] = missing_count
        return missing_units

    @staticmethod
    def get_selected_units_count(obs):
        selected = TerranRecruitStage.get_selected_units(obs)
        selected_units = {}
        for unit in selected:
            selected_units[unit.unit_type] = selected_units.get(unit.unit_type, 0) + 1
        return selected_units

    @staticmethod
    def get_selected_units(obs):
        single_selected = any(obs.observation.single_select[0])
        return obs.observation.single_select if single_selected else obs.observation.multi_select

    @staticmethod
    def get_units_in_queue(obs, unit_types):
        return len([unit for unit in obs.observation.build_queue if unit.unit_type in unit_types])

    @staticmethod
    def get_recruitment_building(unit):
        return UNIT_TO_BUILDING[unit]

    @staticmethod
    def get_units_recruited_at(building_type):
        return [unit for (unit, building) in UNIT_TO_BUILDING.items() if building == building_type]

    @staticmethod
    def get_unit_cost(unit_type):
        return UNIT_TO_COST[unit_type]

    @staticmethod
    def get_unit_to_actions(unit_type):
        return UNIT_TO_ACTIONS[unit_type]
