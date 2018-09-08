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

from pysc2.agents.data.terran_state import TerranState
from pysc2.agents.data.terran_parameters import TerranParameters

from pysc2.lib.point import Point

from sklearn.cluster import KMeans

FUNCTIONS = actions.FUNCTIONS


class Stage(object):

    def __init__(self, remaining_actions: int, state: TerranState, parameters: TerranParameters,
                 stage_provider):
        self.remaining_actions = remaining_actions
        self.queue = []
        self.state = state
        self.parameters = parameters
        self.stage_provider = stage_provider

    def get_next_stage(self):
        next_stage = self.stage_provider.provide_next_stage(self)
        if next_stage is None:
            raise NotImplementedError('Stage {0} is not known to provider'.format(type(self)))
        return next_stage(self.state, self.parameters, self.stage_provider)

    def prepare(self, obs):
        self.queue.append(FUNCTIONS.select_control_group('recall', self.state.CCS_GROUP))

    def process(self, obs):
        raise NotImplementedError('Abstract method')

    def has_next_action(self):
        return self.queue

    def next_action(self):
        return self.queue.pop()

    def ended(self):
        return self.remaining_actions < 1

    @staticmethod
    def units_on_screen(obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    @staticmethod
    def unit_type_selected(obs, unit_type, count=None):
        return \
            (obs.observation.single_select.any() and obs.observation.single_select[0].unit_type == unit_type
             and (count is None or count == 1)) \
            or \
            (obs.observation.multi_select.any() and obs.observation.multi_select[0].unit_type == unit_type
             and (count is None or count == len(obs.observation.multi_select)))

    def move_screen(self, loc):
        self.state.current_loc = loc
        if loc == self.state.current_main_cc_loc:
            self.state.centered_at_cc = True
        else:
            self.state.centered_at_cc = False
        self.queue.append(FUNCTIONS.move_camera(loc))

    def move_screen_to_cc(self):
        self.move_screen(self.state.current_main_cc_loc)

    @staticmethod
    def count_units_on_screen(obs, unit_type=None, only_ready=True):
        return len(
            [unit for unit in obs.observation.feature_units
             if
             (unit_type is None or unit.unit_type == unit_type)
             and
             (not only_ready or unit.build_progress == 0 or unit.build_progress == 100)])

    def select_units(self, obs, unit_type, count: int=None):
        units_on_screen = self.units_on_screen(obs, unit_type)

        if not units_on_screen:
            raise EnvironmentError('No units to select found')

        if count is None:
            unit_to_select = random.choice(units_on_screen)
            unit = self.parameters.screen_point(unit_to_select.x, unit_to_select.y)
            self.queue.append(FUNCTIONS.select_point('select_all_type', unit))
        elif count < 1:
            pass
        else:
            units_to_select = random.choices(units_on_screen, k=count)
            first_unit = self.parameters.screen_point(units_to_select[0].x, units_to_select[0].y)
            self.queue.append(FUNCTIONS.select_point('select', first_unit))
            for unit in units_to_select[1:]:
                next_unit = self.parameters.screen_point(unit.x, unit.y)
                self.queue.append(FUNCTIONS.select_point('toggle', next_unit))

    def select_army(self):
        self.queue.append(FUNCTIONS.select_army("select"))

    @staticmethod
    def can_select_army(obs):
        return FUNCTIONS.select_army.id in obs.observation.available_actions

    def closest_expansion_loc(self, obs, also_visible=False):
        return self.sorted_expansions(obs, also_visible)[0]

    def furthest_expansion_loc(self, obs, also_visible=False):
        return self.sorted_expansions(obs, also_visible, True)[0]

    def sorted_expansions(self, obs, also_visible=False, desc=False):
        expansions = self.all_expansion_locations(obs, also_visible)
        return sorted(expansions, key=self.distance_from_base_to_loc, reverse=desc)

    def all_expansion_locations(self, obs, also_visible=False):
        y, x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.NEUTRAL).nonzero()
        expansions = self.locations_to_points(x, y, screen=False)
        if also_visible:
            return expansions
        visible = self.get_visible_minimap(obs)
        return [expansion for expansion in expansions if expansion not in visible]

    def get_positions_of_enemy_on_minimap(self, obs, only_visible=False, distance_from_visible=None):
        y, x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        enemies = self.locations_to_points(x, y, screen=False)
        if not only_visible:
            return enemies
        visible = self.get_visible_minimap(obs)
        if distance_from_visible is not None:
            return [enemy for enemy in enemies if self.point_in_range_for_any(enemy, visible, distance_from_visible)]
        return [enemy for enemy in enemies if enemy in visible]

    @staticmethod
    def point_in_range_for_any(target, points, dist_range):
        return any([point for point in points if Stage.point_in_range(target, point, dist_range)])

    @staticmethod
    def point_in_range(source, target, dist_range):
        return source.dist(target) <= dist_range

    def get_visible_minimap(self, obs):
        v_y, v_x = (obs.observation.feature_minimap.visibility_map == features.Visibility.VISIBLE).nonzero()
        return self.locations_to_points(v_x, v_y, screen=False)

    def locations_to_points(self, x, y, screen=True):
        transform = self.parameters.screen_point if screen else self.parameters.minimap_point
        return list(map(lambda xy: transform(xy[0], xy[1]), zip(x, y)))

    def distance_from_base_to_loc(self, loc):
        return self.state.current_main_cc_loc.dist(loc)

    def potential_enemy_base_locations(self):
        x = self.state.current_main_cc_loc.x
        y = self.state.current_main_cc_loc.y
        minimap_size = self.state.MINIMAP_SIZE

        return [
            self.parameters.minimap_point(minimap_size - x, minimap_size - y),
            self.parameters.minimap_point(minimap_size - x, y),
            self.parameters.minimap_point(x, minimap_size - y)
        ]

    @staticmethod
    def free_vespene_geyser_on_screen(obs):
        geysers = {units.Neutral.VespeneGeyser, units.Neutral.RichVespeneGeyser}
        refineries = {units.Terran.Refinery, units.Zerg.Extractor, units.Protoss.Assimilator}
        geysers_count = sum([1 for unit in obs.observation.feature_units if unit.unit_type in geysers])
        refineries_count = sum([1 for unit in obs.observation.feature_units if unit.unit_type in refineries])
        return geysers_count > refineries_count
