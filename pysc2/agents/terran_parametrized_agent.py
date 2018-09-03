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
from absl import app

from collections import namedtuple
Point = namedtuple('Point', 'x y')

FUNCTIONS = actions.FUNCTIONS

# Functions
_BUILD_SUPPLYDEPOT = FUNCTIONS.Build_SupplyDepot_screen.id
_NOOP = FUNCTIONS.no_op.id
_SELECT_POINT = FUNCTIONS.select_point.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]

max_count = {
  units.Terran.SCV: 40,
  units.Terran.SupplyDepot: 20,
  units.Terran.Barracks: 1,
  units.Terran.Marine: 100,
  units.Terran.CommandCenter: 3
}

chance_of_building = {
  units.Terran.SCV: 50,
  units.Terran.SupplyDepot: 10,
  units.Terran.Barracks: 5,
  units.Terran.Marine: 60,
  units.Terran.CommandCenter: 3
}

CCS_GROUP = 6
BARRACKS_GROUP = 7


def unit_type_selected(obs, unit_type):
  return \
    (obs.observation.single_select.any() and obs.observation.single_select[0].unit_type == unit_type) \
    or \
    (obs.observation.multi_select.any() and obs.observation.multi_select[0].unit_type == unit_type)


class TerranParametrizedAgent(base_agent.BaseAgent):
  """A parametrized Terran Agent."""

  base_top_left = None
  supply_depot_built = False
  scv_selected = False

  currently_building = None

  current_loc = None
  current_main_cc_loc = None
  initial_cc_set = False
  centered_at_cc = True

  queue = []

  def step(self, obs):
    super(TerranParametrizedAgent, self).step(obs)

    self.queue.append(FUNCTIONS.no_op())

    if self.queue:
      return self.queue.pop(0)

    # print(list(obs.observation.feature_minimap.player_id))
    # print('-'*10)
    # print(list(obs.observation.feature_minimap.player_relative))
    # print('-' * 10)
    # print(list(obs.observation.feature_minimap.selected))
    # raise EnvironmentError
    # print(obs.observation)
    # print(obs.observation.player.minerals)

    if obs.first() or not self.initial_cc_set:
      self.refresh_main_cc_location(obs)
    elif self.current_main_cc_loc is None:
      raise EnvironmentError

    # assign workers if not working

    if self.currently_building is not None:


      if self.currently_building == units.Terran.SCV:
        if obs.observation.player.minerals >= 50:
          if not unit_type_selected(obs, units.Terran.CommandCenter):
            self.queue.append(FUNCTIONS.select_control_group([0], [CCS_GROUP]))
          else:
            if FUNCTIONS.Train_SCV_quick.id in obs.observation.available_actions:
              self.queue.append(FUNCTIONS.Train_SCV_quick("now"))
            else:
              self.currently_building = None


      if self.currently_building == units.Terran.SupplyDepot:
        if not unit_type_selected(obs, units.Terran.SCV):
          self.queue.append(self.select_worker_for_build(obs))
        elif obs.observation.player.minerals >= 100:
          if FUNCTIONS.Build_SupplyDepot_screen.id in obs.observation.available_actions:
            chosen_location = self.location_for_building(obs)
            print(chosen_location)
            self.queue.append(FUNCTIONS.Build_SupplyDepot_screen("now", chosen_location))
            self.queue.append(FUNCTIONS.select_control_group([0], [CCS_GROUP]))
            self.queue.append(self.move_screen(self.current_main_cc_loc))
          else:
            self.currently_building = None  # TODO move screen if no place avail?


      if self.currently_building == units.Terran.Barracks:
        if obs.observation.control_groups[BARRACKS_GROUP][1] >= max_count[units.Terran.Barracks]:
          self.currently_building = None
        elif obs.observation.player.minerals >= 150:
          if not unit_type_selected(obs, units.Terran.SCV):
            self.queue.append(self.select_worker_for_build(obs))
          else:
            if FUNCTIONS.Build_Barracks_screen.id in obs.observation.available_actions:
              chosen_location = self.location_for_building(obs)
              self.queue.append(FUNCTIONS.Build_Barracks_screen("now", chosen_location))
              self.queue.append(FUNCTIONS.select_control_group([0], [CCS_GROUP]))
              self.queue.append(self.move_screen(self.current_main_cc_loc))
            else:
              self.currently_building = None # TODO move screen if no place avail?


    else:
      to_build = random.choices(list(chance_of_building.keys()), list(chance_of_building.values()))[0]
      if to_build == units.Terran.SCV:
        self.currently_building = to_build
      if to_build == units.Terran.SupplyDepot:
        self.currently_building = to_build
      if to_build == units.Terran.Barracks:
        self.currently_building = to_build


    # print('fff')
    # print(obs)
    # # workers to gathering
    # if FUNCTIONS.select_idle_worker.id in obs.observation.available_actions:
    #   return actions.FunctionCall(264, [[0], [0, 0]])
    #   #return actions.FunctionCall(FUNCTIONS.select_idle_worker.id, [[2]])

    if self.queue:
      return self.queue.pop(0)

    return FUNCTIONS.no_op()

  def transform_location(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
      return [x - x_distance, y - y_distance]

    return [x + x_distance, y + y_distance]

  def move_screen(self, loc):
    self.current_loc = loc
    if loc == self.current_main_cc_loc:
      self.centered_at_cc = True
    else:
      self.centered_at_cc = False
    print('move')
    print(loc)
    return FUNCTIONS.move_camera(loc)

  def refresh_main_cc_location(self, obs):
    if unit_type_selected(obs, units.Terran.CommandCenter):
      ccs_y, ccs_x = obs.observation.feature_minimap.selected.nonzero()
      point = Point(ccs_x[0], ccs_y[0])
      self.current_loc = point
      self.current_main_cc_loc = point
      return

    if obs.observation.control_groups[CCS_GROUP][1] > 0:
      self.queue.append(FUNCTIONS.select_control_group([0], [CCS_GROUP]))
      return

    ccs = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.CommandCenter]

    if not ccs:
      raise EnvironmentError

    cc = random.choice(ccs)
    point = Point(cc.x, cc.y)
    self.queue.append(FUNCTIONS.select_point("select_all_type", point))
    self.queue.append(FUNCTIONS.select_control_group([2], [CCS_GROUP]))

  def select_worker_for_build(self, obs):
    if not self.centered_at_cc:
      return self.move_screen(self.current_main_cc_loc)

    scvs = [unit for unit in obs.observation.feature_units if unit.unit_type == units.Terran.SCV]

    if not scvs:
      raise EnvironmentError

    scv = random.choice(scvs)
    return FUNCTIONS.select_point("select", (scv.x, scv.y))

  def location_for_building(self, obs):
    n = units.Neutral
    minerals_types = [n.MineralField, n.MineralField750, n.RichMineralField, n.RichMineralField750]
    minerals = [unit for unit in obs.observation.feature_units if unit.unit_type in minerals_types]
    center_of_building_area = Point(42, 42)

    if minerals:
      sum_x = sum(map(lambda unit: unit.x, minerals))
      sum_y = sum(map(lambda unit: unit.y, minerals))
      minerals_avg_loc_x = sum_x / len(minerals)
      minerals_avg_loc_y = sum_y / len(minerals)
      center_x = 2 * center_of_building_area.x - minerals_avg_loc_x
      center_y = 2 * center_of_building_area.y - minerals_avg_loc_y
      center_of_building_area = Point(center_x, center_y)

    random_x = random.randint(22, 62)
    random_y = random.randint(22, 62)
    random
    self.smart_screen(center_of_building_area)
    return random_x, random_y


def main(unused_argv):
  agent = TerranParametrizedAgent()
  try:
    while True:
      with sc2_env.SC2Env(
              map_name="Dreamcatcher",
              players=[sc2_env.Agent(sc2_env.Race.terran),
                       sc2_env.Bot(sc2_env.Race.zerg,
                                   sc2_env.Difficulty.very_easy)],
              agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),
              step_mul=16,
              game_steps_per_episode=0,
              visualize=True) as env:

        agent.setup(env.observation_spec(), env.action_spec())

        timesteps = env.reset()
        agent.reset()

        while True:
          step_actions = [agent.step(timesteps[0])]
          if timesteps[0].last():
            break
          timesteps = env.step(step_actions)

  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  app.run(main)