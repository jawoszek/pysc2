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

from pysc2.lib.point import Point

from pysc2.agents.stages.stage_provider import StageProvider
from pysc2.agents.data.terran_state import TerranState
from pysc2.agents.data.terran_parameters import TerranParameters

FUNCTIONS = actions.FUNCTIONS

# Functions
_BUILD_SUPPLYDEPOT = FUNCTIONS.Build_SupplyDepot_screen.id
_NOOP = FUNCTIONS.no_op.id
_SELECT_POINT = FUNCTIONS.select_point.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_CC = 18
_SCV = 45

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]



CCS_GROUP = 6
BARRACKS_GROUP = 7

SUPPLY_BUFFER = 6

MINIMAP_SIZE = 64


def unit_type_selected(obs, unit_type):
    return \
        (obs.observation.single_select.any() and obs.observation.single_select[0].unit_type == unit_type) \
        or \
        (obs.observation.multi_select.any() and obs.observation.multi_select[0].unit_type == unit_type)


class TerranAgent(base_agent.BaseAgent):
    """A Terran Agent."""

    def __init__(self):
        super().__init__()
        self.state = TerranState(MINIMAP_SIZE)
        self.parameters = TerranParameters()
        self.stage_provider = StageProvider()
        self.stage = self.stage_provider.provide_next_stage(None)(self.state, self.parameters, self.stage_provider)

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        # return FUNCTIONS.no_op()
        # print(list(obs.observation.control_groups))
        # print(list(obs.observation.single_select))
        # print(list(obs.observation.available_actions))

        if self.stage.has_next_action():
            return self.stage.next_action()

        if self.stage.ended():
            print("stage {0} ended".format(type(self.stage)))
            self.stage = self.stage.get_next_stage()
            print("new stage {0}".format(type(self.stage)))
            self.stage.prepare(obs)
        else:
            self.stage.process(obs)

        # print(list(obs.observation.control_groups))
        # print(list(obs.observation.single_select))
        # print(list(obs.observation.available_actions))

        if self.stage.has_next_action():
            return self.stage.next_action()

        return FUNCTIONS.no_op()


def main(unused_argv):
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="Dreamcatcher",
                    players=[sc2_env.Agent(sc2_env.Race.terran),
                             sc2_env.Bot(sc2_env.Race.zerg,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=MINIMAP_SIZE),
                        use_feature_units=True),
                    step_mul=32,
                    game_steps_per_episode=0,
                    visualize=False,
                    ensure_available_actions=False) as env:

                agent = TerranAgent()
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
