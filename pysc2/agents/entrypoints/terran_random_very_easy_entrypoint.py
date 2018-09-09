
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

from absl import app

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.agents.terran_agent import TerranAgent
from pysc2.agents.data.build_order_provider import BuildOrderProvider, RandomBuildOrderProvider
from websocket import WebSocketTimeoutException

import random

from pysc2.lib.point import Point

FUNCTIONS = actions.FUNCTIONS

DEFAULT_MINIMAP_SIZE = 64
DEFAULT_SCREEN_SIZE = 84
DEFAULT_STEP_MUL = 32


def supply_blocked(build_order):
    current_cap = 15
    order_pairs = build_order.build_order
    for pop, building in order_pairs:
        if pop > current_cap:
            break
        if building == units.Terran.SupplyDepot:
            current_cap += 8
    if current_cap < 50:
        return True
    return False


class TerranRandomVeryEasyEntrypoint(object):

    def __init__(self, minimap_size=DEFAULT_MINIMAP_SIZE,
                 screen_size=DEFAULT_SCREEN_SIZE, step_mul=DEFAULT_STEP_MUL) -> None:
        super().__init__()
        self.build_order_provider = RandomBuildOrderProvider()
        self.minimap_size = minimap_size
        self.screen_size = screen_size
        self.step_mul = step_mul

    def main(self, unused_argv):
        try:
            while True:
                build_order = self.build_order_provider.provide()
                with open("results_random_very_easy.txt", "a") as file:
                    file.write(build_order.storage_format())

                if supply_blocked(build_order):
                    print('Build order with no chances of winning, default lose')
                    with open("results_random_very_easy.txt", "a") as file:
                        file.write(",-1\n")
                    continue

                with sc2_env.SC2Env(
                        map_name="Dreamcatcher",
                        players=[sc2_env.Agent(sc2_env.Race.terran),
                                 sc2_env.Bot(sc2_env.Race.zerg,
                                             sc2_env.Difficulty.very_easy)],
                        agent_interface_format=features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
                            use_feature_units=True),
                        step_mul=self.step_mul,
                        game_steps_per_episode=0,
                        visualize=False,
                        ensure_available_actions=False) as env:

                    agent = TerranAgent(build_order)
                    agent.setup(env.observation_spec(), env.action_spec())

                    timesteps = env.reset()
                    agent.reset()

                    while True:
                        step_actions = [agent.step(timesteps[0])]
                        if timesteps[0].last():
                            print('Finished {0}'.format(timesteps[0].reward))
                            with open("results_random_very_easy.txt", "a") as file:
                                file.write(",{0}\n".format(timesteps[0].reward))
                            break
                        timesteps = env.step(step_actions)

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    while True:
        try:
            entrypoint = TerranRandomVeryEasyEntrypoint()
            app.run(entrypoint.main)
        except (ConnectionError, WebSocketTimeoutException):
            print('TIMEOUT ON APP')
            with open("results_random_very_easy.txt", "a") as file:
                file.write(',0\n')
        except EnvironmentError:
            print('Environmental error')
            with open("results_random_very_easy.txt", "a") as file:
                file.write(',-1\n')
