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


class TerranRefreshStateStage(Stage):

    def __init__(self, state: TerranState, parameters: TerranParameters, stage_provider):
        super().__init__(1, state, parameters, stage_provider)

    def process(self, obs):
        if obs.first():
            ccs = self.units_on_screen(obs, units.Terran.CommandCenter)

            if not ccs:
                raise EnvironmentError

            cc = random.choice(ccs)
            point = self.parameters.screen_point(cc.x, cc.y)
            self.queue.append(FUNCTIONS.select_point('select_all_type', point))
            return

        if self.unit_type_selected(obs, units.Terran.CommandCenter):
            self.queue.append(FUNCTIONS.select_control_group('set', self.state.CCS_GROUP))
            ccs_y, ccs_x = obs.observation.feature_minimap.selected.nonzero()
            point = self.parameters.screen_point(ccs_x[0], ccs_y[0])
            self.state.current_loc = point
            self.state.current_main_cc_loc = point
            self.remaining_actions -= 1
            return

        raise NotImplementedError('Mid-game refresh state not implemented yet')
