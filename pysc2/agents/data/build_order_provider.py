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

from pysc2.lib import actions, features, units
from pysc2.agents.data.terran_build_order import TerranBuildOrder
import random

DEFAULT_BUILD_LENGTH = 20
DEFAULT_RECRUIT_LENGTH = 100

UNITS_TO_CHOOSE = [units.Terran.SCV, units.Terran.Marine]
BUILDINGS_TO_CHOOSE = [units.Terran.SupplyDepot, units.Terran.Barracks, units.Terran.Refinery]


class BuildOrderProvider(object):

    def provide(self) -> TerranBuildOrder:
        raise NotImplementedError('Cannot call methods of abstract class')


class RandomBuildOrderProvider(BuildOrderProvider):

    def __init__(self, recruit_length=DEFAULT_RECRUIT_LENGTH, build_length=DEFAULT_BUILD_LENGTH) -> None:
        super().__init__()
        self.recruit_length = recruit_length
        self.build_length = build_length

    def provide(self) -> TerranBuildOrder:
        recruit_order = [random.choice(UNITS_TO_CHOOSE) for _ in range(0, self.recruit_length)]
        build_order = [random.choice(BUILDINGS_TO_CHOOSE) for _ in range(0, self.build_length)]
        build_order = [(random.randint(10, self.recruit_length-10), building) for building in build_order]
        build_order.sort(key=lambda order: order[0])
        return TerranBuildOrder(recruit_order, build_order)
