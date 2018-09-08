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

from pysc2.agents.stages.terran_refresh_state_stage import TerranRefreshStateStage
from pysc2.agents.stages.terran_build_stage import TerranBuildStage
from pysc2.agents.stages.terran_move_stage import TerranMoveStage
from pysc2.agents.stages.terran_recruit_stage import TerranRecruitStage
from pysc2.agents.stages.terran_scout_stage import TerranScoutStage

STAGE_SUCCESSOR = {
    type(None): TerranRefreshStateStage,
    TerranRefreshStateStage: TerranRecruitStage,
    TerranRecruitStage: TerranMoveStage,
    TerranMoveStage: TerranScoutStage,
    TerranScoutStage: TerranBuildStage,
    TerranBuildStage: TerranRecruitStage
}


class StageProvider(object):

    @staticmethod
    def provide_next_stage(stage):
        return STAGE_SUCCESSOR[type(stage)]
