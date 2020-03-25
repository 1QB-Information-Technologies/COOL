"""-----------------------------------------------------------------------------

Copyright (C) 2019-2020 1QBit
Contact info: Pooya Ronagh <pooya@1qbit.com>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------------"""

from gym.envs.registration import register


register(
    id='SADiscrete-v0',
    entry_point='sagym.sagym:SAGymDiscrete',
    max_episode_steps=250000
)

register(
    id='SAContinuousDestructiveObservation-v0',
    entry_point='sagym.sagym:SAGymContinuousDestructiveObservation',
    max_episode_steps=250000
)

register(
    id='SAContinuous-v0',
    entry_point='sagym.sagym:SAGymContinuous',
    max_episode_steps=10000
)


register(
    id='SAContinuousRandomJ-v0',
    entry_point='sagym.sagym:SAGymContinuousRandomJ',
    max_episode_steps=100000
)
