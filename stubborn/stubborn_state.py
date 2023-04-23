# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

from typing import Mapping
import statistics
import enum
import dataclasses
import functools
import random
import functools
import dataclasses

from frozendict import frozendict
import numpy as np
import more_itertools

from .stubborn_config import StubbornConfig
from .exceptions import SkirmishNotFinishedError, NoFinishedSkirmishesError
from stubborn import county
from stubborn.county import misc
from stubborn.county.typing import Agent, RealNumber
from stubborn.county.constants import ALL_AGENTS


class Move(enum.Enum):
    LEFT = 'left'
    RIGHT = 'right'

    def get_opposite(self) -> Move:
        if self == Move.LEFT:
            return Move.RIGHT
        else:
            assert self == Move.RIGHT
            return Move.LEFT

    def get_opposite_if(self, condition: Any) -> Move:
        return self.get_opposite() if condition else self

    @staticmethod
    def from_neural(neural: int) -> Move:
        if neural == 0:
            return Move.LEFT
        else:
            assert neural == 1
            return Move.RIGHT

    def to_neural(self) -> int:
        if self == Move.LEFT:
            return 0
        else:
            assert self == Move.RIGHT
            return 1

    @staticmethod
    def get_random() -> Move:
        return random.choice((Move.LEFT, Move.RIGHT))



@dataclasses.dataclass(frozen=True, kw_only=True)
class RewardPack:
    left_reward: RealNumber
    right_reward: RealNumber
    estimates_of_left_reward: tuple[RealNumber, RealNumber]
    estimates_of_right_reward: tuple[RealNumber, RealNumber]
    flips: tuple[bool, bool]

    @staticmethod
    def create(stubborn_config: StubbornConfig,
               stubborn_state_or_handicaps: StubbornState | tuple[RealNumber, RealNumber]
               ) -> RewardPack:
        handicaps = (stubborn_state_or_handicaps.handicaps if
                     isinstance(stubborn_state_or_handicaps, StubbornState)
                     else stubborn_state_or_handicaps)
        return RewardPack(
            left_reward=(left_reward := stubborn_config._make_random_reward()),
            right_reward=(right_reward := stubborn_config._make_random_reward()),
            estimates_of_left_reward=stubborn_config._make_reward_estimates(left_reward, handicaps),
            estimates_of_right_reward=stubborn_config._make_reward_estimates(right_reward, handicaps),
            flips=(tuple(random.choices((True, False), k=2)) if stubborn_config.randomize_labels
                   else (False, False))
        )

    @functools.cached_property
    def ideal_move(self) -> Move:
        return Move.from_neural(self.right_reward > self.left_reward)

    @functools.cached_property
    def tempting_moves(self) -> tuple[Move, Move]:
        return tuple(
            Move.from_neural(estimate_of_right_reward > estimate_of_left_reward)
            for estimate_of_left_reward, estimate_of_right_reward in zip(
                self.estimates_of_left_reward, self.estimates_of_right_reward, strict=True
            )
        )

    @functools.cached_property
    def flip_slice_steps(self) -> tuple[int]:
        return tuple((-1 if flip else 1) for flip in self.flips)



@dataclasses.dataclass(frozen=True, kw_only=True)
class StubbornState(county.BaseState):

    stubborn_config: StubbornConfig
    i_turn: int
    reward_pack_by_i_first_turn: dict[int, RewardPack]
    move_pairs: tuple[tuple[Move, Move], ...]
    handicaps: tuple[RealNumber, RealNumber]
    agreed_move_by_i_turn: dict[int, Move]

    def __post_init__(self) -> None:
        assert (sum(self.completed_skirmish_lengths) +
                self.i_skurn == self.i_turn)
        assert len(self.completed_skirmish_lengths) == len(self.agreed_move_by_i_turn)
        assert len(self.move_pairs) == len(self.reward_packs) - 1

    @functools.cached_property
    def agents(self) -> tuple[Agent, ...]:
        return tuple(self.stubborn_config.policy_by_agent)

    @functools.cached_property
    def reward_packs(self) -> tuple[RewardPack, ...]:
        reward_packs = []
        for i in range(self.i_turn + 1):
            try:
                reward_packs.append(self.reward_pack_by_i_first_turn[i])
            except KeyError:
                reward_packs.append(reward_packs[-1])
        assert len(reward_packs) == self.i_turn + 1
        return tuple(reward_packs)

    @functools.cached_property
    def reward_pack(self) -> RewardPack:
        return self.reward_packs[-1]

    @functools.cached_property
    def i_turn_of_current_skirmish_start(self) -> int:
        return max(self.reward_pack_by_i_first_turn)


    @functools.cached_property
    def i_skurn(self) -> int:
        '''Skurn is a turn within the skirmish, i.e. 0 for first turn in skirmish, then 1, 2, etc.'''
        return self.i_turn - self.i_turn_of_current_skirmish_start


    @functools.cached_property
    def completed_skirmish_lengths(self) -> tuple[int, ...]:
        i_turn_pairs = more_itertools.sliding_window(
            sorted(self.reward_pack_by_i_first_turn), 2
        )
        return tuple((new_i_turn - old_i_turn) for old_i_turn, new_i_turn in i_turn_pairs)

    @functools.cached_property
    def mean_completed_skirmish_length(self) -> RealNumber:
        return (statistics.mean(self.completed_skirmish_lengths)
                if self.completed_skirmish_lengths else 1)

    @functools.cached_property
    def _n_agreements_pair_and_n_disagreements_pair(self) -> tuple[tuple[int, int],
                                                                   tuple[int, int]]:
        n_agreements_pair = [0, 0]
        n_disagreements_pair = [0, 0]
        for (i_turn_of_old_skirmish, i_turn_of_new_skirmish) in more_itertools. \
                               sliding_window(sorted(self.reward_pack_by_i_first_turn)[:-1], 2):
            for i_agent in range(2):
                n_agreements_pair[i_agent] += (
                    self.move_pairs[i_turn_of_old_skirmish][i_agent] !=
                    self.move_pairs[i_turn_of_new_skirmish - 1][i_agent]
                )
                n_disagreements_pair[i_agent] += (
                    (i_turn_of_new_skirmish - i_turn_of_old_skirmish >= 2) and
                    (self.move_pairs[i_turn_of_old_skirmish][i_agent] ==
                     self.move_pairs[i_turn_of_new_skirmish - 1][i_agent])
                )
        return (tuple(n_agreements_pair), tuple(n_disagreements_pair))


    @functools.cached_property
    def n_agreements_pair(self) -> tuple[int, int]:
        return self._n_agreements_pair_and_n_disagreements_pair[0]

    @functools.cached_property
    def n_disagreements_pair(self) -> tuple[int, int]:
        return self._n_agreements_pair_and_n_disagreements_pair[1]

    @functools.cached_property
    def agreeabilities(self) -> tuple[RealNumber, RealNumber]:
        '''
        A 2-tuple of an agreeability value for each agent.

        Agreeability is a metric for how likely an agent is TODO THIS METRIC SUCKS
        '''
        return tuple(
            misc.cute_div(
                self.n_agreements_pair[i],
                self.n_agreements_pair[i] + self.n_disagreements_pair[i],
                default=1
            )
            for i in range(2)
        )


    @functools.cached_property
    def left_reward_popularity(self) -> RealNumber:
        return misc.cute_div(
            tuple(self.agreed_move_by_i_turn.values()).count(Move.LEFT),
            len(self.agreed_move_by_i_turn)
        )

    @functools.cached_property
    def right_reward_popularity(self) -> RealNumber:
        return misc.cute_div(
            tuple(self.agreed_move_by_i_turn.values()).count(Move.RIGHT),
            len(self.agreed_move_by_i_turn)
        )

    @functools.cached_property
    def we_played_left_reward_this_skirmish_pair(self) -> tuple[bool, bool]:
        return tuple(
            (self.i_skurn >= 1 and
             self.move_pairs[-1][i].get_opposite_if(self.reward_pack.flips[i]) == Move.LEFT)
            for i in range(2)
        )

    @functools.cached_property
    def we_played_right_reward_this_skirmish_pair(self) -> tuple[bool, bool]:
        return tuple(
            (self.i_skurn >= 1 and
             self.move_pairs[-1][i].get_opposite_if(self.reward_pack.flips[i]) == Move.RIGHT)
            for i in range(2)
        )


    @functools.cached_property
    def _reward_estimates(self):
        return tuple(
            (self.estimates_of_left_reward[i_agent],
             self.estimates_of_right_reward[i_agent])[::self.reward_pack.flip_slice_steps[i_agent]]
            for i_agent in range(2)
        )



    @functools.cached_property
    def observation_by_agent(self) -> dict[Agent, dict]:
        return {
            agent: {
                'i_turn': np.array([self.i_turn], dtype=int),
                'mean_completed_skirmish_length': np.array([self.mean_completed_skirmish_length],
                                                           dtype=np.float32),
                'we_played_left_reward_this_skirmish': np.array(
                    [self.we_played_left_reward_this_skirmish_pair[i_agent]],
                    dtype=bool,
                ),
                'we_played_right_reward_this_skirmish': np.array(
                    [self.we_played_right_reward_this_skirmish_pair[i_agent]],
                    dtype=bool,
                ),
                'i_skurn_capped': np.array(
                    [min(self.i_skurn,
                         self.stubborn_config.i_turn_relative_to_current_skirmish_observation_cap)],
                    dtype=int
                ),
                'handicap': np.array([self.handicaps[i_agent]], dtype=np.float32),
                'reward_estimates': np.array(self._reward_estimates[i_agent], dtype=np.float32),
                'biggest_reward_according_to_estimates': np.array(
                    [1 if (self._reward_estimates[i_agent][1] >=
                           self._reward_estimates[i_agent][0]) else 0],
                    dtype=int,
                ),
                'agreeabilities': np.array(self.agreeabilities, dtype=np.float32),
                'yankee': np.array([self.yankees[i_agent]], dtype=np.float32),
                'zebras': np.array(self.zebras, dtype=np.float32),
            } for i_agent, agent in enumerate(self.agents)
        }

    @functools.cached_property
    def reward_pack_we_just_played_on(self) -> RewardPack:
        wip_reward_pack_by_first_i_turn = dict(self.reward_pack_by_i_first_turn)
        wip_reward_pack_by_first_i_turn.pop(self.i_turn, None)
        return max(wip_reward_pack_by_first_i_turn.items())[1]


    @functools.cached_property
    def reward(self) -> RealNumber:
        if self.we_just_agreed:
            return (self.reward_pack_we_just_played_on.left_reward
                    if self.agreed_move_by_i_turn[self.i_turn - 1] == Move.LEFT
                    else self.reward_pack_we_just_played_on.right_reward)
        else:
            return 0

    @functools.cached_property
    def reward_by_agent(self) -> dict[Agent, RealNumber]:
        return {agent: self.reward for agent in self.agents}

    @functools.cached_property
    def done_by_agent(self) -> dict[Agent, bool]:
        return {agent: (self.i_turn >= (self.stubborn_config.episode_length - 1))
                for agent in self.agents + (ALL_AGENTS,)}


    left_reward = property(lambda self: self.reward_pack.left_reward)
    right_reward = property(lambda self: self.reward_pack.right_reward)
    estimates_of_left_reward = property(lambda self: self.reward_pack.estimates_of_left_reward)
    estimates_of_right_reward = property(lambda self: self.reward_pack.estimates_of_right_reward)

    @staticmethod
    def make_initial(stubborn_config: StubbornConfig) -> StubbornState:

        if None in (stubborn_config.mean_handicap, stubborn_config.handicap_difference):
            raise NotImplementedError

        handicaps = [stubborn_config.mean_handicap - (stubborn_config.handicap_difference / 2),
                     stubborn_config.mean_handicap + (stubborn_config.handicap_difference / 2)]
        for handicap in handicaps:
            assert stubborn_config.min_handicap <= handicap <= stubborn_config.max_handicap


        return StubbornState(
            stubborn_config=stubborn_config,
            i_turn=0,
            handicaps=handicaps,
            reward_pack_by_i_first_turn=frozendict(
                {0: RewardPack.create(stubborn_config, handicaps)}
            ),
            move_pairs=(),
            agreed_move_by_i_turn=frozendict(),
        )

    def step(self, actions: Mapping[Agent, np.ndarray]) -> StubbornState:

        i_turn = self.i_turn + 1

        move_pair = []
        for i, agent in enumerate(self.agents):
            raw_move = Move.from_neural(actions[self.agents[i]][0])
            move_pair.append(
                raw_move.get_opposite_if(self.reward_pack.flips[i])
            )
        move_pairs = self.move_pairs + (move_pair,)

        if move_pairs[-1][0] == move_pairs[-1][1]:
            currently_agreed_move = move_pairs[-1][0]
        elif ((self.i_skurn >= 1) and
              move_pairs[-1][0] != move_pairs[-2][0]):
            # The ole switcheroo.
            assert (move_pairs[-1][0] != move_pairs[-2][0] == move_pairs[-1][1] !=
                    move_pairs[-2][1] == move_pairs[-1][0])
            currently_agreed_move = Move.get_random()
        else:
            currently_agreed_move = None

        if currently_agreed_move is None:
            agreed_move_by_i_turn = self.agreed_move_by_i_turn
            reward_pack_by_i_first_turn = self.reward_pack_by_i_first_turn

        else:
            agreed_move_by_i_turn = self.agreed_move_by_i_turn | frozendict(
                                                           {self.i_turn: currently_agreed_move})
            reward_pack_by_i_first_turn = self.reward_pack_by_i_first_turn | frozendict(
                                            {i_turn: RewardPack.create(self.stubborn_config, self)})

        return StubbornState(
            stubborn_config=self.stubborn_config,
            i_turn=i_turn,
            handicaps=self.handicaps,
            reward_pack_by_i_first_turn=reward_pack_by_i_first_turn,
            move_pairs=move_pairs,
            agreed_move_by_i_turn=agreed_move_by_i_turn,
        )

    @functools.cached_property
    def we_just_agreed(self) -> bool:
        return (self.i_turn - 1) in self.agreed_move_by_i_turn

    @functools.cached_property
    def move_we_just_agreed_on(self) -> Move:
        try:
            return self.agreed_move_by_i_turn[self.i_turn - 1]
        except KeyError as e:
            raise SkirmishNotFinishedError from e

    @functools.cached_property
    def text(self) -> str:
        if not self.we_just_agreed:
            first_line = ''
        else:
            if self.move_we_just_agreed_on == Move.LEFT:
                far_left_text = \
                               f'{self.reward_pack_we_just_played_on.left_reward:+05.1f}'.center(16)
                far_right_text = ' ' * 16
            else:
                assert self.move_we_just_agreed_on == Move.RIGHT
                far_left_text = ' ' * 16
                far_right_text = \
                              f'{self.reward_pack_we_just_played_on.right_reward:+05.1f}'.center(16)

            middle_left_snippet = ''
            middle_right_snippet = ''

            if self.move_pairs[-1][0] == Move.LEFT:
                middle_left_snippet += f'A{self.annotations_for_last_move_pair[0]} '
            else:
                middle_right_snippet += f'A{self.annotations_for_last_move_pair[0]} '

            if self.move_pairs[-1][1] == Move.LEFT:
                middle_left_snippet += f'B{self.annotations_for_last_move_pair[1]}'
            else:
                middle_right_snippet += f'B{self.annotations_for_last_move_pair[1]}'

            middle_left_text = middle_left_snippet.center(13)
            middle_right_text = middle_right_snippet.center(13)

            first_line = (far_left_text + middle_left_text + middle_right_text + far_right_text +
                          '\n\n')

        if self.i_skurn == 0:
            return first_line + (
                f'{self.reward_pack.left_reward:+05.1f} '
                f'(Ae: {self.reward_pack.estimates_of_left_reward[0]:+05.1f} '
                f'Be: {self.reward_pack.estimates_of_left_reward[1]:+05.1f})   '
                f'{self.reward_pack.right_reward:+05.1f} '
                f'(Ae: {self.reward_pack.estimates_of_right_reward[0]:+05.1f} '
                f'Be: {self.reward_pack.estimates_of_right_reward[1]:+05.1f})   '
                f'\nAh: {self.handicaps[0]:+05.1f} '
                f'Bh: {self.handicaps[1]:+05.1f}'
            )

        else:
            middle_left_snippet = ''
            middle_right_snippet = ''

            if self.move_pairs[-1][0] == Move.LEFT:
                middle_left_snippet += f'A{self.annotations_for_last_move_pair[0]}'
            else:
                middle_right_snippet += f'A{self.annotations_for_last_move_pair[0]}'

            if self.move_pairs[-1][1] == Move.LEFT:
                middle_left_snippet += f'B{self.annotations_for_last_move_pair[1]}'
            else:
                middle_right_snippet += f'B{self.annotations_for_last_move_pair[1]}'

            middle_left_text = middle_left_snippet.center(13)
            middle_right_text = middle_right_snippet.center(13)
            far_left_text = far_right_text = ' ' * 16

            return (first_line + far_left_text + middle_left_text + middle_right_text +
                    far_right_text)

    @functools.cached_property
    def annotations_for_last_move_pair(self) -> tuple[str]:
        annotations_for_last_move_pair = []
        for i in range(2):
            last_move_was_ideal = (self.move_pairs[-1][i] ==
                                   self.reward_pack_we_just_played_on.ideal_move)
            last_move_was_tempting = (self.move_pairs[-1][i] ==
                                      self.reward_pack_we_just_played_on.tempting_moves[i])
            annotations_for_last_move_pair.append(
                ' ' if last_move_was_ideal and last_move_was_tempting else
                '!' if last_move_was_ideal and not last_move_was_tempting else
                '?' if not last_move_was_ideal and not last_move_was_tempting else
                ';'
            )
        return tuple(annotations_for_last_move_pair)

    @functools.cached_property
    def n_completed_skirmishes(self) -> int:
        return len(self.completed_skirmish_lengths)

    @functools.cached_property
    def zebras(self) -> tuple[RealNumber, RealNumber]:
        '''
        A 2-tuple of the zebra value for each agent.

        The zebra is a metric that evalutes how often each agent "gets its way" in a disagreement,
        i.e. the ratio between the number of disagreements in which the agent's choice was the one
        that was eventually agreed upon, and the total number of disagreements. It's a number
        between 0 and 1 that expresses how likely the agent is to disagree with another agent and
        ultimately "win".
        '''
        n_times_agent_got_its_way_pair = [0, 0]
        n_disagreements = 0
        for (i_turn_of_old_reward_pack, i_turn_of_new_reward_pack) in more_itertools. \
                                            sliding_window(self.reward_pack_by_i_first_turn, 2):
            first_move_pair = self.move_pairs[i_turn_of_old_reward_pack]
            if first_move_pair[0] == first_move_pair[1]:
                continue
            n_disagreements += 1
            for i_agent, move in enumerate(self.move_pairs[i_turn_of_old_reward_pack]):
                if move == self.agreed_move_by_i_turn[i_turn_of_new_reward_pack - 1]:
                    n_times_agent_got_its_way_pair[i_agent] += 1
        return tuple(misc.cute_div(n_times_agent_got_its_way_pair[i], n_disagreements)
                     for i in range(2))

    @functools.cached_property
    def i_turn_of_last_skirmish_end(self) -> int:
        if len(self.reward_pack_by_i_first_turn) <= 1:
            raise NoFinishedSkirmishesError
        return self.i_turn_of_current_skirmish_start

    @functools.cached_property
    def yankees(self) -> tuple[RealNumber, RealNumber]:
        '''
        A 2-tuple of the yankee value for each agent.

        The yankee is a metric that evalutes how "right" the agent has been in past disagreements.
        For each disagreement, we take the reward advantage of the agent's choice; e.g. if the agent
        selected the left reward, we take `left_reward - right_reward`. We average these numbers for
        all disagreements across the skirmishes that were completed in the game, i.e. excluding any
        ongoing skirmish.
        '''
        data = ([], [])
        try:
            i_turn_of_last_skirmish_end = self.i_turn_of_last_skirmish_end
        except NoFinishedSkirmishesError:
            return (0, 0)
        for move_pair, reward_pack in zip(self.move_pairs[:i_turn_of_last_skirmish_end],
                                          self.reward_packs[:i_turn_of_last_skirmish_end],
                                          strict=True):
            reward_pack: RewardPack
            if misc.all_equal(move_pair):
                continue
            for i_agent in range(2):
                data[i_agent].append((reward_pack.left_reward - reward_pack.right_reward)
                                     if move_pair[i_agent] == Move.LEFT
                                     else (reward_pack.right_reward - reward_pack.left_reward))
        return tuple((statistics.mean(datum) if datum else 0) for datum in data)


    @functools.cached_property
    def glee_by_skurn_pair(self) -> tuple[tuple[RealNumber, ...], tuple[RealNumber, ...]]:
        '''
        A 2-tuple of a `glee_by_skurn` dict for each agent.

        A glee is a metric for how likely an agent is to choose the reward that it sees as having
        the higest reward, according to its estimates. Its value is between 0 and 1. It's a metric
        that is measured per agent per skurn, i.e. you can ask "what is the glee of Agent A in skurn
        2?" This metric aggregates information about the behavior of Agent A in all turns in the
        game that had a skurn of 2. For each of these skurns, it takes the number 1 if in that
        skurn, the agent chose the reward that seems higher to it according to its estimates.
        Otherwise, the value is 0. The glee in skurn `x` would be the averages of these numbers for
        all the turns that are a skurn `x`.
        '''

        n_skurns_to_measure = 5
        n_times_agent_was_gleeful_by_skurn_pair = [[0] * n_skurns_to_measure,
                                                   [0] * n_skurns_to_measure]
        n_opportunities_by_skurn = [0] * n_skurns_to_measure
        for i_turn, (move_pair, reward_pack) in \
                               enumerate(zip(self.move_pairs, self.reward_packs[:-1], strict=True)):

            if i_turn in self.reward_pack_by_i_first_turn:
                assert self.reward_pack_by_i_first_turn[i_turn] == reward_pack
                i_turn_of_reward_pack = i_turn

            i_skurn = i_turn - i_turn_of_reward_pack
            if i_skurn >= n_skurns_to_measure:
                continue
            n_opportunities_by_skurn[i_skurn] += 1
            for i_agent in range(2):
                if ((self.move_pairs[i_turn][i_agent] == Move.LEFT) ==
                    (reward_pack.estimates_of_left_reward[i_agent] >=
                     reward_pack.estimates_of_right_reward[i_agent])):

                    n_times_agent_was_gleeful_by_skurn_pair[i_agent][
                                                       i_skurn] += 1

        glee_by_skurn_pair = []
        for n_times_agent_was_gleeful_by_skurn in n_times_agent_was_gleeful_by_skurn_pair:
            glee_by_skurn_pair.append(glee_by_skurn := [])
            for n_times_agent_was_gleeful, n_opportunities in zip(
                         n_times_agent_was_gleeful_by_skurn, n_opportunities_by_skurn, strict=True):
                if n_opportunities == 0:
                    break
                glee_by_skurn.append(n_times_agent_was_gleeful / n_opportunities)

        return tuple(map(tuple, glee_by_skurn_pair))


    # Documentation for glamor, calculated elsewhere:
    '''

    The glamor is a similar metric to the zebra, in that it's a metric of how likely the agent is to
    get its way in a disagreement. Unlike the zebra, the glamor is measured only on skirmishes that
    begin in a certain `i_turn`, like 30. Also, the glamor measures how likely the agent is to
    insist on its initial solution until the end of the skirmish, and doesn't count instances of a
    switcheroo where the randomly-chosen reward happened to be the one our agent has chosen.

    '''
