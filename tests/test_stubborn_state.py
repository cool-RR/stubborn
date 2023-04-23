# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from stubborn import StubbornState, StubbornConfig, StubbornEnv, Move


def test_stubborn_state():
    stubborn_config = StubbornConfig()
    state = StubbornState.make_initial(stubborn_config)
    assert len(state.agents) == 2

def test_stubborn_env_always_agree():
    stubborn_config = StubbornConfig(randomize_labels=False)
    agents = tuple(stubborn_config.policy_by_agent)
    env = StubbornEnv({'stubborn_config': stubborn_config,})
    for _ in range(stubborn_config.episode_length):
        env.step({agents[0]: [Move.LEFT.to_neural()], agents[1]: [Move.LEFT.to_neural()]})
    state: StubbornState = env.state
    assert set(state.done_by_agent.values()) == {True}
    assert (len(state.reward_pack_by_i_first_turn) - 1 == state.i_turn ==
            stubborn_config.episode_length)


def test_stubborn_env_never_agree():
    stubborn_config = StubbornConfig(randomize_labels=False)
    agents = tuple(stubborn_config.policy_by_agent)
    env = StubbornEnv({'stubborn_config': stubborn_config,})
    for _ in range(stubborn_config.episode_length):
        env.step({agents[0]: [Move.LEFT.to_neural()], agents[1]: [Move.RIGHT.to_neural()]})
    state: StubbornState = env.state
    assert set(state.done_by_agent.values()) == {True}
    assert len(state.reward_pack_by_i_first_turn) == 1
    assert state.i_turn == stubborn_config.episode_length

