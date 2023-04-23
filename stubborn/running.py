# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

from typing import Optional, Iterable, Any
import pathlib
import shlex
import inspect
import yaml
import click
import datetime as datetime_module
import sys
import contextlib

import ray.rllib.env.multi_agent_env
import ray.rllib.algorithms.callbacks
from ray.rllib import algorithms
import ray.tune
import ray.rllib.algorithms.ppo

import stubborn
from stubborn import county
from stubborn.county import misc
from .stubborn_config import StubbornConfig
from .stubborn_env import StubbornEnv
from . import defaults
from .callbackino import Callbackino

from .command_group import *



def rollout_from_tune(config: dict[str, Any]) -> None:

    pure_config = {}
    stubborn_config_dict = {}
    config_arguments = set(inspect.signature(StubbornConfig).parameters)
    for key, value in config.items():
        if key in config_arguments:
            stubborn_config_dict[key] = value
        else:
            pure_config[key] = value

    stubborn_config = StubbornConfig(**stubborn_config_dict)
    return rollout(**pure_config, stubborn_config=stubborn_config,
                  trial_folder=pathlib.Path(ray.tune.get_trial_dir()),
                  report_to_tune=True, write_stdout_to_stdout=False)

def rollout(stubborn_config: StubbornConfig, *,
           trial_folder: Optional[pathlib.Path] = None,
           report_to_tune: bool = False,
           write_stdout_to_stdout: bool = True,
           argv: Optional[Iterable[str]] = None,
           ) -> None:

    trial_folder = trial_folder or pathlib.Path(ray.tune.get_trial_dir())

    samples_folder = trial_folder / 'samples'
    samples_folder.mkdir(parents=True, exist_ok=True)
    # thready_krueger = ThreadyKrueger(trial_folder / 'thready_krueger', start=True,
                                     # interval_seconds=5 * 60)

    output_path: pathlib.Path = trial_folder / 'output.txt'
    reporters = [county.misc.JsonlRolloutReporter(trial_folder / 'rollout.jsonl')]
    if report_to_tune:
        reporters.append(county.misc.TuneRolloutReporter())

    stubborn_env = StubbornEnv({'stubborn_config': stubborn_config})

    hand_coded_policy_class_by_agent = {
        # 'human': HandCodedHumanPolicy,
        # 'robot': HandCodedRobotPolicy,
    }
    hand_coded_policy_class_by_policy_name = {
        stubborn_config.policy_mapping_fn(agent): hand_coded_policy_class
        for agent, hand_coded_policy_class in hand_coded_policy_class_by_agent.items()
    }

    algorithm_config = (
        algorithms.ppo.PPOConfig()
        .training(
            train_batch_size=stubborn_config.train_batch_size,
            lr=stubborn_config.learning_rate,
        )
        .resources(
            num_gpus=0,
            num_gpus_per_worker=0,
        )
        .environment(
            env=StubbornEnv,
            env_config=stubborn_env.config,
            disable_env_checking=True,
        )
        .rollouts(
            create_env_on_local_worker=True,
            num_rollout_workers=2,
        )
        .callbacks(
            Callbackino
        )
        .multi_agent(
            policies={
                policy_name: (
                    ray.rllib.policy.policy.PolicySpec(
                                   policy_class=hand_coded_policy_class_by_policy_name[policy_name])
                    if policy_name in hand_coded_policy_class_by_policy_name
                    else ray.rllib.policy.policy.PolicySpec()
                )
                for policy_name in stubborn_env.policies
            },
            policy_mapping_fn=stubborn_config.policy_mapping_fn,
        )
    )

    algorithm = algorithm_config.build()

    base_agentwise_measures = ('agreeability', 'zebra', 'yankee',
                               *(f'glee_skurn_{i}' for i in range(1)),)
    simple_measure_fields = ('mean_completed_skirmish_length', 'n_completed_skirmishes',
                             'left_reward_popularity', 'right_reward_popularity')
    # simple_fields = ('generation', 'episode_reward', 'datetime')
    agentwise_measure_fields = tuple(f'{agent}_{base_measure}' for agent in
                                     stubborn_config.policy_by_agent for base_measure in
                                     base_agentwise_measures)
    whatever_fields = (
        # 'stubbornness_on_difference_00', 'stubbornness_on_difference_06',
        # 'stubbornness_on_difference_12', 'stubbornness_on_difference_18',
        # 'stubbornness_on_difference_24',

        'insistence_on_stubbornness_0', 'insistence_on_stubbornness_1',
        'insistence_on_stubbornness_2', 'insistence_on_stubbornness_3',
        'insistence_on_stubbornness_4',
    )

    with (trial_folder / 'metadata.yaml').open('w') as yaml_file:
        yaml.dump(
            {
                'stubborn_config': stubborn_config.get_nice_dict(),
                'stubborn_version': stubborn.__version__,
                'observation_fields': list(stubborn_env.observation_space.keys()),
                'algorithm_class': type(algorithm).__name__,
                'argv': sys.argv if argv is None else argv,
            },
            yaml_file,
        )


    with contextlib.ExitStack() as exit_stack:
        if write_stdout_to_stdout:
            exit_stack.enter_context(county.misc.tee_stdout(output_path))
            stream = sys.stdout
        else:
            stream = exit_stack.enter_context(output_path.open('w'))

        for i_generation in range(stubborn_config.n_generations + 1):
            if i_generation == 0:
                # For fast failure in case there's a bug in `Callbackino`:
                Callbackino().on_train_result(algorithm=algorithm, result={})
            (samples_folder / f'{i_generation:04d}').write_text(
                StubbornEnv.sample_episode_to_text(algorithm)
            )
            generation_is_mature = i_generation >= 0.8 * stubborn_config.n_generations
            results = algorithm.train()
            stream.write(f'Generation: {algorithm.iteration:04d}  '
                         f'Episode reward mean: {results["episode_reward_mean"]:04.2f}\n')
            report = {
                'generation': algorithm.iteration,
                'datetime': datetime_module.datetime.now().isoformat(),
                'episode_reward': results['episode_reward_mean'],
                **{whatever_field: results[whatever_field] for whatever_field in whatever_fields},
                **{
                    measure_field: results['custom_metrics'][f'{measure_field}_mean']
                    for measure_field in simple_measure_fields + agentwise_measure_fields
                },
                **{
                    measure_field: results['custom_metrics'][f'{measure_field}_mean']
                    for measure_field in agentwise_measure_fields
                },
                'mature_episode_reward': (results['episode_reward_mean'] if generation_is_mature
                                          else None),
                **{
                    f'mature_{measure_field}': (results['custom_metrics'][f'{measure_field}_mean']
                                                if generation_is_mature else None)
                    for measure_field in simple_measure_fields + agentwise_measure_fields
                },

            }
            for reporter in reporters:
                reporter.report(report)

    # thready_krueger.stop()




@cli.command()
@click.option('-t', '--use-tune', is_flag=True, show_default=True)
@click.option('--n-tune-samples', type=int, default=defaults.DEFAULT_N_TUNE_SAMPLES,
              show_default=True)
@StubbornConfig.add_options_to_click_command
def run(*, use_tune: bool, n_tune_samples: int, **raw_stubborn_config_kwargs) -> None:
    '''
    Run the Stubborn experiment.
    '''
    with contextlib.ExitStack() as exit_stack:
        output_folder = misc.make_output_folder('stubborn')
        click.echo(f'Writing to {shlex.quote(str(output_folder))}')

        exit_stack.enter_context(county.misc.tee_stdout(output_folder / 'stdout'))
        exit_stack.enter_context(county.misc.tee_stderr(output_folder / 'stderr'))
        county.init_ray()

        stubborn_config_kwargs = {}
        for key, values in raw_stubborn_config_kwargs.items():
            if not values:
                continue
            elif len(values) == 1:
                (value,) = values
                stubborn_config_kwargs[key] = value
            else:
                assert len(values) >= 2
                if not use_tune:
                    raise click.UsageError
                stubborn_config_kwargs[key] = ray.tune.grid_search(values)


        if use_tune:
            experiment = ray.tune.Experiment(
                name='stubborn',
                run=rollout_from_tune,
                config={
                    **stubborn_config_kwargs,
                    'argv': sys.argv,
                },
                num_samples=n_tune_samples,
                local_dir=(tune_folder := output_folder / 'tune_rollouts'),
                log_to_file=('stdout', 'stderr'),
                resources_per_trial=ray.tune.PlacementGroupFactory(
                    [{'CPU': 1.0}] + [{'CPU': 1.0}] * 2
                ),
            )
            experiment.dir_name = tune_folder

            analysis = ray.tune.run(
                experiment,
            )

            df = misc.get_mean_dataframe_from_experiment_analysis(
                analysis,
                fields_that_need_last=('squanch',)
            )

            df.to_csv(output_folder / 'tune_analysis.csv')

        else:
            rollout(
                stubborn_config=StubbornConfig(**stubborn_config_kwargs),
                trial_folder=(output_folder / 'rollout')
            )



