# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from typing import Optional

import click
import pandas as pd

from .command_group import cli

from stubborn.county import misc



@cli.command()
@click.option('-p', '--path', 'path_expression', type=str, required=False, show_default=True)
def plot_reward(path_expression: Optional[str]) -> None:
    import plotly.graph_objects as go
    path = misc.parse_output_path(path_expression, 'stubborn', file_name='rollout.jsonl')

    print(f'Making a reward plot for {path.parent.parent.name} ...')

    df = misc.jsonl_to_dataframe(path)
    axis_template = {'title_font': {'size': 60,}, 'tickfont': {'size': 40,}}

    figure = go.Figure(
        data=go.Scatter(x=df.index, y=df['episode_reward'],
                        name='Mean episode reward', mode='markers',
                        marker={'size': 3,}),
        layout=go.Layout(
            title_font_size=28,
            xaxis={**axis_template, 'title': 'Generation'},
            yaxis={**axis_template, 'title': 'Mean episode reward'},
            legend={'font': {'size': 20}}
        )
    )
    figure.show()


@cli.command()
@click.option('-p', '--path', 'path_expression', type=str, required=False, show_default=True)
def plot_insistence(path_expression: Optional[str]) -> None:
    import plotly.graph_objects as go
    path = misc.parse_output_path(path_expression, 'stubborn', file_name='rollout.jsonl')

    print(f'Making an insistence plot for {path.parent.parent.name} ...')

    df = misc.jsonl_to_dataframe(path)
    insistence_df = pd.DataFrame([df[f'insistence_on_stubbornness_{i}'].mean()
                                    for i in range(5)], columns=('confidence',))
    axis_template = {'title_font': {'size': 60,}, 'tickfont': {'size': 20,}}

    figure = go.Figure(
        data=go.Scatter(x=insistence_df.index, y=insistence_df['confidence'],
                        name='Fluff', mode='markers',
                        marker={'size': 20,}),
        layout=go.Layout(
            title_font_size=28,
            xaxis={**axis_template, 'title': r'$\Huge{n}$', 'tick0': 0, 'dtick': 1,},
            yaxis={**axis_template, 'title': r'$\Huge{\zeta_{n,5}}$'},
            legend={'font': {'size': 20}}
        )
    )
    figure.show()


