'''
Copyright 2024 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import annotations

import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy
from plotly import graph_objects as go
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
)

from climateeconomics.core.tools.color_map import ColorMap
from climateeconomics.core.tools.colormaps import available_colormaps
from climateeconomics.core.tools.plotting import (
    TwoAxesInstanciatedChart,
)
from climateeconomics.core.tools.post_proc import get_scenario_value


def create_sankey_diagram_at_year(
    data_dict: dict[str, dict[str, pd.DataFrame]],
    year: int | str,
    colormap: ColorMap | dict | None | str = None,
    constant_width: bool = False,
    years_column: str = GlossaryEnergy.Years,
    split_external: bool = False,
    output_node: str | None = None,
    input_node: str | None = None,
) -> go.Figure:
    """Creates a Sankey diagram showing direct flows between nodes based on their inputs
    and output patterns.

    Args:
        data_dict: Dictionary of dictionaries containing production and consumption dataframes for each actor.
            Format: {
                node: {
                    'output': DataFrame(index=[year1, year2, ...], columns=[prod_type1, prod_type2, ...]),
                    'input': DataFrame(index=[year1, year2, ...], columns=[cons_type1, cons_type2, ...])
                }
            }
        year: The specific year to visualize.

    Returns:
        A plotly Figure object containing the Sankey diagram.
    """
    # Create node labels and mapping
    actors = list(data_dict.keys())
    node_mapping = {actor: idx for idx, actor in enumerate(actors)}

    # Initialize lists for Sankey diagram
    source = []
    target = []
    value = []
    labels = []

    # Compute total consumption by type
    total_consumption = {}
    for actor in data_dict:
        for flow_type in data_dict[actor]["input"].columns:
            if flow_type == years_column:
                continue
            if flow_type not in total_consumption:
                total_consumption[flow_type] = 0.0

            total_consumption[flow_type] += data_dict[actor]["input"][
                data_dict[actor]["input"][years_column] == year
            ][flow_type].to_numpy()[0]

    input_nodes = []
    output_nodes = []

    # Create flows between actors based on matching production and consumption types
    for producer, prod_data in data_dict.items():
        for consumer, cons_data in data_dict.items():
            if producer != consumer:  # Avoid self-loops
                prod_df = prod_data["output"]
                cons_df = cons_data["input"]

                prod_columns = [c for c in prod_df.columns if c != years_column]
                cons_columns = [c for c in cons_df.columns if c != years_column]

                if (
                    year not in prod_df[years_column].to_numpy()
                    or year not in cons_df[years_column].to_numpy()
                ):
                    continue

                prod_df = prod_df[prod_df[years_column] == year]
                cons_df = cons_df[cons_df[years_column] == year]

                # Find matching types between producer and consumer
                common_types = set(prod_columns) & set(cons_columns)

                # Inside the nested loops where flows are created
                for flow_type in common_types:
                    prod_value = prod_df[flow_type].to_numpy()[0]
                    cons_value = cons_df[flow_type].to_numpy()[0]

                    if prod_value > 0 and cons_value > 0:
                        flow_value = cons_value  # assume we have enough production for all consumptions

                        # Handle output node splitting
                        target_node = consumer
                        if consumer == output_node and split_external:
                            net_flow_label = f"{flow_type}"
                            if net_flow_label not in node_mapping:
                                node_mapping[net_flow_label] = len(node_mapping)
                                output_nodes.append(net_flow_label)
                            target_node = net_flow_label

                        # Handle input node splitting
                        source_node = producer
                        if producer == input_node and split_external:
                            in_flow_label = f"{flow_type}"
                            if in_flow_label not in node_mapping:
                                node_mapping[in_flow_label] = len(node_mapping)
                                input_nodes.append(in_flow_label)
                            source_node = in_flow_label

                        source.append(node_mapping[source_node])
                        target.append(node_mapping[target_node])
                        value.append(flow_value)
                        labels.append(flow_type)

    # Update actors in case we added I/O nodes
    actors = list(node_mapping.keys())

    customdata = labels
    hovertemplate = (
        "%{customdata}<br>"
        "From : %{source.label}<br>"
        "To   : %{target.label}<br>"
        "Value: %{value:.2E}<br><extra></extra>"
    )

    # Overload customdata and hovertemplate in case constant_with is True
    if constant_width:
        value_strings = [f"{val:.2f}" for val in value]
        customdata = list(zip(customdata, [f"{val:.2E}" for val in value]))
        value = [1.0] * len(value)
        # Update hovertemplate to include both pieces of information
        hovertemplate = (
            "%{customdata[0]}<br>"
            "From : %{source.label}<br>"
            "To   : %{target.label}<br>"
            "Value: %{customdata[1]}<br><extra></extra>"
        )

    # Handle colormap
    link_data = {
        "source": source,
        "target": target,
        "value": value,
        "hovertemplate": hovertemplate,
        "customdata": customdata,
    }

    if colormap is not None:
        if isinstance(colormap, dict):
            colormap = ColorMap(colormap)
        if isinstance(colormap, str):
            if colormap not in available_colormaps:
                raise ValueError("Requested colormap not available.")
            colormap = available_colormaps[colormap]

        link_data["color"] = [colormap.get_color(label) for label in labels]

    node_color = {}

    if not split_external:
        if output_node in actors:
            output_nodes = [output_node]
        if input_node in actors:
            input_nodes = [input_node]

    middle_nodes = [n for n in actors if n not in input_nodes and n not in output_nodes]

    for node in input_nodes:
        node_color[node] = "red"

    # Middle nodes evenly distributed between 0.3 and 0.7
    for node in middle_nodes:
        node_color[node] = "lightblue"

    # Output nodes at x=1
    for node in output_nodes:
        node_color[node] = "green"

    # Convert to lists in the same order as actors
    node_colors = [node_color[actor] for actor in actors]

    # Create the Sankey diagram with positioned nodes
    fig = go.Figure(
        data=[
            go.Sankey(
                node={
                    "pad": 5,
                    "thickness": 5,
                    "line": {"color": "black", "width": 0.5},
                    "label": actors,
                    "color": node_colors,
                },
                link=link_data,
            )
        ]
    )

    return fig


def create_sankey_with_slider(
    data_dict: dict[str, dict[str, pd.DataFrame]],
    colormap: ColorMap | dict | None | str = None,
    normalized_links: bool = False,
    **kwargs,
) -> go.Figure:
    """Creates an interactive Sankey diagram with a slider to select different years.

    Args:
        data_dict(dict): Dictionary of dictionaries containing production and consumption dataframes for each actor.
            Format: {
                node: {
                    'output': DataFrame(index=[year1, year2, ...], columns=[prod_type1, prod_type2, ...]),
                    'input': DataFrame(index=[year1, year2, ...], columns=[cons_type1, cons_type2, ...])
                }
            }
        colormap(ColorMap, dict, None, str): The colormap to use. Defaults to None.
        normalized_links(bool): Normalize links if it is true.

    Returns:
        A plotly Figure object containing the Sankey diagram with a year slider.

    """
    # Get all available years from the data
    years = set()
    for actor_data in data_dict.values():
        years.update(set(actor_data["output"][GlossaryEnergy.Years].to_numpy()))
        years.update(set(actor_data["input"][GlossaryEnergy.Years].to_numpy()))

    years = sorted(list(years))
    if not years:
        raise ValueError("No valid years found in the data")

    # Create frames for each year using the previous function
    frames = {}
    for year in years:
        # Get the Sankey diagram for this year
        year_fig = create_sankey_diagram_at_year(
            data_dict,
            year,
            colormap=colormap,
            constant_width=normalized_links,
            **kwargs,
        )

        # Create frame from the Sankey data
        frames[year] = year_fig

    # Create the initial figure (using the first year)
    fig = create_slider_figure(frames, method="frames")

    return fig


def create_slider_figure(figures_dict, method="frames"):
    """
    Create a plotly figure with a slider based on a dictionary of figures.

    Parameters:
    -----------
    figures_dict : dict
        Dictionary with keys as labels and values as plotly figure objects
    method : str
        Method to create slider: 'frames', 'visibility', or 'update'

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Figure with slider
    """

    if method == "frames":
        # Approach 1: Using frames
        frames = []
        data = []

        # Create base figure from first plot
        first_key = list(figures_dict.keys())[0]
        fig = figures_dict[first_key]

        # Create frames for each figure
        for key, figure in figures_dict.items():
            frame = go.Frame(data=figure.data, name=str(key))
            frames.append(frame)

            # Add data to base figure (hidden initially)
            for trace in figure.data:
                data.append(trace)

        # Create slider
        sliders = [
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(key)],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                        label=str(key),
                    )
                    for key in figures_dict.keys()
                ],
                currentvalue=dict(visible=True, prefix="Selection: "),
                pad=dict(t=50),
            )
        ]

        # Update layout
        fig.update_layout(
            sliders=sliders,
        )
        fig.frames = frames

    elif method == "visibility":
        # Approach 2: Using visibility
        fig = go.Figure()

        # Add all traces
        for idx, (key, figure) in enumerate(figures_dict.items()):
            for trace in figure.data:
                trace.visible = idx == 0  # Only first figure visible initially
                fig.add_trace(trace)

        # Create slider
        steps = []
        for i, key in enumerate(figures_dict.keys()):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)}],
                label=str(key),
            )
            # Make traces for current figure visible
            start_idx = i * len(list(figures_dict.values())[0].data)
            end_idx = start_idx + len(list(figures_dict.values())[0].data)
            for j in range(start_idx, end_idx):
                step["args"][0]["visible"][j] = True
            steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue=dict(visible=True, prefix="Selection: "),
                pad=dict(t=50),
                steps=steps,
            )
        ]

        fig.update_layout(sliders=sliders)

    elif method == "update":
        # Approach 3: Using update (useful for updating specific properties)
        fig = go.Figure()

        # Add all traces
        for key, figure in figures_dict.items():
            for trace in figure.data:
                fig.add_trace(trace)

        # Create slider
        steps = []
        for i, key in enumerate(figures_dict.keys()):
            step = dict(
                method="update",
                args=[
                    {
                        "visible": [
                            True if i == j else False for j in range(len(figures_dict))
                        ]
                    },
                    {"title": f"Selection: {key}"},  # Update title as example
                ],
                label=str(key),
            )
            steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue=dict(visible=True, prefix="Selection: "),
                pad=dict(t=50),
                steps=steps,
            )
        ]

        fig.update_layout(sliders=sliders)

    return fig


def add_dropdown_menu(fig, sort_by=None, dropdown_name="Select Option"):
    """
    Add a dropdown menu to a plotly figure to control trace visibility.

    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The input plotly figure
    sort_by : dict or None
        If None, creates dropdown options for all traces
        If dict, keys are dropdown labels and values are lists of trace indices to show
    dropdown_name : str
        The title that appears above the dropdown menu

    Returns:
    --------
    plotly.graph_objects.Figure
        Figure with added dropdown menu
    """

    if sort_by is None:
        # Create dropdown options for each trace
        dropdown_buttons = []

        # Add "All" option
        dropdown_buttons.append(
            dict(
                args=[{"visible": [True] * len(fig.data)}], label="All", method="update"
            )
        )

        # Add individual trace options
        for i, trace in enumerate(fig.data):
            # Get trace name, default to trace index if name not set
            trace_name = trace.name if trace.name else f"Trace {i}"

            visibility = [False] * len(fig.data)
            visibility[i] = True

            dropdown_buttons.append(
                dict(args=[{"visible": visibility}], label=trace_name, method="update")
            )

    else:
        # Create dropdown options based on sort_by dictionary
        dropdown_buttons = []

        # Add "All" option
        dropdown_buttons.append(
            {
                "args": [{"visible": [True] * len(fig.data)}],
                "label": "All",
                "method": "update",
            },
        )

        # Add options from sort_by dictionary
        for label, trace_indices in sort_by.items():
            visibility = [False] * len(fig.data)

            # Convert single index to list if necessary
            if isinstance(trace_indices, int):
                trace_indices = [trace_indices]

            # Set visibility for specified traces
            for idx in trace_indices:
                if idx < len(fig.data):
                    visibility[idx] = True
                else:
                    print(f"Warning: Trace index {idx} exceeds number of traces")

            dropdown_buttons.append(
                {
                    "args": [{"visible": visibility}],
                    "label": str(label),
                    "method": "update",
                }
            )

    # Update layout with dropdown menu
    fig.update_layout(
        updatemenus=[
            {
                "buttons": dropdown_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "pad": {"r": 10, "t": 10},
                "bgcolor": "white",
                "bordercolor": "gray",
                "borderwidth": 1,
            }
        ],
        annotations=[
            {
                "text": dropdown_name,
                "x": 0,
                "y": 1.15,
                "yref": "paper",
                "xref": "paper",
                "showarrow": False,
            }
        ],
    )

    return fig


def create_xy_chart(
    execution_engine,
    chart_name: str,
    x_axis_name: str,
    y_axis_name: str,
    data_dict: dict,
    **kwargs,
) -> TwoAxesInstanciatedChart:
    """Create XY chart from data dictionary"""
    new_chart = TwoAxesInstanciatedChart(
        x_axis_name, y_axis_name, chart_name=chart_name, **kwargs
    )

    for data_name, data in data_dict.items():
        x_data = None
        y_data = None
        text_data = None

        if data["data_type"] == "scenario_variable":
            x_data_df = get_scenario_value(
                execution_engine,
                data["x_var_name"],
                data["scenario_name"],
                split_scenario_name=False,
            )
            y_data_df = get_scenario_value(
                execution_engine,
                data["y_var_name"],
                data["scenario_name"],
                split_scenario_name=False,
            )
            x_data = x_data_df[data["x_column_name"]]
            y_data = y_data_df[data["y_column_name"]]
            text_data = y_data_df[data["text_column"]].values.tolist()
        elif data["data_type"] == "csv":
            data_df = pd.read_csv(data["filename"])
            x_data_df = y_data_df = data_df
            x_data = x_data_df[data["x_column_name"]]
            y_data = y_data_df[data["y_column_name"]]
            text_data = y_data_df[data["text_column"]].values.tolist()
        elif data["data_type"] == "dataframe":
            x_data_df = y_data_df = data["data"]
            x_data = x_data_df[data["x_column_name"]]
            y_data = y_data_df[data["y_column_name"]]
            text_data = y_data_df[data["text_column"]].values.tolist()
        elif data["data_type"] == "separate_xy":
            x_data = data["x_data"]
            y_data = data["y_data"]
            text_data = data.get("text")

        if x_data is not None and y_data is not None:
            new_series = InstanciatedSeries(
                x_data * data.get("x_data_scale", 1.0),
                y_data * data.get("y_data_scale", 1.0),
                data_name,
                display_type="scatter",
                marker_symbol=data.get("marker_symbol", "circle"),
                text=text_data,
                **data.get("kwargs", {}),
            )
            new_chart.add_series(new_series)

    return new_chart
