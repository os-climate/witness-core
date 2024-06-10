"""
Copyright 2022 Airbus SAS
Modifications on 2023/04/21-2023/11/03 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pandas as pd
from energy_models.core.stream_type.resources_models.resource_glossary import (
    ResourceGlossary,
)
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)

from climateeconomics.glossarycore import GlossaryCore

RESOURCE_CONSUMPTION_UNIT = ResourceGlossary.UNITS[GlossaryCore.Consumption]


def post_processing_filters(execution_engine, namespace):
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the filters
    """
    filters = []

    chart_list = [
        "Resource Consumption",
    ]
    # The filters are set to False by default since the graphs are not yet
    # mature
    filters.append(ChartFilter("Charts", chart_list, chart_list, "Charts"))

    return filters


def post_processings(execution_engine, namespace, filters):
    """
    WARNING : the execution_engine and namespace arguments are necessary to retrieve the post_processings
    """
    instanciated_charts = []

    # Overload default value with chart filter
    graphs_list = []
    if filters is not None:
        for chart_filter in filters:
            if chart_filter.filter_key == "Charts":
                graphs_list.extend(chart_filter.selected_values)

    # ---
    if "Resource Consumption" in graphs_list:
        chart_name = f"Resource Consumption"
        new_chart = get_chart_resource_consumption(execution_engine, namespace, chart_name=chart_name)
        if new_chart is not None:
            instanciated_charts.append(new_chart)

    return instanciated_charts


def get_chart_resource_consumption(execution_engine, namespace, chart_name="Resource consumption"):
    """! Function to create the resource consumption chart
    @param execution_engine: Execution engine object from which the data is gathered
    @param namespace: String containing the namespace to access the data
    @param chart_name:String, title of the post_proc

    @return new_chart: InstantiatedPlotlyNativeChart Scatter plot
    """

    # Prepare data
    resource_name = namespace.split("Resources.")[-1]
    first_part_ns = namespace.split("Resources.")[0]
    ns_val = None
    # TODO quick fix but need to do a cleaner way but needs deeper reflexion
    ns_list = execution_engine.ns_manager.get_all_namespace_with_name(GlossaryCore.NS_ENERGY_MIX)
    max_length = 0
    longest_object = None

    # get ns_object with longest
    for ns in ns_list:
        if hasattr(ns, "value") and isinstance(ns.value, str):
            if first_part_ns in ns.value and len(ns.value) > max_length:
                max_length = len(ns.value)
                longest_object = ns
    ns_energy_mix = longest_object.value
    index = ns_energy_mix.find(".EnergyMix")
    if index != -1:
        ns_val = ns_energy_mix[:index]
    if ns_val is None:
        raise Exception("variable ns_val is not defined")
    WITNESS_ns = ns_val
    EnergyMix = execution_engine.dm.get_disciplines_with_name(f"{WITNESS_ns}.EnergyMix")[0]
    years = np.arange(
        EnergyMix.get_sosdisc_inputs(GlossaryCore.YearStart), EnergyMix.get_sosdisc_inputs(GlossaryCore.YearEnd) + 1
    )
    # Construct a DataFrame to organize the data
    resource_consumed = pd.DataFrame({GlossaryCore.Years: years})
    energy_list = EnergyMix.get_sosdisc_inputs(GlossaryCore.energy_list)
    for energy in energy_list:
        if energy == "biomass_dry":
            namespace_disc = f"{WITNESS_ns}.AgricultureMix"
        else:
            namespace_disc = f"{WITNESS_ns}.EnergyMix.{energy}"

        energy_disc = execution_engine.dm.get_disciplines_with_name(f"{namespace_disc}")[0]
        techno_list = energy_disc.get_sosdisc_inputs(GlossaryCore.techno_list)
        for techno in techno_list:
            techno_disc = execution_engine.dm.get_disciplines_with_name(f"{namespace_disc}.{techno}")[0]
            consumption_techno = techno_disc.get_sosdisc_outputs("techno_consumption")
            if f"{resource_name} ({RESOURCE_CONSUMPTION_UNIT})" in consumption_techno.columns:
                resource_consumed[f"{energy} {techno}"] = consumption_techno[
                    f"{resource_name} ({RESOURCE_CONSUMPTION_UNIT})"
                ] * techno_disc.get_sosdisc_inputs("scaling_factor_techno_consumption")
    CCUS = execution_engine.dm.get_disciplines_with_name(f"{WITNESS_ns}.CCUS")[0]
    ccs_list = CCUS.get_sosdisc_inputs(GlossaryCore.ccs_list)
    for stream in ccs_list:
        stream_disc = execution_engine.dm.get_disciplines_with_name(f"{WITNESS_ns}.CCUS.{stream}")[0]
        techno_list = stream_disc.get_sosdisc_inputs(GlossaryCore.techno_list)
        for techno in techno_list:
            techno_disc = execution_engine.dm.get_disciplines_with_name(f"{WITNESS_ns}.CCUS.{stream}.{techno}")[0]
            consumption_techno = techno_disc.get_sosdisc_outputs("techno_consumption")
            if f"{resource_name} ({RESOURCE_CONSUMPTION_UNIT})" in consumption_techno.columns:
                resource_consumed[f"{stream} {techno}"] = consumption_techno[
                    f"{resource_name} ({RESOURCE_CONSUMPTION_UNIT})"
                ] * techno_disc.get_sosdisc_inputs("scaling_factor_techno_consumption")

    # Create Figure
    chart_name = f"{resource_name} consumption by technologies"
    new_chart = TwoAxesInstanciatedChart(
        GlossaryCore.Years, f"{resource_name} consumed by techno (Mt)", chart_name=chart_name, stacked_bar=True
    )
    for col in resource_consumed.columns:
        if "category" not in col and col != GlossaryCore.Years:
            legend_title = f"{col}"
            serie = InstanciatedSeries(
                resource_consumed[GlossaryCore.Years].values.tolist(),
                resource_consumed[col].values.tolist(),
                legend_title,
                "bar",
            )
            new_chart.series.append(serie)

    return new_chart
