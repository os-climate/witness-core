"""
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

"""

from energy_models.glossaryenergy import GlossaryEnergy

from climateeconomics.sos_wrapping.post_procs.sectors.emissions.sector_breakdown_emissions import (
    post_processing_filters as ppf_template,
)
from climateeconomics.sos_wrapping.post_procs.sectors.emissions.sector_breakdown_emissions import (
    post_processings as pp_template,
)


def post_processing_filters(execution_engine, namespace):
    return ppf_template(execution_engine, namespace)


def post_processings(execution_engine, scenario_name, chart_filters=None):
    return pp_template(
        execution_engine, scenario_name, sector=GlossaryEnergy.SectorServices, chart_filters=chart_filters
    )
