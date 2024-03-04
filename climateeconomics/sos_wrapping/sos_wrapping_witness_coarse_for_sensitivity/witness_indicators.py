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
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from energy_models.models.renewable.renewable_simple_techno.renewable_simple_techno_disc import RenewableSimpleTechnoDiscipline
from energy_models.models.fossil.fossil_simple_techno.fossil_simple_techno_disc import FossilSimpleTechnoDiscipline
from climateeconomics.glossarycore import GlossaryCore
from energy_models.glossaryenergy import GlossaryEnergy


RENEWABLE_DEFAULT_TECHNO_DICT = RenewableSimpleTechnoDiscipline.techno_infos_dict_default
FOSSIL_DEFAULT_TECHNO_DICT = FossilSimpleTechnoDiscipline.techno_infos_dict_default

class WitnessIndicators(SoSWrapp):
    """
    Utility discipline to analyze Sensitivity Analysis demonstrator outputs in Witness Coarse Storytelling MDA.
    """
    # ontology information
    _ontology_data = {
        'label': 'WITNESS Indicators',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }
    DESC_IN = {GlossaryEnergy.EnergyMeanPriceValue: GlossaryEnergy.EnergyMeanPrice,
               GlossaryCore.EconomicsDfValue: GlossaryCore.EconomicsDf,
               GlossaryCore.TemperatureDfValue: GlossaryCore.TemperatureDf,
               GlossaryCore.NormalizedWelfare: {'type': 'array', 'unit': '-', 'visibility': 'Shared',
                                                'namespace': GlossaryCore.NS_WITNESS,
                                                'description': 'Sum of discounted utilities divided by number of year divided by initial discounted utility'},
               }

    DESC_OUT = {'mean_energy_price_2100': {'type': 'float'},
                'world_net_product_2100': {'type': 'float'},
                'temperature_rise_2100': {'type': 'float'},
                'welfare_indicator': {'type': 'float'},
                }

    def run(self):
        mean_energy_price = self.get_sosdisc_inputs(GlossaryEnergy.EnergyMeanPriceValue)['energy_price'].tolist()[-1]
        world_net_product = self.get_sosdisc_inputs(GlossaryCore.EconomicsDfValue)['output_net_of_d'].tolist()[-1]
        temperature_rise = self.get_sosdisc_inputs(GlossaryCore.TemperatureDfValue)['temp_atmo'].tolist()[-1]
        welfare_indicator = self.get_sosdisc_inputs(GlossaryCore.NormalizedWelfare)[0]

        self.store_sos_outputs_values({
            'mean_energy_price_2100': mean_energy_price,
            'world_net_product_2100': world_net_product,
            'temperature_rise_2100': temperature_rise,
            'welfare_indicator': welfare_indicator,
        })
