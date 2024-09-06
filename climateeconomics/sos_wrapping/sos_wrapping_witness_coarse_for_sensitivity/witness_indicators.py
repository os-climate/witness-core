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
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.models.clean_energy.clean_energy_simple_techno.clean_energy_simple_techno_disc import (
    CleanEnergySimpleTechnoDiscipline,
)
from energy_models.models.fossil.fossil_simple_techno.fossil_simple_techno_disc import (
    FossilSimpleTechnoDiscipline,
)
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

from climateeconomics.glossarycore import GlossaryCore

RENEWABLE_DEFAULT_TECHNO_DICT = CleanEnergySimpleTechnoDiscipline.techno_infos_dict_default
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
               'energy_prices_after_tax': {SoSWrapp.TYPE: 'dataframe',
                                           SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
                                           SoSWrapp.NAMESPACE: GlossaryCore.NS_ENERGY_MIX,
                                           SoSWrapp.UNIT: '$/MWh'},
               GlossaryCore.EconomicsDfValue: GlossaryCore.EconomicsDf,
               GlossaryCore.TemperatureDfValue: GlossaryCore.TemperatureDf,
               GlossaryEnergy.StreamProductionDetailedValue: {SoSWrapp.TYPE: 'dataframe',
                                                              SoSWrapp.VISIBILITY: SoSWrapp.SHARED_VISIBILITY,
                                                              SoSWrapp.NAMESPACE: GlossaryCore.NS_ENERGY_MIX,
                                                              SoSWrapp.UNIT: 'TWh'}
               }

    DESC_OUT = {'mean_energy_price_2100': {SoSWrapp.TYPE: 'float', SoSWrapp.UNIT: '$/MWh'},
                'fossil_energy_price_2100': {SoSWrapp.TYPE: 'float', SoSWrapp.UNIT: '$/MWh'},
                'renewable_energy_price_2100': {SoSWrapp.TYPE: 'float', SoSWrapp.UNIT: '$/MWh'},

                'total_energy_production_2100': {SoSWrapp.TYPE: 'float', SoSWrapp.UNIT: 'PWh'},
                'fossil_energy_production_2100': {SoSWrapp.TYPE: 'float', SoSWrapp.UNIT: 'PWh'},
                'renewable_energy_production_2100': {SoSWrapp.TYPE: 'float', SoSWrapp.UNIT: 'PWh'},

                'world_net_product_2100': {SoSWrapp.TYPE: 'float', SoSWrapp.UNIT: 'T$'},
                'temperature_rise_2100': {SoSWrapp.TYPE: 'float', SoSWrapp.UNIT: 'ºC'},
                }

    def run(self):
        mean_energy_price = self.get_sosdisc_inputs(GlossaryEnergy.EnergyMeanPriceValue)['energy_price'].tolist()[-1]
        prices = self.get_sosdisc_inputs('energy_prices_after_tax')
        prods = self.get_sosdisc_inputs(GlossaryEnergy.StreamProductionDetailedValue)
        fossil_price = prices[GlossaryEnergy.fossil].tolist()[-1]
        renewable_price = prices[GlossaryCore.clean_energy].tolist()[-1]
        total_prod = prods['Total production (uncut)'].tolist()[-1] * 1e-3
        fossil_prod = prods['production fossil (TWh)'].tolist()[-1] * 1e-3
        renewable_prod = prods[f'production {GlossaryCore.clean_energy} (TWh)'].tolist()[-1] * 1e-3
        world_net_product = self.get_sosdisc_inputs(GlossaryCore.EconomicsDfValue)['output_net_of_d'].tolist()[-1]
        temperature_rise = self.get_sosdisc_inputs(GlossaryCore.TemperatureDfValue)['temp_atmo'].tolist()[-1]

        self.store_sos_outputs_values({
            'mean_energy_price_2100': mean_energy_price,
            'fossil_energy_price_2100': fossil_price,
            'renewable_energy_price_2100': renewable_price,
            'total_energy_production_2100': total_prod,
            'fossil_energy_production_2100': fossil_prod,
            'renewable_energy_production_2100': renewable_prod,
            'world_net_product_2100': world_net_product,
            'temperature_rise_2100': temperature_rise,
        })
