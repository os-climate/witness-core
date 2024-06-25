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
import random
import unittest

import pandas as pd
from energy_models.glossaryenergy import GlossaryEnergy as Glossary
from sostrades_core.execution_engine.execution_engine import ExecutionEngine


class IEADataPreparationTest(unittest.TestCase):

    def setUp(self):
        self.name = 'Test'
        self.ee = ExecutionEngine(self.name)

    def test_execute(self):
        self.model_name = 'IEADataPreparation'
        year_start = 2020
        year_end = 2055
        years = [2023, 2030, 2040, 2050]

        ns_dict = {'ns_public': f'{self.name}'}
        self.ee.ns_manager.add_ns_def(ns_dict)

        mod_path = 'climateeconomics.sos_wrapping.post_procs.iea_data_preparation.iea_data_preparation_discipline.IEADataPreparationDiscipline'
        builder = self.ee.factory.get_builder_from_module(
            self.model_name, mod_path)

        self.ee.factory.set_builders_to_coupling_builder(builder)

        self.ee.configure()
        self.ee.display_treeview_nodes()
        CO2_emissions_df = pd.DataFrame({Glossary.Years: years,
                                         Glossary.TotalCO2Emissions: [60, 40, 20, 30]})
        GDP_values = [120, 140, 145, 160]
        GDP_df = pd.DataFrame({Glossary.Years: years,
                               Glossary.OutputNetOfDamage: GDP_values,
                               Glossary.GrossOutput: 0.,
                               Glossary.PerCapitaConsumption: 0.
                               })
        CO2_tax_df = pd.DataFrame({Glossary.Years: years,
                                   Glossary.CO2Tax: [370, 500, 700, 800]})

        energy_production_df = pd.DataFrame({Glossary.Years: years,
                                             Glossary.TotalProductionValue: [40, 70, 80, 10]})

        population_df = pd.DataFrame({Glossary.Years: years,
                                      Glossary.PopulationValue: [8, 8.2, 8.3, 8]})

        temperature_df = pd.DataFrame({Glossary.Years: years,
                                      Glossary.TempAtmo: [2.2, 2.7, 2.75, 2.78],
                                       })

        l_technos_to_add = [f'{Glossary.electricity}_{Glossary.Nuclear}',
                            f'{Glossary.electricity}_{Glossary.Hydropower}',
                            f'{Glossary.electricity}_{Glossary.Solar}',
                            f'{Glossary.electricity}_{Glossary.WindOnshoreAndOffshore}',
                            f'{Glossary.solid_fuel}_{Glossary.CoalExtraction}',
                            f'{Glossary.methane}_{Glossary.FossilGas}',
                            f'{Glossary.biogas}_{Glossary.AnaerobicDigestion}', f'{Glossary.CropEnergy}',
                            f'{Glossary.ForestProduction}'
                            ]

        values_dict = {
            f'{self.name}.{Glossary.YearStart}': year_start,
            f'{self.name}.{Glossary.YearEnd}': year_end,
            f'{self.name}.{self.model_name}.{Glossary.CO2EmissionsGtValue}': CO2_emissions_df,
            f'{self.name}.{self.model_name}.{Glossary.EconomicsDfValue}': GDP_df,
            f'{self.name}.{self.model_name}.{Glossary.CO2TaxesValue}': CO2_tax_df,
            f'{self.name}.{self.model_name}.{Glossary.EnergyProductionValue}': energy_production_df,
            f'{self.name}.{self.model_name}.{Glossary.TemperatureDfValue}': temperature_df,
            f'{self.name}.{self.model_name}.{Glossary.PopulationDfValue}': population_df,
        }
        # random values for techno
        for techno in l_technos_to_add:
            values_dict.update({f'{self.name}.{self.model_name}.{techno}_techno_production' : pd.DataFrame({Glossary.Years: years,
                                Glossary.TechnoProductionValue: [random.randint(15, 100) for _ in range(4)]})})

        values_dict.update({
            f'{self.name}.{self.model_name}.{Glossary.electricity}_energy_prices': pd.DataFrame({
                Glossary.Years: years,
                Glossary.SolarPv: [random.randint(30, 200) for _ in range(4)],
                Glossary.Nuclear: [random.randint(30, 200) for _ in range(4)],
                Glossary.CoalGen: [random.randint(30, 200) for _ in range(4)],
                Glossary.GasTurbine: [random.randint(30, 200) for _ in range(4)]
            })
        })
        self.ee.load_study_from_input_dict(values_dict)

        self.ee.execute()

        disc = self.ee.dm.get_disciplines_with_name(
            f'{self.name}.{self.model_name}')[0]
        gdp_interpolated = disc.get_sosdisc_outputs(f'{Glossary.EconomicsDfValue}_interpolated')
        # check that input value is unchanged
        assert gdp_interpolated.loc[
                   gdp_interpolated[Glossary.Years].isin(years), Glossary.OutputNetOfDamage].tolist() == GDP_values
        # check that the value at 2035 is the expected : 2030 : 140, 2040: 145 => 2035 should be equal to (140+145)/2
        expected_value_2035 = (140+145)/2
        assert gdp_interpolated.loc[gdp_interpolated[Glossary.Years] == 2035, Glossary.OutputNetOfDamage].values[0] == expected_value_2035

        filter = disc.get_chart_filter_list()
        graph_list = disc.get_post_processing_list(filter)
        for graph in graph_list:
            #graph.to_plotly().show()
            pass
