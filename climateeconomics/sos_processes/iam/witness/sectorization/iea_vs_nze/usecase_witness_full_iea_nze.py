'''
Copyright 2023 Capgemini

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
import os.path
from pathlib import Path

import numpy as np
import pandas as pd
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.models.electricity.wind_onshore.wind_onshore_disc import (
    WindOnshoreDiscipline,
)
from energy_models.sos_processes.energy.MDA.energy_process_v0_mda.usecase import (
    Study as datacase_energy,
)
from sostrades_optimization_plugins.models.func_manager.func_manager import (
    FunctionManager,
)
from sostrades_optimization_plugins.models.func_manager.func_manager_disc import (
    FunctionManagerDisc,
)

from climateeconomics.core.tools.ClimateEconomicsStudyManager import (
    ClimateEconomicsStudyManager,
)
from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.sectorization.witness_sect_wo_energy.datacase_witness_wo_energy import (
    DataStudy as datacase_witness,
)

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX
DEFAULT_COARSE_TECHNO_DICT = {GlossaryCore.clean_energy: {'type': 'energy', 'value': [GlossaryCore.CleanEnergySimpleTechno]},
                              GlossaryEnergy.fossil: {'type': 'energy', 'value': [GlossaryEnergy.FossilSimpleTechno]},
                              GlossaryEnergy.carbon_captured: {'type': 'CCUS', 'value': [f'{GlossaryEnergy.direct_air_capture}.{GlossaryEnergy.DirectAirCaptureTechno}',
                                                                           f'{GlossaryEnergy.flue_gas_capture}.{GlossaryEnergy.FlueGasTechno}']},
                              GlossaryEnergy.carbon_storage: {'type': 'CCUS', 'value': [GlossaryEnergy.CarbonStorageTechno]}}
DEFAULT_ENERGY_LIST = [key for key, value in DEFAULT_COARSE_TECHNO_DICT.items(
) if value['type'] == 'energy']
DEFAULT_CCS_LIST = [key for key, value in DEFAULT_COARSE_TECHNO_DICT.items(
) if value['type'] == 'CCUS']

DATA_DIR = Path(__file__).parents[5] / "data"
def create_df_from_csv(filename: str, data_dir=DATA_DIR, **kwargs):
    """Creates a pandas DataFrame from a given filename"""
    return pd.read_csv(str(data_dir / filename), **kwargs)


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=GlossaryCore.YearStartDefault, year_end=GlossaryCore.YearEndDefault, bspline=True, run_usecase=True,
                 execution_engine=None,
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[2], techno_dict=GlossaryEnergy.DEFAULT_TECHNO_DICT):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.energy_list = DEFAULT_ENERGY_LIST
        self.ccs_list = DEFAULT_CCS_LIST
        self.dc_energy = datacase_energy(
            self.year_start, self.year_end, bspline=self.bspline, execution_engine=execution_engine,
            invest_discipline=self.invest_discipline, techno_dict=techno_dict, main_study=False)
        self.sub_study_path_dict = self.dc_energy.sub_study_path_dict

    def setup_process(self):
        datacase_energy.setup_process(self)


    def setup_usecase(self, study_folder_path=None):
        setup_data = {}

        # -- load data from energy pyworld3
        # -- Start with energy to have it at first position in the list...
        self.dc_energy.study_name = self.study_name
        self.energy_mda_usecase = self.dc_energy

        # -- load data from witness
        dc_witness = datacase_witness(
            self.year_start, self.year_end)
        dc_witness.study_name = self.study_name

        witness_input_list = dc_witness.setup_usecase()
        setup_data.update(witness_input_list)

        energy_input_list = self.dc_energy.setup_usecase()
        setup_data.update(energy_input_list)

        self.dict_technos = self.dc_energy.dict_technos
        dspace_energy = self.dc_energy.dspace

        self.merge_design_spaces([dspace_energy, dc_witness.dspace])


        numerical_values_dict = {
            f'{self.study_name}.epsilon0': 1.0,
            f'{self.study_name}.max_mda_iter': 50,
            f'{self.study_name}.tolerance': 1.0e-10,
            f'{self.study_name}.n_processes': 1,
            f'{self.study_name}.linearization_mode': 'adjoint',
            f'{self.study_name}.inner_mda_name': 'MDAGaussSeidel',
            f'{self.study_name}.cache_type': 'SimpleCache'}

        setup_data.update(numerical_values_dict)


        invest_mix = self.get_var_in_values_dict(setup_data, varname="invest_mix")[0]
        setup_data[f'{self.study_name}.EnergyMix.invest_mix'] = invest_mix
        setup_data[f'{self.study_name}.consumers_actors'] = [GlossaryCore.CCUS, GlossaryCore.SectorIndustry, GlossaryCore.SectorServices, GlossaryCore.Crop]

        # loading IEA vs nze data:
        invest_mix = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'invest_mix.csv'))
        setup_data[f'{self.study_name}.EnergyMix.invest_mix'] = invest_mix

        utilisation_ratio = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'utilisation_ratio.csv'))
        for col in utilisation_ratio.columns:
            setup_data[f'{self.study_name}.EnergyMix.{col}.{GlossaryEnergy.UtilisationRatioValue}'] = \
                pd.DataFrame({GlossaryCore.Years: np.arange(self.year_start, self.year_end + 1),
                              GlossaryCore.UtilisationRatioValue: utilisation_ratio[col].values})

        # update capex
        crop_investment_df_NZE = DatabaseWitnessCore.CropInvestmentNZE.value

        # Update Wind Onshore initial capex
        onshore_infos_dict = WindOnshoreDiscipline.techno_infos_dict_default
        onshore_infos_dict["Capex_init"] = 1242.2223847891785  # from data_energy/fitting/windpower.py

        invest_before_year_start_anaerobicdigestion = pd.DataFrame({GlossaryEnergy.Years: np.arange(
            self.year_start - GlossaryEnergy.TechnoConstructionDelayDict['AnaerobicDigestion'], self.year_start),
                                                                    GlossaryEnergy.InvestValue: [0., 1.54817207,
                                                                                                 1.64611214]})
        invest_before_year_start_hydropower = pd.DataFrame({GlossaryEnergy.Years: np.arange(
            self.year_start - GlossaryEnergy.TechnoConstructionDelayDict['Hydropower'], self.year_start),
                                                            GlossaryEnergy.InvestValue: [0., 102.49276698,
                                                                                         98.17710767]})
        invest_before_year_start_windonshore = pd.DataFrame({GlossaryEnergy.Years: np.arange(
            self.year_start - GlossaryEnergy.TechnoConstructionDelayDict['WindOnshore'], self.year_start),
                                                             GlossaryEnergy.InvestValue: [0., 125.73068603,
                                                                                          125.73068603]})
        invest_before_year_start_windoffshore = pd.DataFrame({GlossaryEnergy.Years: np.arange(
            self.year_start - GlossaryEnergy.TechnoConstructionDelayDict['WindOffshore'], self.year_start),
                                                              GlossaryEnergy.InvestValue: [0., 34.26931397,
                                                                                           34.26931397]})
        invest_mix[f'{GlossaryEnergy.methane}.{GlossaryEnergy.Methanation}'] = 0.
        invest_mix[f'{GlossaryEnergy.fuel}.{GlossaryEnergy.ethanol}.{GlossaryEnergy.BiomassFermentation}'] = 0.

        diet_mortality_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'diet_mortality.csv'))
        forest_invest = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'forest_investment.csv'))

        setup_data.update({
            f'{self.study_name}.InvestmentDistribution.invest_mix': invest_mix,
            #f'{self.study_name}.InvestmentDistribution.reforestation_investment': forest_invest,
            f'{self.study_name}.Agriculture.Crop.crop_investment': crop_investment_df_NZE,
            f'{self.study_name}.Agriculture.Forestry.reforestation_cost_per_ha': 3800.,
            f'{self.study_name}.Population.diet_mortality_param_df': diet_mortality_df,
            f'{self.study_name}.EnergyMix.electricity.Hydropower.initial_production': 4444.3,
            # from data_energy/fitting/hydropower.py
            f'{self.study_name}.EnergyMix.biogas.AnaerobicDigestion.initial_production': 507.47,
            # from data_energy/fitting/gaseous_bioenergy.py
            f'{self.study_name}.EnergyMix.electricity.WindOnshore.initial_production': 1555.51,
            # from data_energy/fitting/windpower.py
            f'{self.study_name}.EnergyMix.electricity.WindOffshore.initial_production': 111.08,
            # from data_energy/fitting/windpower.py
            # f'{self.study_name}.EnergyMix.electricity.Hydropower.{GlossaryEnergy.InitialPlantsAgeDistribFactor}': 1.2236,  #result from data_energy/fitting/hydropower.py
            # f'{self.study_name}.EnergyMix.biogas.AnaerobicDigestion.{GlossaryEnergy.InitialPlantsAgeDistribFactor}': 1.0137,  # result from data_energy/fitting/gaseous_bioenergy.py
            # f'{self.study_name}.EnergyMix.electricity.WindOnshore.{GlossaryEnergy.InitialPlantsAgeDistribFactor}': 1.3313,  # result from data_energy/fitting/windpower.py
            # f'{self.study_name}.EnergyMix.electricity.WindOffshore.{GlossaryEnergy.InitialPlantsAgeDistribFactor}': 1.3313,  # result from data_energy/fitting/windpower.py
            f'{self.study_name}.EnergyMix.biogas.AnaerobicDigestion.{GlossaryEnergy.InvestmentBeforeYearStartValue}': invest_before_year_start_anaerobicdigestion,
            f'{self.study_name}.EnergyMix.electricity.Hydropower.{GlossaryEnergy.InvestmentBeforeYearStartValue}': invest_before_year_start_hydropower,
            f'{self.study_name}.EnergyMix.electricity.WindOnshore.{GlossaryEnergy.InvestmentBeforeYearStartValue}': invest_before_year_start_windonshore,
            f'{self.study_name}.EnergyMix.electricity.WindOffshore.{GlossaryEnergy.InvestmentBeforeYearStartValue}': invest_before_year_start_windoffshore,
            f'{self.study_name}.EnergyMix.electricity.WindOnshore.techno_infos_dict': onshore_infos_dict,
        })

        # input for IEA data6
        CO2_emissions_df = create_df_from_csv("IEA_NZE_co2_emissions_Gt.csv")
        GDP_df = create_df_from_csv("IEA_NZE_output_net_of_d.csv")
        # for data integrity, requires values for pc_consumption and gross_output => set to 0
        GDP_df[GlossaryCore.GrossOutput] = 0.
        GDP_df[GlossaryCore.PerCapitaConsumption] = 0.
        CO2_tax_df = create_df_from_csv("IEA_NZE_CO2_taxes.csv")
        population_df = create_df_from_csv("IEA_NZE_population.csv")
        temperature_df = create_df_from_csv("IEA_NZE_temp_atmo.csv")
        energy_production_df = create_df_from_csv("IEA_NZE_energy_production_brut.csv")
        energy_consumption_df = create_df_from_csv("IEA_NZE_energy_final_consumption.csv")
        nuclear_production_df = create_df_from_csv("IEA_NZE_EnergyMix.electricity.Nuclear.techno_production.csv")
        hydro_production_df = create_df_from_csv("IEA_NZE_EnergyMix.electricity.Hydropower.techno_production.csv")
        solar_production_df = create_df_from_csv("IEA_NZE_EnergyMix.electricity.SolarPv.techno_production.csv")
        wind_production_df = create_df_from_csv("IEA_NZE_EnergyMix.electricity.WindXXshore.techno_production.csv")
        coal_production_df = create_df_from_csv("IEA_NZE_EnergyMix_solid_fuel_CoalExtraction_techno_production.csv")
        fossil_gas_production_df = create_df_from_csv(
            "IEA_NZE_EnergyMix.methane.FossilGas.techno_detailed_production.csv")
        biogas_production_df = create_df_from_csv("IEA_NZE_EnergyMix.biogas.energy_production_detailed.csv")
        crop_production_df = create_df_from_csv("IEA_NZE_crop_mix_detailed_production.csv")
        forest_production_df = create_df_from_csv("IEA_NZE_forest_techno_production.csv")
        electricity_prices_df = create_df_from_csv("IEA_NZE_electricity_Technologies_Mix_prices.csv")
        natural_gas_price_df = create_df_from_csv("IEA_NZE_EnergyMix.methane.FossilGas.techno_prices.csv")
        land_use_df = create_df_from_csv("IEA_NZE_Land_Use.land_surface_detail_df.csv")

        setup_data.update({
            **self.set_value_at_namespace("mdo_mode_energy", True, "ns_public"),
            f'{self.study_name}.{GlossaryEnergy.YearStart}': self.year_start,
            f'{self.study_name}.{GlossaryEnergy.YearEnd}': self.year_end,
            f'{self.study_name}.IEA.{GlossaryEnergy.CO2EmissionsGtValue}': CO2_emissions_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.EconomicsDfValue}': GDP_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.CO2TaxesValue}': CO2_tax_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.StreamProductionValue}': energy_production_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.EnergyFinalConsumptionName}': energy_consumption_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.TemperatureDfValue}': temperature_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.PopulationDfValue}': population_df,
            # energy production
            f'{self.study_name}.IEA.{GlossaryEnergy.electricity}_{GlossaryEnergy.Nuclear}_techno_production': nuclear_production_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.electricity}_{GlossaryEnergy.Hydropower}_techno_production': hydro_production_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.electricity}_{GlossaryEnergy.Solar}_techno_production': solar_production_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.electricity}_{GlossaryEnergy.WindOnshoreAndOffshore}_techno_production': wind_production_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.solid_fuel}_{GlossaryEnergy.CoalExtraction}_techno_production': coal_production_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.methane}_{GlossaryEnergy.FossilGas}_techno_production': fossil_gas_production_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.biogas}_{GlossaryEnergy.AnaerobicDigestion}_techno_production': biogas_production_df,
            #f'{self.study_name}.IEA.{GlossaryEnergy.CropEnergy}_techno_production': crop_production_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.ForestProduction}_techno_production': forest_production_df,
            f'{self.study_name}.IEA.land_surface_detail_df': land_use_df,
            # energy prices
            f'{self.study_name}.IEA.{GlossaryEnergy.electricity}_{GlossaryEnergy.StreamPricesValue}': electricity_prices_df,
            f'{self.study_name}.IEA.{GlossaryEnergy.methane}_{GlossaryEnergy.StreamPricesValue}': natural_gas_price_df
        })


        return setup_data


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
