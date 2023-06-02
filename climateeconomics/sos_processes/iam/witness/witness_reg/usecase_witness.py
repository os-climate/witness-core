'''
Copyright 2022 Airbus SAS

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

from os.path import join, dirname
from pandas import DataFrame, concat
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

from sostrades_core.study_manager.study_manager import StudyManager
from climateeconomics.sos_processes.iam.witness_wo_energy.datacase_witness_wo_energy import \
    DataStudy as datacase_witness
from climateeconomics.sos_processes.iam.witness_wo_energy_dev.datacase_witness_wo_energy import \
    DataStudy as datacase_witness_dev
from climateeconomics.sos_processes.iam.witness_wo_energy_thesis.datacase_witness_wo_energy_solow import \
    DataStudy as datacase_witness_thesis
from energy_models.sos_processes.energy.MDA.energy_process_v0_mda.usecase import Study as datacase_energy

from sostrades_core.execution_engine.func_manager.func_manager import FunctionManager
from sostrades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT
from climateeconomics.sos_processes.iam.witness.agriculture_mix_process.usecase import \
    AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT

from climateeconomics.core.tools.ClimateEconomicsStudyManager import ClimateEconomicsStudyManager
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.tools.jsonhandling import convert_to_editable_json, preprocess_data_and_save_json, insert_json_to_mongodb_bis

import cProfile
from io import StringIO
import pstats

INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
AGGR_TYPE = FunctionManagerDisc.AGGR_TYPE
AGGR_TYPE_SUM = FunctionManager.AGGR_TYPE_SUM
AGGR_TYPE_SMAX = FunctionManager.AGGR_TYPE_SMAX

def generate_json_by_discipline(data, json_name):
    json_data = convert_to_editable_json(data)
    json_data_updt = create_fake_regions(json_data, ['US', 'UE'])
    json_data_updt['id'] = json_name.split('.')[-1]
    output_path = join(dirname(__file__), 'data', f'{json_name}.json')
    preprocess_data_and_save_json(json_data_updt, output_path)


def prepare_data(data):
    """
    Prepare data by getting only values
    """

    dict_data = {}
    for disc_id in list(uc_cls.ee.dm.disciplines_dict.keys()):
        disc = dm.get_discipline(disc_id)
        data_in = disc.get_data_in()
        dict_data[disc.sos_name] = {}
        for k,v in data_in.items():
            if not v['numerical']:
                dict_data[disc.sos_name][k] = v['value']
    return dict_data


def create_fake_regions(data, regions_list): 
    """
    Add regions 
    """
    data_updt = {}
    for reg in regions_list:
        data_updt[reg] = data 
    return data_updt


class Study(ClimateEconomicsStudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, bspline=True, run_usecase=False,
                 execution_engine=None,
                 invest_discipline=INVEST_DISCIPLINE_OPTIONS[
                     2], techno_dict=DEFAULT_TECHNO_DICT, agri_techno_list=AGRI_MIX_TECHNOLOGIES_LIST_FOR_OPT,
                 process_level='val'):
        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.bspline = bspline
        self.invest_discipline = invest_discipline
        self.techno_dict = techno_dict
        self.agri_techno_list = agri_techno_list
        self.process_level = process_level
        self.dc_energy = datacase_energy(
            self.year_start, self.year_end, self.time_step, bspline=self.bspline, execution_engine=execution_engine,
            invest_discipline=self.invest_discipline, techno_dict=techno_dict)
        self.sub_study_path_dict = self.dc_energy.sub_study_path_dict

    def setup_constraint_land_use(self):
        func_df = DataFrame(
            columns=['variable', 'parent', 'ftype', 'weight', AGGR_TYPE])
        list_var = []
        list_parent = []
        list_ftype = []
        list_weight = []
        list_aggr_type = []
        list_ns = []
        list_var.extend(
            ['land_demand_constraint'])
        list_parent.extend(['agriculture_constraint'])
        list_ftype.extend([INEQ_CONSTRAINT])
        list_weight.extend([-1.0])
        list_aggr_type.extend(
            [AGGR_TYPE_SUM])
        list_ns.extend(['ns_functions'])
        func_df['variable'] = list_var
        func_df['parent'] = list_parent
        func_df['ftype'] = list_ftype
        func_df['weight'] = list_weight
        func_df[AGGR_TYPE] = list_aggr_type
        func_df['namespace'] = list_ns

        return func_df

    def setup_usecase(self):
        setup_data_list = []


        numerical_values_dict = {
            f'{self.study_name}.epsilon0': 1.0,
            f'{self.study_name}.max_mda_iter': 50,
            f'{self.study_name}.tolerance': 1.0e-10,
            f'{self.study_name}.n_processes': 1,
            f'{self.study_name}.linearization_mode': 'adjoint',
            f'{self.study_name}.sub_mda_class': 'GSPureNewtonMDA',
            f'{self.study_name}.cache_type': 'SimpleCache', }

        database_ids = {'usecase_witness.US.Macroeconomics.database_id': 'Macroeconomics', 'usecase_witness.US.GHGCycle.database_id': 'GHGCycle', 'usecase_witness.US.Damage.database_id': 'Damage', 'usecase_witness.US.Temperature_change.database_id': 'Temperature_change', 'usecase_witness.US.Utility.database_id': 'Utility', 'usecase_witness.US.Policy.database_id': 'Policy', 'usecase_witness.US.Land_Use.database_id': 'Land_Use', 'usecase_witness.US.AgricultureMix.database_id': 'AgricultureMix', 'usecase_witness.US.AgricultureMix.Crop.database_id': 'Crop', 'usecase_witness.US.AgricultureMix.Forest.database_id': 'Forest', 'usecase_witness.US.Population.database_id': 'Population', 'usecase_witness.US.NonUseCapitalDiscipline.database_id': 'NonUseCapitalDiscipline', 'usecase_witness.US.GHGEmissions.database_id': 'GHGEmissions', 'usecase_witness.US.GHGEmissions.Industry.database_id': 'Industry', 'usecase_witness.US.GHGEmissions.Agriculture.database_id': 'Agriculture', 'usecase_witness.US.EnergyMix.methane.database_id': 'methane', 
'usecase_witness.US.EnergyMix.methane.FossilGas.database_id': 'FossilGas', 'usecase_witness.US.EnergyMix.methane.UpgradingBiogas.database_id': 'UpgradingBiogas', 'usecase_witness.US.EnergyMix.methane.Methanation.database_id': 'Methanation', 'usecase_witness.US.EnergyMix.hydrogen.gaseous_hydrogen.database_id': 'gaseous_hydrogen', 'usecase_witness.US.EnergyMix.hydrogen.gaseous_hydrogen.WaterGasShift.database_id': 'WaterGasShift', 'usecase_witness.US.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.SOEC.database_id': 'SOEC', 'usecase_witness.US.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.PEM.database_id': 'PEM', 'usecase_witness.US.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.AWE.database_id': 'AWE', 'usecase_witness.US.EnergyMix.hydrogen.gaseous_hydrogen.PlasmaCracking.database_id': 'PlasmaCracking', 'usecase_witness.US.EnergyMix.biogas.database_id': 'biogas', 'usecase_witness.US.EnergyMix.biogas.AnaerobicDigestion.database_id': 'AnaerobicDigestion', 'usecase_witness.US.EnergyMix.syngas.database_id': 'syngas', 'usecase_witness.US.EnergyMix.syngas.BiomassGasification.database_id': 'BiomassGasification', 'usecase_witness.US.EnergyMix.syngas.SMR.database_id': 'SMR', 'usecase_witness.US.EnergyMix.syngas.CoalGasification.database_id': 'CoalGasification', 'usecase_witness.US.EnergyMix.syngas.Pyrolysis.database_id': 'Pyrolysis', 'usecase_witness.US.EnergyMix.syngas.AutothermalReforming.database_id': 'AutothermalReforming', 'usecase_witness.US.EnergyMix.syngas.CoElectrolysis.database_id': 'CoElectrolysis', 'usecase_witness.US.EnergyMix.fuel.liquid_fuel.database_id': 'liquid_fuel', 'usecase_witness.US.EnergyMix.fuel.liquid_fuel.Refinery.database_id': 'Refinery', 'usecase_witness.US.EnergyMix.fuel.liquid_fuel.FischerTropsch.database_id': 'FischerTropsch', 'usecase_witness.US.EnergyMix.fuel.hydrotreated_oil_fuel.database_id': 'hydrotreated_oil_fuel', 'usecase_witness.US.EnergyMix.fuel.hydrotreated_oil_fuel.HefaDecarboxylation.database_id': 'HefaDecarboxylation', 'usecase_witness.US.EnergyMix.fuel.hydrotreated_oil_fuel.HefaDeoxygenation.database_id': 'HefaDeoxygenation', 'usecase_witness.US.EnergyMix.solid_fuel.database_id': 'solid_fuel', 'usecase_witness.US.EnergyMix.solid_fuel.CoalExtraction.database_id': 'CoalExtraction', 'usecase_witness.US.EnergyMix.solid_fuel.Pelletizing.database_id': 'Pelletizing', 'usecase_witness.US.EnergyMix.electricity.database_id': 'electricity', 'usecase_witness.US.EnergyMix.electricity.WindOffshore.database_id': 'WindOffshore', 'usecase_witness.US.EnergyMix.electricity.WindOnshore.database_id': 'WindOnshore', 'usecase_witness.US.EnergyMix.electricity.SolarPv.database_id': 'SolarPv', 'usecase_witness.US.EnergyMix.electricity.SolarThermal.database_id': 'SolarThermal', 'usecase_witness.US.EnergyMix.electricity.Hydropower.database_id': 'Hydropower', 'usecase_witness.US.EnergyMix.electricity.Nuclear.database_id': 'Nuclear', 'usecase_witness.US.EnergyMix.electricity.CombinedCycleGasTurbine.database_id': 'CombinedCycleGasTurbine', 'usecase_witness.US.EnergyMix.electricity.GasTurbine.database_id': 'GasTurbine', 'usecase_witness.US.EnergyMix.electricity.BiogasFired.database_id': 'BiogasFired', 'usecase_witness.US.EnergyMix.electricity.Geothermal.database_id': 'Geothermal', 'usecase_witness.US.EnergyMix.electricity.CoalGen.database_id': 'CoalGen', 'usecase_witness.US.EnergyMix.electricity.OilGen.database_id': 'OilGen', 'usecase_witness.US.EnergyMix.electricity.BiomassFired.database_id': 'BiomassFired', 'usecase_witness.US.EnergyMix.fuel.biodiesel.database_id': 'biodiesel', 'usecase_witness.US.EnergyMix.fuel.biodiesel.Transesterification.database_id': 'Transesterification', 'usecase_witness.US.EnergyMix.fuel.ethanol.database_id': 'ethanol', 'usecase_witness.US.EnergyMix.fuel.ethanol.BiomassFermentation.database_id': 'BiomassFermentation', 'usecase_witness.US.EnergyMix.hydrogen.liquid_hydrogen.database_id': 'liquid_hydrogen', 'usecase_witness.US.EnergyMix.hydrogen.liquid_hydrogen.HydrogenLiquefaction.database_id': 'HydrogenLiquefaction', 'usecase_witness.US.GHGEmissions.Energy.database_id': 'Energy', 'usecase_witness.US.EnergyMix.database_id': 'EnergyMix', 'usecase_witness.US.CCUS.database_id': 'CCUS', 'usecase_witness.US.Resources.coal_resource.database_id': 'coal_resource', 'usecase_witness.US.Resources.oil_resource.database_id': 'oil_resource', 'usecase_witness.US.Resources.natural_gas_resource.database_id': 'natural_gas_resource', 'usecase_witness.US.Resources.uranium_resource.database_id': 'uranium_resource', 'usecase_witness.US.Resources.copper_resource.database_id': 'copper_resource', 'usecase_witness.US.Resources.database_id': 'Resources', 'usecase_witness.US.InvestmentDistribution.database_id': 'InvestmentDistribution', 'usecase_witness.US.CCUS.carbon_capture.database_id': 'carbon_capture', 'usecase_witness.US.CCUS.carbon_capture.flue_gas_capture.database_id': 'flue_gas_capture', 'usecase_witness.US.CCUS.carbon_capture.direct_air_capture.AmineScrubbing.database_id': 'AmineScrubbing', 'usecase_witness.US.CCUS.carbon_capture.direct_air_capture.CalciumPotassiumScrubbing.database_id': 'CalciumPotassiumScrubbing', 'usecase_witness.US.CCUS.carbon_capture.flue_gas_capture.CalciumLooping.database_id': 'CalciumLooping', 'usecase_witness.US.CCUS.carbon_capture.flue_gas_capture.ChilledAmmoniaProcess.database_id': 
'ChilledAmmoniaProcess', 'usecase_witness.US.CCUS.carbon_capture.flue_gas_capture.CO2Membranes.database_id': 'CO2Membranes', 'usecase_witness.US.CCUS.carbon_capture.flue_gas_capture.MonoEthanolAmine.database_id': 
'MonoEthanolAmine', 'usecase_witness.US.CCUS.carbon_capture.flue_gas_capture.PiperazineProcess.database_id': 'PiperazineProcess', 'usecase_witness.US.CCUS.carbon_capture.flue_gas_capture.PressureSwingAdsorption.database_id': 'PressureSwingAdsorption', 'usecase_witness.US.CCUS.carbon_storage.database_id': 'carbon_storage', 'usecase_witness.US.CCUS.carbon_storage.BiomassBuryingFossilization.database_id': 'BiomassBuryingFossilization', 'usecase_witness.US.CCUS.carbon_storage.DeepOceanInjection.database_id': 'DeepOceanInjection', 'usecase_witness.US.CCUS.carbon_storage.DeepSalineFormation.database_id': 'DeepSalineFormation', 'usecase_witness.US.CCUS.carbon_storage.DepletedOilGas.database_id': 'DepletedOilGas', 'usecase_witness.US.CCUS.carbon_storage.EnhancedOilRecovery.database_id': 'EnhancedOilRecovery', 'usecase_witness.US.CCUS.carbon_storage.GeologicMineralization.database_id': 'GeologicMineralization', 'usecase_witness.US.CCUS.carbon_storage.PureCarbonSolidStorage.database_id': 'PureCarbonSolidStorage', 'usecase_witness.US.EnergyMix.fuel.database_id': 'fuel', 'usecase_witness.US.Energy_demand.database_id': 'Energy_demand', 'usecase_witness.UE.Macroeconomics.database_id': 'Macroeconomics', 'usecase_witness.UE.GHGCycle.database_id': 'GHGCycle', 'usecase_witness.UE.Damage.database_id': 'Damage', 'usecase_witness.UE.Temperature_change.database_id': 'Temperature_change', 'usecase_witness.UE.Utility.database_id': 'Utility', 'usecase_witness.UE.Policy.database_id': 'Policy', 'usecase_witness.UE.Land_Use.database_id': 'Land_Use', 'usecase_witness.UE.AgricultureMix.database_id': 'AgricultureMix', 'usecase_witness.UE.AgricultureMix.Crop.database_id': 'Crop', 'usecase_witness.UE.AgricultureMix.Forest.database_id': 'Forest', 'usecase_witness.UE.Population.database_id': 'Population', 'usecase_witness.UE.NonUseCapitalDiscipline.database_id': 'NonUseCapitalDiscipline', 'usecase_witness.UE.GHGEmissions.database_id': 'GHGEmissions', 'usecase_witness.UE.GHGEmissions.Industry.database_id': 'Industry', 'usecase_witness.UE.GHGEmissions.Agriculture.database_id': 'Agriculture', 'usecase_witness.UE.EnergyMix.methane.database_id': 'methane', 'usecase_witness.UE.EnergyMix.methane.FossilGas.database_id': 'FossilGas', 'usecase_witness.UE.EnergyMix.methane.UpgradingBiogas.database_id': 'UpgradingBiogas', 'usecase_witness.UE.EnergyMix.methane.Methanation.database_id': 'Methanation', 'usecase_witness.UE.EnergyMix.hydrogen.gaseous_hydrogen.database_id': 'gaseous_hydrogen', 'usecase_witness.UE.EnergyMix.hydrogen.gaseous_hydrogen.WaterGasShift.database_id': 'WaterGasShift', 'usecase_witness.UE.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.SOEC.database_id': 'SOEC', 'usecase_witness.UE.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.PEM.database_id': 'PEM', 'usecase_witness.UE.EnergyMix.hydrogen.gaseous_hydrogen.Electrolysis.AWE.database_id': 'AWE', 'usecase_witness.UE.EnergyMix.hydrogen.gaseous_hydrogen.PlasmaCracking.database_id': 'PlasmaCracking', 'usecase_witness.UE.EnergyMix.biogas.database_id': 'biogas', 'usecase_witness.UE.EnergyMix.biogas.AnaerobicDigestion.database_id': 'AnaerobicDigestion', 'usecase_witness.UE.EnergyMix.syngas.database_id': 'syngas', 'usecase_witness.UE.EnergyMix.syngas.BiomassGasification.database_id': 'BiomassGasification', 'usecase_witness.UE.EnergyMix.syngas.SMR.database_id': 'SMR', 'usecase_witness.UE.EnergyMix.syngas.CoalGasification.database_id': 'CoalGasification', 'usecase_witness.UE.EnergyMix.syngas.Pyrolysis.database_id': 'Pyrolysis', 'usecase_witness.UE.EnergyMix.syngas.AutothermalReforming.database_id': 'AutothermalReforming', 'usecase_witness.UE.EnergyMix.syngas.CoElectrolysis.database_id': 'CoElectrolysis', 'usecase_witness.UE.EnergyMix.fuel.liquid_fuel.database_id': 'liquid_fuel', 'usecase_witness.UE.EnergyMix.fuel.liquid_fuel.Refinery.database_id': 'Refinery', 'usecase_witness.UE.EnergyMix.fuel.liquid_fuel.FischerTropsch.database_id': 'FischerTropsch', 'usecase_witness.UE.EnergyMix.fuel.hydrotreated_oil_fuel.database_id': 'hydrotreated_oil_fuel', 'usecase_witness.UE.EnergyMix.fuel.hydrotreated_oil_fuel.HefaDecarboxylation.database_id': 'HefaDecarboxylation', 'usecase_witness.UE.EnergyMix.fuel.hydrotreated_oil_fuel.HefaDeoxygenation.database_id': 'HefaDeoxygenation', 'usecase_witness.UE.EnergyMix.solid_fuel.database_id': 'solid_fuel', 'usecase_witness.UE.EnergyMix.solid_fuel.CoalExtraction.database_id': 'CoalExtraction', 'usecase_witness.UE.EnergyMix.solid_fuel.Pelletizing.database_id': 'Pelletizing', 'usecase_witness.UE.EnergyMix.electricity.database_id': 'electricity', 'usecase_witness.UE.EnergyMix.electricity.WindOffshore.database_id': 'WindOffshore', 'usecase_witness.UE.EnergyMix.electricity.WindOnshore.database_id': 'WindOnshore', 'usecase_witness.UE.EnergyMix.electricity.SolarPv.database_id': 'SolarPv', 'usecase_witness.UE.EnergyMix.electricity.SolarThermal.database_id': 'SolarThermal', 'usecase_witness.UE.EnergyMix.electricity.Hydropower.database_id': 'Hydropower', 'usecase_witness.UE.EnergyMix.electricity.Nuclear.database_id': 'Nuclear', 'usecase_witness.UE.EnergyMix.electricity.CombinedCycleGasTurbine.database_id': 'CombinedCycleGasTurbine', 'usecase_witness.UE.EnergyMix.electricity.GasTurbine.database_id': 'GasTurbine', 'usecase_witness.UE.EnergyMix.electricity.BiogasFired.database_id': 'BiogasFired', 'usecase_witness.UE.EnergyMix.electricity.Geothermal.database_id': 'Geothermal', 'usecase_witness.UE.EnergyMix.electricity.CoalGen.database_id': 'CoalGen', 'usecase_witness.UE.EnergyMix.electricity.OilGen.database_id': 'OilGen', 'usecase_witness.UE.EnergyMix.electricity.BiomassFired.database_id': 'BiomassFired', 'usecase_witness.UE.EnergyMix.fuel.biodiesel.database_id': 'biodiesel', 'usecase_witness.UE.EnergyMix.fuel.biodiesel.Transesterification.database_id': 'Transesterification', 'usecase_witness.UE.EnergyMix.fuel.ethanol.database_id': 'ethanol', 'usecase_witness.UE.EnergyMix.fuel.ethanol.BiomassFermentation.database_id': 'BiomassFermentation', 'usecase_witness.UE.EnergyMix.hydrogen.liquid_hydrogen.database_id': 'liquid_hydrogen', 'usecase_witness.UE.EnergyMix.hydrogen.liquid_hydrogen.HydrogenLiquefaction.database_id': 'HydrogenLiquefaction', 'usecase_witness.UE.GHGEmissions.Energy.database_id': 'Energy', 'usecase_witness.UE.EnergyMix.database_id': 'EnergyMix', 'usecase_witness.UE.CCUS.database_id': 'CCUS', 'usecase_witness.UE.Resources.coal_resource.database_id': 'coal_resource', 'usecase_witness.UE.Resources.oil_resource.database_id': 'oil_resource', 'usecase_witness.UE.Resources.natural_gas_resource.database_id': 'natural_gas_resource', 'usecase_witness.UE.Resources.uranium_resource.database_id': 'uranium_resource', 'usecase_witness.UE.Resources.copper_resource.database_id': 'copper_resource', 'usecase_witness.UE.Resources.database_id': 'Resources', 'usecase_witness.UE.InvestmentDistribution.database_id': 'InvestmentDistribution', 'usecase_witness.UE.CCUS.carbon_capture.database_id': 'carbon_capture', 'usecase_witness.UE.CCUS.carbon_capture.flue_gas_capture.database_id': 
'flue_gas_capture', 'usecase_witness.UE.CCUS.carbon_capture.direct_air_capture.AmineScrubbing.database_id': 'AmineScrubbing', 'usecase_witness.UE.CCUS.carbon_capture.direct_air_capture.CalciumPotassiumScrubbing.database_id': 'CalciumPotassiumScrubbing', 'usecase_witness.UE.CCUS.carbon_capture.flue_gas_capture.CalciumLooping.database_id': 'CalciumLooping', 'usecase_witness.UE.CCUS.carbon_capture.flue_gas_capture.ChilledAmmoniaProcess.database_id': 'ChilledAmmoniaProcess', 'usecase_witness.UE.CCUS.carbon_capture.flue_gas_capture.CO2Membranes.database_id': 'CO2Membranes', 'usecase_witness.UE.CCUS.carbon_capture.flue_gas_capture.MonoEthanolAmine.database_id': 'MonoEthanolAmine', 'usecase_witness.UE.CCUS.carbon_capture.flue_gas_capture.PiperazineProcess.database_id': 'PiperazineProcess', 'usecase_witness.UE.CCUS.carbon_capture.flue_gas_capture.PressureSwingAdsorption.database_id': 'PressureSwingAdsorption', 'usecase_witness.UE.CCUS.carbon_storage.database_id': 'carbon_storage', 'usecase_witness.UE.CCUS.carbon_storage.BiomassBuryingFossilization.database_id': 'BiomassBuryingFossilization', 'usecase_witness.UE.CCUS.carbon_storage.DeepOceanInjection.database_id': 'DeepOceanInjection', 'usecase_witness.UE.CCUS.carbon_storage.DeepSalineFormation.database_id': 'DeepSalineFormation', 'usecase_witness.UE.CCUS.carbon_storage.DepletedOilGas.database_id': 'DepletedOilGas', 'usecase_witness.UE.CCUS.carbon_storage.EnhancedOilRecovery.database_id': 'EnhancedOilRecovery', 'usecase_witness.UE.CCUS.carbon_storage.GeologicMineralization.database_id': 'GeologicMineralization', 'usecase_witness.UE.CCUS.carbon_storage.PureCarbonSolidStorage.database_id': 'PureCarbonSolidStorage', 'usecase_witness.UE.EnergyMix.fuel.database_id': 'fuel', 'usecase_witness.UE.Energy_demand.database_id': 'Energy_demand'}
        
        setup_data_list.append(numerical_values_dict)
        setup_data_list.append(database_ids)
        return setup_data_list

    def run(self, logger_level=None,
            dump_study=False,
            for_test=False):

        profil = cProfile.Profile()
        profil.enable()
        ClimateEconomicsStudyManager.run(
            self, logger_level=logger_level, dump_study=dump_study, for_test=for_test)
        profil.disable()

        result = StringIO()

        ps = pstats.Stats(profil, stream=result)
        ps.sort_stats('cumulative')
        ps.print_stats(500)
        result = result.getvalue()
        print(result)


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()


    data = uc_cls.ee.dm.get_discipline(list(uc_cls.ee.dm.disciplines_dict.keys())[3]).get_data_in()
    dm = uc_cls.ee.dm

    #dict_data = prepare_data(dm)
    #for k,v in dict_data.items():
    #    generate_json_by_discipline(v, k)

    uc_cls.run()

