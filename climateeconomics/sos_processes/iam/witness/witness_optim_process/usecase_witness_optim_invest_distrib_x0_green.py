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
import numpy as np
import pandas as pd
from sos_trades_core.study_manager.study_manager import StudyManager
from sos_trades_core.tools.post_processing.post_processing_factory import PostProcessingFactory
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import Study as witness_optim_sub_usecase
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import OPTIM_NAME, COUPLING_NAME, EXTRA_NAME
from sos_trades_core.execution_engine.func_manager.func_manager_disc import FunctionManagerDisc
from climateeconomics.core.design_variables_translation.witness_bspline.design_var_disc import Design_Var_Discipline
from energy_models.core.energy_study_manager import DEFAULT_TECHNO_DICT


OBJECTIVE = FunctionManagerDisc.OBJECTIVE
INEQ_CONSTRAINT = FunctionManagerDisc.INEQ_CONSTRAINT
EQ_CONSTRAINT = FunctionManagerDisc.EQ_CONSTRAINT
OBJECTIVE_LAGR = FunctionManagerDisc.OBJECTIVE_LAGR
FUNC_DF = FunctionManagerDisc.FUNC_DF
EXPORT_CSV = FunctionManagerDisc.EXPORT_CSV
WRITE_XVECT = Design_Var_Discipline.WRITE_XVECT


class Study(StudyManager):

    def __init__(self, year_start=2020, year_end=2100, time_step=1, bspline=False, run_usecase=False, execution_engine=None,
                 one_invest_discipline=True, techno_dict=DEFAULT_TECHNO_DICT):

        super().__init__(__file__, run_usecase=run_usecase, execution_engine=execution_engine)
        self.year_start = year_start
        self.year_end = year_end
        self.time_step = time_step
        self.optim_name = OPTIM_NAME
        self.coupling_name = COUPLING_NAME
        self.extra_name = EXTRA_NAME
        self.bspline = bspline
        self.one_invest_discipline = one_invest_discipline
        self.techno_dict = techno_dict

        self.witness_uc = witness_optim_sub_usecase(
            self.year_start, self.year_end, self.time_step,  bspline=self.bspline, execution_engine=execution_engine,
            one_invest_discipline=self.one_invest_discipline, techno_dict=techno_dict)
        self.sub_study_path_dict = self.witness_uc.sub_study_path_dict

    def setup_process(self):

        witness_optim_sub_usecase.setup_process(self)

    def setup_usecase(self):
        ns = self.study_name

        values_dict = {}

        self.witness_uc.study_name = f'{ns}.{self.optim_name}'
        self.coupling_name = self.witness_uc.coupling_name
        witness_uc_data = self.witness_uc.setup_usecase()
        for dict_data in witness_uc_data:
            values_dict.update(dict_data)

        # design space WITNESS
        dspace_df = self.witness_uc.dspace

        '''
        Old x0 green
        '''

        invest_techno = {}

        methane_FossilGas_array_mix = [1.88, 5., 5., 5., 5., 5., 5., 5.]
        methane_UpgradingBiogas_array_mix = [
            0.02, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        invest_techno['methane'] = pd.DataFrame({'FossilGas': methane_FossilGas_array_mix,
                                                 'UpgradingBiogas': methane_UpgradingBiogas_array_mix})
        hydrogen_gaseous_hydrogen_WaterGasShift_array_mix = [
            10., 10., 10., 10., 10., 10., 10., 10.]
        hydrogen_gaseous_hydrogen_Electrolysis_SOEC_array_mix = [0.01, 0.011, 0.012, 0.013,
                                                                 0.014, 0.015, 0.016, 0.017]
        hydrogen_gaseous_hydrogen_Electrolysis_PEM_array_mix = [0.01, 0.011, 0.012, 0.013,
                                                                0.014, 0.015, 0.016, 0.017]
        hydrogen_gaseous_hydrogen_Electrolysis_AWE_array_mix = [0.01, 0.011, 0.012, 0.013,
                                                                0.014, 0.015, 0.016, 0.017]
        invest_techno['hydrogen_gaseous_hydrogen'] = pd.DataFrame({'WaterGasShift': hydrogen_gaseous_hydrogen_WaterGasShift_array_mix,
                                                                   'Electrolysis_SOEC': hydrogen_gaseous_hydrogen_Electrolysis_SOEC_array_mix,
                                                                   'Electrolysis_PEM': hydrogen_gaseous_hydrogen_Electrolysis_PEM_array_mix,
                                                                   'Electrolysis_AWE': hydrogen_gaseous_hydrogen_Electrolysis_AWE_array_mix, })
        biogas_AnaerobicDigestion_array_mix = [1., 1., 1., 1., 1., 1., 1., 1.]
        invest_techno['biogas'] = pd.DataFrame(
            {'AnaerobicDigestion': biogas_AnaerobicDigestion_array_mix})
        syngas_BiomassGasification_array_mix = [0.5, 0.53, 0.5618, 0.595508, 0.63123848,
                                                0.66911279, 0.70925956, 0.75181513]
        syngas_SMR_array_mix = [10.,  9.6,  9.216,  8.84736,  8.4934656,
                                8.15372698,  7.8275779,  7.51447478]
        syngas_CoalGasification_array_mix = [0.5, 0.485, 0.47045, 0.4563365, 0.4426464,
                                             0.42936701, 0.416486, 0.40399142]
        invest_techno['syngas'] = pd.DataFrame({'BiomassGasification': syngas_BiomassGasification_array_mix,
                                                'SMR': syngas_SMR_array_mix,
                                                'CoalGasification': syngas_CoalGasification_array_mix, })
        liquid_fuel_Refinery_array_mix = [
            100.,  95.,  90.,  85.,  80.,  75.,  70.,  65.]
        liquid_fuel_FischerTropsch_array_mix = [
            0.1, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1]

        invest_techno['liquid_fuel'] = pd.DataFrame({'Refinery': liquid_fuel_Refinery_array_mix,
                                                     'FischerTropsch': liquid_fuel_FischerTropsch_array_mix})
        solid_fuel_CoalExtraction_array_mix = [
            9.99e+01, 5.00e+01, 2.00e-02, 2.00e-02, 2.00e-02, 2.00e-02, 2.00e-02, 2.00e-02]
        solid_fuel_Pelletizing_array_mix = [
            1., 50., 99.9, 99.9, 99.9, 99.9, 99.9, 99.9]
        invest_techno['solid_fuel'] = pd.DataFrame({'CoalExtraction': solid_fuel_CoalExtraction_array_mix,
                                                    'Pelletizing': solid_fuel_Pelletizing_array_mix})
        biomass_dry_ManagedWood_array_mix = [
            1., 1.03, 1.0609, 1.092727, 1.12550881, 1.15927407, 1.1940523, 1.22987387]
        biomass_dry_UnmanagedWood_array_mix = [1., 0.96, 0.9216, 0.884736, 0.84934656,
                                               0.8153727, 0.78275779, 0.75144748]
        biomass_dry_CropEnergy_array_mix = [1., 2., 2., 2., 2., 2., 2., 2.]

        invest_techno['biomass_dry'] = pd.DataFrame({'ManagedWood': biomass_dry_ManagedWood_array_mix,
                                                     'UnmanagedWood': biomass_dry_UnmanagedWood_array_mix,
                                                     'CropEnergy': biomass_dry_CropEnergy_array_mix, })
        electricity_WindOffshore_array_mix = [
            2.550872, 5., 5., 5., 5., 5., 5., 5.]
        electricity_WindOnshore_array_mix = [2.550872e+00, 1.000000e-06, 1.000000e-06, 1.000000e-06, 1.000000e-06, 1.000000e-06,
                                             1.000000e-06, 1.000000e-06]
        electricity_SolarPv_array_mix = [5., 20., 20., 20., 20., 20., 20., 20.]
        electricity_SolarThermal_array_mix = [
            1.388064e+00, 1.000000e-06, 1.000000e-06, 1.000000e-06, 1.000000e-06, 1.000000e-06, 1.000000e-06, 1.000000e-06]
        electricity_Hydropower_array_mix = [1.5, 5., 5., 5., 5., 5., 5., 5.]
        electricity_Nuclear_array_mix = [
            2.1, 75., 75., 75., 75., 75., 75., 75.]
        electricity_CombinedCycleGasTurbine_array_mix = [
            2.1e+00, 1.0e-06, 1.0e-06, 1.0e-06, 1.0e-06, 1.0e-06, 1.0e-06, 1.0e-06]
        electricity_GasTurbine_array_mix = [
            5.e-01, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06]
        electricity_Geothermal_array_mix = [
            2.46564e-01, 1.00000e-06, 1.00000e-06, 1.00000e-06, 1.00000e-06, 1.00000e-06, 1.00000e-06, 1.00000e-06]
        electricity_CoalGen_array_mix = [1.e-01, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06,
                                         1.e-06, 1.e-06]

        invest_techno['electricity'] = pd.DataFrame({'WindOffshore': electricity_WindOffshore_array_mix,
                                                     'WindOnshore': electricity_WindOnshore_array_mix,
                                                     'SolarPv': electricity_SolarPv_array_mix,
                                                     'SolarThermal': electricity_SolarThermal_array_mix,
                                                     'Hydropower': electricity_Hydropower_array_mix,
                                                     'Nuclear': electricity_Nuclear_array_mix,
                                                     'CombinedCycleGasTurbine': electricity_CombinedCycleGasTurbine_array_mix,
                                                     'GasTurbine': electricity_GasTurbine_array_mix,
                                                     'Geothermal': electricity_Geothermal_array_mix,
                                                     'CoalGen': electricity_CoalGen_array_mix, })
        biodiesel_Transesterification_array_mix = [
            1., 1., 1., 1., 1., 1., 1., 1.]
        invest_techno['biodiesel'] = pd.DataFrame(
            {'Transesterification': biodiesel_Transesterification_array_mix})
        hydrogen_liquid_hydrogen_HydrogenLiquefaction_array_mix = [10.,  9.6,  9.216,  8.84736,  8.4934656,  8.15372698,
                                                                   7.8275779,  7.51447478]
        invest_techno['hydrogen_liquid_hydrogen'] = pd.DataFrame(
            {'HydrogenLiquefaction': hydrogen_liquid_hydrogen_HydrogenLiquefaction_array_mix})
        carbon_capture_direct_air_capture_AmineScrubbing_array_mix = [
            0.5, 1., 1., 1., 1., 1., 1., 1.]
        carbon_capture_direct_air_capture_CalciumPotassiumScrubbing_array_mix = [0.1, 0.103, 0.10609, 0.1092727, 0.11255088,
                                                                                 0.11592741, 0.11940523, 0.12298739]
        carbon_capture_flue_gas_capture_CalciumLooping_array_mix = [10.,  9.6,  9.216,  8.84736,  8.4934656, 8.15372698,  7.8275779,
                                                                    7.51447478]
        carbon_capture_flue_gas_capture_MonoEthanolAmine_array_mix = [
            10., 10., 10., 10., 10., 10., 10., 10.]
        invest_techno['carbon_capture'] = pd.DataFrame({'direct_air_capture_AmineScrubbing': carbon_capture_direct_air_capture_AmineScrubbing_array_mix,
                                                        'direct_air_capture_CalciumPotassiumScrubbing': carbon_capture_direct_air_capture_CalciumPotassiumScrubbing_array_mix,
                                                        'flue_gas_capture_CalciumLooping': carbon_capture_flue_gas_capture_CalciumLooping_array_mix,
                                                        'flue_gas_capture_MonoEthanolAmine': carbon_capture_flue_gas_capture_MonoEthanolAmine_array_mix})
        carbon_storage_DeepSalineFormation_array_mix = [10.,  9.6,  9.216,  8.84736,  8.4934656,
                                                        8.15372698,  7.8275779,  7.51447478]
        carbon_storage_DepletedOilGas_array_mix = [10.,  9.6,  9.216,  8.84736,  8.4934656,
                                                   8.15372698,  7.8275779,  7.51447478]
        carbon_storage_GeologicMineralization_array_mix = [
            2., 2., 2., 2., 2., 2., 2., 2.]
        carbon_storage_Reforestation_array_mix = [
            5., 5., 5., 5., 5., 5., 5., 5.]
        carbon_storage_PureCarbonSolidStorage_array_mix = [
            5., 5., 5., 5., 5., 5., 5., 5.]

        invest_techno['carbon_storage'] = pd.DataFrame({'DeepSalineFormation': carbon_storage_DeepSalineFormation_array_mix,
                                                        'DepletedOilGas': carbon_storage_DepletedOilGas_array_mix,
                                                        'GeologicMineralization': carbon_storage_GeologicMineralization_array_mix,
                                                        'Reforestation': carbon_storage_Reforestation_array_mix,
                                                        'PureCarbonSolidStorage': carbon_storage_PureCarbonSolidStorage_array_mix})

        methane_array_mix = [1.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        hydrogen_gaseous_hydrogen_array_mix = [
            0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        biogas_array_mix = [5.e-02, 1.e-06, 2.e-04,
                            2.e-04, 2.e-04, 2.e-04, 2.e-04, 2.e-04]
        syngas_array_mix = [1.005, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
        liquid_fuel_array_mix = [3.15e+00, 1.00e-06, 1.00e-06, 1.00e-06,
                                 1.00e-06, 1.00e-06, 1.00e-06, 1.00e-06]
        solid_fuel_array_mix = [1.e-05, 1.e-03, 7.e-03,
                                9.e-03, 9.e-03, 5.e-03, 3.e-03, 1.e-03]
        biomass_dry_array_mix = [0.003, 0.01, 0.012,
                                 0.012, 0.011, 0.009, 0.009, 0.009]
        electricity_array_mix = [
            4.49, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5, 7.5]
        biodiesel_array_mix = [2.e-02, 1.e-06, 1.e-06,
                               1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06]
        hydrogen_liquid_hydrogen_array_mix = [
            4.e-01, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06, 1.e-06]
        carbon_capture_array_mix = [2., 2., 2., 2., 2., 2., 2., 2.]
        carbon_storage_array_mix = [0.003, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        energy_mix = pd.DataFrame({'methane': methane_array_mix,
                                   'hydrogen_gaseous_hydrogen': hydrogen_gaseous_hydrogen_array_mix,
                                   'biogas': biogas_array_mix,
                                   'syngas': syngas_array_mix,
                                   'liquid_fuel': liquid_fuel_array_mix,
                                   'solid_fuel': solid_fuel_array_mix,
                                   'biomass_dry': biomass_dry_array_mix,
                                   'electricity': electricity_array_mix,
                                   'biodiesel': biodiesel_array_mix,
                                   'hydrogen_liquid_hydrogen': hydrogen_liquid_hydrogen_array_mix,
                                   })

        ccs_mix = pd.DataFrame({'carbon_capture': carbon_capture_array_mix,
                                'carbon_storage': carbon_storage_array_mix,
                                })
        norm_energy_mix = energy_mix.sum(axis=1).values
        norm_ccs_mix = ccs_mix.sum(axis=1).values
        ccs_percentage_array = np.array([0, 20, 20, 20, 20, 20, 20, 20])
        livestock_usage_factor_array = [15., 15., 15., 15., 15., 15., 15., 15.]

        invest_mix_df = pd.DataFrame()
        for energy, invest_techno_energy in invest_techno.items():
            norm_techno_mix = invest_techno_energy.sum(axis=1)
            if energy in energy_mix.columns:
                mix_energy = energy_mix[energy].values / norm_energy_mix * \
                    (100.0 - ccs_percentage_array) / 100.0
            elif energy in ccs_mix.columns:
                mix_energy = ccs_mix[energy].values / norm_ccs_mix * \
                    ccs_percentage_array / 100.0
            else:
                raise Exception(f'{energy} not in investment_mixes')
            for techno in invest_techno_energy.columns:
                if techno != 'years':
                    invest_mix_df[f'{energy}.{techno}'] = np.maximum(1.0e-6, invest_techno_energy[techno].values *
                                                                     mix_energy / norm_techno_mix)

        invest_list_dspace_values = [
            invest_mix_df[techno].values.tolist() for techno in invest_mix_df.columns]
        invest_list_dspace_values.append(livestock_usage_factor_array)
        dspace_df['value'] = pd.Series(invest_list_dspace_values)
        dspace_size = self.witness_uc.dspace_size
        # optimization functions:
        optim_values_dict = {f'{ns}.epsilon0': 1,
                             f'{ns}.{self.optim_name}.design_space': dspace_df,
                             f'{ns}.{self.optim_name}.objective_name': FunctionManagerDisc.OBJECTIVE_LAGR,
                             f'{ns}.{self.optim_name}.eq_constraints': [],
                             f'{ns}.{self.optim_name}.ineq_constraints': [],

                             # optimization parameters:
                             f'{ns}.{self.optim_name}.max_iter': 500,
                             f'{ns}.warm_start': True,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.warm_start': True,
                             # SLSQP, NLOPT_SLSQP
                             f'{ns}.{self.optim_name}.algo': "L-BFGS-B",
                             f'{ns}.{self.optim_name}.formulation': 'DisciplinaryOpt',
                             f'{ns}.{self.optim_name}.differentiation_method': 'user',
                             f'{ns}.{self.optim_name}.algo_options': {"ftol_rel": 3e-16,
                                                                      "normalize_design_space": False,
                                                                      "maxls": 2 * dspace_size,
                                                                      "maxcor": dspace_size,
                                                                      "pg_tol": 1.e-8,
                                                                      "max_iter": 500,
                                                                      "disp": 110},
                             # f'{ns}.{self.optim_name}.{witness_uc.coupling_name}.linear_solver_MDO':
                             # 'gmres',
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDO_options': {'tol': 1.0e-10,
                                                                                                                   'max_iter': 10000},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.linear_solver_MDA_options': {'tol': 1.0e-10,
                                                                                                                   'max_iter': 50000},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.epsilon0': 1.0,
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.tolerance': 1.0e-10,
                             f'{ns}.{self.optim_name}.parallel_options': {"parallel": False,  # True
                                                                          "n_processes": 32,
                                                                          "use_threading": False,
                                                                          "wait_time_between_fork": 0},
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.sub_mda_class': 'GSorNewtonMDA',
                             f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.max_mda_iter': 50, }
# f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables.{WRITE_XVECT}':
# True}

        #print("Design space dimension is ", dspace_size)

        return [values_dict] + [optim_values_dict]


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.load_data()
    print(
        len(uc_cls.execution_engine.root_process.sos_disciplines[0].sos_disciplines[0].sos_disciplines))
    # df_xvect = pd.read_pickle('df_xvect.pkl')
    # df_xvect.columns = [
    # f'{uc_cls.study_name}.{uc_cls.optim_name}.{uc_cls.coupling_name}.DesignVariables' + col for col in df_xvect.columns]
    # dict_xvect = df_xvect.iloc[-1].to_dict()
    # dict_xvect[f'{uc_cls.study_name}.{uc_cls.optim_name}.eval_mode'] = True
    # uc_cls.load_data(from_input_dict=dict_xvect)
    # f'{ns}.{self.optim_name}.{self.witness_uc.coupling_name}.DesignVariables'
    # uc_cls.execution_engine.root_process.sos_disciplines[0].set_opt_scenario()
    # uc_cls.execution_engine.set_debug_mode()
    uc_cls.run()

#     uc_cls.execution_engine.root_process.sos_disciplines[0].coupling_structure.graph.export_reduced_graph(
#         "reduced.pdf")
#     uc_cls.execution_engine.root_process.sos_disciplines[0].coupling_structure.graph.export_initial_graph(
#         "initial.pdf")
