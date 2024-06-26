"""
Copyright 2022 Airbus SAS
Modifications on 2023/04/19-2023/11/03 Copyright 2023 Capgemini

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

from os.path import dirname, join

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.execution_engine import ExecutionEngine
from sostrades_core.tests.core.abstract_jacobian_unit_test import (
    AbstractJacobianUnittest,
)

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import (
    Study as witness_sub_proc_usecase,
)


class WitnessFullJacobianDiscTest(AbstractJacobianUnittest):
    obj_const = [
        GlossaryCore.WelfareObjective,
        "temperature_objective",
        "CO2_objective",
        "ppm_objective",
        "co2_emissions_objective",
        "EnergyMix.methane.demand_violation",
        "EnergyMix.hydrogen.gaseous_hydrogen.demand_violation",
        "EnergyMix.biogas.demand_violation",
        "EnergyMix.syngas.demand_violation",
        "EnergyMix.liquid_fuel.demand_violation",
        "EnergyMix.solid_fuel.demand_violation",
        "EnergyMix.biomass_dry.demand_violation",
        "EnergyMix.electricity.demand_violation",
        "EnergyMix.biodiesel.demand_violation",
        "EnergyMix.hydrogen.liquid_hydrogen.demand_violation",
        "primary_energies_production",
        "land_demand_constraint_df",
    ]

    def setUp(self):

        self.name = "Test"
        self.ee = ExecutionEngine(self.name)

    def analytic_grad_entry(self):
        return [self.test_01_constraints_wrt_design_var_bspline]

    def test_01_constraints_wrt_design_var_bspline(self):

        self.name = "Test"
        self.ee = ExecutionEngine(self.name)

        coupling_name = "WITNESS_Eval"
        designvariable_name = "DesignVariables"
        extra_name = "WITNESS"
        # retrieve energy process
        chain_builders = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness", "witness"
        )

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(extra_name, after_name=self.ee.study_name)
        self.ee.factory.update_builder_list_with_extra_name(extra_name, builder_list=chain_builders)

        # design variables builder
        design_var_path = (
            "climateeconomics.core.design_variables_translation.witness_bspline.design_var_disc.Design_Var_Discipline"
        )
        design_var_builder = self.ee.factory.get_builder_from_module(f"{designvariable_name}", design_var_path)
        chain_builders.append(design_var_builder)

        #         # function manager builder
        #         fmanager_path = 'sos_trades_core.execution_engine.func_manager.func_manager_disc.FunctionManagerDisc'
        #         fmanager_builder = self.ee.factory.get_builder_from_module(
        #             f'{func_manager_name}', fmanager_path)
        #         chain_builders.append(fmanager_builder)

        # modify namespaces defined in the child process
        self.ee.ns_manager.update_namespace_list_with_extra_ns(coupling_name, after_name=self.ee.study_name)

        ns_dict = {
            GlossaryCore.NS_FUNCTIONS: f"{self.ee.study_name}.{coupling_name}.{extra_name}",
            "ns_public": f"{self.ee.study_name}",
            "ns_optim": f"{self.ee.study_name}",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)

        # create coupling builder
        coupling_builder = self.ee.factory.create_builder_coupling(coupling_name)
        coupling_builder.set_builder_info("cls_builder", chain_builders)
        coupling_builder.set_builder_info("with_data_io", True)

        self.ee.factory.set_builders_to_coupling_builder(coupling_builder)
        self.ee.configure()

        usecase = witness_sub_proc_usecase(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        full_values_dict[f"{usecase.study_name}.WITNESS_Eval.linear_solver_MDO_options"] = {
            "tol": 1.0e-14,
            "max_iter": 50000,
        }
        full_values_dict[f"{usecase.study_name}.WITNESS_Eval.linear_solver_MDA_options"] = {
            "tol": 1.0e-14,
            "max_iter": 50000,
        }
        full_values_dict[f"{usecase.study_name}.WITNESS_Eval.linearization_mode"] = "adjoint"
        full_values_dict[f"{usecase.study_name}.WITNESS_Eval.tolerance"] = 1.0e-12
        full_values_dict[f"{usecase.study_name}.WITNESS_Eval.max_mda_iter"] = 200
        full_values_dict[f"{usecase.study_name}.WITNESS_Eval.sub_mda_class"] = "MDAGaussSeidel"

        input_full_names = []
        nb_poles = 8
        for energy in full_values_dict[f"{self.name}.WITNESS_Eval.WITNESS.{GlossaryCore.energy_list}"]:
            energy_wo_dot = energy.replace(".", "_")
            input_name = f"{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{energy_wo_dot}_array_mix"
            input_full_names.append(input_name)
            full_values_dict[input_name] = np.linspace(1, 2, nb_poles)

            for technology in full_values_dict[
                f"{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{GlossaryCore.techno_list}"
            ]:
                technology_wo_dot = technology.replace(".", "_")
                input_name = f"{self.name}.WITNESS_Eval.WITNESS.EnergyMix.{energy}.{technology}.{energy_wo_dot}_{technology_wo_dot}_array_mix"
                input_full_names.append(input_name)
                full_values_dict[input_name] = np.linspace(3, 4, nb_poles)
        full_values_dict.update(
            {f"{self.name}.design_space": pd.read_csv(join(dirname(__file__), "data/design_space_last_ite.csv"))}
        )
        self.ee.load_study_from_input_dict(full_values_dict)

        disc = self.ee.root_process.proxy_disciplines[0]
        namespace = "Test.WITNESS_Eval.WITNESS"
        output_full_names = [f"{namespace}.CO2_tax_minus_CO2_damage_constraint_df"]
        self.ee.display_treeview_nodes(display_variables=True)

        self.check_jacobian(
            location=dirname(__file__),
            filename="jacobian_constraint_wrt_design_var_bspline.pkl",
            discipline=disc,
            inputs=input_full_names,
            outputs=output_full_names,
            derr_approx="complex_step",
            step=1.0e-15,
            local_data={},
            parallel=True,
        )


if "__main__" == __name__:
    cls = WitnessFullJacobianDiscTest()
    cls.test_01_constraints_wrt_design_var_bspline()
