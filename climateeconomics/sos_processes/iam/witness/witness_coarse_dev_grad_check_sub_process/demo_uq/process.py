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

from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS
from energy_models.glossaryenergy import GlossaryEnergy
from energy_models.sos_processes.energy.MDA.energy_process_v0.usecase import (
    INVEST_DISC_NAME,
)
from energy_models.sos_processes.witness_sub_process_builder import (
    WITNESSSubProcessBuilder,
)

from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_grad_check_sub_process.demo_uq.usecase import (
    Study,
)


class ProcessBuilder(WITNESSSubProcessBuilder):
    """Process builder for the UQ analysis on Witness coarse - Float invest."""

    def get_builders(self):
        coupling_name = "WITNESS_Eval"
        extra_name = "WITNESS"
        self.invest_discipline = INVEST_DISCIPLINE_OPTIONS[2]
        self.techno_dict = GlossaryEnergy.DEFAULT_COARSE_TECHNO_DICT

        chain_builders = self.ee.factory.get_builder_from_process(
            "climateeconomics.sos_processes.iam.witness",
            "witness",
            techno_dict=self.techno_dict,
            invest_discipline=self.invest_discipline,
            process_level=self.process_level,
            use_resources_bool=False,
        )

        # Add Profile Builder discipline
        mods_dict = {
            "InvestmentsProfileBuilderDisc": "energy_models.core.investments.disciplines.investments_profile_builder_disc.InvestmentsProfileBuilderDisc",
        }

        builder_invest_profile = self.create_builder_list(
            mods_dict, associate_namespace=False
        )
        chain_builders.extend(builder_invest_profile)

        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            extra_name, after_name=self.ee.study_name, clean_existing=True
        )
        self.ee.factory.update_builder_list_with_extra_name(
            extra_name, builder_list=chain_builders
        )
        self.ee.ns_manager.update_namespace_list_with_extra_ns(
            coupling_name, after_name=self.ee.study_name, clean_existing=True
        )

        # Add Indicators discipline
        chain_builders.append(
            self.ee.factory.get_builder_from_module(
                "Indicators",
                "climateeconomics.sos_wrapping.sos_wrapping_witness_coarse_for_sensitivity.witness_indicators.WitnessIndicators",
            )
        )

        ns_dict = {
            # GlossaryCore.NS_FUNCTIONS: f"{self.ee.study_name}.{coupling_name}.{extra_name}",
            #'ns_public': f'{self.ee.study_name}',
            # "ns_optim": f"{self.ee.study_name}",
            # GlossaryCore.NS_REFERENCE: f"{self.ee.study_name}.NormalizationReferences",
            "ns_invest": f"{self.ee.study_name}.{coupling_name}.{extra_name}.{INVEST_DISC_NAME}",
            # "ns_witness": f"{self.ee.study_name}.{coupling_name}.{extra_name}",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)

        # create coupling builder
        coupling_builder = self.ee.factory.create_builder_coupling(coupling_name)
        coupling_builder.set_builder_info("cls_builder", chain_builders)

        # DOE builder
        doe_builder = self.ee.factory.create_mono_instance_driver(
            "DOE", coupling_builder
        )

        # UQ builder
        uq_builder = self.ee.factory.add_uq_builder(Study.UQ_NAME)

        ns_dict = {
            "ns_sample_generator": f"{self.ee.study_name}.{Study.SAMPLE_GENERATOR_NAME}",
            "ns_evaluator": f"{self.ee.study_name}.DOE",
            "ns_uncertainty_quantification": f"{self.ee.study_name}.{Study.UQ_NAME}",
        }
        self.ee.ns_manager.add_ns_def(ns_dict)

        return [doe_builder, uq_builder]
