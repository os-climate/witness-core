
# -*- coding: utf-8 -*-
"""
Created on 03/03/2020

@author: NG878B4
"""
# mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8
# Copyright (C) 2020 Airbus SAS.
# All rights reserved.

from climateeconomics.sos_processes.iam.witness.witness_optim_sub_process.usecase_witness_optim_sub import OPTIM_NAME
from energy_models.core.energy_study_manager import DEFAULT_COARSE_TECHNO_DICT_ccs_3
from energy_models.sos_processes.witness_sub_process_builder import WITNESSSubProcessBuilder
from energy_models.core.energy_process_builder import INVEST_DISCIPLINE_OPTIONS


class ProcessBuilder(WITNESSSubProcessBuilder):

    def get_builders(self):

        optim_name = OPTIM_NAME

        # if one invest discipline then we need to setup all subprocesses
        # before get them
        techno_dict = DEFAULT_COARSE_TECHNO_DICT_ccs_3

        coupling_builder = self.ee.factory.get_builder_from_process(
            'climateeconomics.sos_processes.iam.witness', 'witness_optim_sub_process',
            techno_dict=techno_dict, invest_discipline=INVEST_DISCIPLINE_OPTIONS[2])

        # modify namespaces defined in the child process
        for ns in self.ee.ns_manager.ns_list:
            self.ee.ns_manager.update_namespace_with_extra_ns(
                ns, optim_name, after_name=self.ee.study_name)  # optim_name

        #-- set optim builder
        opt_builder = self.ee.factory.create_optim_builder(
            optim_name, [coupling_builder])

        self.ee.post_processing_manager.add_post_processing_module_to_namespace(
            'ns_optim',
            'climateeconomics.sos_wrapping.sos_wrapping_witness.post_proc_witness_optim.post_processing_witness_full')

        return opt_builder
