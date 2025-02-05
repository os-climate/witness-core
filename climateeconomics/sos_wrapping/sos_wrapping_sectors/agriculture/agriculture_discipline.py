'''
Copyright 2024 Capgemini
Modifications on 2023/06/14-2023/11/03 Copyright 2023 Capgemini

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
import logging
from copy import copy, deepcopy

from sostrades_optimization_plugins.models.autodifferentiated_discipline import (
    AutodifferentiedDisc,
)

from climateeconomics.core.core_sectorization.agriculture_model import AgricultureModel
from climateeconomics.glossarycore import GlossaryCore


class AgricultureSectorDiscipline(AutodifferentiedDisc):
    """Agriculture sector for witness sectorized version"""
    _ontology_data = {
        'label': 'Agriculture sector model for WITNESS Sectorized version',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': 'compute food production',
        'icon': 'fas fa-seedling fa-fw',
        'version': '',
    }

    sector_name = AgricultureModel.name
    prod_cap_unit = "G$"

    DESC_IN = {
        GlossaryCore.YearStart: {'type': 'int', 'default': GlossaryCore.YearStartDefault, 'structuring': True,'unit': GlossaryCore.Years, 'visibility': 'Shared', 'namespace': 'ns_public', 'range': [1950, 2080]},
        GlossaryCore.YearEnd: GlossaryCore.get_dynamic_variable(GlossaryCore.YearEndVar),
        "mdo_sectors_invest_level": GlossaryCore.MDOSectorsLevel,

        # Emissions inputs
        GlossaryCore.FoodEmissionsName: GlossaryCore.FoodEmissionsVar,
        GlossaryCore.CropEnergyEmissionsName: GlossaryCore.CropEnergyEmissionsVar,

        f'{GlossaryCore.Forestry}.CO2_land_emission_df': {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': "Shared", 'namespace': GlossaryCore.NS_AGRI, AutodifferentiedDisc.GRADIENTS: True},

    }

    for sub_sector in AgricultureModel.sub_sectors:
        for commun_variable_name, commun_variable_descr, _ in AgricultureModel.sub_sector_commun_variables:
            DESC_IN.update({
                f"{sub_sector}.{commun_variable_name}": GlossaryCore.get_subsector_variable(
                    subsector_name=sub_sector, sector_namespace=GlossaryCore.NS_AGRI, var_descr=commun_variable_descr),
            })

    DESC_OUT = {
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CO2): {'type': 'dataframe', 'unit': 'GtCO2', 'visibility': "Shared", 'namespace': GlossaryCore.NS_WITNESS,
                                                                           'dataframe_descriptor': {
                                                                               GlossaryCore.Years: ('float', None, False),
                                                                               GlossaryCore.Forestry: ('float', None, False),
                                                                               'Crop': ('float', None, False)}},
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.CH4): {'type': 'dataframe', 'unit': 'GtCH4', 'visibility': "Shared", 'namespace': GlossaryCore.NS_WITNESS,
                                                                           'dataframe_descriptor': {
                                                                               GlossaryCore.Years: ('float', None, False),
                                                                               'Crop': ('float', None, False), }},
        GlossaryCore.insertGHGAgriLandEmissions.format(GlossaryCore.N2O): {'type': 'dataframe', 'unit': 'GtN2O', 'visibility': "Shared", 'namespace': GlossaryCore.NS_WITNESS,
                                                                           'dataframe_descriptor': {
                                                                               GlossaryCore.Years: ('float', None, False),
                                                                               },}
    }
    for commun_variable_name, commun_variable_descr, _ in AgricultureModel.sub_sector_commun_variables:
        var_descr = deepcopy(commun_variable_descr)
        var_descr["namespace"] = GlossaryCore.NS_SECTORS
        DESC_OUT.update({f"{sector_name}.{commun_variable_name}": var_descr})

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.model = AgricultureModel()

    def add_additionnal_dynamic_variables(self):
        return {}, {}
    def setup_sos_disciplines(self):  # type: (...) -> None
        dynamic_inputs = {}
        dynamic_outputs = {}

        damage_detailed = GlossaryCore.get_dynamic_variable(GlossaryCore.DamageDetailedDf)
        damage_detailed.update({self.NAMESPACE: GlossaryCore.NS_SECTORS})
        dynamic_outputs[f"{self.sector_name}.{GlossaryCore.DamageDetailedDfValue}"] = damage_detailed

        if "mdo_sectors_invest_level" in self.get_data_in():
            mdo_sectors_invest_level = self.get_sosdisc_inputs("mdo_sectors_invest_level")
            if mdo_sectors_invest_level is not None:
                if mdo_sectors_invest_level == 0:
                    # then we compute invest in subs sector as sub-sector-invest = Net outpput * Share invest sector * Share invest sub sector * Shares invests inside Sub sector
                    # and we go bottom-up until Macroeconomics
                    for sub_sector in AgricultureModel.sub_sectors:
                        dynamic_inputs[f'{self.sector_name}.{sub_sector}.{GlossaryCore.InvestmentDfValue}'] =\
                            GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
                    dynamic_outputs[f'{self.sector_name}.{GlossaryCore.InvestmentDfValue}'] = GlossaryCore.get_dynamic_variable(GlossaryCore.InvestmentDf)
                elif mdo_sectors_invest_level == 2:
                    # then invests in subsectors are in G$
                    dynamic_inputs[f'{self.sector_name}.{self.subsector_name}.{GlossaryCore.InvestmentDetailsDfValue}'] = GlossaryCore.get_dynamic_variable(
                        GlossaryCore.SubSectorInvestDf)
                else:
                    raise NotImplementedError('')
        di, do = self.add_additionnal_dynamic_variables()
        di.update(dynamic_inputs)
        do.update(dynamic_outputs)
        self.add_inputs(di)
        self.add_outputs(do)

