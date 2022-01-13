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
from sos_trades_core.tools.cst_manager.func_manager_common import smooth_maximum,\
    get_dsmooth_dvariable


class PolicyModel():
    '''
    Used to compute carbon emissions from gross output 
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.CO2_tax = pd.DataFrame()
        self.CO2_damage_price = None
        self.CCS_price = None

    def compute_smax(self, param):
        """
        Compute CO2 tax based on ccs_price and co2_damage_price
        """
        self.CO2_damage_price = param['CO2_damage_price']
        self.CCS_price = param['CCS_price']
        self.CO2_tax['years'] = self.CO2_damage_price['years']
        CO2_damage_price_array = self.CO2_damage_price['CO2_damage_price'].values
        CCS_price_array = self.CCS_price['ccs_price_per_tCO2'].values
        l = []

        for elem in range(0, len(CO2_damage_price_array)):
            l.append(smooth_maximum(
                np.array([CO2_damage_price_array[elem], CCS_price_array[elem], 0.0])))

        self.CO2_tax['CO2_tax'] = l

    def compute_CO2_tax_dCCS_dCO2_damage_smooth(self):
        """
        compute dCO2_tax/dCO2_damage and dCO2_tax/dCCS_price
        """
        self.CO2_tax['years'] = self.CO2_damage_price['years']
        CO2_damage_price_array = self.CO2_damage_price['CO2_damage_price'].values
        CCS_price_array = self.CCS_price['ccs_price_per_tCO2'].values
        l_CO2 = []
        l_CCS = []

        for elem in range(0, len(CO2_damage_price_array)):
            dsmooth = get_dsmooth_dvariable(
                np.array([CO2_damage_price_array[elem], CCS_price_array[elem], 0.0]))
            l_CO2.append(dsmooth[0])
            l_CCS.append(dsmooth[1])

        return l_CO2, l_CCS
