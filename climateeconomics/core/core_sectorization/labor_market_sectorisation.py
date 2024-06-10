"""
Copyright 2022 Airbus SAS
Modifications on 2023/09/06-2023/11/03 Copyright 2023 Capgemini

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

import numpy as np
import pandas as pd

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.sos_wrapping.sos_wrapping_sectors.agriculture.agriculture_discipline import (
    AgricultureDiscipline,
)
from climateeconomics.sos_wrapping.sos_wrapping_sectors.industrial.industrial_discipline import (
    IndustrialDiscipline,
)
from climateeconomics.sos_wrapping.sos_wrapping_sectors.services.services_discipline import (
    ServicesDiscipline,
)


class LaborMarketModel:
    """
    Labor Market pyworld3: to compute workforce per sector
    """

    # Units conversion
    conversion_factor = 1.0
    SECTORS_DISC_LIST = [AgricultureDiscipline, ServicesDiscipline, IndustrialDiscipline]
    SECTORS_LIST = [disc.sector_name for disc in SECTORS_DISC_LIST]
    SECTORS_OUT_UNIT = {disc.sector_name: disc.prod_cap_unit for disc in SECTORS_DISC_LIST}

    def __init__(self, inputs_dict):
        """
        Constructor
        """
        self.labor_market_df = None
        self.employment_df = None
        self.configure_parameters(inputs_dict)

    def configure_parameters(self, inputs_dict):
        """
        Configure with inputs_dict from the discipline
        """

        self.year_start = inputs_dict[GlossaryCore.YearStart]  # year start
        self.year_end = inputs_dict[GlossaryCore.YearEnd]  # year end
        self.time_step = inputs_dict[GlossaryCore.TimeStep]
        self.years_range = np.arange(self.year_start, self.year_end + 1, self.time_step)
        self.nb_years = len(self.years_range)
        self.employment_a_param = inputs_dict["employment_a_param"]
        self.employment_power_param = inputs_dict["employment_power_param"]
        self.employment_rate_base_value = inputs_dict["employment_rate_base_value"]
        #         unemployment_rate = inputs_dict['unemployment_rate']
        #         self.unemployment_rate = unemployment_rate['unemployment_rate'].values
        #         self.labor_participation_rate = inputs_dict['labor_participation_rate']
        self.workforce_share_per_sector = inputs_dict["workforce_share_per_sector"]

    def set_coupling_inputs(self, inputs):
        self.working_age_population_df = inputs[GlossaryCore.WorkingAgePopulationDfValue]
        self.working_age_population_df.index = self.working_age_population_df[GlossaryCore.Years].values

    #     def create_dataframes(self):
    #         '''
    #         Create dataframes with years
    #         '''
    #         labor_market_df = pd.DataFrame({GlossaryCore.Years: self.years_range, GlossaryCore.Workforce: self.workforce, 'employed': self.employed})
    #         labor_market_df.index = self.years_range
    #         self.labor_market_df = labor_market_df

    #     def compute_employed(self):
    #         """
    #         Compute the number of employed people (Million people) using as input:
    #         - unemployment rate per year
    #         - labor force participation rate per year
    #         - working age population per year in million of people
    #         employed = working_age_pop * participation_rate *(1-unemployment_rate)
    #         """
    #         working_age_pop = self.working_age_population_df[GlossaryCore.Population1570].values
    #         laborforce_participation_rate = self.labor_participation_rate['participation_rate'].values
    #         unemployment_rate = self.unemployment_rate['unemployment_rate'].values
    #         #compute workforce
    #         workforce = working_age_pop * laborforce_participation_rate/100
    #         #Compute employed: remove unemployed people
    #         employed = workforce* (1- unemployment_rate/100)
    #
    #         return workforce, employed

    #     def compute_employed_persector(self, sector_list):
    #         """ Compute the number of people employed in each sector in million of people
    #         inputs: - share per sector (dataframe with one column per sector)
    #         - number of total employed people (million)
    #         for a sector: employed = share_sector * employed
    #         """
    #         share_per_sector = self.workforce_share_per_sector
    #         employed_df = pd.DataFrame({GlossaryCore.Years: self.years_range, 'employed': sector_employed})
    #         for sector in sector_list:
    #             share_sector = share_per_sector[sector].values
    #             sector_employed =  self.employed * share_sector/100
    #
    #         employed_df.index = self.years_range
    #         return

    def compute_employment_rate(self):
        """
        Compute the employment rate. based on prediction from ILO
        We pyworld3 a recovery from 2020 crisis until 2031 where past level is reached
        For all year not in (2020,2031), value = employment_rate_base_value
        """
        year_covid = 2020
        year_end_recovery = 2031
        # create dataframe
        employment_df = pd.DataFrame(index=self.years_range, columns=[GlossaryCore.Years, GlossaryCore.EmploymentRate])
        employment_df[GlossaryCore.Years] = self.years_range
        # For all years employment_rate = base value
        employment_df[GlossaryCore.EmploymentRate] = self.employment_rate_base_value
        # Compute recovery phase
        years_recovery = np.arange(year_covid, year_end_recovery + 1)
        x_recovery = years_recovery + 1 - year_covid
        employment_rate_recovery = self.employment_a_param * x_recovery**self.employment_power_param
        employment_rate_recovery_df = pd.DataFrame(
            {GlossaryCore.Years: years_recovery, GlossaryCore.EmploymentRate: employment_rate_recovery}
        )
        employment_rate_recovery_df.index = years_recovery
        # Then replace values in original dataframe by recoveries values
        employment_df.update(employment_rate_recovery_df)

        self.employment_df = employment_df
        return employment_df

    def compute_workforce_persector(self):
        """Compute workforce per sector.
        Inputs: - dataframe of share of workforce per sector per year
                - working age population (million) per year
                - dataframe employment rate per year
        output: dataframe with workforce per sector in million per year. 1 column per sector
        """
        workforce_df = self.workforce_share_per_sector.copy(deep=True)
        # drop years for computation
        workforce_df = workforce_df.drop(columns=[GlossaryCore.Years])
        working_age_pop = self.working_age_population_df[GlossaryCore.Population1570].values
        employment_rate = self.employment_df[GlossaryCore.EmploymentRate].values
        # per sector the workforce = share_per_sector * employment_rate *workingagepop
        workforce_df = workforce_df.apply(lambda x: x / 100 * employment_rate * working_age_pop)
        # workforce total is the sum of all sectors
        workforce_df[GlossaryCore.Workforce] = workforce_df.sum(axis=1)
        workforce_df.insert(0, GlossaryCore.Years, self.years_range)
        self.workforce_df = workforce_df
        share = self.workforce_share_per_sector[GlossaryCore.SectorAgriculture].values

        return workforce_df

    def compute_workforce_persector_bis(self):
        """Compute workforce per sector.
        Inputs: - dataframe of share of workforce per sector per year
                - working age population (million) per year
                - dataframe employment rate per year
        output: dataframe with workforce per sector in million per year. 1 column per sector
        """
        workforce_df_dict = {}
        # drop years for computation
        working_age_pop = self.working_age_population_df[GlossaryCore.Population1570].values
        employment_rate = self.employment_df[GlossaryCore.EmploymentRate].values
        sector_list = self.SECTORS_LIST
        workforce_share = self.workforce_share_per_sector
        for sector in sector_list:
            share = workforce_share[sector] / 100
            sector_wf = share * employment_rate * working_age_pop
            workforce_df_dict[sector] = sector_wf
        workforce_df = pd.DataFrame.from_dict(workforce_df_dict)
        # workforce total is the sum of all sectors
        workforce_df[GlossaryCore.Workforce] = workforce_df.sum(axis=1)
        workforce_df.insert(0, GlossaryCore.Years, self.years_range)
        self.workforce_df = workforce_df

        return workforce_df

    # RUN
    def compute(self, inputs):
        """
        Compute all models for year range
        """
        self.inputs = inputs
        self.set_coupling_inputs(inputs)
        self.compute_employment_rate()
        self.compute_workforce_persector_bis()

        return self.workforce_df, self.employment_df

    ### GRADIENTS ###
    def compute_dworkforcetotal_dworkagepop(self):
        """Gradient for workforce wrt working age population"""
        nb_years = self.nb_years
        employment_rate = self.employment_df[GlossaryCore.EmploymentRate].values
        dworkforce_dworkagepop = np.identity(nb_years) * employment_rate

        return dworkforce_dworkagepop

    def compute_dworkforcesector_dworkagepop(self, sector):
        sector_share = self.workforce_share_per_sector[sector].values
        employment_rate = self.employment_df[GlossaryCore.EmploymentRate].values
        # workforce sector = employmentrate * working age pop * share
        dworkforcesector_dworkagepop = np.identity(self.nb_years) * employment_rate * sector_share / 100

        return dworkforcesector_dworkagepop
