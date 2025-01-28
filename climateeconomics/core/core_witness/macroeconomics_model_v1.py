'''
Copyright 2022 Airbus SAS
Modifications on 2023/03/30-2024/01/08 Copyright 2023 Capgemini

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
from sostrades_optimization_plugins.tools.cst_manager.func_manager_common import (
    d_pseudo_abs_obj,
    pseudo_abs_obj,
)

from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class MacroEconomics:
    """
    Economic pyworld3 that compute the evolution of capital, consumption, output...
    """
    def __init__(self, param):
        """
        Constructor
        """
        self.usable_capital_obj_content = None
        self.usable_capital_upper_bound_constraint = None
        self.energy_consumption_households_df = None
        self.consommation_objective_ref = None
        self.year_start: int = 0
        self.year_end = None
        self.productivity_start = None
        self.capital_start_ne = None
        self.population_df = None
        self.productivity_gr_start: float = 0
        self.decline_rate_tfp: float = 0.
        self.depreciation_capital = None
        self.init_rate_time_pref = None
        self.conso_elasticity = None
        self.nb_per = None
        self.years_range = None
        self.nb_years = None
        self.frac_damage_prod = None
        self.init_output_growth = None
        self.output_alpha = None
        self.output_gamma = None
        self.energy_eff_k: float = 0.
        self.energy_eff_cst: float = 0.
        self.energy_eff_xzero: float = 0.
        self.energy_eff_max: float = 0.
        self.capital_utilisation_ratio = None
        self.max_capital_utilisation_ratio = None
        self.employment_a_param = None
        self.employment_power_param = None
        self.employment_rate_base_value = None
        self.usable_capital_ref = None
        self.compute_climate_impact_on_gdp = None
        self.damage_to_productivity = None
        self.sector_list = None
        self.section_list = None
        self.workforce_df = None
        self.damage_fraction_output_df = None
        self.energy_production = None
        self.energy_investment = None
        self.share_non_energy_investment = None
        self.energy_capital = None
        self.population_df = None
        self.working_age_population_df = None
        self.compute_gdp = None
        self.gross_output_in = None
        self.workforce_df = None
        self.usable_capital_lower_bound_constraint = None
        self.economics_detail_df = None
        self.param = param
        self.economics_df = None
        self.damage_df = None
        self.capital_df = None
        self.consommation_objective = None
        self.gdp_percentage_per_section_df = None
        self.sector_gdp_df = None
        self.section_gdp_df = None
        self.dict_sectors_gdp_detailed = None
        self.usable_capital_objective = None
        self.usable_capital_objective_ref = None

        # In the V1 model, we apply the two factors of the COVID pandemic
        # (disability and mortality) to employment_rate and population death_rate individually.
        # In a future study, we could model pandemic as part of an overall health discipline.
        self.pandemic_disability_df = None
        self.set_data()
        self.create_dataframe()
        self.total_gdp_per_group_df = None
        self.percentage_gdp_per_group_df = None
        self.sector_energy_consumption_percentage_df = None
        self.df_gdp_per_country = None
        self.dict_dataframe_energy_consumption_sections = None
        self.dict_energy_consumption_detailed = None

    def set_data(self):

        self.year_start = self.param[GlossaryCore.YearStart]
        self.year_end = self.param[GlossaryCore.YearEnd]
        self.activate_pandemic_effects = self.param['assumptions_dict']['activate_pandemic_effects']
        self.productivity_start = self.param['productivity_start']
        self.capital_start_ne = self.param['capital_start_non_energy']
        self.population_df = self.param[GlossaryCore.PopulationDfValue]
        self.productivity_gr_start = self.param['productivity_gr_start']
        self.decline_rate_tfp = self.param['decline_rate_tfp']
        self.depreciation_capital = self.param['depreciation_capital']
        self.init_rate_time_pref = self.param['init_rate_time_pref']
        self.conso_elasticity = self.param['conso_elasticity']
        self.nb_per = round(
            (self.param[GlossaryCore.YearEnd] -
             self.param[GlossaryCore.YearStart]) +
            1)
        self.years_range = np.arange(
            self.year_start,
            self.year_end + 1)
        self.nb_years = len(self.years_range)
        self.frac_damage_prod = self.param[GlossaryCore.FractionDamageToProductivityValue]
        self.damage_to_productivity = self.param[GlossaryCore.DamageToProductivity]
        self.init_output_growth = self.param['init_output_growth']
        self.output_alpha = self.param['output_alpha']
        self.output_gamma = self.param['output_gamma']
        self.energy_eff_k = self.param['energy_eff_k']
        self.energy_eff_cst = self.param['energy_eff_cst']
        self.energy_eff_xzero = self.param['energy_eff_xzero']
        self.energy_eff_max = self.param['energy_eff_max']
        self.capital_utilisation_ratio = self.param['capital_utilisation_ratio']
        self.max_capital_utilisation_ratio = self.param['max_capital_utilisation_ratio']
        df = self.param[GlossaryCore.PandemicParamDfValue]['disability']
        df.index = self.param[GlossaryCore.PandemicParamDfValue]['param'].values
        self.pandemic_disability_df = pd.DataFrame.from_dict(
            {
                str(year): df.loc[year_range]
                for year_range in df.index[:-1]
                for year_start, year_end in [year_range.split('-')]
                for year in range(int(year_start), int(year_end)+1)
            },
            orient='index',
            columns=[df.name],
        )

        # Employment rate param
        self.employment_a_param = self.param['employment_a_param']
        self.employment_power_param = self.param['employment_power_param']
        self.employment_rate_base_value = self.param['employment_rate_base_value']

        self.usable_capital_ref = self.param['usable_capital_ref']
        self.compute_climate_impact_on_gdp = self.param['assumptions_dict']['compute_climate_impact_on_gdp']
        if not self.compute_climate_impact_on_gdp:
            self.damage_to_productivity = False
        self.sector_list = self.param[GlossaryCore.SectorListValue]
        self.section_list = self.param[GlossaryCore.SectionListValue]
        self.usable_capital_objective_ref = self.param[GlossaryCore.UsableCapitalObjectiveRefName]
        self.consommation_objective_ref = self.param[GlossaryCore.ConsumptionObjectiveRefValue]

    def create_dataframe(self):
        """Create the dataframe and fill it with values at year_start"""
        self.economics_df = pd.DataFrame({GlossaryCore.Years: self.years_range})
        self.workforce_df = pd.DataFrame({GlossaryCore.Years: self.years_range})
        self.capital_df = pd.DataFrame({GlossaryCore.Years: self.years_range})
        self.damage_df = pd.DataFrame({GlossaryCore.Years: self.years_range})

        self.total_gdp_per_group_df = pd.DataFrame()
        self.percentage_gdp_per_group_df = pd.DataFrame()
        self.df_gdp_per_country = pd.DataFrame(columns=[GlossaryCore.CountryName, GlossaryCore.Years, GlossaryCore.GDPName, GlossaryCore.GroupName])
        self.dict_energy_consumption_detailed = {}

    def set_coupling_inputs(self, inputs: dict):
        """
        Set couplings inputs with right index, scaling... 
        """
        self.damage_fraction_output_df = inputs[GlossaryCore.DamageFractionDfValue]
        self.damage_fraction_output_df.index = self.damage_fraction_output_df[GlossaryCore.Years].values
        # Scale energy production
        self.energy_production = inputs[GlossaryCore.StreamProductionValue]
        self.energy_production.index = self.energy_production[GlossaryCore.Years].values
        #Investment in energy
        self.energy_investment = inputs[GlossaryCore.EnergyInvestmentsWoTaxValue]
        self.share_non_energy_investment = inputs[GlossaryCore.ShareNonEnergyInvestmentsValue]
        self.energy_capital = inputs[GlossaryCore.EnergyCapitalDfValue]
        self.energy_capital.index = self.energy_capital[GlossaryCore.Years].values
        # Population dataframes
        self.population_df = inputs[GlossaryCore.PopulationDfValue]
        self.population_df.index = self.population_df[GlossaryCore.Years].values
        self.working_age_population_df = inputs[GlossaryCore.WorkingAgePopulationDfValue]
        self.working_age_population_df.index = self.working_age_population_df[GlossaryCore.Years].values
        self.compute_gdp = inputs['assumptions_dict']['compute_gdp']
        self.gdp_percentage_per_section_df = inputs[GlossaryCore.SectionGdpPercentageDfValue]
        if not self.compute_gdp:
            self.gross_output_in = inputs['gross_output_in']
        self.sector_energy_consumption_percentage_df = inputs[GlossaryCore.SectorEnergyConsumptionPercentageDfName]
        #self.carbon_intensity_df = inputs[GlossaryCore.EnergyCarbonIntensityDfValue]
        # create dictionary where key is sector and value is the energy consumption percebtage for each section per sector
        self.dict_dataframe_energy_consumption_sections = dict(zip(self.sector_list, [inputs[f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_{sector}']
                                                                                       for sector in self.sector_list]))
        self.dict_dataframe_energy_consumption_sections = {key: val.loc[val[GlossaryCore.Years] >= self.year_start] for key, val in self.dict_dataframe_energy_consumption_sections.items()}

    def compute_employment_rate(self):
        """ 
        Compute the employment rate. based on prediction from ILO 
        We pyworld3 a recovery from 2020 crisis until 2031 where past level is reached
        For all year not in (2020,2031), value = employment_rate_base_value
        If `activate_pandemic_effects` is True, additionally reduce by pandemic_disability factor
        """
        year_covid = 2020
        year_end_recovery = 2031
        workforce_df = self.workforce_df
        # For all years employment_rate = base value
        workforce_df[GlossaryCore.EmploymentRate] = self.employment_rate_base_value
        # Compute recovery phase
        years_recovery = np.arange(year_covid, year_end_recovery + 1)
        x_recovery = years_recovery + 1 - year_covid
        employment_rate_recovery = self.employment_a_param * \
            x_recovery**self.employment_power_param
        employment_rate_recovery_df = pd.DataFrame(
            {GlossaryCore.Years: years_recovery, GlossaryCore.EmploymentRate: employment_rate_recovery})
        employment_rate_recovery_df.index = years_recovery
        if self.activate_pandemic_effects:
            employment_rate_recovery_df = employment_rate_recovery_df.mul(self.pandemic_disability_df.rsub(1.0), axis=1)
        # Then replace values in original dataframe by recoveries values
        workforce_df.update(employment_rate_recovery_df)

        self.workforce_df = workforce_df
        return workforce_df

    def compute_workforce(self):
        """ Compute the workforce based on formula: 
        workforce = people in working age * employment_rate 
        inputs : - number of people in working age 
                - employment rate in %
        Output: number of working people in million of people
        """
        self.workforce_df[GlossaryCore.Workforce] = self.workforce_df[GlossaryCore.EmploymentRate].values * \
                                                    self.working_age_population_df[GlossaryCore.Population1570].values

    def compute_productivity_growthrate(self):
        """
        A_g, Growth rate of total factor productivity.
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        """
        prod_growth_rate = self.productivity_gr_start * np.exp(- self.decline_rate_tfp * (self.years_range - self.year_start))
        self.economics_df[GlossaryCore.ProductivityGrowthRate] = prod_growth_rate

    def compute_productivity(self):
        """
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        """
        prod_wo_d = self.productivity_start
        productivity_wo_damage_list = [self.productivity_start]

        damage_fraction_output = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        productivity_growth = self.economics_df[GlossaryCore.ProductivityGrowthRate].values

        for prod_growth, damage_frac_year in zip(productivity_growth[:-1], damage_fraction_output[1:]):
            prod_wo_d = prod_wo_d / (1 - prod_growth / 5)
            productivity_wo_damage_list.append(prod_wo_d)

        productivity_w_damage_list = np.array(productivity_wo_damage_list) * (1 - damage_fraction_output)

        self.economics_df[GlossaryCore.ProductivityWithDamage] = productivity_w_damage_list
        self.economics_df[GlossaryCore.ProductivityWithoutDamage] = productivity_wo_damage_list
        if self.damage_to_productivity:
            self.economics_df[GlossaryCore.Productivity] = productivity_w_damage_list
        else:
            self.economics_df[GlossaryCore.Productivity] = productivity_wo_damage_list

    def compute_capital(self):
        """
        Capital non energy (t) = (1 - depreciation) * Capital non energy (t-1) + Invests non energy(t-1)
        Capital (t) = Capital non energy (t) + Capital energy (t)
        """
        capital_energy = self.energy_capital[GlossaryCore.Capital].values
        capital_non_energy = self.capital_start_ne
        capital_non_energy_list = [capital_non_energy]

        non_energy_invests = self.economics_df[GlossaryCore.NonEnergyInvestmentsValue].values
        for non_energy_invest in non_energy_invests[:-1]:
            capital_non_energy = (1 - self.depreciation_capital) * capital_non_energy + non_energy_invest
            capital_non_energy_list.append(capital_non_energy)

        capital_non_energy = np.array(capital_non_energy_list)

        self.capital_df[GlossaryCore.NonEnergyCapital] = capital_non_energy
        self.capital_df[GlossaryCore.Capital] = capital_non_energy + capital_energy

    def compute_energy_efficiency(self):
        """compute energy_efficiency"""
        years = self.capital_df[GlossaryCore.Years].values
        energy_efficiency = self.energy_eff_cst + self.energy_eff_max / (1 + np.exp(-self.energy_eff_k *
                                                                                    (years - self.energy_eff_xzero)))
        self.capital_df[GlossaryCore.EnergyEfficiency] = energy_efficiency

    def compute_usable_capital(self):
        """compute usable capital = Energy Production Net * capital utilisation ratio * energy efficiency"""
        net_energy_production = self.energy_production[GlossaryCore.TotalProductionValue].values
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        usable_capital = self.capital_utilisation_ratio * net_energy_production * energy_efficiency
        self.capital_df[GlossaryCore.UsableCapital] = usable_capital

    def compute_investment(self):
        """
        Compute I(t) (total Investment) and Ine(t) (Investment in non-energy sectors) in trillions $USD
        Investment Non energy (t) = Share Non Energy investment (t) * Output net of damage (t)
        Investments energy (t) = input energy investments(t)

        Investements (t) = Investments energy (t) + Investments non energy (t)
        """
        self.economics_df[GlossaryCore.EnergyInvestmentsValue] = self.energy_investment[GlossaryCore.EnergyInvestmentsWoTaxValue].values  # T$
        net_output = self.economics_df[GlossaryCore.OutputNetOfDamage].values
        percent_invest_non_energy = self.share_non_energy_investment[GlossaryCore.ShareNonEnergyInvestmentsValue].values
        self.economics_df[GlossaryCore.NonEnergyInvestmentsValue] = percent_invest_non_energy * net_output / 100.
        self.economics_df[GlossaryCore.InvestmentsValue] = self.economics_df[GlossaryCore.EnergyInvestmentsValue].values + \
                                                           self.economics_df[GlossaryCore.NonEnergyInvestmentsValue].values

    def compute_gross_output(self):
        """ Compute the gdp 
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1) 
        output: gdp in trillion dollars
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.economics_df[GlossaryCore.Productivity].values
        working_pop = self.workforce_df[GlossaryCore.Workforce].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        gross_output = productivity * (alpha * capital_u**gamma + (1 - alpha) * working_pop**gamma) ** (1 / gamma)
        self.economics_df[GlossaryCore.GrossOutput] = gross_output


    def set_gross_output(self): 
        """
        Set gross output according to input
        """
        self.economics_df = self.economics_df.merge(self.gross_output_in[[GlossaryCore.Years, GlossaryCore.GrossOutput]],
                                                    on=GlossaryCore.Years, how='left').set_index(self.economics_df.index)

    def get_gdp_percentage_per_section(self):
        '''
        Get default values for gdp percentage per sector from gdp_percentage_per_sector.csv file
        '''
        # the year range for the study can differ from that stated in the csv file
        start_year_csv = self.gdp_percentage_per_section_df[GlossaryCore.Years].values[0]
        if start_year_csv > self.year_start:
            # duplicate first row (start_year_csv - year_start) time
            list_df_to_concat = [self.gdp_percentage_per_section_df.iloc[0:1]] * (start_year_csv - self.year_start)
            # add input dataframe to the list
            list_df_to_concat.append(self.gdp_percentage_per_section_df)
            # concatenate the dataframes using the created list to fill the missing rows
            self.gdp_percentage_per_section_df = pd.concat(list_df_to_concat).reset_index(drop=True)
            # set years of the updated dataframe
            self.gdp_percentage_per_section_df.iloc[0:(start_year_csv - self.year_start)][
                GlossaryCore.Years] = np.arange(self.year_start, start_year_csv)

        elif start_year_csv < self.year_start:
            self.gdp_percentage_per_section_df = self.gdp_percentage_per_section_df[
                self.gdp_percentage_per_section_df[GlossaryCore.Years] > self.year_start - 1]

        end_year_csv = self.gdp_percentage_per_section_df.loc[
            self.gdp_percentage_per_section_df.index[-1], GlossaryCore.Years]

        if end_year_csv > self.year_end:
            self.gdp_percentage_per_section_df = self.gdp_percentage_per_section_df[
                self.gdp_percentage_per_section_df[GlossaryCore.Years] < self.year_end + 1]

        elif end_year_csv < self.year_end:
            list_df_to_concat = [self.gdp_percentage_per_section_df]
            list_df_to_concat.extend([self.gdp_percentage_per_section_df.iloc[-1:]] * (
                    self.year_end - end_year_csv))
            self.gdp_percentage_per_section_df = pd.concat(list_df_to_concat).reset_index(drop=True)
            # fill years with mising years (start at end_year_csv+1, and last element should be year_end)
            self.gdp_percentage_per_section_df.iloc[-(self.year_end - end_year_csv):][GlossaryCore.Years] = np.arange(
                end_year_csv + 1, self.year_end + 1)

    def compute_section_gdp(self):
        """
        Computes the GDP net of damage per section
        """
        # get gdp percentage per section, and compute gdp per section using Net output of damage
        self.get_gdp_percentage_per_section()

        self.section_gdp_df = pd.DataFrame({GlossaryCore.Years: self.years_range})
        for section in self.section_list:
            self.section_gdp_df[section] = self.gdp_percentage_per_section_df[section].values *\
                                           self.economics_df[GlossaryCore.OutputNetOfDamage].values / 100.

    def compute_sector_gdp(self):
        """
        Compute gdp per sector based on gdp per section
        """
        # prepare dictionary with values for each section per sector
        dict_sectors_sections = {
            GlossaryCore.SectorServices: pd.DataFrame({GlossaryCore.Years: self.years_range, **{section: self.section_gdp_df[section].values for section in GlossaryCore.SectionsServices},}),
            GlossaryCore.SectorIndustry: pd.DataFrame({GlossaryCore.Years: self.years_range, **{section: self.section_gdp_df[section].values for section in GlossaryCore.SectionsIndustry},}),
            GlossaryCore.SectorAgriculture: pd.DataFrame({GlossaryCore.Years: self.years_range, **{section: self.section_gdp_df[section].values for section in GlossaryCore.SectionsAgriculture}})
        }
        # create dictionary with sector as key and sum of values for sections
        dict_sum_by_sector = {GlossaryCore.Years: self.years_range}
        dict_sum_by_sector.update({sector: dict_sectors_sections[sector][GlossaryCore.SectionDictSectors[sector]].sum(axis=1) for sector in self.sector_list})
        # create dataframe based on the created dictionnary
        self.sector_gdp_df = pd.DataFrame(data=dict_sum_by_sector)
        self.dict_sectors_gdp_detailed = dict_sectors_sections

    def compute_output_growth(self):
        """
        Compute the output growth between year t and year t-1 
        Output growth of the WITNESS pyworld3 (computed from gross_output_ter)
        """

        gross_output = self.economics_df[GlossaryCore.GrossOutput]

        self.economics_df[GlossaryCore.OutputGrowth] = (gross_output.diff() / gross_output.shift(1)).fillna(0.) * 100

    def compute_output_net_of_damage(self):
        """
        Output net of damages, trillions USD
        """
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        gross_output = self.economics_df[GlossaryCore.GrossOutput].values
        if not self.compute_climate_impact_on_gdp:
            output_net_of_d = gross_output
        else:
            if self.damage_to_productivity:
                damage = 1 - ((1 - damefrac) /
                              (1 - self.frac_damage_prod * damefrac))
                output_net_of_d = (1 - damage) * gross_output
            else:
                output_net_of_d = gross_output * (1 - damefrac)
        self.economics_df[GlossaryCore.OutputNetOfDamage] = output_net_of_d

    def compute_consumption(self):
        """Equation for consumption
        C, Consumption, trillions $USD
        """
        net_output = self.economics_df[GlossaryCore.OutputNetOfDamage].values
        investment = self.economics_df[GlossaryCore.InvestmentsValue].values
        consumption = net_output - investment
        self.economics_df[GlossaryCore.Consumption] = consumption

    def compute_consumption_pc(self):
        """Equation for consumption per capita
        c, Per capita consumption, thousands $USD
        """
        consumption = self.economics_df[GlossaryCore.Consumption].values
        population = self.population_df[GlossaryCore.PopulationValue].values
        consumption_pc = consumption / population * 1000
        self.economics_df[GlossaryCore.PerCapitaConsumption] = consumption_pc

    def compute_usable_capital_lower_bound_constraint(self):
        """
        Lower bound usable capital constraint = capital utilisation ratio * non energy capital - usable capital
        This constraint is only meant to be used when GDP is fixed !
        """
        ne_capital = self.capital_df[GlossaryCore.NonEnergyCapital].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values
        self.usable_capital_lower_bound_constraint = (usable_capital - self.capital_utilisation_ratio * ne_capital) / self.usable_capital_ref if not self.compute_gdp else np.zeros(self.nb_per)

    def compute_usable_capital_upper_bound_constraint(self):
        """
        Constraint : max capital utilisation ratio * non energy capital >= usable capital

        implemented as :  max capital utilisation ratio * non energy capital - usable capital, that should be
        positive when satisfied
        """
        ne_capital = self.capital_df[GlossaryCore.NonEnergyCapital].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values
        diff = self.max_capital_utilisation_ratio * ne_capital - usable_capital
        self.usable_capital_upper_bound_constraint = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.ConstraintUpperBoundUsableCapital: diff / self.usable_capital_ref
        })

    def compute_usable_capital_objective(self):
        """
        usable capital objective = (capital utilisation ratio * non energy capital - usable capital)**2 / usable_capital_objective_ref
        """
        ne_capital = self.capital_df[GlossaryCore.NonEnergyCapital].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values
        self.usable_capital_obj_content = (usable_capital - self.max_capital_utilisation_ratio * ne_capital) / self.usable_capital_objective_ref
        self.usable_capital_objective = pseudo_abs_obj(self.usable_capital_obj_content)#np.array([np.sum(self.usable_capital_obj_content)]) # OK

    def prepare_outputs(self):
        """post processing"""
        self.economics_detail_df = pd.DataFrame.copy(self.economics_df)
        self.economics_df = self.economics_df[GlossaryCore.EconomicsDf['dataframe_descriptor'].keys()]
        self.economics_detail_df = self.economics_detail_df[GlossaryCore.EconomicsDetailDf['dataframe_descriptor'].keys()]

    def compute_damage_from_productivity_loss(self):
        """
        Compute damages due to loss of productivity.

        As GDP ~= productivity x (Usable capital + Labor)², and that we can compute productivity with or without damages,
        we compute the damages on GDP from loss of productivity as
        (productivity wo damage - productivity w damage) x (Usable capital + Labor).
        """
        productivity_w_damage = self.economics_df[GlossaryCore.ProductivityWithDamage].values
        productivity_wo_damage = self.economics_df[GlossaryCore.ProductivityWithoutDamage].values
        applied_productivity = self.economics_df[GlossaryCore.Productivity].values
        gross_output = self.economics_df[GlossaryCore.GrossOutput].values

        estimated_damage_from_productivity_loss = (productivity_wo_damage - productivity_w_damage) / applied_productivity * gross_output
        if self.damage_to_productivity:
            damage_from_productivity_loss = estimated_damage_from_productivity_loss
        else:
            damage_from_productivity_loss = np.zeros_like(estimated_damage_from_productivity_loss)

        self.damage_df[GlossaryCore.DamagesFromProductivityLoss] = damage_from_productivity_loss
        self.damage_df[GlossaryCore.EstimatedDamagesFromProductivityLoss] = estimated_damage_from_productivity_loss

    def compute_damage_from_climate(self):
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        gross_output = self.economics_df[GlossaryCore.GrossOutput].values
        net_output = self.economics_df[GlossaryCore.OutputNetOfDamage].values

        damage_from_climate = np.zeros_like(gross_output)
        if self.compute_climate_impact_on_gdp:
            damage_from_climate = gross_output - net_output
            estimated_damage_from_climate = damage_from_climate
        else:
            if self.damage_to_productivity:
                estimated_damage_from_climate = gross_output * damefrac * (1 - self.frac_damage_prod) / (
                            1 - self.frac_damage_prod * damefrac)
            else:
                estimated_damage_from_climate = gross_output * damefrac

        self.damage_df[GlossaryCore.DamagesFromClimate] = damage_from_climate
        self.damage_df[GlossaryCore.EstimatedDamagesFromClimate] = estimated_damage_from_climate

    def compute_total_damages(self):
        """Damages are the sum of damages from climate + damges from loss of productivity"""

        self.damage_df[GlossaryCore.EstimatedDamages] = self.damage_df[GlossaryCore.EstimatedDamagesFromClimate].values + self.damage_df[GlossaryCore.EstimatedDamagesFromProductivityLoss].values
        self.damage_df[GlossaryCore.Damages] = self.damage_df[GlossaryCore.DamagesFromClimate].values + self.damage_df[GlossaryCore.DamagesFromProductivityLoss].values

    def compute_regionalised_gdp(self):
        """
        Compute regionalised gdp based on the economics_df dataframe computed by the model (that gives total gdp)
        Use linear model to compute share of each region and country in the total gdp
        Compute the gdp per region and per country
        """
        # import linear parameters from database: parameters are computed in the jupyter notebook in data folder
        dict_linear_parameters = DatabaseWitnessCore.LinearParemetersGDPperRegion.value
        breakdown_countries = DatabaseWitnessCore.CountriesPerRegionIMF.value
        # use linear equation y=a*x+b to compute predicted gdp per group per year
        result_total_gdp_per_group = np.array(dict_linear_parameters['a']) * self.years_range + np.array(dict_linear_parameters['b']).reshape(-1, 1)
        gdp_predicted_per_group = result_total_gdp_per_group.T
        # compute percentage of gdp for each group
        percentage_gdp_per_group = gdp_predicted_per_group / gdp_predicted_per_group.sum(axis=1, keepdims=True) * 100
        # compute total based on predicted gdp and on gdp output from model
        total_gdp_per_group = percentage_gdp_per_group * self.economics_df[GlossaryCore.OutputNetOfDamage].values.reshape(-1,1) / 100
        # store data in total gdp
        self.total_gdp_per_group_df[GlossaryCore.Years] = self.years_range
        self.total_gdp_per_group_df[list(breakdown_countries.keys())] = total_gdp_per_group
        self.percentage_gdp_per_group_df[GlossaryCore.Years] = self.years_range
        self.percentage_gdp_per_group_df[list(breakdown_countries.keys())] = percentage_gdp_per_group
        # get percentage of gdp per country in each group
        mean_percentage_gdp_country = DatabaseWitnessCore.GDPPercentagePerCountry.value
        # Iterate over each row to compute gdp of the country
        for index, (_, row) in enumerate(mean_percentage_gdp_country.iterrows()):
            # repeat the years for each country
            df_temp = pd.DataFrame({GlossaryCore.Years: self.total_gdp_per_group_df[GlossaryCore.Years]})
            # compute GDP for each year using the percentage and GDP Value of the correspondant group
            # and convert T$ to G$
            df_temp[GlossaryCore.GDPName] = 1000 * row[GlossaryCore.MeanPercentageName] * self.total_gdp_per_group_df[row[GlossaryCore.GroupName]] / 100
            # Add the country name
            df_temp[GlossaryCore.CountryName] = row[GlossaryCore.CountryName]
            # Add the country group
            df_temp[GlossaryCore.GroupName] = row[GlossaryCore.GroupName]
            # concatenate with the result dataframe
            self.df_gdp_per_country = pd.concat([self.df_gdp_per_country, df_temp]) if index > 0 else df_temp.copy()
        # reset index
        self.df_gdp_per_country.reset_index(drop=True, inplace=True)

    def compute(self, inputs: dict):
        """
        Compute all models for year range
        """

        self.create_dataframe()
        self.set_coupling_inputs(inputs)
        # set gross ouput from input if necessary
        if not self.compute_gdp:
            self.set_gross_output()

        # Employment rate and workforce
        self.compute_employment_rate()
        self.compute_workforce()
        self.compute_energy_efficiency()
        self.compute_usable_capital()
        self.compute_productivity_growthrate()
        self.compute_productivity()
        if self.compute_gdp:
            self.compute_gross_output()
        self.compute_output_net_of_damage()

        self.compute_investment()

        self.compute_capital()

        self.compute_consumption()
        self.compute_consumption_pc()

        self.compute_output_growth()
        self.compute_section_gdp()
        self.compute_sector_gdp()
        self.compute_usable_capital_lower_bound_constraint()
        self.compute_usable_capital_upper_bound_constraint()
        self.compute_usable_capital_objective()
        self.compute_damage_from_productivity_loss()
        self.compute_damage_from_climate()
        self.compute_total_damages()

        self.compute_consumption_objective()

        self.prepare_outputs()

        self.compute_regionalised_gdp()

        self.compute_energy_consumption_per_section()
        self.compute_energy_consumption_households()

        self.capital_df[GlossaryCore.GrossOutput] = self.economics_df[GlossaryCore.GrossOutput].values
        self.capital_df[GlossaryCore.OutputNetOfDamage] = self.economics_df[GlossaryCore.OutputNetOfDamage].values
        self.capital_df[GlossaryCore.PerCapitaConsumption] = self.economics_df[GlossaryCore.PerCapitaConsumption].values

    def compute_energy_consumption_per_section(self):
        """
        Compute energy consumption and emissions per section
        Use energy_consumption_percentage_per_sector_df to compute energy consumption per sector
        Use energy_consumption_percentage_{sector}_sections to compute energy consumption per section per sector
        Use carbon intensity to compute CO2 emissions per section
        Store the data in dictionaries:
        - dict_energy_consumption_detailed: key is sector, value is a dictionary containing detailed and total energy consumption DataFrames
        - dict_sector_emissions_detailed: key is sector, value is a dictionary containing detailed and total energy emissions DataFrames
        """
        self.dict_energy_consumption_detailed = {}
        for sector_name, percentage_sections_of_sector in self.dict_dataframe_energy_consumption_sections.items():

            # intialize dictionary of dictionary
            self.dict_energy_consumption_detailed[sector_name] = {}

            sector_sections_energy_conso_df = pd.DataFrame({GlossaryCore.Years: self.years_range})

            for section in GlossaryCore.SectionDictSectors[sector_name]:
                sector_sections_energy_conso_df[section] = percentage_sections_of_sector[section].values *\
                                                           self.sector_energy_consumption_percentage_df[sector_name].values / 100. *\
                                                           self.energy_production[GlossaryCore.TotalProductionValue].values / 100.

            sector_energy_conso_df = self.energy_production[GlossaryCore.TotalProductionValue].values *\
                                  self.sector_energy_consumption_percentage_df[sector_name].values / 100.

            self.dict_energy_consumption_detailed[sector_name] = {
                "total": sector_energy_conso_df,
                "detailed": sector_sections_energy_conso_df,
            }

            economics_energy_consumption = self.energy_production[GlossaryCore.TotalProductionValue].values * \
                                           (100 - self.sector_energy_consumption_percentage_df[GlossaryCore.Households].values) / 100.

            self.dict_energy_consumption_detailed["total"] = pd.DataFrame({
                GlossaryCore.Years: self.years_range,
                GlossaryCore.TotalEnergyConsumptionAllSectorsName: economics_energy_consumption
            })

    def compute_consumption_objective(self):
        self.consommation_objective = np.array(
            [self.economics_df[GlossaryCore.Consumption].mean()]) / self.consommation_objective_ref

    def compute_energy_consumption_households(self):
        energy_consumption_households = (self.sector_energy_consumption_percentage_df[GlossaryCore.Households].values *
                                         self.energy_production[GlossaryCore.TotalProductionValue].values) / 100.

        self.energy_consumption_households_df = pd.DataFrame({
            GlossaryCore.Years: self.years_range,
            GlossaryCore.TotalProductionValue: energy_consumption_households
        })

    """-------------------Gradient functions-------------------"""

    def _null_derivative(self):
        nb_years = len(self.years_range)
        return np.zeros((nb_years, nb_years))

    def _identity_derivative(self):
        nb_years = len(self.years_range)
        return np.identity(nb_years)

    def d_productivity_w_damage_d_damage_frac_output(self):
        """derivative of productivity with damage wrt damage frac output"""
        nb_years = len(self.years_range)

        productivity_wo_damage = self.economics_detail_df[GlossaryCore.ProductivityWithoutDamage].values

        return np.diag(-productivity_wo_damage)

    def d_productivity_d_damage_frac_output(self):
        """gradient for productivity for damage_df"""
        d_productivity_d_damage_frac_output = self._null_derivative()

        if self.damage_to_productivity:
            d_productivity_d_damage_frac_output = self.d_productivity_w_damage_d_damage_frac_output()
        return d_productivity_d_damage_frac_output

    def d_energy_production(self):
        """
        Derivative of :
        - usable capital
        - gross output
        wrt energy
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.economics_detail_df[GlossaryCore.Productivity].values
        working_pop = self.workforce_df[GlossaryCore.Workforce].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values

        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        d_usable_capital_d_energy = np.diag(self.capital_utilisation_ratio * energy_efficiency)

        d_gross_output_d_energy = np.diag(
            productivity * alpha * usable_capital ** (gamma - 1) * np.diag(d_usable_capital_d_energy) *
            (alpha * usable_capital ** gamma + (1 - alpha) * working_pop ** gamma) ** (1. / gamma - 1.)
        ) if self.compute_gdp else self._null_derivative()

        d_net_output_d_energy = self._d_net_output_d_user_input(d_gross_output_d_energy)
        d_energy_investment_d_energy, d_invest_d_energy, d_non_energy_investment_d_energy = self.d_investment_d_user_input(d_net_output_d_energy)
        d_consumption_d_energy = self.d_consumption_d_user_input(d_net_output_d_energy, d_invest_d_energy)
        d_consumption_pc_d_energy = self.d_consumption_per_capita_d_user_input(d_consumption_d_energy)

        d_damages_d_energy, d_estimated_damages_d_energy = self.d_damages_d_user_input(d_gross_output_d_energy, d_net_output_d_energy)

        d_kne_d_energy = self._d_kne_d_user_input(d_non_energy_investment_d_energy)
        d_ku_obj_d_energy = self._d_ku_obj_d_user_input(d_usable_capital_d_energy, d_kne_d_energy)
        d_ku_ub_contraint = self.d_ku_upper_bound_constraint_d_user_input(d_usable_capital_d_energy, d_kne_d_energy)
        return d_gross_output_d_energy, d_net_output_d_energy, d_usable_capital_d_energy, d_consumption_pc_d_energy,\
               d_estimated_damages_d_energy, d_damages_d_energy, d_energy_investment_d_energy, d_ku_obj_d_energy, d_ku_ub_contraint

    def _d_ku_obj_d_user_input(self, dku_d_user_input, dkne_d_user_input):
        d_ku_obj_content_d_user_input = (dku_d_user_input - self.max_capital_utilisation_ratio * dkne_d_user_input) / self.usable_capital_objective_ref  # OK
        d_ku_obj_d_user_input = d_pseudo_abs_obj(self.usable_capital_obj_content, d_ku_obj_content_d_user_input)
        return d_ku_obj_d_user_input

    def d_workforce_d_workagepop(self):
        """Gradient for workforce wrt working age population"""
        employment_rate = self.workforce_df[GlossaryCore.EmploymentRate].values
        d_workforce_d_workagepop = np.diag(employment_rate)
        return d_workforce_d_workagepop

    def d_working_pop(self):
        """
        Derivative of :
        - usable capital
        - gross output
        - lower bound constraint
        wrt working age population
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        working_pop = self.workforce_df[GlossaryCore.Workforce].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values
        productivity = self.economics_detail_df[GlossaryCore.Productivity].values
        employment_rate = self.workforce_df[GlossaryCore.EmploymentRate].values

        d_gross_output_d_wap = np.diag(
            productivity * (1 - alpha) * working_pop ** (gamma - 1) * employment_rate * (
                alpha * usable_capital ** gamma + (1 - alpha) * working_pop ** gamma
            ) ** (1/gamma - 1)
        ) if self.compute_gdp else self._null_derivative()

        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        dQ_dY = 1 - damefrac if not self.damage_to_productivity else (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)
        if not self.compute_climate_impact_on_gdp:
            dQ_dY = np.ones_like(self.years_range)
        d_net_output_d_wap = np.diag(dQ_dY) @ d_gross_output_d_wap
        _,d_i_d_wap, d_ine_d_wap = self.d_investment_d_user_input(d_net_output_d_wap)
        d_consumption_d_wap = self.d_consumption_d_user_input(d_net_output_d_wap, d_i_d_wap)
        d_consumption_pc_d_wap = self.d_consumption_per_capita_d_user_input(d_consumption_d_wap)

        d_damages_d_wap, d_estimated_damages_d_wap = self.d_damages_d_user_input(d_gross_output_d_wap, d_net_output_d_wap)

        d_kne_d_wap = self._d_kne_d_user_input(d_ine_d_wap)
        d_ku_obj_d_wap = self._d_ku_obj_d_user_input(self._null_derivative(), d_kne_d_wap)
        d_ku_constraint_d_wap = self.d_ku_upper_bound_constraint_d_user_input(self._null_derivative(), d_kne_d_wap)
        return d_gross_output_d_wap, d_net_output_d_wap, d_consumption_pc_d_wap, d_damages_d_wap, d_estimated_damages_d_wap, d_ku_obj_d_wap, d_ku_constraint_d_wap

    def _d_kne_d_user_input(self, d_invest_non_energy_d_user_input):
        d_kne_d_user_input = self._null_derivative()
        for i in range(1, self.nb_years):
            for j in range(self.nb_years):
                d_kne_d_user_input[i, j] = (1 - self.depreciation_capital) * d_kne_d_user_input[i - 1, j] + d_invest_non_energy_d_user_input[i - 1, j]
        return d_kne_d_user_input

    def d_share_invest_non_energy(self):
        """Derivative of the list below wrt to share investments non-energy (snei below)
        - investment non energy
        - ku, kne, lower bound constraint, usable capital obj
        - net ouptut, gross output
        """
        net_output = self.economics_df[GlossaryCore.OutputNetOfDamage].values
        d_ine_dsnei = np.diag(net_output) / 100.
        d_net_output_dnsei = self._null_derivative()
        d_consumption_d_snei = self.d_consumption_d_user_input(d_net_output_dnsei, d_ine_dsnei)
        d_consumption_pc_d_snei = self.d_consumption_per_capita_d_user_input(d_consumption_d_snei)
        d_kne_d_snei = self._d_kne_d_user_input(d_ine_dsnei)

        d_ku_obj_d_snei = self._d_ku_obj_d_user_input(self._null_derivative(), d_kne_d_snei)
        d_ku_ub_constraint_d_snei = self.d_ku_upper_bound_constraint_d_user_input(self._null_derivative(), d_kne_d_snei)
        return d_consumption_pc_d_snei, d_ine_dsnei, d_ku_obj_d_snei, d_ku_ub_constraint_d_snei

    def _d_net_output_d_user_input(self, d_gross_output_d_user_input):
        """derivative of net output wrt X, user should provide the derivative of gross output wrt X"""
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values

        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            dQ_dY = np.diag((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
        elif self.compute_climate_impact_on_gdp and not self.damage_to_productivity:
            dQ_dY = np.diag(1 - damefrac)
        elif not self.compute_climate_impact_on_gdp:
            dQ_dY = np.eye(self.nb_years)
        else:
            raise Exception("Problem")
        dQ_d_user_input = dQ_dY @ d_gross_output_d_user_input
        return dQ_d_user_input

    def _d_energy_investment_d_energy_investement_wo_tax(self):
        """
        energy_investment(t), trillions $USD (including renewable investments)
        Share of the total output
        """
        d_energy_investment_wo_tax_d_energy_investment_wo_tax = np.eye(self.nb_years)
        d_energy_investment_wo_renewable_d_energy_investment_wo_tax = d_energy_investment_wo_tax_d_energy_investment_wo_tax * 1e3 # TODO ? Sure of 1e3 ?

        d_energy_investment_d_energy_investment_wo_tax = d_energy_investment_wo_tax_d_energy_investment_wo_tax

        return d_energy_investment_d_energy_investment_wo_tax, d_energy_investment_wo_renewable_d_energy_investment_wo_tax

    def d_investment_d_energy_investment_wo_tax(self):
        """Derivative of investment wrt share energy investment"""
        d_energy_investment_d_energy_investment_wo_tax, d_energy_investment_wo_renewable_d_energy_investment_wo_tax = \
            self._d_energy_investment_d_energy_investement_wo_tax()

        d_non_energy_investment_d_energy_investment_wo_tax = self._null_derivative()
        d_investment_d_energy_investment_wo_tax = d_energy_investment_d_energy_investment_wo_tax
        return d_investment_d_energy_investment_wo_tax, d_energy_investment_d_energy_investment_wo_tax,\
               d_non_energy_investment_d_energy_investment_wo_tax, d_energy_investment_wo_renewable_d_energy_investment_wo_tax

    def d_investment_d_user_input(self, d_net_output_d_user_input):
        """derivative of investment wrt X, user should provide the derivative of net output wrt X"""
        d_energy_investment_d_user_input = self._null_derivative()
        percent_invest_non_energy = self.share_non_energy_investment[GlossaryCore.ShareNonEnergyInvestmentsValue].values
        d_non_energy_investment_d_user_input = np.diag(percent_invest_non_energy / 100.) @ d_net_output_d_user_input

        d_investment_d_user_input = d_energy_investment_d_user_input + d_non_energy_investment_d_user_input

        return d_energy_investment_d_user_input, d_investment_d_user_input, d_non_energy_investment_d_user_input

    def d_consumption_d_user_input(self, d_net_output_d_user_input, d_investment_d_user_input):
        """derivative of consumption wrt user input"""
        d_consumption_d_user_input = d_net_output_d_user_input - d_investment_d_user_input
        return d_consumption_d_user_input

    def d_consumption_per_capita_d_user_input(self, d_consumption_d_user_input):
        """
        derivative of consumption per capita wrt user input

        consumption per capita = consumption / population * 1000
        """
        d_consumption_per_capita_d_consumption = np.diag(1 / self.population_df[GlossaryCore.PopulationValue].values * 1000)
        d_consumption_per_capita_d_user_input = d_consumption_per_capita_d_consumption @ d_consumption_d_user_input
        return d_consumption_per_capita_d_user_input

    def d_consumption_pc_d_population(self):
        """derivative of pc_consumption wrt population
        consumption_pc = consumption / population * 1000

        """
        consumption = self.economics_detail_df[GlossaryCore.Consumption].values
        population = self.population_df[GlossaryCore.PopulationValue].values

        d_consumption_pc_d_population = np.diag( - consumption * 1000 / population ** 2)
        return d_consumption_pc_d_population

    def d_damage_frac_output(self):
        """derivative of net output wrt damage frac output"""
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        gross_output = self.economics_df[GlossaryCore.GrossOutput].values
        productivity = self.economics_detail_df[GlossaryCore.Productivity].values

        d_gross_output_d_dfo =  np.diag(gross_output / productivity) @ self.d_productivity_d_damage_frac_output()

        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            factor = (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)
            d_factor_d_dfo = np.diag((self.frac_damage_prod - 1) / (1 - self.frac_damage_prod * damefrac) ** 2)
        elif self.compute_climate_impact_on_gdp and not self.damage_to_productivity:
            factor = 1 - damefrac
            d_factor_d_dfo = -np.eye(self.nb_years)
        elif not self.compute_climate_impact_on_gdp:
            factor = np.ones_like(damefrac)
            d_factor_d_dfo = self._null_derivative()
        else:
            raise Exception("Problem")
        d_net_output_d_dfo = np.diag(gross_output) @ d_factor_d_dfo + np.diag(factor) @ d_gross_output_d_dfo
        d_energy_investment_d_dfo, d_invest_d_dfo, d_ine_d_dfo = self.d_investment_d_user_input(d_net_output_d_dfo)
        d_consumption_d_dfo = self.d_consumption_d_user_input(d_net_output_d_dfo, d_invest_d_dfo)
        d_consumption_pc_d_dfo = self.d_consumption_per_capita_d_user_input(d_consumption_d_dfo)


        d_damages_from_productivity_loss_d_dfo, d_estimated_damages_from_productivity_loss_d_dfo = \
            self.d_damages_from_productivity_loss_d_damage_fraction_output(d_gross_output_d_dfo)
        d_estimated_damages_from_climate_d_dfo = self.d_estimated_damages_from_climate_d_damage_frac_output(d_gross_output_d_dfo, d_net_output_d_dfo)
        d_damages_from_climate_d_dfo = self.__d_damages_from_climate_d_user_input(d_gross_output_d_dfo, d_net_output_d_dfo)
        d_estimated_damages_d_dfo = d_estimated_damages_from_climate_d_dfo + d_estimated_damages_from_productivity_loss_d_dfo
        d_damages_d_dfo = d_damages_from_climate_d_dfo + d_damages_from_productivity_loss_d_dfo

        d_kne_d_dfo = self._d_kne_d_user_input(d_ine_d_dfo)
        dku_obj_d_dfo = self._d_ku_obj_d_user_input(self._null_derivative(), d_kne_d_dfo)
        dku_ub_constraint_d_dfo = self.d_ku_upper_bound_constraint_d_user_input(self._null_derivative(), d_kne_d_dfo)
        return d_gross_output_d_dfo, d_net_output_d_dfo, d_consumption_pc_d_dfo, d_estimated_damages_d_dfo, d_damages_d_dfo, d_energy_investment_d_dfo, dku_obj_d_dfo, dku_ub_constraint_d_dfo

    def d_net_output_d_energy_invest(self):
        """derivative of net output wrt share energy_invest"""
        return self._null_derivative()

    def __d_damages_from_climate_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        return derivative

    def __d_estimated_damages_from_climate_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput]

        if self.compute_climate_impact_on_gdp:
            derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        else:
            if self.damage_to_productivity:
                derivative = np.diag(damefrac * (1 - self.frac_damage_prod) /
                                     (1 - self.frac_damage_prod * damefrac)) @ d_gross_output_d_user_input
            else:
                derivative = np.diag(damefrac) @ d_gross_output_d_user_input

        return derivative

    def d_estimated_damages_from_climate_d_damage_frac_output(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput]
        gross_output = self.economics_df[GlossaryCore.GrossOutput].values

        if self.compute_climate_impact_on_gdp:
            derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        else:
            if self.damage_to_productivity:
                derivative = d_gross_output_d_user_input @ np.diag(damefrac * (1 - self.frac_damage_prod) /
                                     (1 - self.frac_damage_prod * damefrac)) + np.diag(gross_output) * \
                             np.diag((1 - self.frac_damage_prod) / (1 - self.frac_damage_prod * damefrac) ** 2)

            else:
                derivative = d_gross_output_d_user_input @ np.diag(damefrac) + np.diag(gross_output)

        return derivative

    def d_damages_from_productivity_loss_d_damage_fraction_output(self, d_gross_output_d_damage_fraction_output):
        gross_output = self.economics_detail_df[GlossaryCore.GrossOutput].values
        productivity_wo_damage = self.economics_detail_df[GlossaryCore.ProductivityWithoutDamage].values
        productivity_w_damage = self.economics_detail_df[GlossaryCore.ProductivityWithDamage].values

        d_productivity_w_damage_d_damage_frac_output = self.d_productivity_w_damage_d_damage_frac_output()
        nb_years = len(self.years_range)
        d_damages_from_productivity_loss_d_damage_fraction_output = self._null_derivative()
        d_estimated_damages_from_productivity_loss_d_damage_fraction_output = self._null_derivative()
        if self.damage_to_productivity:
            for i in range(nb_years):
                d_estimated_damages_from_productivity_loss_d_damage_fraction_output[i] = d_gross_output_d_damage_fraction_output[i] * (productivity_wo_damage[i]/productivity_w_damage[i] - 1) +\
                    - gross_output[i] * productivity_wo_damage[i] / productivity_w_damage[i] ** 2 * d_productivity_w_damage_d_damage_frac_output[i]
        else:
            d_estimated_damages_from_productivity_loss_d_damage_fraction_output = np.diag((productivity_wo_damage - productivity_w_damage)/productivity_wo_damage) @ d_gross_output_d_damage_fraction_output - np.diag(gross_output / productivity_wo_damage) @ d_productivity_w_damage_d_damage_frac_output
        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            d_damages_from_productivity_loss_d_damage_fraction_output = d_estimated_damages_from_productivity_loss_d_damage_fraction_output

        return d_damages_from_productivity_loss_d_damage_fraction_output, d_estimated_damages_from_productivity_loss_d_damage_fraction_output

    def __d_damages_from_productivity_loss_d_user_input(self, d_gross_output_d_user_input):
        productivity_wo_damage = self.economics_detail_df[GlossaryCore.ProductivityWithoutDamage].values
        productivity_w_damage = self.economics_detail_df[GlossaryCore.ProductivityWithDamage].values

        d_damages_from_productivity_loss_d_user_input = self._null_derivative()
        applied_productivity = self.economics_detail_df[GlossaryCore.Productivity].values
        d_estimated_damages_from_prod_loss_d_user_input = np.diag((productivity_wo_damage - productivity_w_damage) / (
                applied_productivity)) @  d_gross_output_d_user_input

        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            d_damages_from_productivity_loss_d_user_input = d_estimated_damages_from_prod_loss_d_user_input

        return d_damages_from_productivity_loss_d_user_input, d_estimated_damages_from_prod_loss_d_user_input

    def d_damages_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        d_damages_from_climate = self.__d_damages_from_climate_d_user_input(d_gross_output_d_user_input, d_net_output_d_user_input)
        d_estimated_damages_from_climate = self.__d_estimated_damages_from_climate_d_user_input(d_gross_output_d_user_input, d_net_output_d_user_input)
        d_damages_from_prod_loss, d_estimated_damages_from_prod_loss = self.__d_damages_from_productivity_loss_d_user_input(d_gross_output_d_user_input)
        d_estimated_damages_d_user_input = self.__d_estimated_damages_d_user_input(d_estimated_damages_from_climate, d_estimated_damages_from_prod_loss)

        d_damages_d_user_input = d_damages_from_prod_loss + d_damages_from_climate
        return d_damages_d_user_input, d_estimated_damages_d_user_input

    def __d_estimated_damages_d_user_input(self, d_estimated_damages_from_climate_d_user_input, d_estimated_damages_from_productivity_loss_d_user_input):
        return d_estimated_damages_from_climate_d_user_input + d_estimated_damages_from_productivity_loss_d_user_input

    def d_ku_upper_bound_constraint_d_user_input(self, d_ku_d_user_input, d_kne_d_user_input):
        return - (d_ku_d_user_input - self.max_capital_utilisation_ratio * d_kne_d_user_input) / self.usable_capital_ref

    def d_gdp_section_d_gdp(self, d_gross_output, section_name: str):
        return d_gross_output @ np.diag(self.gdp_percentage_per_section_df[section_name] / 100)

    def d_gdp_section_energy_consumption_d_energy_prod(self, sector_name: str, section_name: str):
        return np.diag(self.sector_energy_consumption_percentage_df[sector_name].values / 100 * self.dict_dataframe_energy_consumption_sections[sector_name][section_name].values / 100.)

    def d_residential_energy_consumption_d_energy_prod(self):
        return np.diag(self.sector_energy_consumption_percentage_df[GlossaryCore.Households].values / 100)

    """-------------------END of Gradient functions-------------------"""



