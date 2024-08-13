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

from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore
from sostrades_optimization_plugins.tools.cst_manager.func_manager_common import pseudo_abs_obj, d_pseudo_abs_obj


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
        self.share_residential_df = None
        self.consommation_objective_ref = None
        self.year_start: int = 0
        self.year_end = None
        self.time_step = None
        self.productivity_start = None
        self.init_gross_output = None
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
        self.co2_emissions_Gt = None
        self.co2_taxes = None
        self.co2_tax_efficiency = None
        self.co2_invest_limit: float = 0.
        self.employment_a_param = None
        self.employment_power_param = None
        self.employment_rate_base_value = None
        self.usable_capital_ref = None
        self.invest_co2_tax_in_renawables = None
        self.compute_climate_impact_on_gdp = None
        self.damage_to_productivity = None
        self.sector_list = None
        self.section_list = None
        self.energy_investment_wo_renewable = None
        self.energy_investment = None
        self.workforce_df = None
        self.damage_fraction_output_df = None
        self.energy_production = None
        self.co2_emissions_Gt = None
        self.co2_taxes = None
        self.energy_investment_wo_tax = None
        self.share_non_energy_investment = None
        self.energy_capital = None
        self.population_df = None
        self.working_age_population_df = None
        self.compute_gdp = None
        self.gross_output_in = None
        self.workforce_df = None
        self.usable_capital_lower_bound_constraint = None
        self.economics_detail_df = None
        self.energy_investment = None
        self.energy_investment_wo_renewable = None
        self.param = param
        self.economics_df = None
        self.damage_df = None
        self.capital_df = None
        self.energy_wasted_objective = None
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
        self.activate_pandemic_effects = self.param['assumptions_dict']['activate_pandemic_effects']
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
        self.time_step = self.param[GlossaryCore.TimeStep]

        self.productivity_start = self.param['productivity_start']
        self.init_gross_output = self.param[GlossaryCore.InitialGrossOutput['var_name']]
        self.capital_start_ne = self.param['capital_start_non_energy']
        self.population_df = self.param[GlossaryCore.PopulationDfValue]
        self.productivity_gr_start = self.param['productivity_gr_start']
        self.decline_rate_tfp = self.param['decline_rate_tfp']
        self.depreciation_capital = self.param['depreciation_capital']
        self.init_rate_time_pref = self.param['init_rate_time_pref']
        self.conso_elasticity = self.param['conso_elasticity']
        self.nb_per = round(
            (self.param[GlossaryCore.YearEnd] -
             self.param[GlossaryCore.YearStart]) /
            self.param[GlossaryCore.TimeStep] +
            1)
        self.years_range = np.arange(
            self.year_start,
            self.year_end + 1,
            self.time_step)
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
        self.co2_emissions_Gt = self.param[GlossaryCore.CO2EmissionsGtValue]
        self.co2_taxes = self.param[GlossaryCore.CO2TaxesValue]
        self.co2_tax_efficiency = self.param[GlossaryCore.CO2TaxEfficiencyValue]
        self.co2_invest_limit = self.param['co2_invest_limit']
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
        self.invest_co2_tax_in_renawables = self.param['assumptions_dict']['invest_co2_tax_in_renewables']
        self.compute_climate_impact_on_gdp = self.param['assumptions_dict']['compute_climate_impact_on_gdp']
        if not self.compute_climate_impact_on_gdp:
            self.damage_to_productivity = False
        self.sector_list = self.param[GlossaryCore.SectorListValue]
        self.section_list = self.param[GlossaryCore.SectionListValue]
        self.usable_capital_objective_ref = self.param[GlossaryCore.UsableCapitalObjectiveRefName]
        self.consommation_objective_ref = self.param[GlossaryCore.ConsumptionObjectiveRefValue]



    def create_dataframe(self):
        """Create the dataframe and fill it with values at year_start"""
        default_index = np.arange(
            self.year_start, self.year_end + 1, self.time_step)
        param = self.param
        economics_df = pd.DataFrame(
            index=default_index,
            columns=list(GlossaryCore.EconomicsDetailDf['dataframe_descriptor'].keys()))

        for key in economics_df.keys():
            economics_df[key] = 0
        economics_df[GlossaryCore.Years] = self.years_range
        economics_df.loc[param[GlossaryCore.YearStart],
                         GlossaryCore.GrossOutput] = self.init_gross_output
        economics_df.loc[param[GlossaryCore.YearStart],
                         GlossaryCore.Productivity] = self.productivity_start
        economics_df.loc[param[GlossaryCore.YearStart],
                         GlossaryCore.ProductivityWithDamage] = self.productivity_start
        economics_df.loc[param[GlossaryCore.YearStart],
                         GlossaryCore.ProductivityWithoutDamage] = self.productivity_start
        economics_df.loc[param[GlossaryCore.YearStart],
                         GlossaryCore.ProductivityGrowthRate] = self.productivity_gr_start
        economics_df.loc[param[GlossaryCore.YearStart],
                         GlossaryCore.OutputGrowth] = self.init_output_growth
#         economics_df['saving_rate'] = self.saving_rate
        self.economics_df = economics_df
        self.economics_df = self.economics_df.replace(
            [np.inf, -np.inf], np.nan)

        self.energy_investment_wo_renewable = pd.DataFrame(
            index=default_index,
            columns=GlossaryCore.EnergyInvestmentsWoRenewable['dataframe_descriptor'].keys())
        self.energy_investment_wo_renewable[GlossaryCore.Years] = self.years_range

        energy_investment = pd.DataFrame(
            index=default_index,
            columns=GlossaryCore.EnergyInvestments['dataframe_descriptor'].keys())

        for key in energy_investment.keys():
            energy_investment[key] = 0
        energy_investment[GlossaryCore.Years] = self.years_range
        self.energy_investment = energy_investment
        self.energy_investment = self.energy_investment.replace(
            [np.inf, -np.inf], np.nan)
        # workforce_df
        workforce_df = pd.DataFrame(index=default_index, columns=[GlossaryCore.Years,
                                                                  GlossaryCore.EmploymentRate,
                                                                  GlossaryCore.Workforce])
        for key in workforce_df.keys():
            workforce_df[key] = 0
        workforce_df[GlossaryCore.Years] = self.years_range
        self.workforce_df = workforce_df
        #capital df
        self.capital_df = pd.DataFrame(index=default_index,
                                       columns=[GlossaryCore.Years,
                                                GlossaryCore.Capital,
                                                GlossaryCore.NonEnergyCapital,
                                                GlossaryCore.EnergyEfficiency,
                                                GlossaryCore.Emax,
                                                GlossaryCore.UsableCapital,])
        for key in self.capital_df.columns:
            self.capital_df[key] = 0
        self.capital_df[GlossaryCore.Years] = self.years_range
        self.capital_df.loc[param[GlossaryCore.YearStart], GlossaryCore.NonEnergyCapital] = self.capital_start_ne

        self.damage_df = pd.DataFrame(index=default_index,
                                      columns=GlossaryCore.DamageDetailedDf['dataframe_descriptor'].keys())
        for key in self.damage_df.columns:
            self.damage_df[key] = 0
        self.damage_df[GlossaryCore.Years] = self.years_range
        self.total_gdp_per_group_df = pd.DataFrame()
        self.percentage_gdp_per_group_df = pd.DataFrame()
        self.df_gdp_per_country = pd.DataFrame(columns=[GlossaryCore.CountryName, GlossaryCore.Years, GlossaryCore.GDPName, GlossaryCore.GroupName])
        self.dict_energy_consumption_detailed = {}
        self.dict_sector_emissions_detailed = {}
        return economics_df.fillna(0.0), energy_investment.fillna(0.0),

    def set_coupling_inputs(self, inputs: dict):
        """
        Set couplings inputs with right index, scaling... 
        """
        self.share_residential_df = inputs[GlossaryCore.ShareResidentialEnergyDfValue]
        self.damage_fraction_output_df = inputs[GlossaryCore.DamageFractionDfValue]
        self.damage_fraction_output_df.index = self.damage_fraction_output_df[GlossaryCore.Years].values
        # Scale energy production
        self.energy_production = inputs[GlossaryCore.EnergyProductionValue]
        self.co2_emissions_Gt = pd.DataFrame({GlossaryCore.Years: inputs[GlossaryCore.CO2EmissionsGtValue][GlossaryCore.Years].values,
                                              GlossaryCore.TotalCO2Emissions: inputs[GlossaryCore.CO2EmissionsGtValue][GlossaryCore.TotalCO2Emissions].values})
        self.co2_emissions_Gt.index = self.co2_emissions_Gt[GlossaryCore.Years].values
        self.co2_taxes = inputs[GlossaryCore.CO2TaxesValue]
        self.co2_taxes.index = self.co2_taxes[GlossaryCore.Years].values
        self.energy_production.index = self.energy_production[GlossaryCore.Years].values
        self.co2_tax_efficiency.index = self.co2_tax_efficiency[GlossaryCore.Years].values
        #Investment in energy
        self.energy_investment_wo_tax = pd.Series(
            inputs[GlossaryCore.EnergyInvestmentsWoTaxValue][GlossaryCore.EnergyInvestmentsWoTaxValue].values,
            index=self.years_range)
        self.share_non_energy_investment = pd.Series(
            inputs[GlossaryCore.ShareNonEnergyInvestmentsValue][GlossaryCore.ShareNonEnergyInvestmentsValue].values / 100.0, index=self.years_range)
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
        self.workforce_df[GlossaryCore.Workforce] = self.workforce_df[GlossaryCore.EmploymentRate] * \
                                                    self.working_age_population_df[GlossaryCore.Population1570]

    def compute_productivity_growthrate(self):
        """
        A_g, Growth rate of total factor productivity.
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        """
        prod_growth_rate = self.productivity_gr_start * np.exp(- self.decline_rate_tfp * (self.years_range - self.year_start))
        self.economics_df[GlossaryCore.ProductivityGrowthRate] = prod_growth_rate

    def compute_productivity(self, year: int):
        """
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        """
        damage_to_productivity = self.damage_to_productivity
        p_productivity_wo_damage = self.economics_df.loc[year - self.time_step, GlossaryCore.ProductivityWithoutDamage]
        p_productivity_w_damage = self.economics_df.loc[year - self.time_step, GlossaryCore.ProductivityWithDamage]
        p_productivity_gr = self.economics_df.loc[year - self.time_step, GlossaryCore.ProductivityGrowthRate]
        damefrac = self.damage_fraction_output_df.loc[year, GlossaryCore.DamageFractionOutput]
        # we divide the productivity growth rate by 5/time_step because of change in time_step (as
        # advised in Traeger, 2013)
        productivity_wo_damage = p_productivity_wo_damage / (1 - p_productivity_gr / (5 / self.time_step))
        productivity_w_damage = p_productivity_w_damage * (1 - self.frac_damage_prod * damefrac) / (1 - p_productivity_gr / (5 / self.time_step))
        self.economics_df.loc[year, GlossaryCore.ProductivityWithDamage] = productivity_w_damage
        self.economics_df.loc[year, GlossaryCore.ProductivityWithoutDamage] = productivity_wo_damage

        if damage_to_productivity:
            self.economics_df.loc[year, GlossaryCore.Productivity] = productivity_w_damage
        else:
            self.economics_df.loc[year, GlossaryCore.Productivity] = productivity_wo_damage

    def compute_capital(self, year: int):
        """
        K(t+1), Capital for next time period, trillions $USD
        """
        year_start = self.year_start
        if year > self.year_end:
            pass
        elif year == self.year_start: 
            capital = self.capital_start_ne + self.energy_capital.loc[year_start, GlossaryCore.Capital]
            self.capital_df.loc[year_start, GlossaryCore.Capital] = capital
        else: 
            #first compute non energy capital 
            ne_investment = self.economics_df.loc[year - self.time_step, GlossaryCore.NonEnergyInvestmentsValue]
            ne_capital = self.capital_df.loc[year - self.time_step, GlossaryCore.NonEnergyCapital]
            capital_a = ne_capital * (1 - self.depreciation_capital) ** self.time_step + \
                self.time_step * ne_investment
            #Then total capital = ne_capital + energy_capital 
            self.capital_df.loc[year, GlossaryCore.NonEnergyCapital] = capital_a
            # Lower bound for capital
            tot_capital = capital_a + self.energy_capital.loc[year, GlossaryCore.Capital]
            self.capital_df.loc[year, GlossaryCore.Capital] = tot_capital
                                  
            return capital_a

    def compute_energy_efficiency(self):
        """compute energy_efficiency"""
        years = self.capital_df[GlossaryCore.Years].values
        energy_efficiency = self.energy_eff_cst + self.energy_eff_max / (1 + np.exp(-self.energy_eff_k *
                                                                                    (years - self.energy_eff_xzero)))
        self.capital_df[GlossaryCore.EnergyEfficiency] = energy_efficiency

    def compute_usable_capital(self):
        """compute usable capital = Energy Production Net * capital utilisation ratio * energy efficiency"""
        net_energy_production = self.energy_production[GlossaryCore.TotalProductionValue]
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency]
        usable_capital_unbounded = self.capital_utilisation_ratio * net_energy_production * energy_efficiency
        self.capital_df[GlossaryCore.UsableCapital] = usable_capital_unbounded

    def compute_investment(self, year: int):
        """
        Compute I(t) (total Investment) and Ine(t) (Investment in non-energy sectors) in trillions $USD
        Investment Non energy (t) = Share Non Energy investment (t) * Output net of damage (t)
        Investments energy (t) = input energy investments(t)

        Investements (t) = Investments energy (t) + Investments non energy (t)
        """
        net_output = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage]
        self.economics_df.loc[year, GlossaryCore.NonEnergyInvestmentsValue] = self.share_non_energy_investment[year] * net_output
        self.economics_df.loc[year, GlossaryCore.InvestmentsValue] = self.economics_df.loc[year, GlossaryCore.EnergyInvestmentsValue] + \
                                                    self.economics_df.loc[year, GlossaryCore.NonEnergyInvestmentsValue]

    def compute_energy_investment(self, year: int):
        """
        Energy invests  = Energy invest without tax + Added invest in renewables from CO2 tax
        """
        energy_investment_wo_tax = self.energy_investment_wo_tax[year]  # in T$

        self.co2_emissions_Gt[GlossaryCore.TotalCO2Emissions].clip(lower=0.0, inplace=True)

        self.energy_investment_wo_renewable.loc[year, GlossaryCore.EnergyInvestmentsWoRenewableValue] = energy_investment_wo_tax * 10. # in 100G$

        ren_investments = self.compute_energy_renewable_investment(year, energy_investment_wo_tax)  # T$
        energy_investment = energy_investment_wo_tax + ren_investments  # in T$
        self.economics_df.loc[year,
                              [GlossaryCore.EnergyInvestmentsValue,  # T$
                               GlossaryCore.EnergyInvestmentsWoTaxValue,  # T$
                               GlossaryCore.EnergyInvestmentsFromTaxValue]] = \
            [energy_investment,  # T$
             energy_investment_wo_tax, # T$
             ren_investments]  # T$
        self.energy_investment.loc[year, GlossaryCore.EnergyInvestmentsValue] = energy_investment * 10.  # 100G$

        return energy_investment

    def compute_energy_renewable_investment(self, year: int, energy_investment_wo_tax: float):
        """
        computes energy investment for renewable part in T$
        for a given year: returns net CO2 emissions * CO2 taxes * a efficiency factor
        """
        if not self.invest_co2_tax_in_renawables:
            return 0.
        co2_invest_limit = self.co2_invest_limit
        emissions = self.co2_emissions_Gt.loc[year, GlossaryCore.TotalCO2Emissions] * 1e9  # t CO2
        co2_taxes = self.co2_taxes.loc[year, GlossaryCore.CO2Tax]  # $/t
        co2_tax_eff = self.co2_tax_efficiency.loc[year, GlossaryCore.CO2TaxEfficiencyValue] / 100.  # %
        ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$

        # if emissions is zero the right gradient (positive) is not zero but the left gradient is zero
        # when complex step we add ren_invest with the complex step and it is
        # not good
        if ren_investments.real == 0.0:
            ren_investments = 0.0
        # Saturation of renewable invest at n * invest wo tax with n ->
        # co2_invest_limit entry parameter
        if ren_investments > co2_invest_limit * energy_investment_wo_tax and ren_investments != 0.0:
            ren_investments = co2_invest_limit * energy_investment_wo_tax / 10.0 * \
                (9.0 + np.exp(- co2_invest_limit *
                              energy_investment_wo_tax / ren_investments))

        return ren_investments  # T$

    def compute_gross_output(self, year: int):
        """ Compute the gdp 
        inputs: usable capital by year in trill $ , working population by year in million of people,
             productivity by year (no unit), alpha (between 0 and 1) 
        output: gdp in trillion dollars
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.economics_df.loc[year, GlossaryCore.Productivity]

        working_pop = self.workforce_df.loc[year, GlossaryCore.Workforce]
        capital_u = self.capital_df.loc[year, GlossaryCore.UsableCapital]
        # If gamma == 1/2 use sqrt but same formula
        if gamma == 1 / 2:
            output = productivity * (alpha * np.sqrt(capital_u) + (1 - alpha) * np.sqrt(working_pop))**2
        else:
            output = productivity * (alpha * capital_u**gamma + (1 - alpha) * working_pop**gamma) ** (1 / gamma)
        self.economics_df.loc[year, GlossaryCore.GrossOutput] = output

        return output

    def set_gross_output(self): 
        """
        Set gross output according to input
        """
        self.economics_df = self.economics_df.drop(GlossaryCore.GrossOutput, axis=1)
        self.economics_df = self.economics_df.merge(self.gross_output_in[[GlossaryCore.Years, GlossaryCore.GrossOutput]],
                                                    on=GlossaryCore.Years, how='left').set_index(self.economics_df.index)

    def get_gdp_percentage_per_section(self):
        '''
        Get default values for gdp percentage per sector from gdp_percentage_per_sector.csv file
        '''
        # the year range for the study can differ from that stated in the csv file
        start_year_csv = self.gdp_percentage_per_section_df.loc[0, GlossaryCore.Years]
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
        self.section_gdp_df = self.gdp_percentage_per_section_df.copy()
        self.section_gdp_df[self.section_list] = self.section_gdp_df[self.section_list].multiply(self.economics_df.reset_index(drop=True)[GlossaryCore.OutputNetOfDamage], axis='index') / 100.

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

    def compute_output_net_of_damage(self, year: int):
        """
        Output net of damages, trillions USD
        """
        damefrac = self.damage_fraction_output_df.loc[year, GlossaryCore.DamageFractionOutput]
        gross_output = self.economics_df.loc[year, GlossaryCore.GrossOutput]
        if not self.compute_climate_impact_on_gdp:
            output_net_of_d = gross_output
        else:
            if self.damage_to_productivity:
                damage = 1 - ((1 - damefrac) /
                              (1 - self.frac_damage_prod * damefrac))
                output_net_of_d = (1 - damage) * gross_output
            else:
                output_net_of_d = gross_output * (1 - damefrac)
        self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage] = output_net_of_d
        return output_net_of_d

    def compute_consumption(self, year: int):
        """Equation for consumption
        C, Consumption, trillions $USD
        """
        net_output = self.economics_df.loc[year, GlossaryCore.OutputNetOfDamage]
        investment = self.economics_df.loc[year, GlossaryCore.InvestmentsValue]
        consumption = net_output - investment
        # lower bound for conso
        self.economics_df.loc[year, GlossaryCore.Consumption] = consumption

        return consumption

    def compute_consumption_pc(self, year: int):
        """Equation for consumption per capita
        c, Per capita consumption, thousands $USD
        """
        consumption = self.economics_df.loc[year, GlossaryCore.Consumption]
        population = self.population_df.loc[year, GlossaryCore.PopulationValue]
        consumption_pc = consumption / population * 1000
        # Lower bound for pc conso
        self.economics_df.loc[year, GlossaryCore.PerCapitaConsumption] = consumption_pc
        return consumption_pc

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
        Upper bound usable capital constraint = max capital utilisation ratio * non energy capital - usable capital
        """
        ne_capital = self.capital_df[GlossaryCore.NonEnergyCapital].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values
        self.usable_capital_upper_bound_constraint = - (usable_capital - self.max_capital_utilisation_ratio * ne_capital) / self.usable_capital_ref

    def compute_usable_capital_objective(self):
        """
        usable capital objective = (capital utilisation ratio * non energy capital - usable capital)**2 / usable_capital_objective_ref
        """
        ne_capital = self.capital_df[GlossaryCore.NonEnergyCapital].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values
        self.usable_capital_obj_content = (usable_capital - self.capital_utilisation_ratio * ne_capital) / self.usable_capital_objective_ref
        self.usable_capital_objective = pseudo_abs_obj(self.usable_capital_obj_content)#np.array([np.sum(self.usable_capital_obj_content)]) # OK

    def prepare_outputs(self):
        """post processing"""
        self.economics_df = self.economics_df.fillna(0.0)
        self.economics_df = self.economics_df.replace(
            [np.inf, -np.inf], np.nan)
        self.economics_detail_df = pd.DataFrame.copy(self.economics_df)
        self.economics_df = self.economics_df[GlossaryCore.EconomicsDf['dataframe_descriptor'].keys()]
        self.economics_detail_df = self.economics_detail_df[GlossaryCore.EconomicsDetailDf['dataframe_descriptor'].keys()]

        self.energy_investment = self.energy_investment.fillna(0.0)

        self.energy_investment_wo_renewable = self.energy_investment_wo_renewable.fillna(0.)

    def compute_damage_from_productivity_loss(self):
        """
        Compute damages due to loss of productivity.

        As GDP ~= productivity x (Usable capital + Labor)², and that we can compute productivity with or without damages,
        we compute the damages on GDP from loss of productivity as
        (productivity wo damage - productivity w damage) x (Usable capital + Labor).
        """
        productivity_w_damage = self.economics_df[GlossaryCore.ProductivityWithDamage]
        productivity_wo_damage = self.economics_df[GlossaryCore.ProductivityWithoutDamage]
        applied_productivity = self.economics_df[GlossaryCore.Productivity]
        gross_output = self.economics_df[GlossaryCore.GrossOutput]

        estimated_damage_from_productivity_loss = (productivity_wo_damage - productivity_w_damage) / applied_productivity * gross_output
        if self.damage_to_productivity:
            damage_from_productivity_loss = estimated_damage_from_productivity_loss
        else:
            damage_from_productivity_loss = np.zeros_like(estimated_damage_from_productivity_loss)

        self.damage_df[GlossaryCore.DamagesFromProductivityLoss] = damage_from_productivity_loss
        self.damage_df[GlossaryCore.EstimatedDamagesFromProductivityLoss] = estimated_damage_from_productivity_loss

    def compute_damage_from_climate(self):
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput]
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

        self.damage_df[GlossaryCore.EstimatedDamages] = self.damage_df[GlossaryCore.EstimatedDamagesFromClimate] + self.damage_df[GlossaryCore.EstimatedDamagesFromProductivityLoss]
        self.damage_df[GlossaryCore.Damages] = self.damage_df[GlossaryCore.DamagesFromClimate] + self.damage_df[GlossaryCore.DamagesFromProductivityLoss]

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
        for _, row in mean_percentage_gdp_country.iterrows():
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
            self.df_gdp_per_country = pd.concat([self.df_gdp_per_country, df_temp])
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

        year_start = self.year_start
        # YEAR START
        self.compute_capital(year_start)
        self.compute_output_net_of_damage(year_start)
        self.compute_energy_investment(year_start)
        self.compute_investment(year_start)
        self.compute_consumption(year_start)
        self.compute_consumption_pc(year_start)
        # for year 0 compute capital +1
        self.compute_capital(year_start + 1)
        self.compute_productivity_growthrate()

        # Then iterate over years from year_start + tstep:
        for year in self.years_range[1:]:
            # First independant variables
            self.compute_productivity(year)
            # Then others:
            if self.compute_gdp:
                self.compute_gross_output(year)
            self.compute_output_net_of_damage(year)
            self.compute_energy_investment(year)
            self.compute_investment(year)
            self.compute_consumption(year)
            self.compute_consumption_pc(year)
            # capital t+1 :
            self.compute_capital(year+1)

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

            # Create a temporary DataFrame to compute energy consumption per section
            merged_df_energy_prod = pd.merge(self.energy_production, self.sector_energy_consumption_percentage_df[[GlossaryCore.Years, sector_name]],
                                             on=GlossaryCore.Years, how='inner')
            # division by 100 for percentages
            merged_df_energy_prod['energy_consumption_sector'] = merged_df_energy_prod[GlossaryCore.TotalProductionValue] * merged_df_energy_prod[sector_name] / 100.
            merged_df_energy_prod = pd.merge(merged_df_energy_prod, percentage_sections_of_sector, on=GlossaryCore.Years, how='inner')

            # Extracting list of sections from DataFrame
            list_sections = percentage_sections_of_sector.drop(columns=[GlossaryCore.Years]).columns

            # Multiply each section's percentage by energy consumption sector and store in the dictionary
            # division by 100 for percentages
            # no further conversion needed, we convert TWh * kg/kWh = 1e12 Wh * kg/1e3 Wh = 1e9 kg = Mt
            merged_df_energy_prod[list_sections] = percentage_sections_of_sector[list_sections].apply(lambda col: col * merged_df_energy_prod['energy_consumption_sector'] / 100.)

            # list of columns to store in final dictionary
            list_columns_to_store = [GlossaryCore.Years] + list_sections.to_list()

            # Store detailed energy consumption data
            self.dict_energy_consumption_detailed[sector_name]["detailed"] = merged_df_energy_prod[list_columns_to_store]

            # Compute total energy consumption for sector
            total_energy_consumption_sector = pd.DataFrame(columns=[GlossaryCore.Years, GlossaryCore.TotalEnergyConsumptionSectorName])
            total_energy_consumption_sector[GlossaryCore.TotalEnergyConsumptionSectorName] = merged_df_energy_prod[list_sections].sum(axis=1)
            total_energy_consumption_sector[GlossaryCore.Years] = merged_df_energy_prod[GlossaryCore.Years]
            self.dict_energy_consumption_detailed[sector_name]["total"] = total_energy_consumption_sector

        # Compute total energy consumption and emissions across all sectors
        self.dict_energy_consumption_detailed["total"] = pd.concat([detailed_data["total"].set_index(GlossaryCore.Years) for detailed_data in
                   self.dict_energy_consumption_detailed.values()], axis=1).sum(axis=1).to_frame(
            name=GlossaryCore.TotalEnergyConsumptionAllSectorsName).reset_index()

    def compute_consumption_objective(self):
        self.consommation_objective = np.array(
            [self.economics_df[GlossaryCore.Consumption].mean()]) / self.consommation_objective_ref

    def compute_energy_consumption_households(self):
        energy_consumption_households = (self.share_residential_df[GlossaryCore.ShareSectorEnergy].values *
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

        d_productivity_w_damage_d_damage_frac_output = self._null_derivative()
        p_productivity_gr = self.economics_detail_df[GlossaryCore.ProductivityGrowthRate].values
        p_productivity = self.economics_detail_df[GlossaryCore.ProductivityWithDamage].values

        # first line stays at zero since derivatives of initial values are zero
        for i in range(1, nb_years):
            d_productivity_w_damage_d_damage_frac_output[i, i] = (1 - self.frac_damage_prod * self.damage_fraction_output_df.loc[
                self.years_range[i], GlossaryCore.DamageFractionOutput]) * \
                                                        d_productivity_w_damage_d_damage_frac_output[i - 1, i] / (
                                                                    1 - (p_productivity_gr[i - 1] /
                                                                         (5 / self.time_step))) - \
                                                        self.frac_damage_prod * \
                                                        p_productivity[i - 1] / \
                                                        (1 - (p_productivity_gr[i - 1] / (5 / self.time_step)))
            for j in range(1, i):
                d_productivity_w_damage_d_damage_frac_output[i, j] = (1 - self.frac_damage_prod *
                                                             self.damage_fraction_output_df.loc[self.years_range[
                                                                                                    i], GlossaryCore.DamageFractionOutput]) * \
                                                            d_productivity_w_damage_d_damage_frac_output[i - 1, j] / \
                                                            (1 - (p_productivity_gr[i - 1] / (5 / self.time_step)))
        return d_productivity_w_damage_d_damage_frac_output

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
        d_gross_output_d_energy[0, 0] = 0.

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
        d_ku_obj_content_d_user_input = (dku_d_user_input - self.capital_utilisation_ratio * dkne_d_user_input) / self.usable_capital_objective_ref  # OK
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
        d_gross_output_d_wap[0, 0] = 0.

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

        self.co2_emissions_Gt[GlossaryCore.TotalCO2Emissions].clip(
            lower=0.0, inplace=True)

        d_ren_investments_d_energy_investment_wo_tax = self._d_ren_investments_d_energy_investment_wo_tax(
            d_energy_investment_wo_tax_d_energy_investment_wo_tax)
        d_energy_investment_d_energy_investment_wo_tax = d_energy_investment_wo_tax_d_energy_investment_wo_tax + \
                                                       d_ren_investments_d_energy_investment_wo_tax

        return d_energy_investment_d_energy_investment_wo_tax, d_energy_investment_wo_renewable_d_energy_investment_wo_tax

    def d_investment_d_energy_investment_wo_tax(self):
        """Derivative of investment wrt share energy investment"""
        d_energy_investment_d_energy_investment_wo_tax, d_energy_investment_wo_renewable_d_energy_investment_wo_tax = \
            self._d_energy_investment_d_energy_investement_wo_tax()

        d_non_energy_investment_d_energy_investment_wo_tax = self._null_derivative()
        d_investment_d_energy_investment_wo_tax = d_energy_investment_d_energy_investment_wo_tax
        return d_investment_d_energy_investment_wo_tax, d_energy_investment_d_energy_investment_wo_tax,\
               d_non_energy_investment_d_energy_investment_wo_tax, d_energy_investment_wo_renewable_d_energy_investment_wo_tax

    def _d_ren_investments_d_energy_investment_wo_tax(self, denergy_investment_wo_tax):
        """
        computes gradients for energy investment for renewable part by energy_investment_wo_tax
        for a given year: returns net CO2 emissions * CO2 taxes * a efficiency factor
        """
        energy_investment_wo_tax = self.economics_detail_df[GlossaryCore.EnergyInvestmentsWoTaxValue].values  # T$
        co2_invest_limit = self.co2_invest_limit
        # t CO2
        emissions = self.co2_emissions_Gt[GlossaryCore.TotalCO2Emissions].values * 1e9
        co2_taxes = self.co2_taxes[GlossaryCore.CO2Tax].values  # $/t
        co2_tax_eff = self.co2_tax_efficiency[GlossaryCore.CO2TaxEfficiencyValue].values / 100.  # %

        ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$

        nb_years = len(self.years_range)
        # derivative matrix initialization
        dren_investments = self._null_derivative()
        if self.invest_co2_tax_in_renawables: # TODO: what follows is wrong
            for i in range(nb_years):
                # if emissions is zero the right gradient (positive) is not zero but the left gradient is zero
                # when complex step we add ren_invest with the complex step and it is
                # not good
                if ren_investments[i].real == 0.0:
                    ren_investments[i] = 0.0

                # Saturation of renewable invest at n * invest wo tax with n ->
                # co2_invest_limit entry parameter
                if ren_investments[i] > co2_invest_limit * energy_investment_wo_tax[i] and ren_investments[i] != 0.0:
                    u = co2_invest_limit * energy_investment_wo_tax[i] / 10.0
                    u_prime = co2_invest_limit * \
                        denergy_investment_wo_tax[i] / 10.0
                    v = 9.0 + np.exp(- co2_invest_limit *
                                     energy_investment_wo_tax[i] / ren_investments[i])
                    v_prime = (- co2_invest_limit *
                               denergy_investment_wo_tax[i] / ren_investments[i]) * (v - 9.0)

                    dren_investments[i] = u_prime * v + v_prime * u

        return dren_investments

    def d_energy_investment_d_co2_tax(self):
        """derivative of energy investment wrt co2 tax"""
        energy_investment_wo_tax = self.energy_investment_wo_tax.values
        dren_investments = self.d_renewable_investments_d_co2_tax(
            energy_investment_wo_tax)
        denergy_investment = dren_investments

        return denergy_investment

    def d_renewable_investments_d_co2_tax(self, energy_investment_wo_tax):
        """derivative of renewable investment wrt co2 tax"""
        nb_years = len(self.years_range)
        co2_invest_limit = self.co2_invest_limit
        # t CO2
        emissions = self.co2_emissions_Gt[GlossaryCore.TotalCO2Emissions].values * 1e9
        co2_taxes = self.co2_taxes[GlossaryCore.CO2Tax].values  # $/t
        co2_tax_eff = self.co2_tax_efficiency[GlossaryCore.CO2TaxEfficiencyValue].values / 100.  # %

        ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$
        d_ren_investments_dco2_taxes = np.diag(emissions * co2_tax_eff / 1e12)

        # derivative matrix initialization
        dren_investments = self._null_derivative()
        if self.invest_co2_tax_in_renawables:
            for i in range(nb_years):

                # if emissions is zero the right gradient (positive) is not zero but the left gradient is zero
                # when complex step we add ren_invest with the complex step and it
                # is not good
                if ren_investments[i].real == 0.0:
                    ren_investments[i] = 0.0
                    d_ren_investments_dco2_taxes[i] = np.zeros(nb_years)

                dren_investments[i] = d_ren_investments_dco2_taxes[i]

                # Saturation of renewable invest at n * invest wo tax with n ->
                # co2_invest_limit entry parameter
                if ren_investments[i] > co2_invest_limit * energy_investment_wo_tax[i] and ren_investments[i] != 0.0:
                    v = np.exp(- co2_invest_limit *
                               energy_investment_wo_tax[i] / ren_investments[i])
                    v_prime = (d_ren_investments_dco2_taxes[i] * co2_invest_limit *
                               energy_investment_wo_tax[i] / (ren_investments[i]**2))

                    dren_investments[i] = co2_invest_limit * \
                        energy_investment_wo_tax[i] / 10.0 * v_prime * v

        return dren_investments

    def d_investment_d_user_input(self, d_net_output_d_user_input):
        """derivative of investment wrt X, user should provide the derivative of net output wrt X"""
        d_energy_investment_d_user_input = self._null_derivative()
        self._d_ren_investments_d_energy_investment_wo_tax(d_energy_investment_d_user_input)
        d_non_energy_investment_d_user_input = np.diag(self.share_non_energy_investment.values) @ d_net_output_d_user_input

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
        """derivative of net output wrt damage frac output #todo: refactor!!!"""
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


    def d_investment_d_co2emissions(self):
        """derivative of investments wrt co2 emissions"""
        nb_years = len(self.years_range)
        co2_invest_limit = self.co2_invest_limit
        # t CO2
        emissions = self.co2_emissions_Gt[GlossaryCore.TotalCO2Emissions].values * 1e9
        co2_taxes = self.co2_taxes[GlossaryCore.CO2Tax].values  # $/t
        co2_tax_eff = self.co2_tax_efficiency[GlossaryCore.CO2TaxEfficiencyValue].values / 100.  # %
        energy_investment_wo_tax = self.economics_detail_df[GlossaryCore.EnergyInvestmentsWoTaxValue].values

        dren_investments = self._null_derivative()

        if self.invest_co2_tax_in_renawables:
            ren_investments = emissions * co2_taxes * co2_tax_eff / 1e12  # T$
            dren_investments = np.identity(nb_years) * co2_taxes * co2_tax_eff / 1e12 * 1e9

            for i in range(nb_years):
                if ren_investments[i] > co2_invest_limit * energy_investment_wo_tax[i] and ren_investments[i] != 0.0:
                    g_prime = co2_taxes[i] * co2_tax_eff[i] / 1e12 * 1e9
                    f_prime = g_prime * co2_invest_limit * energy_investment_wo_tax[i] / 10.0 * (co2_invest_limit * energy_investment_wo_tax[i] / ren_investments[i]**2) *\
                        np.exp(- co2_invest_limit *
                               energy_investment_wo_tax[i] / ren_investments[i])
                    dren_investments[i, i] = f_prime
                if ren_investments[i].real == 0.0:
                    dren_investments[i, i] = 0.0

        denergy_invest = dren_investments
        dinvestment = denergy_invest
        return denergy_invest, dinvestment

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

    """-------------------END of Gradient functions-------------------"""




