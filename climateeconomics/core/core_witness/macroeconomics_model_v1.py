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

from climateeconomics.glossarycore import GlossaryCore
from climateeconomics.database.database_witness_core import DatabaseWitnessCore

class MacroEconomics:
    """
    Economic pyworld3 that compute the evolution of capital, consumption, output...
    """
    GDP_PERCENTAGE_PER_SECTOR_FILE = 'gdp_percentage_per_sector.csv'
    DATA_FOLDER = 'data'

    def __init__(self, param):
        """
        Constructor
        """
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
        self.lo_capital = None
        self.lo_conso = None
        self.lo_per_capita_conso = None
        self.hi_per_capita_conso = None
        self.nb_per = None
        self.years_range: int = 0
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
        self.delta_capital_cons = None
        self.usable_cap_sqrt = None
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
        self.dict_sectors_detailed = None
        self.usable_capital_objective = None
        self.usable_capital_objective_ref = None
        self.set_data()
        self.create_dataframe()
        self.total_gdp_per_group_df = None
        self.percentage_gdp_per_group_df = None
        self.sector_energy_consumption_percentage_df = None
        self.carbon_intensity_df = None
        self.df_gdp_per_country = None
        self.dict_dataframe_energy_consumption_sections = None
        self.dict_energy_consumption_detailed = None
        self.dict_sector_energy_emissions_detailed = None
        self.dict_dataframe_non_energy_emissions_sections = None
        self.dict_non_energy_emissions_detailed = None
        self.dict_total_emissions_detailed = None


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
        self.lo_capital = self.param['lo_capital']
        self.lo_conso = self.param['lo_conso']
        self.lo_per_capita_conso = self.param['lo_per_capita_conso']
        self.hi_per_capita_conso = self.param['hi_per_capita_conso']
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
                                                GlossaryCore.UsableCapital,
                                                GlossaryCore.UsableCapitalUnbounded])
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
        self.damage_fraction_output_df = inputs[GlossaryCore.DamageFractionDfValue]
        self.damage_fraction_output_df.index = self.damage_fraction_output_df[GlossaryCore.Years].values
        # Scale energy production
        self.energy_production = inputs[GlossaryCore.EnergyProductionValue].copy(deep=True)
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
        self.carbon_intensity_df = inputs[GlossaryCore.EnergyCarbonIntensityDfValue]
        # create dictionary where key is sector and value is the energy consumption percebtage for each section per sector
        self.dict_dataframe_energy_consumption_sections = dict(zip(self.sector_list, [inputs[f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_{sector}']
                                                                                       for sector in self.sector_list]))
        self.dict_dataframe_non_energy_emissions_sections = dict(zip(self.sector_list, [inputs[f'{GlossaryCore.SectionNonEnergyEmissionGdpDfValue}_{sector}']
                                                                                       for sector in self.sector_list]))



      
    def compute_employment_rate(self):
        """ 
        Compute the employment rate. based on prediction from ILO 
        We pyworld3 a recovery from 2020 crisis until 2031 where past level is reached
        For all year not in (2020,2031), value = employment_rate_base_value
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
            self.capital_df.loc[year, GlossaryCore.Capital] = max(tot_capital, self.lo_capital)
                                  
            return capital_a

    def compute_energy_usage(self):
        """Wasted energy is the overshoot of energy production not used by usable capital"""
        non_energy_capital = self.capital_df[GlossaryCore.NonEnergyCapital]
        net_energy_production = self.energy_production[GlossaryCore.TotalProductionValue]  # PWh
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency]
        optimal_energy_production = self.max_capital_utilisation_ratio * non_energy_capital / self.capital_utilisation_ratio / energy_efficiency  # Pwh
        self.economics_df[GlossaryCore.OptimalEnergyProduction] = optimal_energy_production * 1e3
        self.economics_df[GlossaryCore.UsedEnergy] = np.minimum(net_energy_production, optimal_energy_production) * 1e3
        self.economics_df[GlossaryCore.UnusedEnergy] = np.maximum(net_energy_production - optimal_energy_production, 0.) * 1e3
        # Energy_wasted = max((Enet - Eoptimal),0.)
        self.economics_df[GlossaryCore.EnergyWasted] = (net_energy_production - optimal_energy_production) * 1e3 #TWh
        self.economics_df.loc[self.economics_df[GlossaryCore.EnergyWasted] < 0., GlossaryCore.EnergyWasted] = 0.

    def compute_energy_wasted_objective(self):
        """Computes normalized energy wasted constraint. Ewasted=max(Enet - Eoptimal, 0)
        Normalize by total energy since Energy wasted is around 10% of total energy => have constraint around 0.1
        which can be compared to the negative welfare objective (same order of magnitude)
        """
        # total energy is supposed to be > 0.
        energy_wasted_objective = self.economics_df[GlossaryCore.EnergyWasted].values.sum() * 1e-3 / \
                                  self.energy_production[GlossaryCore.TotalProductionValue].values.sum()  # PWh / PWh

        self.energy_wasted_objective = np.array([energy_wasted_objective])

    def compute_energy_efficiency(self):
        """compute energy_efficiency"""
        years = self.capital_df[GlossaryCore.Years].values
        energy_efficiency = self.energy_eff_cst + self.energy_eff_max / (1 + np.exp(-self.energy_eff_k *
                                                                                    (years - self.energy_eff_xzero)))
        self.capital_df[GlossaryCore.EnergyEfficiency] = energy_efficiency

    def compute_unbounded_usable_capital(self):
        """compute unbounded usable capital = Energy Production Net * capital utilisation ratio * energy efficiency"""
        net_energy_production = self.energy_production[GlossaryCore.TotalProductionValue]
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency]
        usable_capital_unbounded = self.capital_utilisation_ratio * net_energy_production * energy_efficiency
        self.capital_df[GlossaryCore.UsableCapitalUnbounded] = usable_capital_unbounded

    def compute_usable_capital(self, year: int):
        """  Usable capital is the part of the capital stock that can be used in the production process. 
        To be usable the capital needs enough energy.
        K_u = min (max capital utilisation ratio * Kne, Unbounded Usable Capital)
        E is energy in Twh and K is capital in trill dollars constant 2020
        Output: usable capital in trill dollars constant 2020
        """
        ne_capital = self.capital_df.loc[year, GlossaryCore.NonEnergyCapital]

        usable_capital_unbounded = self.capital_df.loc[year, GlossaryCore.UsableCapitalUnbounded]
        upper_bound = self.max_capital_utilisation_ratio * ne_capital

        usable_capital = upper_bound if np.real(usable_capital_unbounded) > np.real(
            upper_bound) else usable_capital_unbounded

        self.capital_df.loc[year, GlossaryCore.UsableCapital] = usable_capital

    def compute_investment(self, year: int):
        """Compute I(t) (total Investment) and Ine(t) (Investment in non-energy sectors) in trillions $USD """
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
            GlossaryCore.SectorServices: {section: self.section_gdp_df[section].values for section in GlossaryCore.SectionsServices},
            GlossaryCore.SectorIndustry: {section: self.section_gdp_df[section].values for section in GlossaryCore.SectionsIndustry},
            GlossaryCore.SectorAgriculture: {section: self.section_gdp_df[section].values for section in GlossaryCore.SectionsAgriculture}
        }
        # create dictionary with sector as key and sum of values for sections
        dict_sum_by_sector = {GlossaryCore.Years: self.years_range}
        dict_sum_by_sector.update({
            sector: np.sum(list(sub_dict.values()), axis=0)
            for sector, sub_dict in dict_sectors_sections.items()
        })
        # create dataframe based on the created dictionnary
        self.sector_gdp_df = pd.DataFrame(data=dict_sum_by_sector)
        self.dict_sectors_detailed = dict_sectors_sections

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
        self.economics_df.loc[year, GlossaryCore.Consumption] = max(
            consumption, self.lo_conso)
        return consumption

    def compute_consumption_pc(self, year: int):
        """Equation for consumption per capita
        c, Per capita consumption, thousands $USD
        """
        consumption = self.economics_df.loc[year, GlossaryCore.Consumption]
        population = self.population_df.loc[year, GlossaryCore.PopulationValue]
        consumption_pc = consumption / population * 1000
        # Lower bound for pc conso
        self.economics_df.loc[year, GlossaryCore.PerCapitaConsumption] = max(
            consumption_pc, self.lo_per_capita_conso)
        return consumption_pc

    def compute_usable_capital_lower_bound_constraint(self):
        """
        Lower bound usable capital constraint = capital utilisation ratio * non energy capital - usable capital
        This constraint is only meant to be used when GDP is fixed !
        """
        ne_capital = self.capital_df[GlossaryCore.NonEnergyCapital].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values
        self.delta_capital_cons = (usable_capital - self.capital_utilisation_ratio * ne_capital) / self.usable_capital_ref if not self.compute_gdp else np.zeros(self.nb_per)

    def compute_usable_capital_objective(self):
        """
        usable capital objective = (capital utilisation ratio * non energy capital - usable capital)**2 / usable_capital_objective
        """
        ne_capital = self.capital_df[GlossaryCore.NonEnergyCapital].values
        usable_capital_unbouded = self.capital_df[GlossaryCore.UsableCapitalUnbounded].values
        self.usable_cap_sqrt = (usable_capital_unbouded - self.capital_utilisation_ratio * ne_capital)
        self.usable_capital_objective = np.array([np.sum(np.power(self.usable_cap_sqrt,2))/ (self.nb_years * self.usable_capital_objective_ref)])


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
        self.compute_unbounded_usable_capital()

        year_start = self.year_start
        # YEAR START
        self.compute_capital(year_start)
        self.compute_usable_capital(year_start)
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
            self.compute_usable_capital(year)
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
        self.compute_usable_capital_objective()
        self.compute_energy_usage()
        self.compute_energy_wasted_objective()
        self.compute_damage_from_productivity_loss()
        self.compute_damage_from_climate()
        self.compute_total_damages()

        self.compute_consumption_objective()

        self.prepare_outputs()

        self.compute_regionalised_gdp()

        self.compute_energy_consumption_and_emissions_per_section()
        self.compute_non_energy_emissions_per_section()
        self.compute_total_emissions()

        return self.economics_detail_df, self.economics_df, self.damage_df,self.energy_investment, \
            self.energy_investment_wo_renewable, self.workforce_df, \
            self.capital_df, self.sector_gdp_df, self.energy_wasted_objective, self.total_gdp_per_group_df,\
            self.percentage_gdp_per_group_df, self.df_gdp_per_country


    def compute_energy_consumption_and_emissions_per_section(self):
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
        self.dict_sector_energy_emissions_detailed = {}
        for sector_name, percentage_sections_of_sector in self.dict_dataframe_energy_consumption_sections.items():

            # intialize dictionary of dictionary
            self.dict_energy_consumption_detailed[sector_name] = {}
            self.dict_sector_energy_emissions_detailed[sector_name] = {}

            # Create a temporary DataFrame to compute energy consumption per section
            merged_df_energy_prod = pd.merge(self.energy_production, self.sector_energy_consumption_percentage_df[[GlossaryCore.Years, sector_name]],
                                             on=GlossaryCore.Years, how='inner')
            # 1e3 to convert PWh to TWh, division by 100 for percentages
            merged_df_energy_prod['energy_consumption_sector'] = merged_df_energy_prod[GlossaryCore.TotalProductionValue] * merged_df_energy_prod[sector_name] * 1e3 / 100.
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

            # Multiply each section's percentage by carbon intensity and store in the dictionary for GHG emissions
            energy_emissions_detailed_df = pd.DataFrame(columns=list_columns_to_store)
            energy_emissions_detailed_df[GlossaryCore.Years] = merged_df_energy_prod[GlossaryCore.Years]
            energy_emissions_detailed_df[list_sections] = merged_df_energy_prod[list_sections].multiply(
                self.carbon_intensity_df[GlossaryCore.EnergyCarbonIntensityDfValue], axis=0)
            self.dict_sector_energy_emissions_detailed[sector_name]["detailed"] = energy_emissions_detailed_df

            # Compute total energy emissions for sector
            total_energy_emissions_sector = pd.DataFrame(columns=[GlossaryCore.Years, "total_energy_emissions_sector"])
            total_energy_emissions_sector["total_energy_emissions_sector"] = energy_emissions_detailed_df[list_sections].sum(axis=1)
            total_energy_emissions_sector[GlossaryCore.Years] = energy_emissions_detailed_df[GlossaryCore.Years]
            self.dict_sector_energy_emissions_detailed[sector_name]["total"] = total_energy_emissions_sector

        # Compute total energy consumption and emissions across all sectors
        self.dict_energy_consumption_detailed["total"] = pd.concat([detailed_data["total"].set_index(GlossaryCore.Years) for detailed_data in
                   self.dict_energy_consumption_detailed.values()], axis=1).sum(axis=1).to_frame(
            name=GlossaryCore.TotalEnergyConsumptionAllSectorsName).reset_index()
        self.dict_sector_energy_emissions_detailed["total"] = pd.concat(
            [detailed_data["total"].set_index(GlossaryCore.Years) for detailed_data in
             self.dict_sector_energy_emissions_detailed.values()], axis=1).sum(axis=1).to_frame(
            name=GlossaryCore.TotalEnergyEmissionsAllSectorsName).reset_index()

    def compute_non_energy_emissions_per_section(self):
        """
        Compute non-energy emissions per section.
        Use self.dict_dataframe_non_energy_emissions_sections to compute emissions per sector.
        Each value in self.dict_dataframe_non_energy_emissions_sections is a DataFrame with a column "years" and columns for each section of the sector, measured in tCO2eq/M$.
        Use self.economics_detail_df to multiply emissions rates by GDP of each section.
        Store the data in dictionaries:
            - dict_non_energy_emissions_detailed: key is sector, value is a dictionary containing detailed and total non-energy emissions DataFrames.
        The detailed DataFrame contains emissions data for each section along with the years column.
        The total DataFrame contains the total emissions for each sector along with the years column.
        """
        # Initialize dictionaries
        self.dict_non_energy_emissions_detailed = {}

        # Loop through each sector
        for sector_name, emissions_df in self.dict_dataframe_non_energy_emissions_sections.items():
            # Create temporary DataFrame to compute emissions per section
            merged_df_emissions = pd.DataFrame()
            merged_df_emissions[GlossaryCore.Years] = emissions_df[GlossaryCore.Years]
            # Extract list of sections from DataFrame
            list_sections = emissions_df.drop(columns=[GlossaryCore.Years]).columns

            # Multiply each section's emissions rate by GDP of the section and store in the dictionary
            # tCO2eq / M$ * T$ = 1e6 M$ * tCo2eq / M$ = 1e6 * tCO2eq = MtCO2eq
            merged_df_emissions[list_sections] = emissions_df[list_sections].apply(
                lambda col: col * self.dict_sectors_detailed[sector_name][col.name], axis=0)

            # List of columns to store in final dictionary
            list_columns_to_store = [GlossaryCore.Years] + list_sections.to_list()

            # Store detailed emissions data
            self.dict_non_energy_emissions_detailed[sector_name] = {
                "detailed": merged_df_emissions[list_columns_to_store]
            }

            # Compute total emissions for the sector
            total_emissions_sector = pd.DataFrame(columns=[GlossaryCore.Years, GlossaryCore.TotalNonEnergyEmissionsSectorName])
            total_emissions_sector[GlossaryCore.TotalNonEnergyEmissionsSectorName] = merged_df_emissions[list_sections].sum(axis=1)
            total_emissions_sector[GlossaryCore.Years] = merged_df_emissions[GlossaryCore.Years]
            self.dict_non_energy_emissions_detailed[sector_name]["total"] = total_emissions_sector

        # Compute total non-energy emissions across all sectors
        total_non_energy_emissions_all_sectors = pd.concat(
            [detailed_data["total"].set_index(GlossaryCore.Years) for detailed_data in
             self.dict_non_energy_emissions_detailed.values()], axis=1
        ).sum(axis=1).to_frame(name=GlossaryCore.TotalNonEnergyEmissionsAllSectorsName).reset_index()

        # Store total non-energy emissions in the dictionary
        self.dict_non_energy_emissions_detailed["total"] =  total_non_energy_emissions_all_sectors

    def compute_total_emissions(self):
        """
        Compute total emissions

        This method computes total emissions by summing non-energy and energy emissions for each sector and aggregating
        them into a detailed dictionary and a total emissions dataframe.

        Returns:
            None
        """
        # Initialize dictionary to store total emissions
        self.dict_total_emissions_detailed = {}

        # Create total emissions dataframe
        total_emissions_dataframe = pd.DataFrame(columns=[GlossaryCore.Years, GlossaryCore.TotalEmissionsName])
        years = self.dict_non_energy_emissions_detailed["total"][GlossaryCore.Years]
        total_emissions_dataframe[GlossaryCore.Years] = years
        total_emissions_dataframe[GlossaryCore.TotalEmissionsName] = self.dict_non_energy_emissions_detailed["total"][GlossaryCore.TotalNonEnergyEmissionsAllSectorsName] + self.dict_sector_energy_emissions_detailed["total"][GlossaryCore.TotalEnergyEmissionsAllSectorsName]
        self.dict_total_emissions_detailed["total"] = total_emissions_dataframe
        sector_list = [sector_name for sector_name in list(self.dict_non_energy_emissions_detailed.keys()) if sector_name != "total"]
        # Iterate over sectors to compute total emissions
        for sector_name in sector_list:
            self.dict_total_emissions_detailed[sector_name] = {}

            # Compute total emissions for the sector
            dataframe_total_sector = pd.DataFrame(columns=[GlossaryCore.Years, GlossaryCore.TotalEmissionsName])
            dataframe_total_sector[GlossaryCore.Years] = years
            dataframe_total_sector[GlossaryCore.TotalEmissionsName] = self.dict_non_energy_emissions_detailed[sector_name]["total"][GlossaryCore.TotalNonEnergyEmissionsSectorName] + self.dict_sector_energy_emissions_detailed[sector_name]["total"][GlossaryCore.TotalEnergyEmissionsSectorName]
            self.dict_total_emissions_detailed[sector_name]["total"] = dataframe_total_sector

            # Compute total emissions for each section within the sector
            dataframe_total_sector_sections = pd.DataFrame()
            dataframe_total_sector_sections[GlossaryCore.Years] = years
            non_energy_emissions_df = self.dict_non_energy_emissions_detailed[sector_name]["detailed"].drop(columns=[GlossaryCore.Years])
            energy_emissions_df = self.dict_sector_energy_emissions_detailed[sector_name]["detailed"].drop(columns=[GlossaryCore.Years])
            columns_to_sum = energy_emissions_df.columns
            dataframe_total_sector_sections[columns_to_sum] = non_energy_emissions_df[columns_to_sum] + energy_emissions_df[columns_to_sum]
            self.dict_total_emissions_detailed[sector_name]["detailed"] = dataframe_total_sector_sections


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

    def d_Y_Ku_Ew_Constraint_d_energy(self):
        """
        Derivative of :
        - usable capital
        - Non-energy capital
        - gross output
        - lower bound constraint
        - Energy_wasted
        wrt energy
        """
        alpha = self.output_alpha
        gamma = self.output_gamma
        productivity = self.economics_detail_df[GlossaryCore.Productivity].values
        working_pop = self.workforce_df[GlossaryCore.Workforce].values
        usable_capital = self.capital_df[GlossaryCore.UsableCapital].values

        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        d_Ku_d_E = np.diag(self.capital_utilisation_ratio * energy_efficiency)

        index_zeros = self.economics_detail_df[GlossaryCore.UnusedEnergy].values > 0.

        d_Ku_d_E[index_zeros, index_zeros] = 0.

        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        dQ_dY = 1 - damefrac if not self.damage_to_productivity else (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)
        d_Kne_dE = self._null_derivative()
        dY_dE = np.diag(
            productivity * alpha * usable_capital ** (gamma - 1) * np.diag(d_Ku_d_E) *
            (alpha * usable_capital ** gamma + (1 - alpha) * working_pop ** gamma) ** (1. / gamma - 1.)
        ) if self.compute_gdp else self._null_derivative()
        dY_dE[0, 0] = 0.

        for i in range(1, self.nb_per + 1):
            for j in range(1, i):
                dY_dE[i - 1, j] = productivity[i-1] * alpha * usable_capital[i-1] ** (gamma - 1) *\
                    d_Ku_d_E[i - 1, j] * (alpha * usable_capital[i-1] ** gamma + (1 - alpha) * working_pop[i-1] ** gamma) ** (1. / gamma - 1.) if self.compute_gdp else 0.
                if i < self.nb_per:
                    d_Kne_dE[i, j] = (1 - self.depreciation_capital) * d_Kne_dE[i - 1, j] + \
                                     self.share_non_energy_investment.values[i-1] * dQ_dY[i-1] * dY_dE[i - 1, j]
                    d_Ku_d_E[i, j] = index_zeros[i] * self.max_capital_utilisation_ratio * d_Kne_dE[i, j]

        d_lower_bound_constraint_dE = (d_Ku_d_E - self.capital_utilisation_ratio * d_Kne_dE) / self.usable_capital_ref if not self.compute_gdp else self._null_derivative()
        dKunbouded_d_E = np.diag(self.capital_utilisation_ratio * energy_efficiency)
        d_Ku_obj_d_E = np.sum(2 * (dKunbouded_d_E - self.capital_utilisation_ratio * d_Kne_dE) *
                              self.usable_cap_sqrt.reshape(-1, 1) / (self.usable_capital_objective_ref * self.nb_years), axis=0)
        # TODO use d_Ku_d_E
        # Energy_wasted Ew = E - KNE * k where k = max_capital_utilisation_ratio/capital_utilisation_ratio/energy_efficiency*1.e3
        # energy_efficiency is function of the years. Eoptimal in TWh
        k = np.diag(self.max_capital_utilisation_ratio / self.capital_utilisation_ratio / energy_efficiency * 1.e3)
        d_Ew_dE = self._identity_derivative() * 1.e3 - np.matmul(k, d_Kne_dE) # Enet converted from PWh to TWh
        # Since Ewasted = max(Enet - Eoptimal, 0.), gradient should be 0 when Enet - Eoptimal <=0, ie when Ewasted =0
        # => put to 0 the lines of the gradient matrix corresponding to the years where Ewasted=0
        matrix_of_years_E_is_wasted = (self.economics_df[GlossaryCore.EnergyWasted].values > 0.).astype(int)
        d_Ew_dE = np.transpose(np.multiply(matrix_of_years_E_is_wasted, np.transpose(d_Ew_dE)))

        return dY_dE, d_Ku_d_E, d_lower_bound_constraint_dE, d_Ew_dE, d_Ku_obj_d_E

    def d_workforce_d_workagepop(self):
        """Gradient for workforce wrt working age population"""
        employment_rate = self.workforce_df[GlossaryCore.EmploymentRate].values
        d_workforce_d_workagepop = np.diag(employment_rate)
        return d_workforce_d_workagepop

    def d_gross_output_d_working_pop(self):
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

        index_zeros = self.economics_detail_df[GlossaryCore.UnusedEnergy].values > 0.

        d_Y_d_wap = np.diag(
            productivity * (1 - alpha) * working_pop ** (gamma - 1) * employment_rate * (
                alpha * usable_capital ** gamma + (1 - alpha) * working_pop ** gamma
            ) ** (1/gamma - 1)
        ) if self.compute_gdp else self._null_derivative()
        d_Y_d_wap[0, 0] = 0.

        d_Ku_d_wap = self._null_derivative()
        d_Kne_d_wap = self._null_derivative()

        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        dQ_dY = 1 - damefrac if not self.damage_to_productivity else (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)

        for i in range(1, self.nb_per + 1):
            for j in range(1, i):
                a = productivity[i-1]
                b = alpha  * usable_capital[i-1] ** (gamma - 1) * d_Ku_d_wap[i - 1, j] + (1 - alpha) * working_pop[i - 1] * (gamma - 1) * employment_rate[i - 1] * (i-1 == j)
                c = (alpha * usable_capital[i-1] ** gamma + (1 - alpha) * working_pop[i-1] ** gamma) ** (1/gamma - 1)

                if i -1 != j and self.compute_gdp:
                    d_Y_d_wap[i - 1, j] = a * b * c

                if i < self.nb_per:
                    d_Kne_d_wap[i, j] = (1 - self.depreciation_capital) * d_Kne_d_wap[i - 1, j] +\
                            self.share_non_energy_investment.values[i - 1] * dQ_dY[i - 1] * d_Y_d_wap[i - 1, j]
                    d_Ku_d_wap[i, j] = index_zeros[i] * self.max_capital_utilisation_ratio * d_Kne_d_wap[i, j]

        d_lower_bound_constraint_d_wap = (d_Ku_d_wap - self.capital_utilisation_ratio * d_Kne_d_wap) / self.usable_capital_ref if not self.compute_gdp else self._null_derivative()
        d_Ku_unbouded_d_wap = self._null_derivative()
        d_Ku_obj_d_wap = np.sum(2 * (d_Ku_unbouded_d_wap - self.capital_utilisation_ratio * d_Kne_d_wap)
                                * self.usable_cap_sqrt.reshape(-1, 1)/ (self.usable_capital_objective_ref * self.nb_years), axis=0)
        # Energy_wasted Ew = E - KNE * k where k = max_capital_utilisation_ratio/capital_utilisation_ratio/energy_efficiency * 1e3
        # energy_efficiency is function of the years. Eoptimal in TWh
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        k = np.diag(self.max_capital_utilisation_ratio / self.capital_utilisation_ratio / energy_efficiency * 1.e3)
        d_Ew_d_wap = - np.matmul(k, d_Kne_d_wap)
        # Since Ewasted = max(Enet - Eoptimal, 0.), gradient should be 0 when Enet - Eoptimal <=0, ie when Ewasted =0
        # => put to 0 the lines of the gradient matrix corresponding to the years where Ewasted=0
        matrix_of_years_E_is_wasted = (self.economics_df[GlossaryCore.EnergyWasted].values > 0.).astype(int)
        d_Ew_d_wap = np.transpose(np.multiply(matrix_of_years_E_is_wasted, np.transpose(d_Ew_d_wap)))

        return d_Ku_d_wap, d_Ew_d_wap, d_Y_d_wap, d_lower_bound_constraint_d_wap, d_Ku_obj_d_wap

    def d_net_output_d_user_input(self, d_gross_output_d_user_input):
        """derivative of net output wrt X, user should provide the derivative of gross output wrt X"""
        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values

        if self.damage_to_productivity:
            dQ_dY = np.diag((1 - damefrac) / (1 - self.frac_damage_prod * damefrac))
        else:
            dQ_dY = np.diag(1 - damefrac)
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
        consumption = self.economics_detail_df[GlossaryCore.Consumption].values
        dconsumption = d_net_output_d_user_input - d_investment_d_user_input
        # find index where lower bound reached
        theyears = np.where(consumption == self.lo_conso)[0]
        # Then for these years derivative = 0
        dconsumption[theyears] = 0
        return dconsumption

    def d_consumption_per_capita_d_user_input(self, d_consumption_d_user_input):
        """
        derivative of consumption per capita wrt user input

        consumption per capita = consumption / population * 1000
        """
        pc_consumption = self.economics_df[GlossaryCore.PerCapitaConsumption].values

        d_consumption_per_capita_d_consumption = np.diag(1 / self.population_df[GlossaryCore.PopulationValue].values * 1000)
        d_consumption_per_capita_d_user_input = d_consumption_per_capita_d_consumption @ d_consumption_d_user_input
        # find index where lower bound reached and set it to zero
        theyears = np.where(pc_consumption == self.lo_per_capita_conso)[0]
        d_consumption_per_capita_d_user_input[theyears] = 0
        return d_consumption_per_capita_d_user_input

    def d_consumption_pc_d_population(self):
        """derivative of pc_consumption wrt population
        consumption_pc = consumption / population * 1000

        # Lower bound for pc conso
        self.economics_df.loc[year, GlossaryCore.PerCapitaConsumption] = max(
            consumption_pc, self.lo_per_capita_conso)
        """
        consumption_pc = self.economics_detail_df[GlossaryCore.PerCapitaConsumption].values
        consumption = self.economics_detail_df[GlossaryCore.Consumption].values
        population = self.population_df[GlossaryCore.PopulationValue].values

        d_consumption_pc_d_population = np.diag( - consumption * 1000 / population ** 2)
        # find index where lower bound reached and set them to 0
        theyears = np.where(consumption_pc == self.lo_per_capita_conso)[0]
        d_consumption_pc_d_population[theyears] = 0
        return d_consumption_pc_d_population

    def d_gross_output_d_damage_frac_output(self):
        """derivative of gross output wrt damage frac output"""

        d_productivity_d_damage_frac_output = self.d_productivity_d_damage_frac_output()

        working_pop = self.workforce_df[GlossaryCore.Workforce].values
        capital_u = self.capital_df[GlossaryCore.UsableCapital].values
        gamma = self.output_gamma
        alpha = self.output_alpha

        d_gross_output_d_productivity = np.diag(
            (alpha * capital_u ** gamma + (1 - alpha) * working_pop ** gamma) ** (1 / gamma))

        d_gross_output_d_productivity[0, 0] = 0  # at year start gross output is an input
        d_Y_d_dfo = d_gross_output_d_productivity @ d_productivity_d_damage_frac_output if self.compute_gdp else self._null_derivative()
        productivity = self.economics_detail_df[GlossaryCore.Productivity].values

        ############
        index_zeros = self.economics_detail_df[GlossaryCore.UnusedEnergy].values > 0.

        damefrac = self.damage_fraction_output_df[GlossaryCore.DamageFractionOutput].values
        Y = self.economics_df[GlossaryCore.GrossOutput].values
        dQ_dY = 1 - damefrac if not self.damage_to_productivity else (1 - damefrac) / (1 - self.frac_damage_prod * damefrac)

        d_dQ_dY_d_dfo = damefrac * 0 - 1. if not self.damage_to_productivity else ((self.frac_damage_prod - 1)
                                                                                   / ((1 - self.frac_damage_prod * damefrac) ** 2))
        d_Kne_d_dfo = self._null_derivative()
        d_Ku_d_dfo = self._null_derivative()
        for i in range(1, self.nb_per):
            for j in range(i):
                dQ_im1_d_dfo_j = dQ_dY[i - 1] * d_Y_d_dfo[i - 1, j] + Y[i - 1] * d_dQ_dY_d_dfo[i - 1] * (i - 1 == j)
                d_Kne_d_dfo[i, j] = (1 - self.depreciation_capital) * d_Kne_d_dfo[i - 1, j] + \
                                    self.share_non_energy_investment.values[i - 1] * dQ_im1_d_dfo_j
                d_Ku_d_dfo[i, j] = index_zeros[i] * self.max_capital_utilisation_ratio * d_Kne_d_dfo[i, j]
                d_Y_d_dfo[i, j] += productivity[i] * alpha * d_Ku_d_dfo[i, j] * (
                        capital_u[i] ** (gamma - 1) * (alpha * capital_u[i] ** gamma + (1-alpha) * working_pop[i] ** gamma) ** (1/gamma - 1)
                ) if self.compute_gdp else 0.

        d_lower_bound_constraint_d_dfo = (d_Ku_d_dfo - self.capital_utilisation_ratio * d_Kne_d_dfo) / self.usable_capital_ref if not self.compute_gdp else self._null_derivative()
        dKunbouded_d_E = self._null_derivative()
        d_Ku_obj_d_dfo = np.sum(2 * (dKunbouded_d_E - self.capital_utilisation_ratio * d_Kne_d_dfo)
                                * self.usable_cap_sqrt.reshape(-1, 1) / (self.usable_capital_objective_ref * self.nb_years), axis=0)

        # Energy_wasted Ew = E - KNE * k where k = max_capital_utilisation_ratio/capital_utilisation_ratio/energy_efficiency*1e3
        # energy_efficiency is function of the years. Eoptimal in TWh
        energy_efficiency = self.capital_df[GlossaryCore.EnergyEfficiency].values
        k = np.diag(self.max_capital_utilisation_ratio / self.capital_utilisation_ratio / energy_efficiency * 1.e3)
        d_Ew_d_dfo = - np.matmul(k, d_Kne_d_dfo)
        # Since Ewasted = max(Enet - Eoptimal, 0.), gradient should be 0 when Enet - Eoptimal <=0, ie when Ewasted =0
        # => put to 0 the lines of the gradient matrix corresponding to the years where Ewasted=0
        matrix_of_years_E_is_wasted = (self.economics_df[GlossaryCore.EnergyWasted].values > 0.).astype(int)
        d_Ew_d_dfo = np.transpose(np.multiply(matrix_of_years_E_is_wasted, np.transpose(d_Ew_d_dfo)))


        return d_Y_d_dfo, d_Ku_d_dfo, d_Ew_d_dfo, d_lower_bound_constraint_d_dfo, d_Ku_obj_d_dfo

    def d_net_output_d_damage_frac_output(self, d_gross_output_d_damage_frac_output):
        """derivative of net output wrt damage frac output #todo: refactor!!!"""
        nb_years = len(self.years_range)
        d_net_output_d_damage_frac_output = self._null_derivative()
        for i in range(nb_years):
            gross_output = self.economics_detail_df.loc[self.years_range[i], GlossaryCore.GrossOutput]
            damage_frac_output = self.damage_fraction_output_df.loc[self.years_range[i], GlossaryCore.DamageFractionOutput]
            for j in range(0, i + 1):
                if i == j:
                    if self.damage_to_productivity:
                        d_net_output_d_damage_frac_output[i, j] = (self.frac_damage_prod - 1) / ((self.frac_damage_prod * damage_frac_output - 1)**2) * gross_output + \
                                            (1 - damage_frac_output) / (1 - self.frac_damage_prod *
                                              damage_frac_output) * d_gross_output_d_damage_frac_output[i, j]
                    else:
                        d_net_output_d_damage_frac_output[i, j] = - gross_output + \
                                            (1 - damage_frac_output) * d_gross_output_d_damage_frac_output[i, j]
                else:
                    if self.damage_to_productivity:
                        d_net_output_d_damage_frac_output[i, j] = (
                            1 - damage_frac_output) / (1 - self.frac_damage_prod * damage_frac_output) * d_gross_output_d_damage_frac_output[i, j]
                    else:
                        d_net_output_d_damage_frac_output[i, j] = (
                            1 - damage_frac_output) * d_gross_output_d_damage_frac_output[i, j]
        return d_net_output_d_damage_frac_output

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

    def d_investment_d_share_investment_non_energy(self):
        """Derivative of Investment wrt to share investments non-energy"""
        net_output = self.economics_detail_df[GlossaryCore.OutputNetOfDamage].values
        d_non_energy_investment_d_share_investment_non_energy = np.diag(net_output / 100.0)
        d_investment_d_share_investment_non_energy = d_non_energy_investment_d_share_investment_non_energy

        return d_investment_d_share_investment_non_energy, d_non_energy_investment_d_share_investment_non_energy

    def d_net_output_d_energy_invest(self):
        """derivative of net output wrt share energy_invest"""
        return self._null_derivative()

    def grad_energy_wasted_objective(self, grad_sum_energy_wasted, grad_sum_energy_total):
        """
        gradient of the Energy_wasted objective as function of the gradients of:
        - the sum of the energy_wasted
        - the sum of the total energy production
        grad(Ew_obj) = grad(sum_year(Ew)/sum_year(Etotal)) =
        ((sum_year(Etotal) * grad(sum_year(Ewasted) - (sum_year(Ewasted) * grad(sum_year(Etotal)))/
        (sum_year(Etotal))^2
        where grad(sum_year(E)) = sum_year(grad(E))
        grad(sum_year(Ewasted)) = sum_year(grad(Ewasted)) = np.ones(self.years_range) @ grad(Ewasted)
        grad(sum_year(Etotal)) = sum_year(grad(Etotal)) = np.ones(self.years_range) @ grad(Etotal)
        """
        sum_ewasted = self.economics_df[GlossaryCore.EnergyWasted].values.sum()
        sum_etotal = self.energy_production[GlossaryCore.TotalProductionValue].values.sum()
        # sumetotal is supposed > 0 otherwise no energy in the system => cannot work
        grad_energy_wasted_obj = (sum_etotal * grad_sum_energy_wasted - sum_ewasted * grad_sum_energy_total) / \
                                 sum_etotal ** 2

        return grad_energy_wasted_obj * 1e-3

    def d_damages_from_climate_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
        """
        damages_from_climate = gross output - net output
        """
        derivative = d_gross_output_d_user_input - d_net_output_d_user_input
        return derivative

    def d_estimated_damages_from_climate_d_user_input(self, d_gross_output_d_user_input, d_net_output_d_user_input):
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

    def d_damages_from_productivity_loss_d_user_input(self, d_gross_output_d_user_input):
        productivity_wo_damage = self.economics_detail_df[GlossaryCore.ProductivityWithoutDamage].values
        productivity_w_damage = self.economics_detail_df[GlossaryCore.ProductivityWithDamage].values

        d_damages_from_productivity_loss_d_user_input = self._null_derivative()
        applied_productivity = self.economics_detail_df[GlossaryCore.Productivity].values
        d_estimated_damages_from_prod_loss_d_user_input = np.diag((productivity_wo_damage - productivity_w_damage) / (
                applied_productivity)) @  d_gross_output_d_user_input

        if self.compute_climate_impact_on_gdp and self.damage_to_productivity:
            d_damages_from_productivity_loss_d_user_input = d_estimated_damages_from_prod_loss_d_user_input

        return d_damages_from_productivity_loss_d_user_input, d_estimated_damages_from_prod_loss_d_user_input

    def d_damages_d_user_input(self, d_damages_from_climate_d_user_input, d_damages_from_productivity_loss_d_user_input):
        return d_damages_from_climate_d_user_input + d_damages_from_productivity_loss_d_user_input

    def d_estimated_damages_d_user_input(self, d_estimated_damages_from_climate_d_user_input, d_estimated_damages_from_productivity_loss_d_user_input):
        return d_estimated_damages_from_climate_d_user_input + d_estimated_damages_from_productivity_loss_d_user_input

    def d_consumption_objective_d_consumption(self, d_consumption):
        return d_consumption.mean(axis=0) / self.consommation_objective_ref

    """-------------------END of Gradient functions-------------------"""

    def compute_consumption_objective(self):
        self.consommation_objective = np.array([self.economics_df[GlossaryCore.Consumption].mean()]) / self.consommation_objective_ref
