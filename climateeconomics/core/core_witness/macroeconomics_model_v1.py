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

from sostrades_optimization_plugins.models.differentiable_model import (
    DifferentiableModel,
)

from climateeconomics.database.database_witness_core import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


class MacroEconomics(DifferentiableModel):
    """
    Economic pyworld3 that compute the evolution of capital, consumption, output...
    """

    def set_coupling_inputs(self, inputs: dict):
        """Set couplings inputs with right index, scaling... """
        if not self.inputs['assumptions_dict']['compute_gdp']:
            self.gross_output_in = inputs['gross_output_in']
        # create dictionary where key is sector and value is the energy consumption percebtage for each section per sector
        self.dict_dataframe_energy_consumption_sections = dict(zip(self.inputs[GlossaryCore.SectorListValue], [inputs[f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_{sector}']
                                                                                       for sector in self.inputs[GlossaryCore.SectorListValue]]))
        self.dict_dataframe_energy_consumption_sections = {key: val.loc[val[GlossaryCore.Years] >= self.year_start] for key, val in self.dict_dataframe_energy_consumption_sections.items()}

    def compute_employment_rate(self):
        """ 
        Compute the employment rate. based on prediction from ILO 
        We pyworld3 a recovery from 2020 crisis until 2031 where past level is reached
        For all year not in (2020,2031), value = employment_rate_base_value
        """
        year_covid = 2020
        year_end_recovery = 2031
        employment_rates = []
        for year in self.years:
            if year < year_end_recovery:
                x_recovery = year + 1 - year_covid
                employment_rate = self.inputs['employment_a_param'] * \
                                           x_recovery ** self.inputs['employment_power_param']
            else:
                employment_rate = self.inputs['employment_rate_base_value']
            employment_rates.append(employment_rate)

        self.outputs[f"{GlossaryCore.WorkforceDfValue}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.WorkforceDfValue}:{GlossaryCore.EmploymentRate}"] = self.np.array(employment_rates)


    def compute_workforce(self):
        """ Compute the workforce based on formula: 
        workforce = people in working age * employment_rate 
        inputs : - number of people in working age 
                - employment rate in %
        Output: number of working people in million of people
        """
        self.outputs[f"{GlossaryCore.WorkforceDfValue}:{GlossaryCore.Workforce}"] = \
            self.outputs[f"{GlossaryCore.WorkforceDfValue}:{GlossaryCore.EmploymentRate}"] * \
            self.inputs[f"{GlossaryCore.WorkingAgePopulationDfValue}:{GlossaryCore.Population1570}"]

    def compute_productivity_growthrate(self):
        """
        A_g, Growth rate of total factor productivity.
        Returns:
            :returns: A_g(0) * exp(-Δ_a * (t-1))
        """
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.ProductivityGrowthRate}"] = \
            self.inputs['productivity_gr_start'] * \
            self.np.exp(- self.inputs['decline_rate_tfp'] * (self.years - self.year_start))

    def compute_productivity(self):
        """
        productivity
        if damage_to_productivity= True add damage to the the productivity
        if  not: productivity evolves independently from other variables (except productivity growthrate)
        """
        prod_wo_d = self.inputs['productivity_start']
        productivity_wo_damage_list =  [self.inputs['productivity_start']]

        damage_fraction_output = self.inputs[f"{GlossaryCore.DamageFractionDfValue}:{GlossaryCore.DamageFractionOutput}"]
        productivity_growth = self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.ProductivityGrowthRate}"]

        for prod_growth, damage_frac_year in zip(productivity_growth[:-1], damage_fraction_output[1:]):
            prod_wo_d = prod_wo_d / (1 - prod_growth / 5)
            productivity_wo_damage_list.append(prod_wo_d)

        productivity_wo_damage_list = self.np.array(productivity_wo_damage_list)
        productivity_w_damage_list = self.np.array(productivity_wo_damage_list) * (1 - damage_fraction_output)

        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.ProductivityWithDamage}"] = productivity_w_damage_list
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.ProductivityWithoutDamage}"] = productivity_wo_damage_list
        if self.inputs[GlossaryCore.DamageToProductivity]:
            self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Productivity}"] = productivity_w_damage_list
        else:
            self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Productivity}"] = productivity_wo_damage_list

    def compute_energy_efficiency(self):
        """compute energy_efficiency"""
        self.outputs[f"{GlossaryCore.CapitalDfValue}:{GlossaryCore.EnergyEfficiency}"] = \
            self.inputs['energy_eff_cst'] + \
            self.inputs['energy_eff_max'] / (1 + self.np.exp(-self.inputs['energy_eff_k'] * (self.years - self.inputs['energy_eff_xzero'])))

    def set_gross_output(self):
        """
        Set gross output according to input
        """
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.GrossOutput}"] = self.inputs[f"gross_output_in:{GlossaryCore.GrossOutput}"]

    def compute_sectors_and_subsectors_gdp(self):
        """Computes the subsectors and sectors gdp"""
        # get gdp percentage per section, and compute gdp per section using Net output of damage
        self.outputs[f"{GlossaryCore.SectorGdpDfValue}:{GlossaryCore.Years}"] = self.years
        for sector, sector_sections in zip(GlossaryCore.SectorsPossibleValues,
                                           [GlossaryCore.SectionsServices, GlossaryCore.SectionsAgriculture , GlossaryCore.SectionsIndustry]):
            self.outputs[f"{sector}.{GlossaryCore.SectionGdpDfValue}:{GlossaryCore.Years}"] = self.years
            for section in sector_sections:
                self.outputs[f"{sector}.{GlossaryCore.SectionGdpDfValue}:{section}"] = \
                    self.inputs[f"{GlossaryCore.SectionGdpPercentageDfValue}:{section}"] *\
                    self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.OutputNetOfDamage}"] / 100.
            self.outputs[f"{sector}.{GlossaryCore.SectionGdpDfValue}:Total"] = self.sum_cols(
                self.get_cols_output_dataframe(df_name=f"{sector}.{GlossaryCore.SectionGdpDfValue}", expect_years=True)
            )
            self.outputs[f"{GlossaryCore.SectorGdpDfValue}:{sector}"] = self.outputs[f"{sector}.{GlossaryCore.SectionGdpDfValue}:Total"]

    def compute_consumption(self):
        """Equation for consumption
        C, Consumption, trillions $USD
        """
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Consumption}"] = \
            self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.OutputNetOfDamage}"] - \
            self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.InvestmentsValue}"]

    def compute_consumption_pc(self):
        """Equation for consumption per capita
        c, Per capita consumption, thousands $USD
        """
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.PerCapitaConsumption}"] = \
            self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Consumption}"] / \
            self.inputs[f"{GlossaryCore.PopulationDfValue}:{GlossaryCore.PopulationValue}"] * 1000

    def prepare_outputs(self):
        """post processing"""
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Years}"] = self.years
        for key in GlossaryCore.EconomicsDf['dataframe_descriptor'].keys():
            self.outputs[f"{GlossaryCore.EconomicsDfValue}:{key}"] = \
                self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{key}"]

        self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.Years}"] = self.years
        for key in GlossaryCore.DamageDf['dataframe_descriptor'].keys():
            self.outputs[f"{GlossaryCore.DamageDfValue}:{key}"] = \
                self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{key}"]

    def compute_damage_from_productivity_loss(self):
        """
        Compute damages due to loss of productivity.

        As GDP ~= productivity x (Usable capital + Labor)², and that we can compute productivity with or without damages,
        we compute the damages on GDP from loss of productivity as
        (productivity wo damage - productivity w damage) x (Usable capital + Labor).
        """
        productivity_w_damage = self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.ProductivityWithDamage}"]
        productivity_wo_damage = self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.ProductivityWithoutDamage}"]
        applied_productivity = self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Productivity}"]
        gross_output_y = self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.GrossOutput}"]

        estimated_damage_from_productivity_loss = (productivity_wo_damage - productivity_w_damage) / applied_productivity * gross_output_y
        if self.inputs[GlossaryCore.DamageToProductivity]:
            damage_from_productivity_loss = estimated_damage_from_productivity_loss
        else:
            damage_from_productivity_loss = self.np.zeros_like(estimated_damage_from_productivity_loss)

        self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromProductivityLoss}"] = damage_from_productivity_loss
        self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromProductivityLoss}"] = estimated_damage_from_productivity_loss

    def compute_damage_from_climate(self):
        damefrac = self.inputs[f"{GlossaryCore.DamageFractionDfValue}:{GlossaryCore.DamageFractionOutput}"]
        gross_output_y = self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.GrossOutput}"]
        net_output = self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.OutputNetOfDamage}"]

        damage_from_climate = self.np.zeros_like(gross_output_y)
        if self.inputs['assumptions_dict']['compute_climate_impact_on_gdp']:
            damage_from_climate = gross_output_y - net_output
            estimated_damage_from_climate = damage_from_climate
        else:
            if self.inputs[GlossaryCore.DamageToProductivity]:
                estimated_damage_from_climate = gross_output_y * damefrac * (1 - self.inputs[GlossaryCore.FractionDamageToProductivityValue]) / (
                            1 - self.inputs[GlossaryCore.FractionDamageToProductivityValue] * damefrac)
            else:
                estimated_damage_from_climate = gross_output_y * damefrac

        self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromClimate}"] = damage_from_climate
        self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromClimate}"] = estimated_damage_from_climate

    def compute_total_damages(self):
        """Damages are the sum of damages from climate + damges from loss of productivity"""

        self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamages}"] = self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromClimate}"] + self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.EstimatedDamagesFromProductivityLoss}"]
        self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.Damages}"] = self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromClimate}"] + self.outputs[f"{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.DamagesFromProductivityLoss}"]

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
        result_total_gdp_per_group = self.np.array(dict_linear_parameters['a']) * self.years + self.np.array(dict_linear_parameters['b']).reshape(-1, 1)
        gdp_predicted_per_group = result_total_gdp_per_group.T
        # compute percentage of gdp for each group
        percentage_gdp_per_group = gdp_predicted_per_group / gdp_predicted_per_group.sum(axis=1, keepdims=True) * 100
        # compute total based on predicted gdp and on gdp output from model
        total_gdp_per_group = percentage_gdp_per_group * self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.OutputNetOfDamage}"].reshape(-1,1) / 100
        # store data in total gdp
        self.outputs[f"{GlossaryCore.PercentageGDPGroupDFName}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.TotalGDPGroupDFName}:{GlossaryCore.Years}"] = self.years
        for group, percentage_gdp_group, gdp_group in zip(breakdown_countries.keys(), percentage_gdp_per_group.T, total_gdp_per_group.T):
            self.outputs[f'{GlossaryCore.PercentageGDPGroupDFName}:{group}'] = percentage_gdp_group
            self.outputs[f'{GlossaryCore.TotalGDPGroupDFName}:{group}'] = gdp_group

        # get percentage of gdp per country in each group
        mean_percentage_gdp_country = DatabaseWitnessCore.GDPPercentagePerCountry.value
        # Iterate over each row to compute gdp of the country

        self.outputs[f"{GlossaryCore.GDPCountryDFName}:{GlossaryCore.Years}"] = self.years
        for (country_name, country_group, country_mean_percentage) in mean_percentage_gdp_country[["country_name", "group", "mean_percentage"]].values:
            # repeat the years for each country
            # compute GDP for each year using the percentage and GDP Value of the correspondant group
            # and convert T$ to G$
            self.outputs[f"{GlossaryCore.GDPCountryDFName}:{country_name}"] = \
                1e3 * country_mean_percentage * self.outputs[f"{GlossaryCore.TotalGDPGroupDFName}:{country_group}"] / 100

    def compute(self,):
        """
        Compute all models for year range
        """
        self.configure()
        # set gross ouput from input if necessary
        if not self.inputs['assumptions_dict']['compute_gdp']:
            self.set_gross_output()

        # Employment rate and workforce
        self.compute_employment_rate()
        self.compute_workforce()
        self.compute_energy_efficiency()

        self.compute_productivity_growthrate()
        self.compute_productivity()

        self.compute_main_temporal_loop()

        self.compute_consumption()
        self.compute_consumption_pc()

        self.compute_sectors_and_subsectors_gdp()

        self.compute_damage_from_productivity_loss()
        self.compute_damage_from_climate()
        self.compute_total_damages()

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
        """
        conversion_factor = GlossaryCore.conversion_dict[GlossaryCore.EnergyConsumptionDf['unit']][GlossaryCore.SectionEnergyConsumptionDf['unit']]
        for sector in GlossaryCore.SectorsPossibleValues:
            self.outputs[f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}:{GlossaryCore.Years}"] = self.years
            for section in GlossaryCore.SectionDictSectors[sector]:
                self.outputs[f"{sector}.{GlossaryCore.SectionEnergyConsumptionDfValue}:{GlossaryCore.Years}"] = \
                    self.inputs[f'{GlossaryCore.SectorEnergyConsumptionPercentageDfName}_{sector}:{section}'] / 100. *\
                    self.outputs[f"{GlossaryCore.Macroeconomics}.{GlossaryCore.EnergyConsumptionValue}:Total"] * conversion_factor

    def compute_energy_consumption_households(self):
        """Compute energy consumption of households"""
        conversion_factor = GlossaryCore.conversion_dict[GlossaryCore.EnergyConsumptionDf['unit']][GlossaryCore.ResidentialEnergyConsumptionDf['unit']]
        self.outputs[f"{GlossaryCore.ResidentialEnergyConsumptionDfValue}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.ResidentialEnergyConsumptionDfValue}:{GlossaryCore.TotalProductionValue}"] = \
            self.inputs[f"{GlossaryCore.SectorEnergyConsumptionPercentageDfName}:{GlossaryCore.Households}"] * \
            self.outputs[f"{GlossaryCore.Macroeconomics}.{GlossaryCore.EnergyConsumptionValue}:Total"] / 100. * conversion_factor

    def configure(self):
        self.year_start = self.inputs[GlossaryCore.YearStart]
        self.year_end = self.inputs[GlossaryCore.YearEnd]
        self.years = self.np.arange(self.year_start, self.year_end + 1)

    def compute_main_temporal_loop(self):
        # data loading

        energy_efficiency = self.outputs[f"{GlossaryCore.CapitalDfValue}:{GlossaryCore.EnergyEfficiency}"]
        productivity = self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Productivity}"]
        share_non_energy_invest = self.inputs[f"{GlossaryCore.ShareNonEnergyInvestmentsValue}:{GlossaryCore.ShareNonEnergyInvestmentsValue}"]

        energy_market_ratio = self.inputs[f"{GlossaryCore.EnergyMarketRatioAvailabilitiesValue}:Total"]
        workforce = self.outputs[f"{GlossaryCore.WorkforceDfValue}:{GlossaryCore.Workforce}"]
        damefrac = self.inputs[f"{GlossaryCore.DamageFractionDfValue}:{GlossaryCore.DamageFractionOutput}"]

        capital_non_energy_y = self.inputs['capital_start_non_energy']
        capital_non_energy_list = [capital_non_energy_y]

        gross_output_list = []
        net_output_list = []

        energy_demands_list = []
        energy_consumption_list = []
        non_energy_investments_list = []
        usable_capital_list = []

        for (year, energy_efficiency_y, productivity_y, snei_y, energy_market_ratio_y, workforce_y, damage_fraction_y) in \
                zip(self.years, energy_efficiency, productivity, share_non_energy_invest, energy_market_ratio, workforce, damefrac):

            energy_demand_y = capital_non_energy_y * self.inputs['capital_utilisation_ratio'] / energy_efficiency_y # TWh = T$ * / (T$/TWh)
            energy_consumption_y = energy_demand_y * energy_market_ratio_y / 100.
            usable_capital_y = energy_consumption_y * energy_efficiency_y
            gross_output_y = productivity_y * \
            (self.inputs['output_alpha'] * usable_capital_y ** self.inputs['output_gamma'] +
             (1 - self.inputs['output_alpha']) * workforce_y ** self.inputs['output_gamma']) ** (1 / self.inputs['output_gamma'])

            if not self.inputs['assumptions_dict']['compute_climate_impact_on_gdp']:
                output_net_of_d_y = gross_output_y
            else:
                if self.inputs[GlossaryCore.DamageToProductivity]:
                    damage = 1 - ((1 - damage_fraction_y) /
                                  (1 - self.inputs[GlossaryCore.FractionDamageToProductivityValue] * damage_fraction_y))
                    output_net_of_d_y = (1 - damage) * gross_output_y
                else:
                    output_net_of_d_y = gross_output_y * (1 - damage_fraction_y)

            non_energy_investment_y = output_net_of_d_y * snei_y / 100.

            gross_output_list.append(gross_output_y)
            net_output_list.append(output_net_of_d_y)
            energy_demands_list.append(energy_demand_y)
            energy_consumption_list.append(energy_consumption_y)
            non_energy_investments_list.append(non_energy_investment_y)
            usable_capital_list.append(usable_capital_y)

            if year < self.year_end:
                # next year :
                capital_non_energy_y = (1 - self.inputs["depreciation_capital"]) * capital_non_energy_y + non_energy_investment_y
                capital_non_energy_list.append(capital_non_energy_y)

        self.outputs[f"{GlossaryCore.CapitalDfValue}:{GlossaryCore.NonEnergyCapital}"] = self.np.array(capital_non_energy_list)
        self.outputs[f"{GlossaryCore.CapitalDfValue}:{GlossaryCore.Capital}"] = \
            self.outputs[f"{GlossaryCore.CapitalDfValue}:{GlossaryCore.NonEnergyCapital}"] + \
            self.inputs[f"{GlossaryCore.EnergyCapitalDfValue}:{GlossaryCore.Capital}"]
        self.outputs[f"{GlossaryCore.CapitalDfValue}:{GlossaryCore.UsableCapital}"] = self.np.array(usable_capital_list)

        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.GrossOutput}"] = self.np.array(gross_output_list)
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.OutputNetOfDamage}"] = self.np.array(net_output_list)

        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.EnergyInvestmentsValue}"] = self.inputs[f"{GlossaryCore.EnergyInvestmentsWoTaxValue}:{GlossaryCore.EnergyInvestmentsWoTaxValue}"]
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.NonEnergyInvestmentsValue}"] = self.np.array(non_energy_investments_list)
        self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.InvestmentsValue}"] = \
            self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.EnergyInvestmentsValue}"] + \
            self.outputs[f"{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.NonEnergyInvestmentsValue}"]

        self.outputs[f"{GlossaryCore.Macroeconomics}_{GlossaryCore.EnergyDemandValue}:Total"] = self.np.array(energy_demands_list)
        self.outputs[f"{GlossaryCore.Macroeconomics}.{GlossaryCore.EnergyConsumptionValue}:Total"] = self.np.array(energy_consumption_list)

