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

import autograd.numpy as np
from energy_models.core.stream_type.energy_models.biomass_dry import BiomassDry
from energy_models.glossaryenergy import GlossaryEnergy

from climateeconomics.core.tools.differentiable_model import DifferentiableModel
from climateeconomics.glossarycore import GlossaryCore


class ForestAutodiff(DifferentiableModel):
    """
    Forest model class 
    basic for now, to evolve 

    """

    def compute(self):
        """Computation methods"""

        self.initialize_years()
        self.compute_yields()
        self.compute_managed_wood_surface()
        self.compute_reforestation_deforestation_surface()
        self.compute_capital_loss()
        self.compute_deforestation_biomass()
        self.compute_managed_wood_production()
        self.sumup_global_surface_data()
        self.compute_global_CO2_production()
        self.compute_biomass_dry_production()

        self.compute_land_use_required()


        self.compute_forest_constraint_evolution()
        self.compute_production_for_energy()
        self.compute_price_in_d_per_mwh()
        self.compute_carbon_emissions()
        self.compute_carbon_consumption()

        self.compute_economical_output_and_damages()
        self.rescale_techno_production_and_consumption()
        self.compute_coupling_dfs()

        return self.get_dataframes()

    def compute_managed_wood_surface(self):
        """
        compute managed wood delta and cumulative surface from investments
        Will be recalculate with deforestation impact in compute_reforestation_deforestation_surface method
        """
        construction_delay = self.inputs['params'][GlossaryCore.ConstructionDelay]
        mw_cost = self.inputs['params']['managed_wood_price_per_ha']
        # managed wood from past invest. invest in G$ - surface in Gha.
        mw_from_past_invest = self.inputs[f'managed_wood_invest_before_year_start:{GlossaryCore.InvestmentsValue}'] / mw_cost
        # managed wood from actual invest
        mw_from_invest = self.inputs[f'managed_wood_investment:{GlossaryCore.InvestmentsValue}'] / mw_cost
        # concat all managed wood form invest
        managed_wood_yearly_surface_variation = np.concatenate([mw_from_past_invest, mw_from_invest])
        # remove value that exceed year_end
        managed_wood_yearly_surface_variation = managed_wood_yearly_surface_variation[:-construction_delay]
        self.outputs['managed_wood_df:delta_surface'] = managed_wood_yearly_surface_variation
        managed_wood_total_surface = self.inputs['managed_wood_initial_surface'] + np.cumsum(managed_wood_yearly_surface_variation)
        self.outputs['managed_wood_df:cumulative_surface'] = managed_wood_total_surface

    def compute_managed_wood_production(self):
        """
        compute data concerning managed wood : surface taken, production, CO2 absorbed, as delta and cumulative
        """

        # Biomass production part
        # Gha * m3/ha
        cubic_meter_production_wo_climate_change = (self.inputs['managed_wood_initial_surface'] * self.inputs['params']['actual_yield_year_start'] +
                                  (self.outputs['managed_wood_df:cumulative_surface'] - self.inputs['managed_wood_initial_surface'])
                                  * self.inputs['params']['managed_wood_yield_year_start'])
        cubic_meter_production = (self.inputs['managed_wood_initial_surface'] * self.outputs['yields:actual'] +
                                  (self.outputs['managed_wood_df:cumulative_surface'] - self.inputs['managed_wood_initial_surface'])
                                  * self.outputs['yields:managed wood'])
        wasted_production =  cubic_meter_production_wo_climate_change - cubic_meter_production
        self.outputs['managed_wood_df:delta_wood_production (Mt)'] = self.outputs['managed_wood_df:delta_surface'] * self.outputs['yields:managed wood'] * (
                                                                     1 - self.inputs['params']['residues_density_percentage']) * self.inputs['params']['wood_density']
        # Gm3* kg/m3 => Mt
        self.outputs['managed_wood_df:wood_production (Mt)'] = cubic_meter_production * (1 - self.inputs['params']['residues_density_percentage']) * self.inputs['params']['wood_density']
        self.outputs['managed_wood_df:wasted_wood_production (Mt)'] = wasted_production * (1 - self.inputs['params']['residues_density_percentage']) * self.inputs['params']['wood_density']

        self.outputs['managed_wood_df:wood_production_for_energy (Mt)'] = self.outputs['managed_wood_df:wood_production (Mt)'] * self.inputs['params']['wood_percentage_for_energy']
        self.outputs['managed_wood_df:wood_production_for_industry (Mt)'] = self.outputs['managed_wood_df:wood_production (Mt)'] * (1 - self.inputs['params']['wood_percentage_for_energy'])

        self.outputs['managed_wood_df:delta_residues_production (Mt)'] = self.outputs['managed_wood_df:delta_surface'] * self.outputs['yields:managed wood'] * self.inputs['params']['residues_density_percentage'] * self.inputs['params']['residues_density']
        self.outputs['managed_wood_df:residues_production (Mt)'] = cubic_meter_production * self.inputs['params']['residues_density_percentage'] * self.inputs['params']['residues_density']
        self.outputs['managed_wood_df:wasted_residues_production (Mt)'] = wasted_production * self.inputs['params']['residues_density_percentage'] * self.inputs['params']['residues_density']
        self.outputs['managed_wood_df:residues_production_for_energy (Mt)'] = self.outputs['managed_wood_df:residues_production (Mt)'] * \
                                                                              self.inputs['params']['wood_percentage_for_energy']
        self.outputs['managed_wood_df:residues_production_for_industry (Mt)'] = self.outputs['managed_wood_df:residues_production (Mt)'] * \
                                                                                (1 - self.inputs['params']['wood_percentage_for_energy'])

        self.outputs['managed_wood_df:delta_biomass_production (Mt)'] = self.outputs['managed_wood_df:delta_wood_production (Mt)'] + \
                                                                        self.outputs['managed_wood_df:delta_residues_production (Mt)']
        self.outputs['managed_wood_df:biomass_production (Mt)'] = self.outputs['managed_wood_df:wood_production (Mt)'] + \
                                                                  self.outputs['managed_wood_df:residues_production (Mt)']

        # CO2 part
        self.outputs['managed_wood_df:delta_CO2_emitted'] = - \
                                                        self.outputs['managed_wood_df:delta_surface'] * self.inputs['params']['CO2_per_ha'] / 1000
        # CO2 emitted is delta cumulate
        self.outputs['managed_wood_df:CO2_emitted'] = - \
                                                  (self.outputs['managed_wood_df:cumulative_surface'] -
                                                   self.inputs['managed_wood_initial_surface']) * self.inputs['params']['CO2_per_ha'] / 1000

    def compute_reforestation_deforestation_surface(self):
        """
        compute land use and due to reforestation et deforestation activities
        CO2 is not computed here because surface limit need to be taken into account before.
        """

        # forest surface is in Gha, deforestation_surface is in Mha,
        # deforested_surface is in Gha
        self.outputs['forest_surface_detail_df:delta_deforestation_surface'] = - self.inputs[f'deforestation_investment:{GlossaryCore.InvestmentsValue}'] / self.inputs['params']['deforestation_cost_per_ha']

        # forested surface
        # invest in G$, coest_per_ha in $/ha --> Gha
        self.outputs['forest_surface_detail_df:delta_reforestation_surface'] = self.inputs['reforestation_investment:reforestation_investment'] / self.inputs['params']['reforestation_cost_per_ha']

        self.outputs['forest_surface_detail_df:deforestation_surface'] = np.cumsum(self.outputs['forest_surface_detail_df:delta_deforestation_surface'])
        self.outputs['forest_surface_detail_df:reforestation_surface'] = np.cumsum(self.outputs['forest_surface_detail_df:delta_reforestation_surface'])

        delta_unmanaged_forest_surface = self.outputs['forest_surface_detail_df:delta_reforestation_surface'] + self.outputs['forest_surface_detail_df:delta_deforestation_surface']
        self.outputs['forest_surface_detail_df:unmanaged_forest'] = np.maximum(self.inputs['initial_unmanaged_forest_surface'] + np.cumsum(delta_unmanaged_forest_surface), 0)


    def compute_deforestation_biomass(self):
        """
        compute biomass produce by deforestation. It is a one time production.
        We use actual yield because deforestation is done on actual forest not managed ones
        """

        deforestation_production_wo_climate_change = - self.outputs['forest_surface_detail_df:delta_deforestation_surface'] * self.inputs['params']['actual_yield_year_start'] * self.inputs['params']['wood_density']
        deforestation_production = - self.outputs['forest_surface_detail_df:delta_deforestation_surface'] * self.outputs['yields:unmanaged wood'] * self.inputs['params']['wood_density']
        self.outputs['biomass_dry_detail_df:deforestation (Mt)'] = deforestation_production
        self.outputs['biomass_dry_detail_df:deforestation_wasted (Mt)'] = deforestation_production_wo_climate_change - deforestation_production

        self.outputs['biomass_dry_detail_df:deforestation_for_energy'] = self.outputs['biomass_dry_detail_df:deforestation (Mt)'] * \
                                                                  self.inputs['params']['wood_percentage_for_energy']
        self.outputs['biomass_dry_detail_df:deforestation_for_industry'] = self.outputs['biomass_dry_detail_df:deforestation (Mt)'] - \
                                                                    self.outputs['biomass_dry_detail_df:deforestation_for_energy']
        self.outputs['biomass_dry_detail_df:deforestation_price_per_ton'] = self.outputs['yields:unmanaged wood'] * self.inputs['params']['wood_density'] / self.inputs['params']['deforestation_cost_per_ha']
        self.outputs['biomass_dry_detail_df:deforestation_price_per_MWh'] = self.outputs['biomass_dry_detail_df:deforestation_price_per_ton'] / \
                                                                     self.inputs['params']['biomass_dry_calorific_value']

    def sumup_global_surface_data(self):
        """
        managed wood and unmanaged wood impact forest_surface_detail_df
        """
        self.outputs['forest_surface_detail_df:delta_global_forest_surface'] = self.outputs['forest_surface_detail_df:delta_reforestation_surface'] + \
                                                                        self.outputs['forest_surface_detail_df:delta_deforestation_surface']
        self.outputs['forest_surface_detail_df:global_forest_surface'] = self.outputs['managed_wood_df:cumulative_surface'] + \
                                                                  self.outputs['forest_surface_detail_df:unmanaged_forest'] + \
                                                                  self.inputs['initial_protected_forest_surface']
        self.outputs['forest_surface_detail_df:protected_forest_surface'] = self.zeros_array + self.inputs['initial_protected_forest_surface']

    def compute_global_CO2_production(self):
        """
        compute the global CO2 production in Gt
        """
        # in Gt of CO2

        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:delta_CO2_emitted'] = -self.outputs['forest_surface_detail_df:delta_global_forest_surface'] * \
                                                           self.inputs['params']['CO2_per_ha'] / 1000
        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:delta_CO2_deforestation'] = -self.outputs['forest_surface_detail_df:delta_deforestation_surface'] * \
                                                                 self.inputs['params']['CO2_per_ha'] / 1000
        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:delta_CO2_reforestation'] = -self.outputs['forest_surface_detail_df:delta_reforestation_surface'] * \
                                                                 self.inputs['params']['CO2_per_ha'] / 1000

        # remove CO2 managed surface from global emission because CO2_per_ha
        # from managed forest = 0
        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:CO2_deforestation'] = - self.outputs['forest_surface_detail_df:deforestation_surface'] * \
                                                           self.inputs['params']['CO2_per_ha'] / 1000
        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:CO2_reforestation'] = -self.outputs['forest_surface_detail_df:reforestation_surface'] * \
                                                           self.inputs['params']['CO2_per_ha'] / 1000
        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:initial_CO2_land_use_change'] = self.zeros_array + self.inputs['initial_co2_emissions']
        # global sum up
        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:global_CO2_emitted'] = self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:CO2_deforestation'] + \
                                                            self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:initial_CO2_land_use_change']
        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:global_CO2_captured'] = self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:CO2_reforestation']
        self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:emitted_CO2_evol_cumulative'] = self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:global_CO2_emitted'] + \
                                                                     self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:global_CO2_captured']

        self.outputs[f'CO2_land_emission_df:{GlossaryCore.Years}'] = self.years
        self.outputs['CO2_land_emission_df:emitted_CO2_evol_cumulative'] = self.outputs[f'{GlossaryCore.CO2EmissionsDetailDfValue}:emitted_CO2_evol_cumulative']



    def compute_biomass_dry_production(self):
        """
        compute total biomass dry prod
        """

        self.outputs['biomass_dry_detail_df:biomass_dry_for_energy (Mt)'] = self.outputs['managed_wood_df:wood_production_for_energy (Mt)'] + \
                                                                     self.outputs['managed_wood_df:residues_production_for_energy (Mt)'] + \
                                                                     self.outputs['biomass_dry_detail_df:deforestation_for_energy']

        self.compute_price()

        self.managed_wood_part = self.outputs['managed_wood_df:biomass_production (Mt)'] / (
                self.outputs['managed_wood_df:biomass_production (Mt)'] + self.outputs['biomass_dry_detail_df:deforestation (Mt)'])
        self.deforestation_part = self.outputs['biomass_dry_detail_df:deforestation (Mt)'] / (
                self.outputs['managed_wood_df:biomass_production (Mt)'] + self.outputs['biomass_dry_detail_df:deforestation (Mt)'])

        self.outputs['biomass_dry_detail_df:price_per_ton'] = self.outputs['biomass_dry_detail_df:managed_wood_price_per_ton'] * self.managed_wood_part + \
                                                       self.outputs['biomass_dry_detail_df:deforestation_price_per_ton'] * \
                                                       self.deforestation_part

        self.outputs['biomass_dry_detail_df:managed_wood_price_per_MWh'] = self.outputs['biomass_dry_detail_df:managed_wood_price_per_ton'] / \
                                                                    self.inputs['params']['biomass_dry_calorific_value']
        self.outputs['biomass_dry_detail_df:price_per_MWh'] = self.outputs['biomass_dry_detail_df:price_per_ton'] / \
                                                       self.inputs['params']['biomass_dry_calorific_value']

    def compute_price(self):
        """
        compute price as in techno_type
        """

        crf = self.compute_capital_recovery_factor()

        self.outputs['biomass_dry_detail_df:managed_wood_transport ($/t)'] = self.inputs['transport_cost:transport']

        # Factory cost including CAPEX OPEX
        # $/ha * ha/m3 * m3/kg * 1000 = $/t
        mean_density = (1 - self.inputs['params']['residues_density_percentage']) * self.inputs['params']['wood_density'] +\
                       self.inputs['params']['residues_density_percentage'] * self.inputs['params']['residues_density']

        self.outputs['biomass_dry_detail_df:managed_wood_capex ($/t)'] = \
            self.inputs['params']['managed_wood_price_per_ha'] * (crf + 0.045) / \
            self.outputs['yields:managed wood'] / mean_density * 1000

        self.outputs['biomass_dry_detail_df:managed_wood_price_per_ton'] = \
            (self.outputs['biomass_dry_detail_df:managed_wood_capex ($/t)'] +
             self.outputs['biomass_dry_detail_df:managed_wood_transport ($/t)']) * \
            self.inputs['margin:margin'] / 100.0

    def compute_capital_recovery_factor(self):
        """
        Compute annuity factor with the Weighted averaged cost of capital
        and the lifetime of the selected solution
        """
        wacc = self.inputs['params']['WACC']
        crf = (wacc * (1.0 + wacc) ** 100) / \
              ((1.0 + wacc) ** 100 - 1.0)

        return crf

    def compute_carbon_emissions(self):
        '''
        Compute the carbon emissions from the technology taking into account 
        CO2 from production + CO2 from primary resources 
        '''
        # CO2 emissions
        if 'CO2_from_production' not in self.inputs['params']:
            self.outputs['CO2_emissions:production'] = self.zeros_array + self.get_theoretical_co2_prod(unit='kg/kWh')
        elif self.inputs['params']['CO2_from_production'] == 0.0:
            self.outputs['CO2_emissions:production'] = self.zeros_array + 0.0
        else:
            if self.inputs['params']['CO2_from_production_unit'] == 'kg/kg':
                self.outputs['CO2_emissions:production'] = self.zeros_array + self.inputs['params']['CO2_from_production'] / \
                                                           self.inputs['params']['biomass_dry_high_calorific_value']
            elif self.inputs['params']['CO2_from_production_unit'] == 'kg/kWh':
                self.outputs['CO2_emissions:production'] = self.zeros_array + self.inputs['params']['CO2_from_production']

        # Add carbon emission from input energies (resources or other
        # energies)

        co2_emissions_frominput_energies = self.compute_CO2_emissions_from_input_resources()

        # Add CO2 from production + C02 from input energies
        self.outputs['CO2_emissions:Forest'] = self.outputs['CO2_emissions:production'] + \
                                               co2_emissions_frominput_energies

    def get_theoretical_co2_prod(self, unit='kg/kWh'):
        '''
        Get the theoretical CO2 production for a given technology,
        '''
        return 0.0

    def compute_CO2_emissions_from_input_resources(self):
        '''
        Need to take into account  CO2 from electricity/fuel production
        '''
        return 0.0

    def compute_production_for_energy(self):
        # techno production in TWh
        self.outputs[f'techno_production:{BiomassDry.name} ({BiomassDry.unit})'] = (self.outputs['managed_wood_df:wood_production_for_energy (Mt)'] +
                                                                                    self.outputs['biomass_dry_detail_df:deforestation_for_energy']) * self.inputs['params']['biomass_dry_calorific_value'] + self.outputs['managed_wood_df:residues_production_for_energy (Mt)'] * self.inputs['params']['residue_calorific_value']

    def compute_carbon_consumption(self):
        # CO2 consumed
        self.outputs[f'techno_consumption:{GlossaryEnergy.carbon_capture} (Mt)'] = \
            -self.inputs['params']['CO2_from_production'] / self.inputs['params']['biomass_dry_high_calorific_value'] * \
            self.outputs[f'techno_production:{BiomassDry.name} ({BiomassDry.unit})']

        self.outputs[f'techno_consumption_woratio:{GlossaryEnergy.carbon_capture} (Mt)'] = \
            - self.inputs['params']['CO2_from_production'] / self.inputs['params']['biomass_dry_high_calorific_value'] * \
            self.outputs[f'techno_production:{BiomassDry.name} ({BiomassDry.unit})']

    def compute_forest_constraint_evolution(self):
        # compute forest constrain evolution: reforestation + deforestation
        self.outputs['forest_surface_detail_df:forest_constraint_evolution'] = self.outputs['forest_surface_detail_df:reforestation_surface'] + \
                                                                        self.outputs['forest_surface_detail_df:deforestation_surface']

    def compute_price_in_d_per_mwh(self):
        # Prices in $/MWh
        self.outputs['techno_prices:Forest'] = self.outputs['biomass_dry_detail_df:price_per_MWh']
        self.outputs['techno_prices:Forest_wotaxes'] = self.outputs['biomass_dry_detail_df:price_per_MWh']

    def compute_economical_output_and_damages(self):
        """
        Net output = Net production * Prices
        Damages = Wasted production * Prices
        Gross output = Net output + Damages
        """
        # Forest net output breakdown
        # Mt * $/ton = M$ -> so divide by 1e3 to get G$
        self.outputs[f'{GlossaryCore.EconomicsDetailDfValue}:{GlossaryCore.Years}'] = self.years
        self.outputs[f'{GlossaryCore.EconomicsDetailDfValue}:Deforestation'] = self.outputs['biomass_dry_detail_df:deforestation (Mt)'] * self.outputs['biomass_dry_detail_df:deforestation_price_per_ton'] / 1e3
        self.outputs[f'{GlossaryCore.EconomicsDetailDfValue}:Managed wood'] = self.outputs['managed_wood_df:wood_production (Mt)'] * self.outputs['biomass_dry_detail_df:managed_wood_price_per_ton'] / 1e3
        self.outputs[f'{GlossaryCore.EconomicsDetailDfValue}:Residues'] = self.outputs['managed_wood_df:residues_production (Mt)'] * self.outputs['biomass_dry_detail_df:managed_wood_price_per_ton'] / 1e3

        net_output = self.outputs[f'{GlossaryCore.EconomicsDetailDfValue}:Deforestation'] + \
                     self.outputs[f'{GlossaryCore.EconomicsDetailDfValue}:Managed wood'] + \
                     self.outputs[f'{GlossaryCore.EconomicsDetailDfValue}:Residues']

        # Forest economical damages detail
        # Mt * $/ton = M$ -> so divide by 1e3 to get G$
        self.outputs[f'{GlossaryCore.DamageDetailedDfValue}:{GlossaryCore.Years}'] = self.years
        self.outputs[f'{GlossaryCore.DamageDetailedDfValue}:Deforestation'] = self.outputs['biomass_dry_detail_df:deforestation_wasted (Mt)'] * self.outputs['biomass_dry_detail_df:deforestation_price_per_ton'] / 1e3
        self.outputs[f'{GlossaryCore.DamageDetailedDfValue}:Managed wood'] = self.outputs['managed_wood_df:wasted_wood_production (Mt)'] * self.outputs['biomass_dry_detail_df:managed_wood_price_per_ton'] / 1e3
        self.outputs[f'{GlossaryCore.DamageDetailedDfValue}:Residues'] = self.outputs['managed_wood_df:wasted_residues_production (Mt)'] * self.outputs['biomass_dry_detail_df:managed_wood_price_per_ton'] / 1e3

        total_damages = self.outputs[f'{GlossaryCore.DamageDetailedDfValue}:Deforestation'] + \
                        self.outputs[f'{GlossaryCore.DamageDetailedDfValue}:Managed wood'] + \
                        self.outputs[f'{GlossaryCore.DamageDetailedDfValue}:Residues']

        # Forest economical output coupling variable
        self.outputs[f"{GlossaryCore.Forest}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.Forest}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.OutputNetOfDamage}"] = net_output
        self.outputs[f"{GlossaryCore.Forest}.{GlossaryCore.ProductionDfValue}:{GlossaryCore.GrossOutput}"] = net_output + total_damages

        # Forest economical damages coupling variable
        self.outputs[f"{GlossaryCore.Forest}.{GlossaryCore.DamageDfValue}:{GlossaryCore.Years}"] = self.years
        self.outputs[f"{GlossaryCore.Forest}.{GlossaryCore.DamageDfValue}:{GlossaryCore.Damages}"] = total_damages


    def compute_yields(self):
        """yields are impact by climate change"""
        self.outputs[f'yields:{GlossaryCore.Years}'] = self.years
        self.outputs['yields:actual'] = self.inputs['params']['actual_yield_year_start'] * (1 - self.inputs[f'{GlossaryCore.CropProductivityReductionName}:{GlossaryCore.CropProductivityReductionName}'] / 100)
        self.outputs['yields:managed wood'] = self.inputs['params']['managed_wood_yield_year_start'] * (1 - self.inputs[f'{GlossaryCore.CropProductivityReductionName}:{GlossaryCore.CropProductivityReductionName}'] / 100)
        self.outputs['yields:unmanaged wood'] = self.inputs['params']['unmanaged_wood_yield_year_start'] * (1 - self.inputs[f'{GlossaryCore.CropProductivityReductionName}:{GlossaryCore.CropProductivityReductionName}'] / 100)

    def initialize_years(self):
        self.years = np.arange(self.inputs[GlossaryCore.YearStart], self.inputs[GlossaryCore.YearEnd] + 1)
        self.zeros_array = self.years * 0.
        self.outputs[f'forest_surface_detail_df:{GlossaryCore.Years}'] = self.years
        self.outputs[f'managed_wood_df:{GlossaryCore.Years}'] = self.years
        self.outputs[f'biomass_dry_detail_df:{GlossaryCore.Years}'] = self.years

        # output dataframes:
        self.outputs[f'techno_production:{GlossaryCore.Years}'] = self.years
        self.outputs[f'techno_prices:{GlossaryCore.Years}'] = self.years
        self.outputs[f'techno_consumption:{GlossaryCore.Years}'] = self.years
        self.outputs[f'techno_consumption_woratio:{GlossaryCore.Years}'] = self.years
        self.outputs[f'land_use_required:{GlossaryCore.Years}'] = self.years
        self.outputs[f'CO2_emissions:{GlossaryCore.Years}'] = self.years
        self.outputs[f'forest_lost_capital:{GlossaryCore.Years}'] = self.years

    def compute_land_use_required(self):
        # compute land_use for energy
        self.outputs[f'land_use_required:{GlossaryCore.Years}'] = self.years
        self.outputs['land_use_required:Forest (Gha)'] = self.outputs['managed_wood_df:cumulative_surface']

    def rescale_techno_production_and_consumption(self):
        self.outputs[f'techno_production:{BiomassDry.name} ({BiomassDry.unit})'] /= self.inputs['scaling_factor_techno_production']
        self.outputs[f'techno_consumption:{GlossaryEnergy.carbon_capture} (Mt)'] /= self.inputs['scaling_factor_techno_consumption']
        self.outputs[f'techno_consumption_woratio:{GlossaryEnergy.carbon_capture} (Mt)'] /= self.inputs['scaling_factor_techno_consumption']

    def compute_coupling_dfs(self):
        self.outputs[f'biomass_dry_df:{GlossaryCore.Years}'] = self.years
        self.outputs['biomass_dry_df:price_per_MWh'] = self.outputs['biomass_dry_detail_df:price_per_MWh']
        self.outputs['biomass_dry_df:biomass_dry_for_energy (Mt)'] = self.outputs['biomass_dry_detail_df:biomass_dry_for_energy (Mt)']

        self.outputs[f'forest_surface_df:{GlossaryCore.Years}'] = self.years
        self.outputs['forest_surface_df:global_forest_surface'] = self.outputs['forest_surface_detail_df:global_forest_surface']
        self.outputs['forest_surface_df:forest_constraint_evolution'] = self.outputs['forest_surface_detail_df:forest_constraint_evolution']

    def compute_capital_loss(self):
        self.outputs[f'forest_lost_capital:{GlossaryCore.Years}'] = self.years
        self.outputs['forest_lost_capital:deforestation'] = - self.outputs['forest_surface_detail_df:delta_deforestation_surface'] * self.inputs['params']['reforestation_cost_per_ha']

        
