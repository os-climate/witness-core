'''
mode: python; py-indent-offset: 4; tab-width: 4; coding: utf-8
Copyright (C) 2020 Airbus SAS
'''
import unittest
import time
import numpy as np
import pandas as pd
from sos_trades_core.execution_engine.execution_engine import ExecutionEngine
from climateeconomics.sos_processes.iam.witness.witness_dev.usecase_witness import Study as Study_open


class TestGlobalEnergyValues(unittest.TestCase):
    """
    This test class has the objective to test order of magnitude of some key values in energy models in 2020
    """

    def setUp(self):
        '''
        Initialize third data needed for testing
        '''
        self.dirs_to_del = []
        self.namespace = 'MyCase'
        self.study_name = f'{self.namespace}'
        self.name = 'Test'
        self.energymixname = 'EnergyMix'
        self.ee = ExecutionEngine(self.name)
        repo = 'climateeconomics.sos_processes.iam.witness'
        builder = self.ee.factory.get_builder_from_process(
            repo, 'witness_dev')
        self.ee.factory.set_builders_to_coupling_builder(builder)
        self.ee.configure()
        usecase = Study_open(execution_engine=self.ee)
        usecase.study_name = self.name
        values_dict = usecase.setup_usecase()

        self.ee.display_treeview_nodes()
        full_values_dict = {}
        for dict_v in values_dict:
            full_values_dict.update(dict_v)

        self.ee.load_study_from_input_dict(full_values_dict)

    def test_01_check_global_production_values(self):
        '''
        Test order of magnitude of raw energy production with values from ourworldindata
        https://ourworldindata.org/energy-mix?country=

        '''
        self.ee.execute()

        # These emissions are in Gt
        energy_production = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.energy_production_brut_detailed')

        '''
        Theory in 2019 from ourwolrdindata  expressed in TWh (2020 is a covid year)
        we need to substract energy own use to get same hypthesis than our models (enrgy own_use is substracted from raw production
        '''
        oil_product_production = 49472. - 2485.89
        wind_production = 1590.19  # in 2020
        nuclear_production = 2616.61
        hydropower_production = 4355.
        trad_biomass_production = 13222.
        other_renew_production = 1614.
        modern_biofuels_production = 1043.  # in 2020
        # in 2020
        # https://ourworldindata.org/renewable-energy#solar-energy-generation
        solar_production = 844.37
        coal_production = 43752. - 952.78
        gas_production = 39893. - 3782.83
        total_production = 171240.

        '''
        Oil production
        '''

        computed_oil_production = energy_production['production fuel.liquid_fuel (TWh)'].loc[
            energy_production['years'] == 2020].values[0]

        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_oil_production,
                             oil_product_production * 1.1)
        self.assertGreaterEqual(
            computed_oil_production, oil_product_production * 0.9)

        '''
        Gas production
        '''
        fossil_gas_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.methane.FossilGas.techno_production')
        computed_gas_production = fossil_gas_prod['methane (TWh)'].loc[
            fossil_gas_prod['years'] == 2020].values[0] * 1000.0

        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_gas_production,
                             gas_production * 1.1)
        self.assertGreaterEqual(
            computed_gas_production, gas_production * 0.9)

        '''
        Coal production
        '''
        coal_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.solid_fuel.CoalExtraction.techno_production')
        computed_coal_production = coal_prod['solid_fuel (TWh)'].loc[
            coal_prod['years'] == 2020].values[0] * 1000.0

        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_coal_production,
                             coal_production * 1.1)
        self.assertGreaterEqual(
            computed_coal_production, coal_production * 0.9)

        '''
        Biomass production , the value is traditional biomass consumption , but we know that we do not consume all the biomass that we can produce 
        Waiting for a specific value to compare
        '''
#
        computed_biomass_production = energy_production['production biomass_dry (TWh)'].loc[
            energy_production['years'] == 2020].values[0]

        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_biomass_production,
                             trad_biomass_production * 1.1)
        self.assertGreaterEqual(
            computed_biomass_production, trad_biomass_production * 0.9)

        '''
        Biofuel production
        '''

        computed_biodiesel_production = energy_production['production fuel.biodiesel (TWh)'].loc[
            energy_production['years'] == 2020].values[0]

        computed_biogas_production = energy_production['production biogas (TWh)'].loc[
            energy_production['years'] == 2020].values[0]

        computed_biofuel_production = computed_biodiesel_production + \
            computed_biogas_production
        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_biofuel_production,
                             modern_biofuels_production * 1.1)
        # we compare in TWh and must be near 30% of error because some biofuels
        # are missing
        self.assertGreaterEqual(
            computed_biofuel_production, modern_biofuels_production * 0.7)

        '''
        Solar production
        '''
        elec_solar_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.SolarPv.techno_production')

        elec_solarth_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.SolarThermal.techno_production')

        computed_solar_production = elec_solar_prod['electricity (TWh)'].loc[
            elec_solar_prod['years'] == 2020].values[0] * 1000.0 + \
            elec_solarth_prod['electricity (TWh)'].loc[
            elec_solarth_prod['years'] == 2020].values[0] * 1000.0

        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_solar_production,
                             solar_production * 1.1)
        self.assertGreaterEqual(
            computed_solar_production, solar_production * 0.9)

        '''
        Wind production
        '''
        elec_windonshore_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.WindOnshore.techno_production')
        elec_windoffshore_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.WindOffshore.techno_production')

        computed_wind_production = elec_windonshore_prod['electricity (TWh)'].loc[
            elec_windonshore_prod['years'] == 2020].values[0] * 1000.0 + \
            elec_windoffshore_prod['electricity (TWh)'].loc[
            elec_windoffshore_prod['years'] == 2020].values[0] * 1000.0

        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_wind_production,
                             wind_production * 1.1)
        self.assertGreaterEqual(
            computed_wind_production, wind_production * 0.9)

        '''
        Nuclear production
        '''
        elec_nuclear_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.Nuclear.techno_production')

        computed_nuclear_production = elec_nuclear_prod['electricity (TWh)'].loc[
            elec_nuclear_prod['years'] == 2020].values[0] * 1000.0

        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_nuclear_production,
                             nuclear_production * 1.1)
        self.assertGreaterEqual(
            computed_nuclear_production, nuclear_production * 0.9)

        '''
        Hydropower production
        '''
        elec_hydropower_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.Hydropower.techno_production')

        computed_hydropower_production = elec_hydropower_prod['electricity (TWh)'].loc[
            elec_hydropower_prod['years'] == 2020].values[0] * 1000

        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(computed_hydropower_production,
                             hydropower_production * 1.1)
        self.assertGreaterEqual(
            computed_hydropower_production, hydropower_production * 0.9)

    def test_02_check_global_co2_emissions_values(self):
        '''
        Test order of magnitude of co2 emissions with values from ourworldindata
        https://ourworldindata.org/emissions-by-fuel

        '''
        self.ee.execute()

        # These emissions are in Gt

        co2_emissions_by_energy = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.co2_emissions_by_energy')

        '''
        Theory in 2020 from ourwolrdindata  expressed in Mt
        '''
        oil_co2_emissions = 11.07e3  # expressed in Mt
        coal_co2_emissions = 13.97e3  # expressed in Mt
        gas_co2_emissions = 7.4e3  # expressed in Mt
        total_co2_emissions = 34.81e3  # billions tonnes

        '''
        Methane CO2 emissions are emissions from methane energy + gasturbine from electricity
        '''
        elec_gt_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.GasTurbine.techno_detailed_production')
        elec_cgt_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.CombinedCycleGasTurbine.techno_detailed_production')

        wgs_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.hydrogen.gaseous_hydrogen.WaterGasShift.techno_detailed_production')

        computed_methane_co2_emissions = co2_emissions_by_energy['methane'].loc[co2_emissions_by_energy['years'] == 2020].values[0] + \
            elec_gt_prod['CO2 from Flue Gas (Mt)'].loc[elec_gt_prod['years']
                                                       == 2020].values[0] +\
            elec_cgt_prod['CO2 from Flue Gas (Mt)'].loc[elec_gt_prod['years']
                                                        == 2020].values[0] +\
            wgs_prod['CO2 from Flue Gas (Mt)'].loc[wgs_prod['years']
                                                   == 2020].values[0] * 0.75

        # we compare in Mt and must be near 10% of error
        self.assertLessEqual(computed_methane_co2_emissions,
                             gas_co2_emissions * 1.1)
        self.assertGreaterEqual(
            computed_methane_co2_emissions, gas_co2_emissions * 0.9)

        print(
            f'Methane CO2 emissions : ourworldindata {gas_co2_emissions} Mt vs WITNESS {computed_methane_co2_emissions} TWh')
        '''
        Coal CO2 emissions are emissions from coal energy + CoalGeneration from electricity + SMR + 
        '''
        elec_coal_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.CoalGen.techno_detailed_production')

        computed_coal_co2_emissions = co2_emissions_by_energy['solid_fuel'].loc[co2_emissions_by_energy['years'] == 2020].values[0] + \
            elec_coal_prod['CO2 from Flue Gas (Mt)'].loc[elec_coal_prod['years']
                                                         == 2020].values[0] +\
            wgs_prod['CO2 from Flue Gas (Mt)'].loc[wgs_prod['years']
                                                   == 2020].values[0] * 0.25
        # we compare in Mt and must be near 10% of error
        self.assertLessEqual(computed_coal_co2_emissions,
                             coal_co2_emissions * 1.1)
        self.assertGreaterEqual(
            computed_coal_co2_emissions, coal_co2_emissions * 0.9)

        print(
            f'Coal CO2 emissions : ourworldindata {coal_co2_emissions} Mt vs WITNESS {computed_coal_co2_emissions} TWh')
        '''
        Oil CO2 emissions are emissions from oil energy 
        '''

        computed_oil_co2_emissions = co2_emissions_by_energy['fuel.liquid_fuel'].loc[
            co2_emissions_by_energy['years'] == 2020].values[0]
        # we compare in Mt and must be near 10% of error
        self.assertLessEqual(computed_oil_co2_emissions,
                             oil_co2_emissions * 1.1)
        self.assertGreaterEqual(
            computed_oil_co2_emissions, oil_co2_emissions * 0.9)

        print(
            f'Oil CO2 emissions : ourworldindata {oil_co2_emissions} Mt vs WITNESS {computed_oil_co2_emissions} TWh')
        '''
        Total CO2 emissions are emissions from oil energy 
        '''
        sources = self.ee.dm.get_value(
            'Test.CCUS.CO2_emissions_by_use_sources')
        sinks = self.ee.dm.get_value('Test.CCUS.CO2_emissions_by_use_sinks')[
            'CO2_resource removed by energy mix (Gt)'].values[0]
        sources_sum = sources.loc[sources['years'] == 2020][[
            col for col in sources.columns if col != 'years']].sum(axis=1)[0]
        computed_total_co2_emissions = (sources_sum - sinks) * 1000
        # we compare in Mt and must be near 10% of error

        print(
            f'Total CO2 emissions : ourworldindata {total_co2_emissions} Mt vs WITNESS {computed_total_co2_emissions} TWh')
        self.assertLessEqual(computed_total_co2_emissions,
                             total_co2_emissions * 1.1)
        self.assertGreaterEqual(
            computed_total_co2_emissions, total_co2_emissions * 0.9)

    def test_03_check_net_production_values(self):
        '''
        Test order of magnitude of net energy production with values from Energy Balances IEA 2019

        '''
        self.ee.execute()

        # These emissions are in Gt
        net_energy_production = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.energy_production_detailed')

        energy_production = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.energy_production_brut_detailed')
        '''
        Theory in 2019 from Energy Balances IEA 2019  expressed in TWh 
        '''

        '''
        Coal balances 
        '''
        print('----------  Coal balances -------------')

        coal_energy_own_use = 952.78
        print(
            f'Energy own use for coal production is {coal_energy_own_use} TWh and now taken into account into raw production')
        energy_production_raw_coal_iea = 46666 - coal_energy_own_use  # TWH
        coal_raw_prod = energy_production['production solid_fuel (TWh)'][0]
        error_coalraw_prod = np.abs(
            energy_production_raw_coal_iea - coal_raw_prod) / energy_production_raw_coal_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_coalnet_prod,
#                              10.0)

        print('coal raw production error : ', error_coalraw_prod, ' %',
              f'IEA :{energy_production_raw_coal_iea} TWh vs WITNESS :{coal_raw_prod} TWh')

        # elec plants needs
        elec_plants = self.ee.dm.get_value(f'{self.name}.{self.energymixname}.electricity.energy_consumption')[
            'solid_fuel (TWh)'][0] * 1000.0

        elec_plants_coal_IEA = 20194.44  # TWh

        error_elec_plants = np.abs(
            elec_plants_coal_IEA - elec_plants) / elec_plants_coal_IEA * 100.0
        # we compare in TWh and must be near 10% of error
        self.assertLessEqual(error_elec_plants,
                             10.0)

        print('coal used by electricity plants error : ', error_elec_plants, ' %',
              f'IEA :{elec_plants_coal_IEA} TWh vs WITNESS :{elec_plants} TWh')

        # syngas plants needs
        syngas_plants = self.ee.dm.get_value(f'{self.name}.{self.energymixname}.syngas.energy_consumption')[
            'solid_fuel (TWh)'][0] * 1000.0

        liquefaction_plants_coal_IEA = 264.72  # TWh

        error_syngas_plants = np.abs(
            liquefaction_plants_coal_IEA - syngas_plants) / liquefaction_plants_coal_IEA * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_syngas_plants,
#                              10.0)

        print('coal used by syngas plants error : ', error_syngas_plants, ' %',
              f'IEA :{liquefaction_plants_coal_IEA} TWh vs WITNESS :{syngas_plants} TWh')

        coal_used_by_energy = energy_production[
            'production solid_fuel (TWh)'][0] - net_energy_production[
            'production solid_fuel (TWh)'][0]
        # chp plants and heat plantstechnology not implemented
        chp_plants = 8222.22 + 289  # TWh

        print('CHP and heat plants not implemented corresponds to ',
              chp_plants / coal_used_by_energy * 100.0, ' % of coal used by energy : ', chp_plants, ' TWh')
        # coal to gas technology not implemented
        gas_works = 196.11  # Twh

        coal_total_final_consumption = net_energy_production[
            'production solid_fuel (TWh)'][0]
        print('Coal to gas plants not implemented corresponds to ',
              gas_works / coal_used_by_energy * 100.0, ' % of coal used by energy')
        coal_total_final_consumption = net_energy_production[
            'production solid_fuel (TWh)'][0]
        coal_total_final_consumption_iea = 11055  # TWH

        error_coalnet_prod = np.abs(
            coal_total_final_consumption_iea - coal_total_final_consumption) / coal_total_final_consumption_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_coalnet_prod,
#                              10.0)

        print('coal net production error : ', error_coalnet_prod, ' %',
              f'IEA :{coal_total_final_consumption_iea} TWh vs WITNESS :{coal_total_final_consumption} TWh')
        print('CHP and heat plants not taken into account for coal consumption explains the differences')

        '''
        Gas balances
        '''
        print('----------  Gas balances -------------')

        energy_own_use = 3732.83

        print('Energy industry own use covers the amount of fuels used by the energy producing industries (e.g. for heating, lighting and operation of all equipment used in the extraction process, for traction and for distribution)')
        print(
            f'Energy own use for methane production is {energy_own_use} TWh and now taken into account into raw production')

        energy_production_raw_gas_iea = 40000 - energy_own_use  # TWH
        gas_raw_prod = energy_production['production methane (TWh)'][0]
        error_gasraw_prod = np.abs(
            energy_production_raw_gas_iea - gas_raw_prod) / energy_production_raw_gas_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_coalnet_prod,
#                              10.0)

        print('gas raw production error : ', error_gasraw_prod, ' %',
              f'IEA :{energy_production_raw_gas_iea} TWh vs WITNESS :{gas_raw_prod} TWh')

        # elec plants needs
        elec_plants = self.ee.dm.get_value(f'{self.name}.{self.energymixname}.electricity.energy_consumption')[
            'methane (TWh)'][0] * 1000.0

        elec_plants_gas_IEA = 10833.33  # TWh
        chp_plants_iea = 3887.05 + 709  # TWh
        error_elec_plants = np.abs(
            elec_plants_gas_IEA - elec_plants) / elec_plants_gas_IEA * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_elec_plants,
#                              10.0)

        print('gas used by electricity plants error : ',
              error_elec_plants, ' %',
              f'IEA :{elec_plants_gas_IEA } TWh vs WITNESS :{elec_plants} TWh')

        methane_used_by_energy = energy_production[
            'production methane (TWh)'][0] - net_energy_production[
            'production methane (TWh)'][0]
        print('CHP and heat plants not implemented corresponds to ',
              chp_plants_iea / methane_used_by_energy * 100.0, ' % of methane used by energy : ', chp_plants_iea, ' TWh')
        # syngas plants needs
        syngas_plants = self.ee.dm.get_value(f'{self.name}.{self.energymixname}.syngas.energy_consumption')[
            'methane (TWh)'][0] * 1000.0

        liquefaction_plants_methane_IEA = 202.74  # TWh
        other_transformation = 277.5  # TWH
        # other transformaton includes  the transformation of natural gas for
        # hydrogen manufacture

        error_syngas_plants = np.abs(
            liquefaction_plants_methane_IEA + other_transformation - syngas_plants) / liquefaction_plants_methane_IEA * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_syngas_plants,
#                              10.0)

        print('methane used by syngas plants error : ',
              error_syngas_plants, ' %',
              f'IEA :{liquefaction_plants_methane_IEA + other_transformation} TWh vs WITNESS :{syngas_plants} TWh')

        methane_total_final_consumption = net_energy_production[
            'production methane (TWh)'][0]
        methane_total_final_consumption_iea = 19001  # TWH

        error_methanenet_prod = np.abs(
            methane_total_final_consumption_iea - methane_total_final_consumption) / methane_total_final_consumption_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_coalnet_prod,
#                              10.0)
        print('methane net production error : ', error_methanenet_prod, ' %',
              f'IEA :{methane_total_final_consumption_iea} TWh vs WITNESS :{methane_total_final_consumption} TWh')
        print('CHP and heat plants not taken into account for methane consumption explains some differences')
        '''
        Electricity balances
        '''
        print('----------  Electricity balances -------------')

        net_elec_prod = net_energy_production[
            'production electricity (TWh)'][0]

        net_elec_prod_iea = 22847.66  # TWh

        error_net_elec_prod = np.abs(
            net_elec_prod_iea - net_elec_prod) / net_elec_prod_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_coalnet_prod,
#                              10.0)

        print('Net electricity production error : ', error_net_elec_prod, ' %',
              f'IEA :{net_elec_prod_iea} TWh vs WITNESS :{net_elec_prod} TWh')

        energy_production_raw_hydro_iea = 4222.22  # TWH
        elec_hydropower_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.Hydropower.techno_production')

        computed_hydropower_production = elec_hydropower_prod['electricity (TWh)'].loc[
            elec_hydropower_prod['years'] == 2020].values[0] * 1000

        error_hydropowerraw_prod = np.abs(
            energy_production_raw_hydro_iea - computed_hydropower_production) / energy_production_raw_hydro_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_coalnet_prod,
#                              10.0)

        print('hydropower raw production error : ', error_hydropowerraw_prod, ' %',
              f'IEA :{energy_production_raw_hydro_iea} TWh vs WITNESS :{computed_hydropower_production} TWh')

        energy_production_raw_wind_iea = 1427.41  # TWH

        elec_windonshore_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.WindOnshore.techno_production')
        elec_windoffshore_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.WindOffshore.techno_production')

        computed_wind_production = elec_windonshore_prod['electricity (TWh)'].loc[
            elec_windonshore_prod['years'] == 2020].values[0] * 1000.0 + \
            elec_windoffshore_prod['electricity (TWh)'].loc[
            elec_windoffshore_prod['years'] == 2020].values[0] * 1000.0

        error_wind_prod = np.abs(
            energy_production_raw_wind_iea - computed_wind_production) / energy_production_raw_wind_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_wind_prod,
#                              10.0)

        print('Wind raw production error : ', error_wind_prod, ' %',
              f'IEA :{energy_production_raw_wind_iea} TWh vs WITNESS :{computed_wind_production} TWh')

        elec_solar_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.SolarPv.techno_production')

        computed_solarpv_production = elec_solar_prod['electricity (TWh)'].loc[
            elec_solar_prod['years'] == 2020].values[0] * 1000
        energy_production_solarpv_iea = 680.9  # TWh
        error_solarpv_prod = np.abs(
            energy_production_solarpv_iea - computed_solarpv_production) / energy_production_solarpv_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_solarpv_prod,
#                              10.0)

        print('Solar PV raw production error : ', error_solarpv_prod, ' %',
              f'IEA :{energy_production_solarpv_iea} TWh vs WITNESS :{computed_solarpv_production} TWh')

        elec_solarth_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.SolarThermal.techno_production')

        computed_solarth_production = elec_solarth_prod['electricity (TWh)'].loc[
            elec_solarth_prod['years'] == 2020].values[0] * 1000
        energy_production_solarth_iea = 13.36  # TWh
        error_solarth_prod = np.abs(
            energy_production_solarth_iea - computed_solarth_production) / energy_production_solarth_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_solarpv_prod,
#                              10.0)

        print('Solar Thermal raw production error : ', error_solarth_prod, ' %',
              f'IEA :{energy_production_solarth_iea} TWh vs WITNESS :{computed_solarth_production} TWh')

        elec_geoth_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.Geothermal.techno_production')

        computed_geoth_production = elec_geoth_prod['electricity (TWh)'].loc[
            elec_geoth_prod['years'] == 2020].values[0] * 1000.0

        energy_production_geoth_iea = 91.09  # TWh
        error_geoth_prod = np.abs(
            energy_production_geoth_iea - computed_geoth_production) / energy_production_geoth_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_solarpv_prod,
#                              10.0)

        print('Geothermal raw production error : ', error_geoth_prod, ' %',
              f'IEA :{energy_production_geoth_iea} TWh vs WITNESS :{computed_geoth_production} TWh')

        elec_coalgen_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.CoalGen.techno_production')

        computed_coalgen_production = elec_coalgen_prod['electricity (TWh)'].loc[
            elec_coalgen_prod['years'] == 2020].values[0] * 1000.0

        energy_production_coalgen_iea = 9914.45  # TWh
        error_geoth_prod = np.abs(
            energy_production_coalgen_iea - computed_coalgen_production) / energy_production_coalgen_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_solarpv_prod,
#                              10.0)

        print('Coal generation raw production error : ', error_geoth_prod, ' %',
              f'IEA :{energy_production_coalgen_iea} TWh vs WITNESS :{computed_coalgen_production} TWh')

        elec_oilgen_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.OilGen.techno_production')
        computed_oilgen_production = elec_oilgen_prod['electricity (TWh)'].loc[
            elec_oilgen_prod['years'] == 2020].values[0] * 1000.0
        energy_production_oilgen_iea = 747  # TWh
        error_oil_prod = np.abs(
            energy_production_oilgen_iea - computed_oilgen_production) / energy_production_oilgen_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_solarpv_prod,
#                              10.0)

        print('Oil generation raw production error : ', error_oil_prod, ' %',
              f'IEA :{energy_production_oilgen_iea} TWh vs WITNESS :{computed_oilgen_production} TWh')

        elec_gt_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.GasTurbine.techno_production')
        elec_cgt_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.CombinedCycleGasTurbine.techno_production')

        computed_gasgen_production = elec_gt_prod['electricity (TWh)'].loc[
            elec_gt_prod['years'] == 2020].values[0] * 1000.0 + elec_cgt_prod['electricity (TWh)'].loc[
            elec_cgt_prod['years'] == 2020].values[0] * 1000.0

        energy_production_gasgen_iea = 6346  # TWh
        error_gasgen_prod = np.abs(
            energy_production_gasgen_iea - computed_gasgen_production) / energy_production_gasgen_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_solarpv_prod,
#                              10.0)

        print('Gas generation raw production error : ', error_gasgen_prod, ' %',
              f'IEA :{energy_production_gasgen_iea} TWh vs WITNESS :{computed_gasgen_production} TWh')

        elec_nuclear_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.Nuclear.techno_production')

        computed_nuclear_production = elec_nuclear_prod['electricity (TWh)'].loc[
            elec_nuclear_prod['years'] == 2020].values[0] * 1000.0

        energy_production_nuclear_iea = 2789.69  # TWh
        error_geoth_prod = np.abs(
            energy_production_nuclear_iea - computed_nuclear_production) / energy_production_nuclear_iea * 100.0
        # we compare in TWh and must be near 10% of error
#         self.assertLessEqual(error_solarpv_prod,
#                              10.0)

        print('Nuclear raw production error : ', error_geoth_prod, ' %',
              f'IEA :{energy_production_nuclear_iea} TWh vs WITNESS :{computed_nuclear_production} TWh')

        energy_production_oilgen_iea = 747  # TWh
        energy_production_biofuelgen_iea = 542.56  # TWh

        print(
            f'Technologies of electricity generation with oil ({energy_production_oilgen_iea} TWh) and biofuel ({energy_production_biofuelgen_iea} TWh) are not yet implemented')
        '''
        Biofuels and waste balances
        '''
        print('----------  Biomass dry balances -------------')

        print('We consider biomass_dry equals to the sum of primary solid biofuels (no municipal/industiral waste) but in the doc they do not consider crop residues')
        biomass_dry_raw_prod_iea = (
            48309940) / 3600  # TWh 1414648 + 1142420 +
        biomass_dry_net_prod_iea = (36537355) / 3600  # TWh + 150882 + 519300

#         managed_wood_prod = self.ee.dm.get_value(
#             f'{self.name}.{self.energymixname}.biomass_dry.ManagedWood.techno_production')
#
#         computed_managed_wood_prod = managed_wood_prod['biomass_dry (TWh)'].loc[
#             managed_wood_prod['years'] == 2020].values[0] * 1000.0
#
#         unmanaged_wood_prod = self.ee.dm.get_value(
#             f'{self.name}.{self.energymixname}.biomass_dry.UnmanagedWood.techno_production')
#
#         computed_unmanaged_wood_prod = unmanaged_wood_prod['biomass_dry (TWh)'].loc[
#             unmanaged_wood_prod['years'] == 2020].values[0] * 1000.0
#
#         crop_energy_prod = self.ee.dm.get_value(
#             f'{self.name}.{self.energymixname}.biomass_dry.CropEnergy.techno_production')
#
#         computed_crop_energy_prod = crop_energy_prod['biomass_dry (TWh)'].loc[
#             crop_energy_prod['years'] == 2020].values[0] * 1000.0
#
        biomass_dry_net_prod = net_energy_production[
            'production biomass_dry (TWh)'][0]  # - computed_crop_energy_prod
#
        biomass_dry_raw_prod = energy_production[
            'production biomass_dry (TWh)'][0]

        error_biomassdry_raw_prod = np.abs(
            biomass_dry_raw_prod_iea - biomass_dry_raw_prod) / biomass_dry_raw_prod_iea * 100.0

        print('Biomass dry raw production error : ', error_biomassdry_raw_prod, ' %',
              f'IEA :{biomass_dry_raw_prod_iea} TWh vs WITNESS :{biomass_dry_raw_prod} TWh')

        error_biomassdry_net_prod = np.abs(
            biomass_dry_net_prod_iea - biomass_dry_net_prod) / biomass_dry_net_prod_iea * 100.0

        print('Biomass dry net production error : ', error_biomassdry_net_prod, ' %',
              f'IEA :{biomass_dry_net_prod_iea} TWh vs WITNESS :{biomass_dry_net_prod} TWh')
#
#         biomass_dry_elec_plants = 3650996 / 3600  # TWh
#         biomass_dry_chp_plants = (2226110 + 324143) / 3600  # TWh
#         biomass_dry_otherrtransf = 5220384 / 3600  # TWh
#
#         print('CHP and heat plants using biomass are not implemented corresponds to ',
#               biomass_dry_chp_plants / biomass_dry_raw_prod_iea * 100.0, ' % of biomass raw production : ', biomass_dry_chp_plants, ' TWh')
#         print('Electricity plants using biomass are not implemented corresponds to ',
#               biomass_dry_elec_plants / biomass_dry_raw_prod_iea * 100.0, ' % of biomass raw production : ', biomass_dry_elec_plants, ' TWh')
#
#         biogas_cons = self.ee.dm.get_value(
#             f'{self.name}.{self.energymixname}.biogas.energy_consumption')
#
#         biomass_by_biogas_cons = biogas_cons['wet_biomass (Mt)'].loc[
#             biogas_cons['years'] == 2020].values[0] * 1000 * 3.6  # 3.6 is calorific value of biomass_dry
#
#         syngas_cons = self.ee.dm.get_value(
#             f'{self.name}.{self.energymixname}.solid_fuel.energy_consumption')
#
#         biomass_by_syngas_cons = syngas_cons['biomass_dry (TWh)'].loc[
#             syngas_cons['years'] == 2020].values[0] * 1000
#
#         solid_fuel_cons = self.ee.dm.get_value(
#             f'{self.name}.{self.energymixname}.solid_fuel.energy_consumption')
#
#         biomass_by_solid_fuel_cons = solid_fuel_cons['biomass_dry (TWh)'].loc[
#             solid_fuel_cons['years'] == 2020].values[0] * 1000
#
#         biomass_dry_otherrtransf_witness = biomass_by_solid_fuel_cons + biomass_by_syngas_cons
#         biomass_dry_otherrtransf_with_ana = biomass_by_biogas_cons + \
#             biomass_dry_otherrtransf_witness
#
#         error_biomassdry_otherrtransf_prod = np.abs(
#             biomass_dry_otherrtransf - biomass_dry_otherrtransf_witness) / biomass_dry_otherrtransf * 100.0
#
#         print('Biomass dry other transformation production error : ', error_biomassdry_otherrtransf_prod, ' %',
#               f'IEA :{biomass_dry_otherrtransf} TWh vs WITNESS :{biomass_dry_otherrtransf_witness} TWh')
#
#         error_biomassdry_otherrtransf_with_ana_prod = np.abs(
#             biomass_dry_otherrtransf - biomass_dry_otherrtransf_with_ana) / biomass_dry_otherrtransf * 100.0
#
#         print('Biomass dry other transformation (adding anaerobic digestion) production error : ', error_biomassdry_otherrtransf_with_ana_prod, ' %',
# f'IEA :{biomass_dry_otherrtransf} TWh vs WITNESS with anaerobic
# digestion :{biomass_dry_otherrtransf_with_ana} TWh')

        print('----------  liquid biofuels balances -------------')

        print('IEA biofuels includes bioethanol (ethanol produced from biomass), biomethanol (methanol produced from biomass), bioETBE (ethyl-tertio-butyl-ether produced on the basis of bioethanol) and bioMTBE (methyl-tertio-butyl-ether produced on the basis of biomethanol')
        print('and biodiesel (a methyl-ester produced from vegetable or animal oil, of diesel quality), biodimethylether (dimethylether produced from biomass), Fischer Tropsch (Fischer Tropsch produced from biomass), cold pressed bio-oil (oil produced from oil seed through mechanical processing only) ')

        raw_biodiesel_prod = energy_production[
            'production fuel.biodiesel (TWh)'][0]
        raw_hydrotreated_oil_fuel_prod = energy_production[
            'production fuel.hydrotreated_oil_fuel (TWh)'][0]

        raw_liquid_fuel = raw_biodiesel_prod + \
            raw_hydrotreated_oil_fuel_prod
        liquidbiofuels_raw_prod_iea = 131224 * 1e6 * 11.9 / 1e9  # in kt

        error_liquid_fuel_raw_prod = np.abs(
            liquidbiofuels_raw_prod_iea - raw_liquid_fuel) / liquidbiofuels_raw_prod_iea * 100.0

        print('Liquid fuels raw production error : ', error_liquid_fuel_raw_prod, ' %',
              f'IEA :{liquidbiofuels_raw_prod_iea} TWh vs WITNESS :{raw_liquid_fuel} TWh')
        print(
            'A lot of biofuels are not implemented (no details of specific biofuels productions ')
        print('----------  Biogases balances -------------')

        print('In IEA, biogas are mainly gases from the anaerobic digestion but also can be produced from thermal processes (pyrolysis) or from syngas')
        print('WITNESS model considers only anaerobic digestion')

        raw_biogas_prod = energy_production[
            'production biogas (TWh)'][0]
        biogas_raw_prod_iea = 1434008 / 3600

        error_biogas_raw_prod = np.abs(
            biogas_raw_prod_iea - raw_biogas_prod) / biogas_raw_prod_iea * 100.0

        print('Biogas raw production error : ', error_biogas_raw_prod, ' %',
              f'IEA :{biogas_raw_prod_iea} TWh vs WITNESS :{raw_biogas_prod} TWh')

        print(
            f'Biogas is used in energy industry mainly for electricity plants {448717/3600} TWh and CHP plants {385127/3600} TWh')
        print('These technologies are not yet implemented in WITNESS models, then :')
        biogas_net_prod_iea = 521188 / 3600
        net_biogas_prod = net_energy_production[
            'production biogas (TWh)'][0]
        error_biogas_net_prod = np.abs(
            biogas_net_prod_iea - net_biogas_prod) / biogas_net_prod_iea * 100.0

        print('Biogas net production error : ', error_biogas_net_prod, ' %',
              f'IEA :{biogas_net_prod_iea} TWh vs WITNESS :{net_biogas_prod} TWh')

        '''
        Oil balances 
        '''
        print('----------  Oil balances -------------')

        iea_data_oil = {'kerosene': (14082582 + 2176724) / 3600,
                        # gasoline + diesel
                        'gasoline': (41878252 + 56524612) / 3600,
                        #'diesel': 56524612 / 3600,
                        #'naphtas' :11916946/3600,
                        'heating_oil': 16475667 / 3600,  # equivalent to fuel oil
                        #'other_oil_products' :25409482/3600,
                        'liquefied_petroleum_gas': 5672984 / 3600,  # LPG/ethane
                        'fuel.liquid_fuel': 190442343 / 3600  # total of crude oil
                        }

        raw_refinery_prod = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.fuel.liquid_fuel.Refinery.techno_production')

        raw_refinery_prod_2020 = raw_refinery_prod.loc[
            raw_refinery_prod['years'] == 2020] * 1000.0

        for oil_name, oil_prod in iea_data_oil.items():

            oil_prod_witness = raw_refinery_prod_2020[
                f'{oil_name} (TWh)'].values[0]
            error_oil_prod = np.abs(
                oil_prod - oil_prod_witness) / oil_prod * 100.0

            print(f'{oil_name} raw production error : ', error_oil_prod, ' %',
                  f'IEA :{oil_prod} TWh vs WITNESS :{oil_prod_witness} TWh')

        print(
            'WITNESS model only takes for now raw liquid_fuel production which is correct')

        net_liquid_fuel_prod = net_energy_production[
            'production fuel.liquid_fuel (TWh)'][0]
        liquid_fuel_net_prod_iea = 168375005 / 3600

        error_liquid_fuel_net_prod = np.abs(
            liquid_fuel_net_prod_iea - net_liquid_fuel_prod) / liquid_fuel_net_prod_iea * 100.0

        print('Liquid fuel net production error : ', error_liquid_fuel_net_prod, ' %',
              f'IEA :{liquid_fuel_net_prod_iea} TWh vs WITNESS :{net_liquid_fuel_prod} TWh')

        liquid_fuel_own_use = 2485.89  # TWH
        liquid_fuel_raw_prod = raw_refinery_prod_2020[
            f'fuel.liquid_fuel (TWh)'].values[0]
        energy_production_raw_liquidfuel_iea = 52900 - liquid_fuel_own_use
        print(
            f'Energy own use for liquid fuel production is {liquid_fuel_own_use} TWh')

        print('Liquid fuel raw production error : ', error_liquid_fuel_net_prod, ' %',
              f'IEA :{energy_production_raw_liquidfuel_iea} TWh vs WITNESS :{liquid_fuel_raw_prod} TWh')
        chp_plants = 159.62 + 99.81  # TWh

        print('CHP and heat plants not implemented corresponds to ',
              chp_plants / liquid_fuel_raw_prod * 100.0, ' % of total raw liquid fuel production : ', chp_plants, ' TWh')

        oil_elec_plants = 1591.67  # TWh

        # elec plants needs
        elec_plants_oil = self.ee.dm.get_value(f'{self.name}.{self.energymixname}.electricity.energy_consumption')[
            'fuel.liquid_fuel (TWh)'][0] * 1000.0

        error_oil_cons = np.abs(
            oil_elec_plants - elec_plants_oil) / oil_elec_plants * 100.0

        print('Liquid fuel consumption from elec error : ', error_oil_cons, ' %',
              f'IEA :{oil_elec_plants} TWh vs WITNESS :{elec_plants_oil} TWh')

        print('----------------- Total production -------------------')

        total_raw_prod_iea = 173340  # TWh
        total_raw_prod = energy_production['Total production'][0]
        error_total_raw_prod = np.abs(
            total_raw_prod_iea - total_raw_prod) / total_raw_prod_iea * 100.0

        print('Total raw production error : ', error_total_raw_prod, ' %',
              f'IEA :{total_raw_prod_iea} TWh vs WITNESS :{total_raw_prod} TWh')

        total_net_prod_iea = 116103  # TWh
        total_net_prod = net_energy_production['Total production'][0]
        error_total_net_prod = np.abs(
            total_net_prod_iea - total_net_prod) / total_net_prod_iea * 100.0

        print('Total net production error : ', error_total_net_prod, ' %',
              f'IEA :{total_net_prod_iea} TWh vs WITNESS :{total_net_prod} TWh')

    def test_04_check_prices_values(self):
        '''
        Test order of magnitude of prices

        '''
        self.ee.execute()

        # These emissions are in Gt
        energy_prices = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.energy_prices')

        energy_prices_after_tax = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.energy_prices_after_tax')
        '''
        Energy prices
        '''
        print('Comparison of prices coming from globalpetrolprices.com')

        elec_price_iea = 137  # $/MWh
        elec_price = energy_prices[
            'electricity'][0]

        error_elec_price = np.abs(
            elec_price_iea - elec_price) / elec_price_iea * 100.0

        print('Electricity price error in 2021: ', error_elec_price, ' %',
              f'globalpetrolprices.com :{elec_price_iea} $/MWh vs WITNESS :{elec_price} $/MWh')

        ng_price_iea_2022 = 1.17 / 0.657e-3 / 13.9  # $/MWh
        ng_price_iea_2021 = 0.8 / 0.657e-3 / 13.9  # $/MWh
        ng_price = energy_prices[
            'methane'][0]

        error_ng_price = np.abs(
            ng_price_iea_2021 - ng_price) / ng_price_iea_2021 * 100.0

        print('Natural Gas/Methane price error in 2021 : ', error_ng_price, ' %',
              f'globalpetrolprices.com :{ng_price_iea_2021} $/MWh vs WITNESS :{ng_price} $/MWh')

        kerosene_price_iea = 0.92 / 0.0095  # $/MWh in 2022
        kerosene_price_iea_2021 = 2.8 / 39.5 * 1000  # $/MWh in 2021
        kerosene_price = energy_prices[
            'fuel.liquid_fuel'][0]

        error_kerosene_price = np.abs(
            kerosene_price_iea_2021 - kerosene_price) / kerosene_price_iea_2021 * 100.0

        print('kerosene price error in 2021 : ', error_kerosene_price, ' %',
              f'globalpetrolprices.com :{kerosene_price_iea_2021} $/MWh vs WITNESS :{kerosene_price} $/MWh')

        print('hydrogen prices details have been found on IEA website :https://www.iea.org/data-and-statistics/charts/global-average-levelised-cost-of-hydrogen-production-by-energy-source-and-technology-2019-and-2050 ')

        hydrogen_prices = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.hydrogen.gaseous_hydrogen.energy_detailed_techno_prices')
        smr_price_iea = 1.6 / 33.3 * 1000  # between 0.7 and 1.6 $/kg mean : 1.15 $/kg
        # between 1.9 and 2.5 $/kg mean : 2.2 $/kg
        coal_gas_price_iea = 2.5 / 33.3 * 1000
        wgs_price = hydrogen_prices[
            'WaterGasShift'][0]
        wgs_price_iea = 0.75 * smr_price_iea + 0.25 * coal_gas_price_iea
        error_wgs_price = np.abs(
            wgs_price_iea - wgs_price) / wgs_price_iea * 100.0

        print('Hydrogen price by watergas shift (coal and gas) error in 2021: ', error_wgs_price, ' %',
              f'IEA :{wgs_price_iea} $/MWh vs WITNESS :{wgs_price} $/MWh')

        electrolysis_price_iea = 7.7 / 33.3 * 1000  # between 3.2 and 7.7 $/kg

        electrolysis_price = hydrogen_prices[
            'Electrolysis.SOEC'][0]
        error_electrolysis_price = np.abs(
            electrolysis_price_iea - electrolysis_price) / electrolysis_price_iea * 100.0

        print('Hydrogen price by Electrolysis error in 2021: ', error_electrolysis_price, ' %',
              f'IEA :{electrolysis_price_iea} $/MWh vs WITNESS :{electrolysis_price} $/MWh')

        biogas_price_gazpack = 30 / 0.293  # 30 $/mbtu

        biogas_price = energy_prices[
            'biogas'][0]

        error_biogas_price = np.abs(
            biogas_price_gazpack - biogas_price) / biogas_price_gazpack * 100.0

        print('Biogas price error in 2019: ', error_biogas_price, ' %',
              f'gazpack.nl/ :{biogas_price_gazpack} $/MWh vs WITNESS :{biogas_price} $/MWh')
        # between 50 and 100 $ /tonne
        coal_price_ourworldindata = 50 * 1e-3 / 4.86 * 1e3

        coal_price = energy_prices[
            'solid_fuel'][0]

        error_coal_price = np.abs(
            coal_price_ourworldindata - coal_price) / coal_price_ourworldindata * 100.0

        print('Coal price error in 2021: ', error_coal_price, ' %',
              f'ourworldindata.com :{coal_price_ourworldindata} $/MWh vs WITNESS :{coal_price} $/MWh')

        biodiesel_price_neste = 1500 / 10.42

        biodiesel_price = energy_prices[
            'fuel.biodiesel'][0]

        error_biodiesel_price = np.abs(
            biodiesel_price_neste - biodiesel_price) / biodiesel_price_neste * 100.0

        print('Biodiesel price error in 2021: ', error_biodiesel_price, ' %',
              f'neste.com :{biodiesel_price_neste} $/MWh vs WITNESS :{biodiesel_price} $/MWh')

        biomass_price_statista = 35 / 3.6

        biomass_price = energy_prices[
            'biomass_dry'][0]

        error_biomass_price = np.abs(
            biomass_price_statista - biomass_price) / biomass_price_statista * 100.0

        print('Biomass price error in 2021: ', error_biomass_price, ' %',
              f'US statista.com :{biomass_price_statista} $/MWh vs WITNESS :{biomass_price} $/MWh')

        hefa_price_iea = 1.2 / 780e-3 / 12.2 * 1000

        hefa_price = energy_prices[
            'fuel.hydrotreated_oil_fuel'][0]

        error_hefa_price = np.abs(
            hefa_price_iea - hefa_price) / hefa_price_iea * 100.0

        print('HEFA price error in 2020: ', error_hefa_price, ' %',
              f'IEA :{hefa_price_iea} $/MWh vs WITNESS :{hefa_price} $/MWh')

        print('------------- Electricity prices --------------')

        elec_detailed_prices = self.ee.dm.get_value(
            f'{self.name}.{self.energymixname}.electricity.energy_detailed_techno_prices')

        elec_detailed_prices['Nuclear'].values[0]


if '__main__' == __name__:
    t0 = time.time()
    cls = TestGlobalEnergyValues()
    cls.setUp()
    cls.test_03_check_net_production_values()
    print(f'Time : {time.time() - t0} s')
