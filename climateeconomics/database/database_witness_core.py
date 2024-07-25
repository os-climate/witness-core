'''
Copyright 2023 Capgemini

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
import json
from datetime import date
from os.path import dirname, join

import pandas as pd

from climateeconomics.database.collected_data import ColectedData, HeavyCollectedData

data_folder = join(dirname(dirname(__file__)), "data")


class DatabaseWitnessCore:
    '''Stocke les valeurs utilisées dans witness core'''

    FoodWastePercentage = ColectedData(
        value=30,
        unit="%",
        description="Share of food production that is not consumed (wasted). From harvest to retail: 13%, Retail and conso:17% ",
        link="https://www.fao.org/platform-food-loss-waste/flw-data/en",
        source="Food and Agriculture Organization of the United Nations",
        last_update_date=date(2024, 2, 28)
    )

    ENSOTemperatureAnomaly = ColectedData(
        value=+0.25,
        unit="°C",
        description="Global temperature anomaly due to El Nino - Southern Oscillation phenomenon",
        link="https://berkeleyearth.org/global-temperature-report-for-2023/#:~:text=Annual%20Temperature%20Anomaly&text=As%20a%20result%2C%202023%20is,C%20(2.7%20%C2%B0F).",
        source="BerkleyEarth",
        last_update_date=date(2024, 2, 27)
    )

    TemperatureAnomalyPreIndustrialYearStart = HeavyCollectedData(
        value=join(data_folder, "temp_anomaly_pre_industrial.csv"),
        unit="°C",
        description="Global average temperature anomaly historic (2010-2023) relative to 1850-1900 average",
        link="https://berkeleyearth.org/global-temperature-report-for-2023/#:~:text=Annual%20Temperature%20Anomaly&text=As%20a%20result%2C%202023%20is,C%20(2.7%20%C2%B0F).",
        source="BerkleyEarth",
        last_update_date=date(2024, 2, 27),
        critical_at_year_start=True,
        column_to_pick="Warming"
    )

    # Data for sectorization
    InvestInduspercofgdp2020 = ColectedData(
        value=5.831,
        unit="%",
        description="Investment in Industry sector as percentage of GDP for year start (2020)",
        link="",
        source="Computed from World bank,IMF and IEA data",
        last_update_date=date(2023, 10, 23),
    )

    InvestServicespercofgdpYearStart = ColectedData(
        value=19.231,
        unit="%",
        description="Investment in Services sector as percentage of GDP for year start (2020)",
        link="",
        source="Computed from World bank and IMF data",
        last_update_date=date(2023, 10, 23),
    )

    InvestAgriculturepercofgdpYearStart = ColectedData(
        value=0.4531,
        unit="%",
        description="Investment in Agriculture sector as percentage of GDP for year start (2020)",
        link="",
        source="Computed from World bank, IMF, and FAO gross capital formation data",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareAgricultureYearStart = ColectedData(
        value=2.1360,
        unit="%",
        description="Share of net energy production dedicated to Agriculture sector in % for year start (2020)",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareIndustryYearStart = ColectedData(
        value=28.9442,
        unit="%",
        description="Share of net energy production dedicated to Industry sector in % for year start (2020)",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareServicesYearStart = ColectedData(
        value=36.9954,
        unit="%",
        description="Share of net energy production dedicated to Services sector in % for year start (2020)",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareResidentialYearStart = ColectedData(
        value=21.00,
        unit="%",
        description="Share of net energy production dedicated to Residential in % for year start (2020)",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareOtherYearStart = ColectedData(
        value=10.9230,
        unit="%",
        description="Share of net energy production dedicated to other consumption in % for year start (2020)",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    ShareGlobalGDPAgriculture2021 = ColectedData(
        value=4.30,
        unit="%",
        description="Share of global GDP coming from the agriculture sector [%]",
        link="https://ourworldindata.org/grapher/agriculture-share-gdp?tab=table",
        source="World Bank and OECD",
        last_update_date=date(2023, 12, 1),
    )

    ShareGlobalGDPIndustry2021 = ColectedData(
        value=27.59,
        unit="%",
        description="Share of global GDP coming from the industry sector [%]",
        link="https://www.statista.com/statistics/256563/share-of-economic-sectors-in-the-global-gross-domestic-product/#:~:text=In%202021%2C%20agriculture%20contributed%204.3,the%20global%20gross%20domestic%20product.",
        source="World Bank",
        last_update_date=date(2023, 12, 1),
    )

    ShareGlobalGDPServices2021 = ColectedData(
        value=64.43,
        unit="%",
        description="Share of global GDP coming from the services sector [%]",
        link="https://www.statista.com/statistics/256563/share-of-economic-sectors-in-the-global-gross-domestic-product/#:~:text=In%202021%2C%20agriculture%20contributed%204.3,the%20global%20gross%20domestic%20product.",
        source="World Bank",
        last_update_date=date(2023, 12, 1),
    )

    WorldGDPForecastSSP3 = HeavyCollectedData(
        value=join(data_folder, "economics_df_ssp3.csv"),
        unit="G$",
        description="World GDP from 2020 to 2100 according to SSP3 scenario",
        link="",
        source="IPCC",
        last_update_date=date(2023, 12, 1),
    )  # todo : complete information

    WorldPopulationForecast = HeavyCollectedData(
        value=join(data_folder, "population_df.csv"),
        unit="Million people",
        description="World population forecast GDP from 2020 to 2100",
        link="",
        source="",
        last_update_date=date(2023, 12, 1),
    )  # todo : complete information

    CO2PreIndustrialConcentration = ColectedData(
        value=280,
        unit="ppm",
        description="Concentration of CO2 in atmosphere for pre-industrial era",
        link="https://archive.ipcc.ch/ipccreports/tar/wg1/016.htm",
        source="IPCC",
        last_update_date=date(2024, 1, 12),
    )

    CH4PreIndustrialConcentration = ColectedData(
        value=700,
        unit="ppb",
        description="Concentration of CH4 in atmosphere for pre-industrial era",
        link="https://archive.ipcc.ch/ipccreports/tar/wg1/016.htm",
        source="IPCC",
        last_update_date=date(2024, 1, 12),
    )

    N2OPreIndustrialConcentration = ColectedData(
        value=270,
        unit="ppb",
        description="Concentration of N2O in atmosphere for pre-industrial era",
        link="https://archive.ipcc.ch/ipccreports/tar/wg1/016.htm",
        source="IPCC",
        last_update_date=date(2023, 12, 1),
    )

    HistoricCO2Concentration = HeavyCollectedData(
        value=join(data_folder, "co2_annmean_mlo.csv"),
        unit="PPM",
        description="Concentration of CO2 in atmosphere from 1961 to 2023",
        link="https://gml.noaa.gov/ccgg/trends/data.html",
        source="Earth System Research Laboratorie; Global Monitoring Laboratory",
        last_update_date=date(2023, 1, 18),
        column_to_pick="mean"
    )

    HistoricCH4Concentration = HeavyCollectedData(
        value=join(data_folder, "ch4_annmean_gl.csv"),
        unit="PPB",
        description="Concentration of CH4 in atmosphere from 1984 to 2022",
        link="https://gml.noaa.gov/ccgg/trends/data.html",
        source="Earth System Research Laboratorie; Global Monitoring Laboratory",
        last_update_date=date(2023, 1, 18),
        column_to_pick="mean"
    )

    HistoricN2OConcentration = HeavyCollectedData(
        value=join(data_folder, "n2o_annmean_gl.csv"),
        unit="PPB",
        description="Concentration of N2O in atmosphere from 1984 to 2022",
        link="https://gml.noaa.gov/ccgg/trends/data.html",
        source="Earth System Research Laboratorie; Global Monitoring Laboratory",
        last_update_date=date(2023, 1, 18),
        column_to_pick="mean",
        critical_at_year_start=True
    )

    CumulativeCO2Emissions = ColectedData(
        value=1772.8,
        unit="Gt",
        description="Running sum of CO2 emissions produced from fossil fuels and industry since the first year of recording 1750 until 2022, measured in Giga tonnes. Land-use change is not included ",
        link="https://ourworldindata.org/grapher/cumulative-co-emissions?tab=table",
        source="Our World in Data",
        last_update_date=date(2024, 2, 7),
    )

    LinearParemetersGDPperRegion = ColectedData(
        value={'a': [[1588.20633202],
                     [1959.02654051],
                     [365.116917],
                     [310.57653261],
                     [326.54836759],
                     [164.98219368]],
               'b': [-3149086.04111858, -3913390.07132115, -727002.61494862,
                     -616562.71581522, -648579.38856917, -328849.45974308]},
        unit="$",
        description="Linear parameters for the equation y=ax+b for each region to compute GDP share percentage for each region",
        link="Check macroeconomics documentation and jupyter notebook",
        source="",
        last_update_date=date(2024, 3, 18)
    )

    # read json for countries per region
    with open(join(dirname(dirname(__file__)) , 'data', 'countries_per_region.json'), 'r') as fp:
        countries_per_region = json.load(fp)

    CountriesPerRegionIMF = ColectedData(
        value=countries_per_region,
        unit="",
        description="breakdown of countries according to IMF",
        link="https://www.imf.org/en/Publications/WEO/weo-database/2023/April/groups-and-aggregates",
        source="World Economic Outlook : International Monetary Fund",
        last_update_date=date(2024, 3, 18)
    )

    gdp_percentage_per_country = pd.read_csv(join(dirname(dirname(__file__)) , 'data', 'mean_gdp_country_percentage_in_group.csv'))
    GDPPercentagePerCountry = ColectedData(
        value=gdp_percentage_per_country,
        unit="%",
        description="mean percentage GDP of each country in the group",
        link="",
        source="mean percentages were computed based on official GDP data from international organizations and on the IMF grouping",
        last_update_date=date(2024, 3, 18)
    )
    energy_consumption_services = pd.read_csv(join(dirname(dirname(__file__)) , 'data', 'energy_consumption_percentage_services_sections.csv'))
    energy_consumption_agriculture = pd.read_csv(join(dirname(dirname(__file__)) , 'data', 'energy_consumption_percentage_agriculture_sections.csv'))
    energy_consumption_industry = pd.read_csv(join(dirname(dirname(__file__)) , 'data', 'energy_consumption_percentage_industry_sections.csv'))
    energy_consumption_household = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'energy_consumption_percentage_household_sections.csv'))


    EnergyConsumptionPercentageSectionsDict = ColectedData(
        value= {"Agriculture": energy_consumption_agriculture,
                 "Services": energy_consumption_services,
                 "Industry": energy_consumption_industry,
                "Household": energy_consumption_household},
        unit="%",
        description="energy consumption of each section for all sectors",
        link="",
        source="", # multiples sources TODO
        last_update_date=date(2024,3,26)
    )

    non_energy_emissions_services = pd.read_csv(join(dirname(dirname(__file__)) , 'data', 'non_energy_emission_gdp_services_sections.csv'))
    non_energy_emissions_agriculture = pd.read_csv(join(dirname(dirname(__file__)) , 'data', 'non_energy_emission_gdp_agriculture_sections.csv'))
    non_energy_emissions_industry = pd.read_csv(join(dirname(dirname(__file__)) , 'data', 'non_energy_emission_gdp_industry_sections.csv'))
    non_energy_emissions_household = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'non_energy_emission_gdp_household_sections.csv'))

    SectionsNonEnergyEmissionsDict = ColectedData(
        value={"Agriculture": non_energy_emissions_agriculture,
                 "Services": non_energy_emissions_services,
                 "Industry": non_energy_emissions_industry,
               "Household": non_energy_emissions_household},
        unit="tCO2eq/M$",
        description="Non energy CO2 emission per $GDP",
        link="",
        source="", # multiples sources TODO
        last_update_date=date(2024,3,26)
    )

    energy_consumption_per_sector = pd.read_csv(join(dirname(dirname(__file__)) , 'data', 'energy_consumption_percentage_per_sector.csv'))

    EnergyConsumptionPercentageSectorDict = ColectedData(
        value=energy_consumption_per_sector,
        unit="%",
        description="energy consumption of each sector",
        link="",
        source="", # multiples sources TODO
        last_update_date=date(2024,3,26)
    )

    atmosphere_total_mass_kg = 5.1480 * 10 ** 18
    molar_mass_atmosphere = 0.02897  # kg/mol
    n_moles_in_atmosphere = atmosphere_total_mass_kg / molar_mass_atmosphere
    kg_to_gt = 10 ** (-12)
    molar_mass_co2, molar_mass_ch4, molar_mass_n2o = 0.04401, 0.016_04, 0.044_013  # kg/mol

    pp_to_gt = {
        "CO2": n_moles_in_atmosphere * molar_mass_co2 * kg_to_gt * 10 ** -6,  # ppm
        "CH4": n_moles_in_atmosphere * molar_mass_ch4 * kg_to_gt * 10 ** -6 * 1e-3,  # ppb
        "N2O": n_moles_in_atmosphere * molar_mass_n2o * kg_to_gt * 10 ** -6 * 1e-3,  # ppb
    }
    del atmosphere_total_mass_kg, molar_mass_atmosphere, n_moles_in_atmosphere, kg_to_gt, molar_mass_co2, molar_mass_ch4, molar_mass_n2o

    PandemicParamsDf = HeavyCollectedData(
        value=join(data_folder, "pandemic_param.csv"),
        unit="%",
        description="Pandemic Mortality Rate by Age & Rate of Excess Mortality by Age",
        link="?",
        source="?",
        last_update_date=date(2023, 3, 1),
    )

    ProductionCropForEnergy = HeavyCollectedData(
        value=join(data_folder, "crop_energy_historic_prod.csv"),
        unit="TWh",
        description="Production of crop for energy (2010-2020)",
        link="https://www.iea.org/articles/what-does-net-zero-emissions-by-2050-mean-for-bioenergy-and-land-use",
        source="IEA Global bioenergy supply in the Net Zero Scenario, 2010-2050", # considered as crop for our model : Conventional bioenergy crops, short-rotation woody crops. So we sum them and then convert exajoules (EJ) to TWh
        last_update_date=date(2024, 7, 24),
        critical_at_year_start=True,
        column_to_pick="prod"
    )

    CropInvestmentNZE = HeavyCollectedData(
        value=join(data_folder, "crop_investment_nze.csv"),
        unit="G$",
        description="Investment obtained based on the expected production in NZE scenario, production values are in the link below",
        link="https://www.iea.org/articles/what-does-net-zero-emissions-by-2050-mean-for-bioenergy-and-land-use",
        source="Investments are obtained manualy based on production of IEA Global bioenergy supply in the Net Zero Scenario, 2010-2050",
        last_update_date=date(2024, 4, 24)
    )

    InvestFossilYearStart = HeavyCollectedData(
        value=join(data_folder, "fossil_energy_historic_invests.csv"),
        unit="G$",
        description="Investment in fossil energy 2015-2023",
        link="https://www.iea.org/data-and-statistics/charts/global-energy-investment-in-clean-energy-and-in-fossil-fuels-2015-2023",
        source="World energy investment - IEA",
        last_update_date=date(2024, 7, 23),
        critical_at_year_start=True,
        column_to_pick="Fossil fuels"
    )

    InvestCleanEnergyYearStart = HeavyCollectedData(
        value=join(data_folder, "renewable_energy_historic_invests.csv"),
        unit="G$",
        description="Investment in clean energy 2015-2023",
        link="https://www.iea.org/data-and-statistics/charts/global-energy-investment-in-clean-energy-and-in-fossil-fuels-2015-2023",
        source="World energy investment - IEA",
        last_update_date=date(2024, 7, 23),
        critical_at_year_start=True,
        column_to_pick="Clean energy"
    )

    InvestCCUSYearStart = HeavyCollectedData(
        value=join(data_folder, "ccus_historic_invests.csv"),
        unit="G$",
        description="Investment in all CCUS in G US$. \n Data is mocked until 2020 but invest are so low it wont change anything ",
        link=["https://iea.blob.core.windows.net/assets/181b48b4-323f-454d-96fb-0bb1889d96a9/CCUS_in_clean_energy_transitions.pdf",
              "https://www.mckinsey.com/industries/oil-and-gas/our-insights/global-energy-perspective-2023-ccus-outlook",
              "https://about.bnef.com/blog/ccus-market-outlook-2023-announced-capacity-soars-by-50/"],
        source="",
        last_update_date=date(2024, 7, 23),
        critical_at_year_start=True,
        column_to_pick="invests"
    )