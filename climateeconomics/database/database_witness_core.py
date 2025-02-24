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
import datetime
import json
import os
from datetime import date
from os.path import dirname, join

import pandas as pd

from climateeconomics.database.collected_data import ColectedData, HeavyCollectedData

data_folder = join(dirname(dirname(__file__)), "data")


class DatabaseWitnessCore:
    '''Stocke les valeurs utilisées dans witness core'''

    ShareInvestNonEnergy = ColectedData(
        value=21.5,
        unit="%",
        description="Share of the GDP that is invest in other sectors than energy sector",
        link="https://data.imf.org/?sk=1ce8a55f-cfa7-4bc0-bce2-256ee65ac0e4",
        source="International Monetary Fund",
        last_update_date=date(2024, 9, 25)
    )

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
        description="Global average temperature anomaly historic wrt 1850-1900 average",
        link="https://climate.copernicus.eu/climate-indicators/temperature",
        source="Copernicus",
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
    with open(join(dirname(dirname(__file__)), 'data', 'countries_per_region.json'), 'r') as fp:
        countries_per_region = json.load(fp)

    CountriesPerRegionIMF = ColectedData(
        value=countries_per_region,
        unit="",
        description="breakdown of countries according to IMF",
        link="https://www.imf.org/en/Publications/WEO/weo-database/2023/April/groups-and-aggregates",
        source="World Economic Outlook : International Monetary Fund",
        last_update_date=date(2024, 3, 18)
    )

    gdp_percentage_per_country = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'mean_gdp_country_percentage_in_group.csv'))
    GDPPercentagePerCountry = ColectedData(
        value=gdp_percentage_per_country,
        unit="%",
        description="mean percentage GDP of each country in the group",
        link="",
        source="mean percentages were computed based on official GDP data from international organizations and on the IMF grouping",
        last_update_date=date(2024, 3, 18)
    )
    energy_consumption_services = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'energy_consumption_percentage_services_sections.csv'))
    energy_consumption_agriculture = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'energy_consumption_percentage_agriculture_sections.csv'))
    energy_consumption_industry = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'energy_consumption_percentage_industry_sections.csv'))
    energy_consumption_household = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'energy_consumption_percentage_household_sections.csv'))

    EnergyConsumptionPercentageSectionsDict = ColectedData(
        value={"Agriculture": energy_consumption_agriculture,
                 "Services": energy_consumption_services,
                 "Industry": energy_consumption_industry,
                "Household": energy_consumption_household},
        unit="%",
        description="energy consumption of each section for all sectors",
        link="",
        source="",  # multiples sources TODO
        last_update_date=date(2024, 3, 26)
    )

    non_energy_emissions_services = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'non_energy_emission_gdp_services_sections.csv'))
    non_energy_emissions_agriculture = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'non_energy_emission_gdp_agriculture_sections.csv'))
    non_energy_emissions_industry = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'non_energy_emission_gdp_industry_sections.csv'))
    non_energy_emissions_household = pd.read_csv(join(dirname(dirname(__file__)), 'data', 'non_energy_emission_gdp_household_sections.csv'))

    SectionsNonEnergyEmissionsDict = ColectedData(
        value={"Agriculture": non_energy_emissions_agriculture,
                 "Services": non_energy_emissions_services,
                 "Industry": non_energy_emissions_industry,
               "Household": non_energy_emissions_household},
        unit="tCO2eq/M$",
        description="Non energy CO2 emission per $GDP",
        link="",
        source="",  # multiples sources TODO
        last_update_date=date(2024, 3, 26)
    )

    EnergyConsumptionPercentageSectorDict = HeavyCollectedData(
        value=join(data_folder, 'energy_consumption_percentage_per_sector.csv'),
        unit="%",
        description="energy consumption of each sector",
        link="",
        source="",  # multiples sources TODO
        last_update_date=date(2024, 3, 26),
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
        source="IEA Global bioenergy supply in the Net Zero Scenario, 2010-2050",  # considered as crop for our model : Conventional bioenergy crops, short-rotation woody crops. So we sum them and then convert exajoules (EJ) to TWh
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

    PopulationYearStart = HeavyCollectedData(
        value=join(data_folder, "population_by_age.csv"),
        unit="people",
        description="repartition of the population by age",
        link="https://population.un.org/dataportal/home?df=8b604e23-cef9-4a48-b4cb-b1b5f3aefea9",
        critical_at_year_start=True,
        column_to_pick=["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100_over"],
        source="United Nations Population division",
        last_update_date=date(2024, 8, 14),
    )

    MacroProductivityStart = HeavyCollectedData(
        value=join(data_folder, "productivity.csv"),
        unit="-",
        description="Productivity factor at year start",
        column_to_pick="productivity",
        source="Witness MDA scenario starting in 2020, scenario business as usual fossil 40% with damage. Took prod with damage",
        link="",
        critical_at_year_start=True,
        last_update_date=date(2024, 7, 25),
    )

    MacroInitGrossOutput = HeavyCollectedData(
        value=join(data_folder, "world-gpp-ppp.csv"),
        column_to_pick="GDP, PPP (Trillion USD$2020)",
        unit="G$",
        description="Global GDP",
        link=["https://data.worldbank.org/indicator/NY.GDP.MKTP.PP.CD", "https://www.usinflationcalculator.com/"],
        critical_at_year_start=True,
        source="World Bank",
        last_update_date=date(2024, 8, 21),
    )

    MacroNonEnergyCapitalStart = HeavyCollectedData(
        value=join(data_folder, "capitals.csv"),
        unit="G$ constant $ 2020",
        description="Global energy capital ",
        link="",
        critical_at_year_start=True,
        source="",
        column_to_pick="Capital non energy",
        last_update_date=date(2024, 8, 21),
    )

    MacroProductivityGrowthStart = HeavyCollectedData(
        value=join(data_folder, "productivity.csv"),
        unit="-",
        description="Productivity growth at year start",
        link="",
        critical_at_year_start=True,
        column_to_pick="productivity_gr",
        source="Witness MDA scenario starting in 2020, scenario business as usual fossil 40% with damage.",
        last_update_date=date(2024, 7, 25),
    )

    SectorServiceCapitalStart = ColectedData(
        value=281.2092,
        unit="G$",
        description="Sector Service capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    SectorServiceProductivityStart = ColectedData(
        value=0.1328496,
        unit="-",
        description="Sector Service capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    SectorServiceProductivityGrowthStart = ColectedData(
        value=0.00161432,
        unit="-",
        description="Sector Service capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    SectorIndustryCapitalStart = ColectedData(
        value=88.5051,
        unit="G$",
        description="Sector Industry capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    SectorIndustryProductivityStart = ColectedData(
        value=0.4903228,
        unit="-",
        description="Sector Industry capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    SectorIndustryProductivityGrowthStart = ColectedData(
        value=0.00019,
        unit="-",
        description="Sector Industry capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    SectorAgricultureCapitalStart = ColectedData(
        value=6.92448579,
        unit="G$",
        description="Sector Agriculture capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    SectorAgricultureCapital = HeavyCollectedData(
        value=join(data_folder, "agriculture_sector_capital.csv"),
        unit="Trillion US$ 2015",
        description="Sector of Agriculture, forestry and fishing",
        link="https://www.fao.org/statistics/highlights-archive/highlights-detail/agricultural-investments-and-capital-stock.-global-and-regional-trends-(2012-2022)/en",
        critical_at_year_start=True,
        source="FAO annual global reports",
        last_update_date=date(2024, 12, 13),
        column_to_pick="capital"
    )

    SectorAgricultureInvest = HeavyCollectedData(
        value=join(data_folder, "agriculture_sector_investments.csv"),
        unit="Trillion US$ 2015",
        description="Sector of Agriculture, forestry and fishing",
        link="https://www.fao.org/statistics/highlights-archive/highlights-detail/agricultural-investments-and-capital-stock.-global-and-regional-trends-(2012-2022)/en",
        critical_at_year_start=True,
        source="FAO annual global reports",
        last_update_date=date(2024, 12, 13),
        column_to_pick="investment"
    )

    SectorAgricultureProductivityStart = ColectedData(
        value=1.31162,
        unit="-",
        description="Sector Agriculture capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    SectorAgricultureProductivityGrowthStart = ColectedData(
        value=0.0027844,
        unit="-",
        description="Sector Agriculture capital",
        year_value=2020,
        link="",
        critical_at_year_start=False,
        source="",
        last_update_date=date(2024, 7, 26),
    )

    ForestEmissions = HeavyCollectedData(
        value=join(data_folder, "forest_emissions.csv"),
        unit="GtCO2",
        description="Forest emissions",
        link="https://www.nasa.gov/science-research/earth-science/nasa-satellites-help-quantify-forests-impacts-on-global-carbon-budget/",
        critical_at_year_start=True,
        source="NASA; Credits: Harris et al. 2021 / Global Forest Watch / World Resources Institute",
        last_update_date=date(2024, 7, 26),
        column_to_pick="emissions"
    )

    OceanWarmingAnomalySincePreindustrial = HeavyCollectedData(
        value=join(data_folder, "ocean_temp_anomaly_pre_industrial.csv"),
        unit="°C",
        description="Warming anomaly of ocean since pre-industrial era",
        link="https://www.ncei.noaa.gov/access/monitoring/monthly-report/global/202406",
        critical_at_year_start=True,
        column_to_pick="global ocean SST anomaly",
        source="NOAA National Centers for Environmental Information, Monthly Global Climate Report for June 2024, published online July 2024, retrieved on August 11, 2024 from https://www.ncei.noaa.gov/access/monitoring/monthly-report/global/202406.",
        last_update_date=date(2024, 8, 11),
    )

    IEANZEFinalEnergyConsumption = HeavyCollectedData(
        value=join(data_folder, "IEA_NZE_energy_final_consumption.csv"),
        unit="-",
        description="Final Consumption of energy for IEA NZE scenario",
        link="",
        critical_at_year_start=False,
        column_to_pick="Final Consumption",
        source="IEA",
        last_update_date=date(2025, 1, 9),
    )

    IEANZEEnergyProduction = HeavyCollectedData(
        value=join(data_folder, "IEA_NZE_energy_production_brut.csv"),
        unit="-",
        description="Final Consumption of energy for IEA NZE scenario",
        link="",
        critical_at_year_start=False,
        column_to_pick="Total production",
        source="IEA",
        last_update_date=date(2025, 1, 9),
    )

    IEANZEGDPNetOfDamage = HeavyCollectedData(
        value=join(data_folder, "IEA_NZE_output_net_of_d.csv"),
        unit="-",
        description="GDP net of damage for IEA NZE scenario",
        link="",
        critical_at_year_start=False,
        column_to_pick="output_net_of_d",
        source="IEA",
        last_update_date=date(2025, 1, 9),
    )

    @classmethod
    def get_reforestation_invest_before_year_start(cls, year_start: int, construction_delay: int,
                                                   is_available_at_year: bool = False):

        path_to_csv = os.path.join(data_folder, "forest_invests") + ".csv"
        df = pd.read_csv(path_to_csv)
        heavy_collected_data = HeavyCollectedData(
            value=path_to_csv,
            description="",
            unit="G$",
            link="",
            source="",
            last_update_date=datetime.datetime.today(),
            critical_at_year_start=True,
            column_to_pick="investment"
        )
        out_df = df
        if is_available_at_year:
            return construction_delay == 0 or (heavy_collected_data.is_available_at_year(
                year_start - construction_delay) and heavy_collected_data.is_available_at_year(year_start - 1))
        if construction_delay > 0:
            out_df = heavy_collected_data.get_between_years(year_start=year_start - construction_delay,
                                                            year_end=year_start - 1)
        return out_df, heavy_collected_data
