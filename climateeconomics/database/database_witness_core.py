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
from os.path import join, dirname
from datetime import date
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

    TemperatureAnomalyPreIndustrialYearStart = ColectedData(
        value=+1.3,
        unit="°C",
        description="Global average temperature anomaly relative to 1850-1900 average",
        link="https://berkeleyearth.org/global-temperature-report-for-2023/#:~:text=Annual%20Temperature%20Anomaly&text=As%20a%20result%2C%202023%20is,C%20(2.7%20%C2%B0F).",
        source="BerkleyEarth",
        last_update_date=date(2024, 2, 27)
    )

    # Data for sectorization
    InvestInduspercofgdp2020 = ColectedData(
        value=5.831,
        unit="%",
        description="Investment in Industry sector as percentage of GDP for year 2020",
        link="",
        source="Computed from World bank,IMF and IEA data",
        last_update_date=date(2023, 10, 23),
    )

    InvestServicespercofgdp2020 = ColectedData(
        value=19.231,
        unit="%",
        description="Investment in Services sector as percentage of GDP for year 2020",
        link="",
        source="Computed from World bank and IMF data",
        last_update_date=date(2023, 10, 23),
    )

    InvestAgriculturepercofgdp2020 = ColectedData(
        value=0.4531,
        unit="%",
        description="Investment in Agriculture sector as percentage of GDP for year 2020",
        link="",
        source="Computed from World bank, IMF, and FAO gross capital formation data",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareAgriculture2020 = ColectedData(
        value=2.1360,
        unit="%",
        description="Share of net energy production dedicated to Agriculture sector in % in 2020",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareIndustry2020 = ColectedData(
        value=28.9442,
        unit="%",
        description="Share of net energy production dedicated to Industry sector in % in 2020",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareServices2020 = ColectedData(
        value=36.9954,
        unit="%",
        description="Share of net energy production dedicated to Services sector in % in 2020",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareResidential2020 = ColectedData(
        value=21.00,
        unit="%",
        description="Share of net energy production dedicated to Residential in % in 2020",
        link="",
        source="IEA",
        last_update_date=date(2023, 10, 23),
    )

    EnergyshareOther2020 = ColectedData(
        value=10.9230,
        unit="%",
        description="Share of net energy production dedicated to other consumption in % in 2020",
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

    C02YearStartConcentration = ColectedData(
        value=414,
        unit="ppm",
        description="Concentration of CO2 in atmosphere for year start (2020)",
        link="https://gml.noaa.gov/ccgg/trends/",
        source="US Department of Commerce",
        last_update_date=date(2024, 1, 12),
    )

    CH4YearStartConcentration = ColectedData(
        value=1872,
        unit="ppb",
        description="Concentration of CH4 in atmosphere for year start (2020)",
        link="https://gml.noaa.gov/ccgg/trends_ch4/",
        source="US Department of Commerce",
        last_update_date=date(2024, 1, 12),
    )

    N2OYearStartConcentration = ColectedData(
        value=333,
        unit="ppb",
        description="Concentration of N2O in atmosphere for year start (2020)",
        link="https://gml.noaa.gov/ccgg/trends_n2o/",
        source="US Department of Commerce",
        last_update_date=date(2024, 1, 12),
    )

    HistoricCO2Concentration = HeavyCollectedData(
        value=join(data_folder, "co2_annmean_mlo.csv"),
        unit="PPM",
        description="Concentration of CO2 in atmosphere from 1961 to 2023",
        link="https://gml.noaa.gov/ccgg/trends/data.html",
        source="Earth System Research Laboratorie; Global Monitoring Laboratory",
        last_update_date=date(2023, 1, 18),
    )

    HistoricCH4Concentration = HeavyCollectedData(
        value=join(data_folder, "ch4_annmean_gl.csv"),
        unit="PPB",
        description="Concentration of CH4 in atmosphere from 1984 to 2022",
        link="https://gml.noaa.gov/ccgg/trends/data.html",
        source="Earth System Research Laboratorie; Global Monitoring Laboratory",
        last_update_date=date(2023, 1, 18),
    )

    HistoricN2OConcentration = HeavyCollectedData(
        value=join(data_folder, "n2o_annmean_gl.csv"),
        unit="PPB",
        description="Concentration of N2O in atmosphere from 1984 to 2022",
        link="https://gml.noaa.gov/ccgg/trends/data.html",
        source="Earth System Research Laboratorie; Global Monitoring Laboratory",
        last_update_date=date(2023, 1, 18),
    )

    CumulativeCO2Emissions = ColectedData(
        value=1772.8,
        unit="Gt",
        description="Running sum of CO2 emissions produced from fossil fuels and industry since the first year of recording 1750 until 2022, measured in Giga tonnes. Land-use change is not included ",
        link="https://ourworldindata.org/grapher/cumulative-co-emissions?tab=table",
        source="Our World in Data",
        last_update_date=date(2024, 2, 7),
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
