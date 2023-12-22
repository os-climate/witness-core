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
