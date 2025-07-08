'''
Copyright 2025 Capgemini

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
from datetime import date
from os.path import dirname, join

from climateeconomics.database.collected_data import HeavyCollectedData

data_folder = dirname(__file__)



class StDB:
    # Story telling scenario names :

    # classic story telling : optim plays with invest in different technos
    UC1 = "- Damage, - Tax"
    UC2 = "+ Damage, + Tax, Fossil only"
    UC3 = "+ Damage, + Tax, No CCUS"
    UC4 = "+ Damage, + Tax, CCUS"

    # prod vs demand : optim tries to match energy prod and demand
    USECASE2 = '- damage - tax, fossil 100%'
    USECASE2B = '+ damage + tax, fossil 100%'
    USECASE3 = '- damage + tax, IEA'
    USECASE4 = '+ damage + tax, fossil 40%'
    USECASE5 = '+ damage - tax, STEP inspired'
    USECASE6 = '+ damage - tax, NZE inspired'
    USECASE7 = '+ damage + tax, NZE'

    FullFossilShareEnergyInvest = HeavyCollectedData(
        value=join(data_folder, "uc1_percentage_of_gdp_energy_invest.csv"),
        unit="%",
        description="Percentage of GDP that is invested in energy sector",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    FullFossilEnergyInvestMix = HeavyCollectedData(
        value=join(data_folder, "uc2_techno_invest_percentage.csv"),
        unit="%",
        description="Distribution of investments between energy production technos",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )


    BusineesAsUsualShareEnergyInvest = HeavyCollectedData(
        value=join(data_folder, "uc3_percentage_of_gdp_energy_invest.csv"),
        unit="%",
        description="Percentage of GDP that is invested in energy sector",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    BAUEnergyInvestMix = HeavyCollectedData(
        value=join(data_folder, "uc3_techno_invest_percentage.csv"),
        unit="%",
        description="Distribution of investments between energy production technos",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    UC4EnergyInvestMix = HeavyCollectedData(
        value=join(data_folder, "uc4_techno_invest_percentage.csv"),
        unit="%",
        description="Distribution of investments between energy production technos",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    UC7EnergyInvestMix = HeavyCollectedData(
        value=join(data_folder, "uc7_techno_invest_percentage.csv"),
        unit="%",
        description="Distribution of investments between energy production technos",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    NZEShareEnergyInvest = HeavyCollectedData(
        value=join(data_folder, "uc3_percentage_of_gdp_energy_invest.csv"),
        unit="%",
        description="Percentage of GDP that is invested in energy sector",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    NZEShareCCUSInvest = HeavyCollectedData(
        value=join(data_folder, "uc7_percentage_of_gdp_energy_invest.csv"),
        unit="%",
        description="Percentage of GDP that is invested in CCUS sector",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    NZEEnergyInvestMix = HeavyCollectedData(
        value=join(data_folder, "uc7_techno_invest_percentage.csv"),
        unit="%",
        description="Distribution of investments between energy production technos",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    NoCCUSInvestShare = HeavyCollectedData(
        value=join(data_folder, "no_ccus_invest.csv"),
        unit="%",
        description="Percentage of GDP that is invested in CCUS sector",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )
