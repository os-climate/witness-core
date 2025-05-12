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
import datetime
import json
import os
from datetime import date
from os.path import dirname, join

import pandas as pd

from climateeconomics.database.collected_data import ColectedData, HeavyCollectedData

data_folder = dirname(__file__)



class StoryTellingDatabase:

    FullFossilShareEnergyInvest = HeavyCollectedData(
        value=join(data_folder, "uc1_percentage_of_gdp_energy_invest.csv"),
        unit="%",
        description="Percentage of GDP that is invested in energy sector",
        link="",
        source="",
        last_update_date=date(2025, 5, 5),
    )

    FullFossilEnergyInvestMix = HeavyCollectedData(
        value=join(data_folder, "uc1_techno_invest_percentage.csv"),
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
