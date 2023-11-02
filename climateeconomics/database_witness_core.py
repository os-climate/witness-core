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
from datetime import date

class ColectedData:
    def __init__(self,
                 value,
                 description: str,
                 link: str,
                 source: str,
                 last_update_date: date):
        self.value = value
        self.description = description
        self.link = link
        self.source = source
        self.last_update_date = last_update_date


class DatabaseWitnessCore:
    """Stocke les valeurs utilisées dans witness core"""
    #Data for sectorization
    InvestInduspercofgdp2020 = ColectedData(5.831,
                                    description="Investment in Industry sector as percentage of GDP for year 2020",
                                    link="",
                                    source="Computed from World bank,IMF and IEA data",
                                    last_update_date=date(2023, 10, 23))

    InvestServicespercofgdp2020 = ColectedData(19.231,
                                    description="Investment in Services sector as percentage of GDP for year 2020",
                                    link="",
                                    source="Computed from World bank and IMF data",
                                    last_update_date=date(2023, 10, 23))


    InvestAgriculturepercofgdp2020 = ColectedData(0.4531,
                                    description="Investment in Agriculture sector as percentage of GDP for year 2020",
                                    link="",
                                    source="Computed from World bank, IMF, and FAO gross capital formation data",
                                    last_update_date=date(2023, 10, 23))

    EnergyshareAgriculture2020 = ColectedData(2.1360,
                                                  description="Share of net energy production dedicated to Agriculture sector in % in 2020",
                                                  link="",
                                                  source="IEA",
                                                  last_update_date=date(2023, 10, 23))

    EnergyshareIndustry2020 = ColectedData(28.9442,
                                              description="Share of net energy production dedicated to Industry sector in % in 2020",
                                              link="",
                                              source="IEA",
                                              last_update_date=date(2023, 10, 23))

    EnergyshareServices2020 = ColectedData(36.9954,
                                              description="Share of net energy production dedicated to Services sector in % in 2020",
                                              link="",
                                              source="IEA",
                                              last_update_date=date(2023, 10, 23))

    EnergyshareResidential2020 = ColectedData(21.00,
                                           description="Share of net energy production dedicated to Residential in % in 2020",
                                           link="",
                                           source="IEA",
                                           last_update_date=date(2023, 10, 23))

    EnergyshareOther2020 = ColectedData(10.9230,
                                           description="Share of net energy production dedicated to other consumption in % in 2020",
                                           link="",
                                           source="IEA",
                                           last_update_date=date(2023, 10, 23))



