'''
Copyright 2024 Capgemini
Modifications on 2023/04/19-2024/06/24 Copyright 2023 Capgemini

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
from climateeconomics.calibration.crop.tools import CalibrationData
from climateeconomics.glossarycore import GlossaryCore

input_calibration_datas = [
    CalibrationData(  # share of total final energy consumption in agriculture
        varname=f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}',
        column_name=GlossaryCore.TotalProductionValue,
        year=2022,
        value=4 / 100.,
        unit='T$',
        source='FAO',
        link="https://www.fao.org/4/x8054e/x8054e05.htm") *
    CalibrationData(  # total global final energy consumption
        varname=f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.EnergyProductionValue}',
        column_name=GlossaryCore.TotalProductionValue,
        year=2022,
        value=172.821,
        unit='PWh',
        source='IEA',
        link="https://www.iea.org/world/energy-mix"),
    CalibrationData(  # Share of people working in agriculture relative to global workforce
        varname=GlossaryCore.WorkforceDfValue,
        column_name=GlossaryCore.SectorAgriculture,
        year=2022,
        value=26 / 100,
        unit='',
        source='World Bank',
        link="https://data.worldbank.org/indicator/SL.AGR.EMPL.ZS") *
    CalibrationData(  # global labor force
        varname=GlossaryCore.WorkforceDfValue,
        column_name=GlossaryCore.SectorAgriculture,
        year=2022,
        value=3.55 * 1000,
        unit='million people',
        source='World Bank',
        link="https://data.worldbank.org/indicator/SL.TLF.TOTL.IN"),
]
