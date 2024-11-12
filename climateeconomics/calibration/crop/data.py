"""
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
"""
from climateeconomics.calibration.crop.tools import (
    CalibrationData,
    solve_share_prod_waste,
)
from climateeconomics.glossarycore import GlossaryCore

output_calibration_datas = [
    CalibrationData(
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.Milk,
        year=2022,
        value=930 ,
        unit='Mt',
        source='FAO',
        link='https://www.fao.org/dairy-production-products/production/milk-production/en'),
    # rice
    CalibrationData(
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.RiceAndMaize,
        year=2022,
        value=516.02 ,
        unit='Mt',
        source='FAS',
        link='https://fas.usda.gov/data/production/commodity/0422110') +
    # maize
    CalibrationData(
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.RiceAndMaize,
        year=2022,
        value=1163.497,
        unit='Mt',
        source='Food and Agriculture Organization of the United Nations (2023)',
        link='https://ourworldindata.org/grapher/maize-production?tab=table&time=2021..latest&showSelectionOnlyInTable=1&country=European+Union~OWID_WRL'),
    CalibrationData(
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.Fish,
        year=2022,
        value=223.2,
        unit='Mt',
        source='2024 edition of The State of World Fisheries and Aquaculture (SOFIA) ',
        link='https://www.fao.org/newsroom/detail/fao-report-global-fisheries-and-aquaculture-production-reaches-a-new-record-high/en'),
    CalibrationData(
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.Eggs,
        year=2022,
        value=87,
        unit='Mt',
        source='Statista',
        link='https://www.statista.com/statistics/263972/egg-production-worldwide-since-1990/#:~:text=The%20production%20volume%20of%20eggs,increased%20by%20over%20100%20percent.'),
    # fruits
    CalibrationData(
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.FruitsAndVegetables,
        year=2022,
        value=933.07,
        unit='Mt',
        source='FAO',
        link='https://www.statista.com/statistics/262266/global-production-of-fresh-fruit/#:~:text=In%202022%2C%20the%20global%20production,million%20metric%20tons%20in%202000.') +
    CalibrationData(  # vegetables
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.FruitsAndVegetables,
        year=2022,
        value=1173,
        unit='Mt',
        source='FAO',
        link='https://www.statista.com/statistics/262266/global-production-of-fresh-fruit/#:~:text=In%202022%2C%20the%20global%20production,million%20metric%20tons%20in%202000.'),
    CalibrationData(  # white meat : poultry (or chicken)
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.WhiteMeat,
        year=2022,
        value=139.219,
        unit='Mt',
        source='UN Food and Agriculture Organization (FAO)',
        link='https://ourworldindata.org/agricultural-production') +
    CalibrationData(  # white meat : pork / pig
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.WhiteMeat,
        year=2022,
        value=122.585,
        unit='Mt',
        source='UN Food and Agriculture Organization (FAO)',
        link='https://ourworldindata.org/agricultural-production') +
    CalibrationData(  # white meat : turkey
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.WhiteMeat,
        year=2022,
        value=5.081,
        unit='Mt',
        source='UN Food and Agriculture Organization (FAO)',
        link='https://ourworldindata.org/agricultural-production') +
    CalibrationData(  # white meat : rabbit
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.WhiteMeat,
        year=2022,
        value=0.756,
        unit='Mt',
        source='UN Food and Agriculture Organization (FAO)',
        link='https://ourworldindata.org/agricultural-production'),
    CalibrationData(  # red meat : duck
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.RedMeat,
        year=2022,
        value=6.068,
        unit='Mt',
        # metric selector : Food available for consumption
        source='UN Food and Agriculture Organization (FAO)',
        link='https://ourworldindata.org/agricultural-production') +
    CalibrationData(  # red meat : beef and buffalo
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.RedMeat,
        year=2022,
        value=76.249,
        unit='Mt',
        source='UN Food and Agriculture Organization (FAO)',
        # metric selector : production
        link='https://ourworldindata.org/agricultural-production') +
    CalibrationData(  # red meat : lamb and mutton
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.RedMeat,
        year=2022,
        value=10.272,
        unit='Mt',
        source='UN Food and Agriculture Organization (FAO)',
        # metric selector : production
        link='https://ourworldindata.org/agricultural-production') +
    CalibrationData(  # red meat : goat and sheep
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.RedMeat,
        year=2022,
        value=16.639,
        unit='Mt',
        source='UN Food and Agriculture Organization (FAO)',
        # metric selector : production
        link='https://ourworldindata.org/agricultural-production') +
    CalibrationData(  # red meat : horse
        varname=GlossaryCore.FoodTypeProductionName,
        column_name=GlossaryCore.RedMeat,
        year=2022,
        value=0.775,
        unit='Mt',
        source='UN Food and Agriculture Organization (FAO)',
        # metric selector : production
        link='https://ourworldindata.org/agricultural-production'),

    # workforce fisheries
    CalibrationData(  # red meat : horse
        varname="workforce_breakdown",
        column_name=GlossaryCore.Fish,
        year=2022,
        value=61.8,
        unit='million people',
        source='FAO',
        link='https://openknowledge.fao.org/server/api/core/bitstreams/66538eba-9c85-4504-8438-c1cf0a0a3903/content/sofia/2024/fisheries-aquaculture-employment.html'),

    # total available calories for consumption
    CalibrationData(
        varname=GlossaryCore.CaloriesPerCapitaValue,
        column_name='kcal_pc',
        year=2022, # actually for 2021 but guess it didnt change much
        value=2959,
        unit='kcal/person/day',
        source='Food and Agriculture Organization of the United Nations (2023) and other sources',
        link='https://openknowledge.fao.org/server/api/core/bitstreams/66538eba-9c85-4504-8438-c1cf0a0a3903/content/sofia/2024/fisheries-aquaculture-employment.html'),
]

input_calibration_datas = [
    CalibrationData(
        varname=f'{GlossaryCore.SectorAgriculture}.{GlossaryCore.InvestmentDfValue}',
        column_name=GlossaryCore.InvestmentsValue,
        year=2022,
        value=0.597,
        unit='T$',
        source='FAO',
        link="https://openknowledge.fao.org/server/api/core/bitstreams/a18945c6-aca1-4628-aefb-4c9fca4043ed/content"),
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


    # Fish waste at production distribution and by consumers
    # Reasonning for finding our value given datas from the report
    # total_waste = food_prod * (share prod waste / 100) + food_prod * (1 - share_prod waste /100) * share consumers waste / 100
    #
    # share consumers waste = 2.043 / 23.813 * 100
    # 14.78 / 100 = share prod waste / 100 + share consumers waste/ 100 - share prod waste * share consumers / 10000
    CalibrationData(
        varname=GlossaryCore.FoodTypeWasteAtProductionShareName,
        column_name=GlossaryCore.Fish,
        year=2022,  # actually 2021 in the report but guess it didnt change much
        value=solve_share_prod_waste(14.78, 6.78),
        unit='%',
        source='World Economic Forum',
        link="https://www3.weforum.org/docs/WEF_Investigating_Global_Aquatic_Food_Loss_and_Waste_2024.pdf"),
    CalibrationData(
        varname=GlossaryCore.FoodTypeWasteByConsumersShareName,
        column_name=GlossaryCore.Fish,
        year=2022,  # actually 2021 in the report but guess it didnt change much
        value=6.78,
        unit='%',
        source='World Economic Forum',
        link="https://www3.weforum.org/docs/WEF_Investigating_Global_Aquatic_Food_Loss_and_Waste_2024.pdf"),

    # waste fruits and vegetables
    # weighted average of fruits and vegetables respective prod
    CalibrationData(
        varname=GlossaryCore.FoodTypeWasteByConsumersShareName,
        column_name=GlossaryCore.FruitsAndVegetables,
        year=2022,
        value= 933 / (933 + 1173) * 12 + 1173 / (933 + 1173) * 25,
        unit='%',
        source='FAO',
        link="https://www.toogoodtogo.com/en-au/about-food-waste;https://www.sciencedirect.com/science/article/pii/S0921344920302305#tbl0001"),
    CalibrationData(
        varname=GlossaryCore.FoodTypeWasteAtProductionShareName,
        column_name=GlossaryCore.FruitsAndVegetables,
        year=2022,  # actually 2021 in the report but guess it didnt change much
        value=solve_share_prod_waste(14.78, 6.78),
        unit='%',
        source='FAO',
        link="https://www.toogoodtogo.com/en-au/about-food-waste"),
]

total_foodchain_wastes = { # accounting waste a production and distribution and waste at home by consumers
    GlossaryCore.RedMeat: 5,
    GlossaryCore.Eggs: 7,
    GlossaryCore.Cereals: 24,
    GlossaryCore.RiceAndMaize: 24,  # rice and maize are cereals some same value assumed
}
assumed_consumers_waste_share = 6.78

for food_type, total_waste in total_foodchain_wastes.items():
    input_calibration_datas.append(
        CalibrationData(
            varname=GlossaryCore.FoodTypeWasteAtProductionShareName,
            column_name=food_type,
            year=2022,
            value=solve_share_prod_waste(total_waste, min(total_waste, assumed_consumers_waste_share)),
            unit='%',
            source='',
            link="https://www.sciencedirect.com/science/article/pii/S0921344920302305#tbl0001"),
    )
    input_calibration_datas.append(
        CalibrationData(  # assumed
            varname=GlossaryCore.FoodTypeWasteByConsumersShareName,
            column_name=food_type,
            year=2022,
            value=assumed_consumers_waste_share,
            unit='%',
            source='',
            link=""),
    )

# needs of workforce and energy. Aiming for 0 unused workforce and energy in model with 2022 data
food_types = GlossaryCore.DefaultFoodTypes
unused_workforce_data = []
for food_type in food_types:
    unused_workforce_data.append(
        CalibrationData(
            varname="unused_workforce" + "_breakdown",
            column_name=food_type,
            year=2022,
            value=0.001,
            unit='',
            source='',
            link=''),
    )

unused_energy_data = []
for food_type in food_types:
    unused_workforce_data.append(
        CalibrationData(
            varname="unused_energy" + "_breakdown",
            column_name=food_type,
            year=2022,
            value=0.001,
            unit='',
            source='',
            link=''),
    )