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
