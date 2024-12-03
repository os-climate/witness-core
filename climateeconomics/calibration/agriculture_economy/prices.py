#https://www.statista.com/statistics/675826/average-prices-meat-beef-worldwide/
from climateeconomics.database import DatabaseWitnessCore

#https://fred.stlouisfed.org/series/PFISHUSDM

agri_gdp_ppp_2021 = DatabaseWitnessCore.MacroInitGrossOutput.get_value_at_year(2021) * DatabaseWitnessCore.ShareGlobalGDPAgriculture2021.value /100
print(agri_gdp_ppp_2021)

# https://www.statista.com/statistics/502286/global-meat-and-seafood-market-value/
# meat market value 2021 : 0.897 T$