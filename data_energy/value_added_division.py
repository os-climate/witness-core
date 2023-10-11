import pandas as pd
from os.path import join, dirname
"read DataFrame"
data_folder = join(dirname(dirname(__file__)), 'data_energy', 'data')
data=pd.read_csv(join(data_folder, 'energy_value_division.csv'))
"we only keep data for 2017 and we calculate the average for each country and then the global value"
dt = data[data['Year'] == 2017]
mean_per_country = dt.groupby('Country')['Value USD (million USD)'].mean()
general_average = mean_per_country.mean()
"calculate the average of all years in the dataframe "
mean_per_country_and_year = data.groupby(['Country', 'Year'])['Value USD (million USD)'].mean()
mean_per_year = mean_per_country_and_year.groupby('Year').mean()
"calculate the weight of each country for 2017"
"Step 1: Calculate the total sum of values for each country"
total_values_per_country = dt.groupby('Country')['Value USD (million USD)'].sum()
total_value = total_values_per_country.sum()
weights_per_country = total_values_per_country / total_value
total_weight=weights_per_country.sum()
"weighted average by country without taking the sector into account"
weighted_avg = total_values_per_country* weights_per_country / total_weight
