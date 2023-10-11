import pandas as pd
from os.path import join, dirname
"read DataFrame"
data_folder = join(dirname(dirname(__file__)), 'data_energy', 'data')
data=pd.read_csv(join(data_folder, 'value_added_by_sector.csv'))
"we only keep data for 2017 and we calculate the average for each country and then the global value"
dt = data[data['Year'] == 2017]
mean_per_country = dt.groupby('Country')['Value USD (Million USD)'].mean()
general_average = mean_per_country.mean()
# "calculate the average of all years in the dataframe "
# mean_per_country_and_year = data.groupby(['Country', 'Year'])['Value USD (Million USD)'].mean()
# mean_per_year = mean_per_country_and_year.groupby('Year').mean()

"calculate the weight of each country for 2017"
"Step 1: Calculate the total sum of values for each country"
total_values_per_country = dt.groupby('Country')['Value USD (Million USD)'].sum()

"Step 2: Divide the total sum by the total weight to get the weight per country"
total_value = total_values_per_country.sum()
weights_per_country = total_values_per_country / total_value
total_weight=weights_per_country.sum()
grouped_data = dt.groupby(['Country', 'ISIC Section'])
weighted_avg_by_group = {}
section_sums={}
for name, group in grouped_data:
    weighted_avg = group['Value USD (Million USD)'] * weights_per_country[name[0]]
    weighted_avg_by_group[name] = weighted_avg
    country_name = name[0]
    section_name = name[1]

    weighted_avg_value = weighted_avg.values[0]


   # print(f"Country: {country_name}, Section: {section_name}, Weighted Value: {weighted_avg_value}")
    if section_name in section_sums:
        section_sums[section_name] += weighted_avg_value
    else:
        section_sums[section_name] = weighted_avg_value


section_weighted_averages = {}
for section, section_sum in section_sums.items():
    section_weighted_average = section_sum / total_weight
    section_weighted_averages[section] = section_weighted_average
    print(f"Section: {section}, Weighted Average: {section_weighted_average}")