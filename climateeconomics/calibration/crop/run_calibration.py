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
import json

from climateeconomics.calibration.crop.shares_of_waste import to_export as to_export_shares_of_waste
from climateeconomics.calibration.crop.productions import to_export as to_export_productions
from climateeconomics.calibration.crop.emissions import to_export as to_export_emissions
from climateeconomics.calibration.crop.land_use import to_export as to_export_land_use
from climateeconomics.calibration.crop.capital import to_export as to_export_capital

# All calibration is done with 2021 data as we found all the data we needed for this year.

# Road map calibration of crop discipline :
# 1. Shares of waste
# 2. Productions
# 3. Emissions
# 4. Land use
# 5. Capital

# you will find details on methodology in the corresponding files

# Save the dictionaries to a JSON file
data_to_save = {
    **to_export_shares_of_waste,
    **to_export_productions,
    **to_export_emissions,
    **to_export_land_use,
    **to_export_capital,
}

with open('output_calibration.json', 'w') as json_file:
    json.dump(data_to_save, json_file, indent=4)