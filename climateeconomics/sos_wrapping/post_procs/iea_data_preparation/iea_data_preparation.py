'''
Copyright 2024 Capgemini

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
from copy import deepcopy

import numpy as np
import pandas as pd

from energy_models.glossaryenergy import GlossaryEnergy as Glossary




class IEADataPreparation:
    """
    IEA Data Preparation Model
    """

    def __init__(self):
        '''
        Constructor
        '''
        self.year_start = None
        self.year_end = None
        self.CO2_emissions_df_in = None
        self.CO2_emissions_df_out = None
        self.population_in = None
        self.population_out = None
        self.gdp_in = None
        self.gdp_out = None
        self.energy_prices_in = None
        self.energy_prices_out = None
        self.CO2_tax_in = None
        self.CO2_tax_out = None
        self.energy_production_in = None
        self.energy_production_out = None
        self.temperature_in = None
        self.temperature_out = None
        self.dict_df_in = None
        self.dict_df_out = None

    def configure_parameters(self, input_dict, variables_to_store):
        """
        Configure parameters of model
        """

        self.year_start = input_dict[Glossary.YearStart]
        self.year_end = input_dict[Glossary.YearEnd]
        self.dict_df_in = {key_name: input_dict[key_name] for key_name in variables_to_store if key_name in input_dict}
    def compute(self):
        """
        Interpolate between year start and year end all the dataframes
        """
        self.dict_df_out = self.interpolate_dataframes(self.dict_df_in)


    def interpolate_dataframes(self, dataframes_dict):
        """
        Interpolates missing values in multiple DataFrames using linear interpolation.

        This function takes a dictionary of DataFrames that each contain a 'years' column and
        other columns with values to be interpolated. The function ensures the 'years' column
        spans from 2020 to 2050 and performs linear interpolation on the missing values.

        Parameters:
        dataframes_dict (dict of str: pd.DataFrame): Dictionary of DataFrames to be interpolated.
                                                     Each DataFrame must contain a 'years' column.

        Returns:
        dict of str: pd.DataFrame: Dictionary of DataFrames with interpolated values from 2020 to 2050.

        Raises:
        ValueError: If a DataFrame does not contain the 'years' column.
        """

        # Dictionary to store the interpolated dataframes
        interpolated_dfs_dict = {}

        for key, df_original in dataframes_dict.items():
            print(key)
            # copy input dataframe to avoid issues of memory link
            df = df_original.copy()
            # Check if 'years' is in the columns
            if Glossary.Years not in df.columns:
                raise ValueError(f"The column 'years' is missing in the DataFrame with key '{key}'.")

            # Ensure the 'years' column is of type int
            df[Glossary.Years] = df[Glossary.Years].astype(int)

            # Set the index to 'years' to facilitate interpolation
            df.set_index(Glossary.Years, inplace=True)

            # Create a range of years from year_start to year_end
            full_range = pd.Series(np.arange(self.year_start, self.year_end), name=Glossary.Years)

            # Reindex the DataFrame to include all the years in the range
            df = df.reindex(full_range)

            # Perform linear interpolation for each column independently
            df = df.apply(lambda col: col.interpolate(method='linear'))

            # Perform linear backward extrapolation for each column independently
            for column in df.columns:
                # Find the first valid index
                first_valid_index = df[column].first_valid_index()
                if first_valid_index is not None and first_valid_index > self.year_start:
                    # Calculate the slope using the first two valid data points
                    next_valid_index = df[column].index[df[column].index > first_valid_index][0]
                    slope = (df[column][next_valid_index] - df[column][first_valid_index]) / \
                            (next_valid_index - first_valid_index)
                    # Extrapolate backward
                    df.loc[self.year_start:first_valid_index - 1, column] = df.loc[first_valid_index, column] + \
                                                                       slope * (df.loc[
                                                                                self.year_start:first_valid_index - 1].index - first_valid_index)

            # Reset the index to bring 'years' back as a column
            df.reset_index(inplace=True)

            # Add the interpolated DataFrame to the dictionary
            interpolated_dfs_dict[key+'_interpolated'] = df

        return interpolated_dfs_dict
