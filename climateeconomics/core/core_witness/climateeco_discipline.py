'''
Copyright 2022 Airbus SAS
Modifications on 2023/05/02-2024/06/24 Copyright 2023 Capgemini

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
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

from climateeconomics.glossarycore import GlossaryCore


class ClimateEcoDiscipline(SoSWrapp):
    """
    Climate Economics Discipline
    """

    assumptions_dict_default = {'compute_gdp': True,
                                'compute_climate_impact_on_gdp': True,
                                'activate_climate_effect_population': True,
                                'activate_pandemic_effects': False,
                                                }

    YEAR_START_DESC_IN = {'type': 'int', 'default': GlossaryCore.YearStartDefault,
                          'unit': 'year', 'visibility': 'Shared', 'namespace': 'ns_public', 'range': [1950, 2080]}
    TIMESTEP_DESC_IN = {'type': 'int', 'default': 1, 'unit': 'year per period',
                        'visibility': 'Shared', 'namespace': 'ns_public', 'user_level': 2}
    ALPHA_DESC_IN = {'type': 'float', 'range': [0., 1.], 'default': 0.5, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS,
                     'user_level': 1, 'unit': '-'}
    GWP_100_default = {GlossaryCore.CO2: 1.0,
                       GlossaryCore.CH4: 28.,
                       GlossaryCore.N2O: 265.}

    GWP_20_default = {GlossaryCore.CO2: 1.0,
                      GlossaryCore.CH4: 85.,
                      GlossaryCore.N2O: 265.}
    ASSUMPTIONS_DESC_IN = {
        'var_name': 'assumptions_dict', 'type': 'dict', 'default': assumptions_dict_default, 'visibility': 'Shared', 'namespace': GlossaryCore.NS_WITNESS, 'structuring': True, 'unit': '-'}

    desc_in_default_pandemic_param = GlossaryCore.PandemicParamDf
    # https://stackoverflow.com/questions/13905741/accessing-class-variables-from-a-list-comprehension-in-the-class-definition
    global_data_dir = join(Path(__file__).parents[2], 'data')

    # ontology information
    _ontology_data = {
        'label': 'WITNESS Climate Economics Model',
        'type': 'Research',
        'source': 'SoSTrades Project',
        'validated': '',
        'validated_by': 'SoSTrades Project',
        'last_modification_date': '',
        'category': '',
        'definition': '',
        'icon': '',
        'version': '',
    }

    def _run(self):
        """
        Run user-defined model.

        Returns:
            local_data (Dict): outputs of the model run
        """
        check_range_before_run = self.get_sosdisc_inputs(GlossaryCore.CheckRangeBeforeRunBoolName)
        inputs = self.get_sosdisc_inputs()
        if check_range_before_run:
            dict_ranges = self.get_ranges_input_var()
            self.check_ranges(inputs, dict_ranges)

        self.run()

        outputs = self.get_sosdisc_outputs()
        if check_range_before_run:
            dict_ranges = self.get_ranges_output_var()
            self.check_ranges(outputs, dict_ranges)

        return self.local_data

    def get_greataxisrange(self, serie):
        """
        Get the lower and upper bound of axis for graphs
        min_value: lower bound
        max_value: upper bound
        """
        min_value = serie.values.min()
        max_value = serie.values.max()
        min_range = self.get_value_axis(min_value, 'min')
        max_range = self.get_value_axis(max_value, 'max')

        return min_range, max_range

    def get_value_axis(self, value, min_or_max):
        """
        if min: if positive returns 0, if negative returns 1.1*value
        if max: if positive returns is 1.1*value, if negative returns 0
        """
        if min_or_max == 'min':
            if value >= 0:
                value_out = 0
            else:
                value_out = value * 1.1
            return value_out

        elif min_or_max == "max":
            if value >= 0:
                value_out = value * 1.1
            else:
                value_out = 0
            return value_out

    def get_ranges_input_var(self):
        '''
        Get available ranges of input data.
        '''
        return self.get_ranges_var(self.DESC_IN)

    def get_ranges_output_var(self):
        '''
        Get available ranges of output data.
        '''
        return self.get_ranges_var(self.DESC_OUT)

    def get_ranges_var(self, DESC):
        """
        Get available ranges of input or output data.

        DESC: [dataframe descriptor] from which the data range will be recovered (self.DESC_IN or self.DESC_OUT)

        Returns:
            dict: Dictionary containing ranges for each variable.
                  For DataFrame variables, it includes ranges for each column.

        Note:
            This method looks into the DESC attribute, which is a dictionary
            describing input or output variables, and extracts the available ranges.
        """

        # Initialize an empty dictionary to store variable ranges
        dict_ranges = {}
        # Loop through input variables
        for var_name, dict_data in DESC.items():
            # Check if the variable type is a DataFrame
            if dict_data[self.TYPE] == 'dataframe':
                if self.DATAFRAME_DESCRIPTOR in dict_data:
                    range_dict_df = {}
                    # Extract ranges from DataFrame descriptor and store them in a dictionary
                    for variable, (_, variable_range, _) in dict_data[self.DATAFRAME_DESCRIPTOR].items():
                        if variable_range is not None:
                            range_dict_df[variable] = variable_range
                    # Store the type of each column of the DataFrame
                    dict_ranges[var_name] = range_dict_df

            # For other types, check if the range is defined in DESC_IN
            else:
                # Check if the range is specified in DESC_IN
                if self.RANGE in dict_data:
                    range_data = dict_data[self.RANGE]
                    if range_data is not None:
                        dict_ranges[var_name] = range_data
        # Return the dictionary of ranges
        return dict_ranges

    def check_ranges(self, data, ranges):
        """
        Check value ranges for each variable with a defined range.

        Args:
            data (dict): Dictionary with variable values.
            ranges (dict): Dictionary with possible value ranges for each variable.

        Raises:
            ValueError: If a variable is outside the specified range.
            TypeError: If the variable type is not supported.
        """
        # Iterate through each variable in the provided data
        for key, value in data.items():
            # Check if the variable has a defined range
            if key in ranges:
                variable_range = ranges[key]
                if variable_range is not None:
                    # If the variable is a nested dictionary, apply recursion
                    if isinstance(value, dict) and isinstance(variable_range, dict):
                        # Recursion for nested dictionaries
                        self.check_ranges(value, variable_range)
                    # If the variable is of type float or int, check if it is within the specified range
                    elif isinstance(value, (float, int)):
                        if not (variable_range[0] <= value <= variable_range[1]):
                            raise ValueError(
                                f"The value of '{key}' ({value}) is outside the specified range {variable_range}")
                    # If the variable is a DataFrame, check each column's values against the specified range
                    elif isinstance(value, pd.DataFrame):
                        # Check for DataFrames
                        for column in value.columns:
                            column_range = variable_range.get(column)
                            if column_range:
                                # Check if all values of the column are in the specified range
                                if not value[column].between(column_range[0], column_range[1]).all():
                                    raise ValueError(
                                        f"The values in column '{column}' of '{key}' are outside the specified range {column_range}. Values={value[column]}")
                    # If the variable is a NumPy array or a list, check if all values are within the specified range
                    elif isinstance(value, (np.ndarray, list)):
                        # Check for arrays
                        if not np.all(np.logical_and(variable_range[0] <= value, value <= variable_range[1])):
                            raise ValueError(f"The values of '{key}' are outside the specified range {variable_range}. Value={value}")
                    # If the variable type is not supported, raise a TypeError
                    else:
                        raise TypeError(f"Unsupported type for variable '{key}'")
