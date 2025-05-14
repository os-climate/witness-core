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
from os.path import isfile
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class ColectedData:
    def __init__(
        self,
        value,
        unit: str,
        description: str,
        link: Union[str, list[str]],
        source: str,
        last_update_date: date,
        critical_at_year_start: bool = False,
        year_value: Union[int, None] = None,
        do_check: bool = True
    ):
        self.value = value
        self.unit = unit
        self.description = description
        self.link = link
        self.source = source
        self.last_update_date = last_update_date
        self.critical_at_year_start = critical_at_year_start
        self.year_value = year_value
        if do_check and critical_at_year_start and year_value is None:
            raise Exception(
                "Value is critical for year start, please specify what is the year of the value")

    @property
    def value(self):
        """getter of the value"""
        return self.__value

    @value.setter
    def value(self, val):
        self.__value = val

    @property
    def gui_description(self) -> str:
        """returns a description for displaying in GUI"""
        gui_descr = f"Defaults values infos :" \
                    f"{self.description}.\n" \
                    f"source : {self.source} ({self.link})\n" \
                    f"Lastly checked on {self.last_update_date.isoformat()}"
        return gui_descr


class HeavyCollectedData(ColectedData):
    """
    Class meant to store collected data that are heavy in terms of memory usage and loading time, like dataframe.

    The getter for the value has been overload to only read csv at this moment, and to avoid reading all csv when
    importing the Database. Also, once the getter has been called, the loaded value is cached to avoid
    new reading of a csv next time getter is called.
    """

    def __init__(
        self,
        value: str,
        unit: str,
        description: str,
        link: Union[str, list[str]],
        source: str,
        last_update_date: date,
        critical_at_year_start: bool = False,
        column_to_pick: Union[str, list[str]] = None,

    ):
        """

        :param value:
        :param unit:
        :param description:
        :param link:
        :param source:
        :param last_update_date:
        :param critical_at_year_start: If the data is used to initiate value at year start of discipline, set to True
        :param column_to_pick: when the dataframe is called at a specific, indicates which column to collect the value
        """
        if critical_at_year_start and column_to_pick is None:
            raise Exception("Dataframe is critical for year start, please specify in which column should the year start value be picked")
        super().__init__(value, unit, description, link, source, last_update_date, critical_at_year_start=critical_at_year_start, do_check=False)
        self.__cached_value = None
        self.column_to_pick = column_to_pick

    @property
    def value(self):
        """getter of the value"""
        if self.__cached_value is not None:
            return self.__cached_value
        self.__cached_value = pd.read_csv(self.__value)
        return self.__cached_value

    @value.setter
    def value(self, val: str):
        if not isinstance(val, str):
            raise ValueError("value must be a path for HeavyCollectedData")
        if not isfile(val):
            raise ValueError(f"{val} must be a file")
        self.__value = val

    def get_value_at_year(self, year: int, column: str = None) -> float:
        """Returns the dataframe value at the selected year. Interpolate data if needed when possible"""
        if column is None:
            column = self.column_to_pick
        year = int(year)
        df = self.value
        years_int = df['years'].values.astype(int)
        if year in years_int:
            return float(df.loc[df["years"] == year, column].values[0])

        if years_int.min() <= year <= years_int.max():
            # we will interpolate missing year data
            f = interp1d(x=years_int, y=df[column].values)
            return float(f(year))
        else:
            raise Exception("Donnée indisponible pour cette année")

    def get_df_at_year(self, year: int):
        """Returns a dataframe at the selected year, interpolation is applied if necessary"""
        if not isinstance(self.column_to_pick, list):
            raise TypeError('column_to_pick must be a list of string when calling this method')
        out = pd.DataFrame({
            col: self.get_value_at_year(year, col) for col in self.column_to_pick
        }, index=[0])
        return out

    def is_available_at_year(self, year: int) -> bool:
        """Indicate if data is available or can be interpolated at specified year"""
        year = int(year)
        df = self.value
        years_int = df['years'].values.astype(int)

        return years_int.min() <= year <= years_int.max()

    def get_between_years(self, year_start: int, year_end: int, column: str = None) -> pd.DataFrame:
        """Returns the dataframe between selected years. Interpolate data if needed when possible"""
        if column is None:
            column = self.column_to_pick
        df = self.value
        years = df["years"].values
        values = df[column].values
        f_interp = interp1d(x=years, y=values)
        all_years = np.arange(year_start, year_end + 1)
        if year_start < years.min() or year_end > years.max():
            raise ValueError("Données indisponible pour ces années")
        if year_start == year_end:
            out = pd.DataFrame({
                "years": all_years,
                column: values[0]
            })
        else:
            out = pd.DataFrame({
                "years": all_years,
                column: f_interp(all_years)
            })
        return out

    def get_all_cols_between_years(self, year_start: int, year_end: int) -> pd.DataFrame:
        df = self.value
        columns = list(df.columns)
        columns.remove("years")
        dfs = [self.get_between_years(year_start, year_end, column=col) for col in columns]
        out = pd.DataFrame({"years": dfs[0]["years"],
                            **{col: df[col] for col, df in zip(columns, dfs)}
        })
        return out
