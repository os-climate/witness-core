from os.path import isfile
import pandas as pd
from datetime import date


class ColectedData:
    def __init__(
        self,
        value,
        unit: str,
        description: str,
        link: str,
        source: str,
        last_update_date: date,
    ):
        self.value = value
        self.unit = unit
        self.description = description
        self.link = link
        self.source = source
        self.last_update_date = last_update_date

    @property
    def value(self):
        """getter of the value"""
        return self.__value

    @value.setter
    def value(self, val):
        self.__value = val


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
        link: str,
        source: str,
        last_update_date: date,
    ):
        super().__init__(value, unit, description, link, source, last_update_date)
        self.__cached_value = None

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
