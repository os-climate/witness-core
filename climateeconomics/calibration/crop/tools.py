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
from typing import Union

import numpy as np


def solve_share_prod_waste(total_share_waste, share_waste):
    # total_share_waste / 100 = share prod waste / 100 + share consumers waste/ 100 - share prod waste * share consumers / 10000
    # total_share_waste = share prod waste + share consumers waste - share prod waste * share consumers / 100
    # share prod waste ( 1 - share consumers / 100) = total_share_waste - share consumers waste
    # share prod waste  = total_share_waste - share consumers waste / ( 1 - share consumers / 100)
    share_prod_waste = (total_share_waste - share_waste) / (1 - share_waste / 100)
    return int(share_prod_waste * 100) / 100

class CalibrationData:
    def __init__(self,
                 varname: str,
                 column_name: str,
                 year: int,
                 value: float,
                 source: str,
                 unit: str,
                 link: str,
                 ):
        self.varname = varname
        self.key = column_name
        self.year = year
        self.value = value
        self.source = source
        self.link = link
        self.unit = unit

    def __add__(self, other):
        if self.varname != other.varname:
            raise ValueError('Cannot add two CalibrationData with different outputvarname')
        if self.key != other.key:
            raise ValueError('Cannot add two CalibrationData with different key')
        if self.year != other.year:
            raise ValueError('Cannot add two CalibrationData with different year')
        return CalibrationData(
            varname=self.varname,
            column_name=self.key,
            year=self.year,
            value=self.value + other.value,
            source=self.source + ';' + other.source,
            link=self.link + ';' + other.link,
            unit=self.unit,
        )

    def __mul__(self, other):
        if self.varname != other.varname:
            raise ValueError('Cannot multiply two CalibrationData with different outputvarname')
        if self.key != other.key:
            raise ValueError('Cannot multiply two CalibrationData with different key')
        if self.year != other.year:
            raise ValueError('Cannot multiply two CalibrationData with different year')
        return CalibrationData(
            varname=self.varname,
            column_name=self.key,
            year=self.year,
            value=self.value * other.value,
            source=self.source + ';' + other.source,
            link=self.link + ';' + other.link,
            unit=self.unit,
        )



class DesignVar:
    def __init__(self, varname: str, key: str, initial_value: float, min_value: Union[float, None] = None, max_value: Union[float, None] = None, allowed_variation_pct: Union[float, None] = None):
        self.varname = varname
        self.key = key
        self.min_value = min_value
        self.max_value = max_value
        self.value = initial_value
        self.initial_value = initial_value
        self.allowed_variation = allowed_variation_pct
        self.lb, self.ub = self._compute_bounds()
        self.optimal_value = None

    def __repr__(self):
        return f"\nDesignVar({self.varname:<30}, {self.key:<23}, {np.round(self.value, 2):<4}, {np.round(self.lb, 2):>4}, {np.round(self.ub, 2):<4})"

    def _compute_bounds(self) -> tuple:
        if self.allowed_variation is not None:
            return self.value * (1 - self.allowed_variation / 100), self.value * (1 + self.allowed_variation / 100)
        else:
            return self.min_value, self.max_value

    def to_dict(self):
        return {
            'varname': self.varname,
            'key': self.key,
            'optimal_value': self.optimal_value,
        }

    def get_bounds(self) -> tuple:
        return self.lb, self.ub


class DesignSpace:
    def __init__(self, design_vars: list[DesignVar]):
        self.design_vars = design_vars
        self.dict_design_vars = {i: design_var for i, design_var in enumerate(design_vars)}

    def get_x(self) -> np.ndarray:
        return np.array([design_var.value for design_var in self.design_vars])

    def set_x(self, x):
        for i in range(len(x)):
            self.dict_design_vars[i].value = x[i]

    def set_x_opt(self, x):
        for i in range(len(x)):
            self.dict_design_vars[i].optimal_value = x[i]

    def get_bounds(self) -> list[tuple]:
        return [design_var.get_bounds() for design_var in self.design_vars]

    def get_updated_input_data_from_x(self, x: np.ndarray) -> list[tuple]:
        return [(self.dict_design_vars[i].varname, self.dict_design_vars[i].key, x[i]) for i in range(len(x))]

    def __repr__(self):
        return f"DesignSpace({self.design_vars})"

    def to_dict(self):
        return [design_var.to_dict() for design_var in self.design_vars]

    def dump(self, path):
        with open(path, 'w') as f:
            f.write(str(self.to_dict()))


