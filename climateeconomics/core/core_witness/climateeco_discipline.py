'''
Copyright 2022 Airbus SAS

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
from sos_trades_core.execution_engine.sos_discipline import SoSDiscipline

class ClimateEcoDiscipline(SoSDiscipline):
    """
    Climate Economics Discipline
    """
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
            if value >=0:
                value_out = 0
            else: 
                value_out = value*1.1
        
        elif min_or_max == "max":
            if value >= 0:
                value_out = value*1.1
            else: 
                value_out = 0 
            
        return value_out 
            
            
            
        
        
        
    