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
from scipy.optimize import LinearConstraint
import numpy as np
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

from climateeconomics.calibration.crop.data import CalibrationData, output_calibration_datas, input_calibration_datas, \
    unused_workforce_data, unused_energy_data
from climateeconomics.calibration.crop.tools import DesignVar, DesignSpace
from climateeconomics.glossarycore import GlossaryCore

from climateeconomics.sos_processes.iam.witness.crop_2.usecase import Study

year_start_calibration = 2022
year_end_calibration = 2023
study = Study(year_start=year_start_calibration, year_end=year_end_calibration)
input_data = study.setup_usecase()

# FIRST STEP: CALIBRATION OF CAPEXES AND INVEST SHARES TO MATCH ACTUAL PRODUCTION

for calibration_data in input_calibration_datas:
    calibration_data: CalibrationData
    matchink_key_input_datas = list(filter(lambda x: calibration_data.varname in x, input_data.keys()))
    if len(matchink_key_input_datas) != 1:
        raise ValueError(f"Expected 1 matching key, got {len(matchink_key_input_datas)}")
    key = matchink_key_input_datas[0]
    input_data[key][calibration_data.key] = calibration_data.value


food_types = GlossaryCore.DefaultFoodTypes

design_vars = []
design_vars.extend([
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.Fish, initial_value=8.80, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.RedMeat, initial_value=8.90, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.WhiteMeat, initial_value=13.06, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.Eggs, initial_value=4.58, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.FruitsAndVegetables, initial_value=8.95, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.Milk, initial_value=8.96, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.Cereals, initial_value=26.68, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.RiceAndMaize, initial_value=11.05, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareInvestFoodTypesName, key=GlossaryCore.OtherFood, initial_value=8.97, min_value=1, max_value=100),
])
"""
design_vars.extend([
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.Fish, initial_value=11.11, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.RedMeat, initial_value=11.11, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.WhiteMeat, initial_value=11.11, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.Eggs, initial_value=11.11, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.FruitsAndVegetables, initial_value=11.11, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.Milk, initial_value=11.11, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.Cereals, initial_value=11.11, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.RiceAndMaize, initial_value=11.11, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareEnergyUsageFoodTypesName, key=GlossaryCore.OtherFood, initial_value=11.11, min_value=1, max_value=100),
])
design_vars.extend([
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.Fish, initial_value=11.00, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.RedMeat, initial_value=11.34, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.WhiteMeat, initial_value=11.34, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.Eggs, initial_value=10.92, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.FruitsAndVegetables, initial_value=12.75, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.Milk, initial_value=11.56, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.Cereals, initial_value=12.97, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.RiceAndMaize, initial_value=7.13, min_value=1, max_value=100),
    DesignVar(varname=GlossaryCore.ShareWorkforceFoodTypesName, key=GlossaryCore.OtherFood, initial_value=10.94, min_value=1, max_value=100),
])

design_vars.extend([
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.Fish, initial_value=290, allowed_variation_pct=5),
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.RedMeat, initial_value=225, allowed_variation_pct=5),
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.WhiteMeat, initial_value=150, allowed_variation_pct=5),
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.Eggs, initial_value=112.5, allowed_variation_pct=5),
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.FruitsAndVegetables, initial_value=80, allowed_variation_pct=5),
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.Milk, initial_value=75, allowed_variation_pct=5),
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.Cereals, initial_value=37.5, allowed_variation_pct=5),
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.RiceAndMaize, initial_value=30, allowed_variation_pct=5),
    DesignVar(varname=GlossaryCore.FoodTypeCapexName, key=GlossaryCore.OtherFood, initial_value=100, min_value=5, max_value=400),
])
"""

#design_vars.extend([DesignVar(varname=GlossaryCore.FoodTypeEnergyNeedName, key=food_type, min_value=0.001, max_value=10 ** 4, initial_value=1000) for food_type in food_types])
#design_vars.extend([DesignVar(varname=GlossaryCore.FoodTypeWorkforceNeedName, key=food_type, min_value=0.001, max_value=10 ** 4, initial_value=0.001) for food_type in food_types])
design_vars.append(DesignVar(varname=GlossaryCore.FoodTypeKcalByProdUnitName, key=GlossaryCore.OtherFood, min_value=0.001, max_value=10 ** 4, initial_value=0.001))

# equality constraint
equality_constraints = []
# 1 : shares variables : sum must equal 100
for i in range(3):
    A = np.zeros((1, len(design_vars)))
    A[0, i * len(food_types): (i + 1) * len(food_types)] = 1
    b = np.ones(1) * 100
    equality_constraints.append(LinearConstraint(A, lb=b*0.99, ub=b*1.01))

# 1 : shares variables : they must deviate to much from each other
# share of investment, energy, and workforce attributed for one food type must be close to each other
"""
for i in range(len(food_types)):
    A = np.zeros((3, len(design_vars)))
    A[0, i + 0 * len(food_types)] = 1
    A[0, i + 1 * len(food_types)] = -1
    A[1, i + 0 * len(food_types)] = 1
    A[1, i + 2 * len(food_types)] = -1
    A[1, i + 1 * len(food_types)] = 1
    A[1, i + 2 * len(food_types)] = -1
    b = np.ones(3) * 15 # % max deviation
    equality_constraints.append(LinearConstraint(A, lb=-b, ub=b))
"""

design_space = DesignSpace(design_vars)

def loss_function(x: np.ndarray, print_report=False, show_graph=False):
    design_space.set_x(x)
    input_data_to_update = design_space.get_updated_input_data_from_x(x)
    for (variable_name, key, value) in input_data_to_update:
        matchink_key_input_datas = list(filter(lambda x: variable_name in x, input_data.keys()))
        if len(matchink_key_input_datas) != 1:
            raise ValueError(f"Expected 1 matching key, got {len(matchink_key_input_datas)}")
        matching_key = matchink_key_input_datas[0]
        input_data[matching_key][key] = value

    study = Study(data=input_data)
    study.load_data()
    study.run()
    ouputs = study.ee.dm.get_data_dict_values()

    loss = 0
    report = "CALIBRATION REPORT\n"

    # to match : productions
    for calibration_data in output_calibration_datas:
        calibration_data: CalibrationData
        matchink_key_output_datas = list(filter(lambda x: calibration_data.varname == x.split('.')[-1], ouputs.keys()))
        if len(matchink_key_output_datas) != 1:
            raise ValueError(f"Expected 1 matching key, got {len(matchink_key_output_datas)}")
        matching_key = matchink_key_output_datas[0]
        model_value = ouputs[matching_key][calibration_data.key].values[0]
        relative_error = ( (model_value - calibration_data.value) / calibration_data.value)
        report += "{:<20} {:<20} {:<10} {:<20}\n".format(
            calibration_data.varname,
            calibration_data.key,
            calibration_data.year,
            f"relative error {int(relative_error * 100)}%"
        )
        loss += (relative_error) ** 2 / len(output_calibration_datas)

    # to minimize : unused workforce and energy
    """
    for calibration_data in unused_workforce_data:
        calibration_data: CalibrationData
        matchink_key_output_datas = list(filter(lambda x: calibration_data.varname == x.split('.')[-1], ouputs.keys()))
        if len(matchink_key_output_datas) != 1:
            raise ValueError(f"Expected 1 matching key, got {len(matchink_key_output_datas)}")
        matching_key = matchink_key_output_datas[0]
        model_loss_capital = ouputs[matching_key][calibration_data.key].values[0]

        report += "{:<20} {:<20} {:<10} {:<20}\n".format(
            calibration_data.varname,
            calibration_data.key,
            calibration_data.year,
            f"{model_loss_capital}"
        )
        loss += model_loss_capital / len(unused_workforce_data) / 100
        # to minimize : unused workforce and energy
    for calibration_data in unused_energy_data:
        calibration_data: CalibrationData
        matchink_key_output_datas = list(
            filter(lambda x: calibration_data.varname == x.split('.')[-1], ouputs.keys()))
        if len(matchink_key_output_datas) != 1:
            raise ValueError(f"Expected 1 matching key, got {len(matchink_key_output_datas)}")
        matching_key = matchink_key_output_datas[0]
        model_loss_capital = ouputs[matching_key][calibration_data.key].values[0]

        report += "{:<20} {:<20} {:<10} {:<20}\n".format(
            calibration_data.varname,
            calibration_data.key,
            calibration_data.year,
            f"{model_loss_capital}"
        )
        loss += model_loss_capital / len(unused_energy_data)

    """
    if print_report:
        print(report)
    print("error_calibration:", loss)
    if show_graph:
        ppf = PostProcessingFactory()
        post_procs = ppf.get_all_post_processings(execution_engine=study.ee, filters_only=False, as_json=False)
        for list_chart in post_procs.values():
            for pp in list_chart[0].post_processings:
                pp.to_plotly().show()
    return loss


print(design_space)
from scipy.optimize import minimize

x0 = design_space.get_x()
result = minimize(loss_function, x0, bounds=design_space.get_bounds(), constraints=equality_constraints,)


x_optimal = result.x

design_space.set_x_opt(x_optimal)
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M")
design_space.dump(f"calibration_optimale_{ts}.json")
print(design_space)

loss_function(x_optimal, print_report=True, show_graph=True)