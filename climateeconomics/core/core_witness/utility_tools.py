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

from typing import Tuple

import autograd.numpy as np
import plotly.graph_objects as go
from autograd import jacobian

from climateeconomics.glossarycore import GlossaryCore


def get_inputs_for_utility_all_sectors(inputs_dict: dict):
    years = inputs_dict[GlossaryCore.PopulationDfValue][GlossaryCore.Years].to_numpy()
    population = inputs_dict[GlossaryCore.PopulationDfValue][GlossaryCore.PopulationValue].to_numpy()
    energy_price = inputs_dict[GlossaryCore.EnergyMeanPriceValue][GlossaryCore.EnergyPriceValue].to_numpy()

    return years, population, energy_price


def get_inputs_for_utility_per_sector(inputs_dict: dict, sector: str):
    consumption = inputs_dict[GlossaryCore.SectorizedConsumptionDfValue][sector].to_numpy()
    init_rate_time_pref = inputs_dict[f"{sector}_init_rate_time_pref"]
    scurve_stretch = inputs_dict[f"{sector}_strech_scurve"]
    scurve_shift = inputs_dict[f"{sector}_shift_scurve"]

    return consumption, init_rate_time_pref, scurve_shift, scurve_stretch


def compute_utility_discount_rate(years_range: np.ndarray, year_start: int, init_rate_time_pref: float) -> np.ndarray:
    """
    Compute utility discount rate.

    :param years_range: Array of years
    :param year_start: Starting year
    :param init_rate_time_pref: Initial rate of time preference
    :return: Array of utility discount rates

    The discount rate is calculated as:
    rr(t) = 1/((1+prstp)**(tstep*(t.val-1)))
    """
    t = ((years_range - year_start)) + 1
    u_discount_rate = 1 / ((1 + init_rate_time_pref) ** ((t - 1)))
    return u_discount_rate


def compute_quantity_pc(consumption_pc: np.ndarray, energy_price: np.ndarray) -> np.ndarray:
    """
    Compute utility per capita based on consumption and energy prices.

    :param consumption_pc: Array of consumption values
    :param energy_price: Array of energy prices
    :return: Array of utility per capita values

    Consumption = Quantity (of "things" consumed") * Price ("average price of things consumed")
    We consider that the average price of things that are consumed is driven by energy price.
    """
    consumption_pc_year_start = consumption_pc[0]
    quantity_year_start = consumption_pc_year_start / energy_price[0]
    quantity = consumption_pc / energy_price
    utility_quantity_pc = quantity / quantity_year_start
    return utility_quantity_pc



def compute_utility_per_capita(quantity_pc: np.ndarray, scurve_shift: float, scurve_stretch: float) -> np.ndarray:
    """
    Compute utility per capita based on consumption and energy prices.

    :param quantity_pc: Array with the utility quantity to apply the s-curve transformation
    :param scurve_shift: S-curve shift parameter
    :param scurve_stretch: S-curve stretch parameter
    :return: Array of utility per capita values transformed by s-curve
    """

    return s_curve_function(quantity_pc, scurve_shift, scurve_stretch)



def compute_utility_population(utility: np.ndarray, population: np.ndarray) -> np.ndarray:
    """
    Compute utility for the entire population.

    :param utility: Array of utility values
    :param population: Array of population values
    :return: Array of utility values for the population

    Utility quantity population (year) = Utility quantity per capita(year) * population (year)
    """
    pop_ratio = population / population[0]
    return pop_ratio * utility


def compute_utility_quantities(years: np.ndarray, consumption: np.ndarray, energy_price: np.ndarray,
                               population: np.ndarray, init_rate_time_pref: float,
                               scurve_shift: float, scurve_stretch: float):
    year_start = int(years[0])
    year_end = int(years[-1])
    years_range = np.arange(year_start, year_end + 1)

    consumption_pc = consumption / population
    quantity_pc = compute_quantity_pc(consumption_pc, energy_price)
    utility_pc = s_curve_function(quantity_pc, scurve_shift, scurve_stretch)
    discount_rate = compute_utility_discount_rate(years_range, year_start, init_rate_time_pref)
    discounted_utility_pc = utility_pc * discount_rate
    pop_ratio = population / population[0]
    discounted_utility_pop = pop_ratio * discounted_utility_pc

    return {GlossaryCore.UtilityDiscountRate: discount_rate, GlossaryCore.UtilityQuantity: quantity_pc,
            GlossaryCore.PerCapitaUtilityQuantity: utility_pc,
            GlossaryCore.DiscountedUtilityQuantityPerCapita: discounted_utility_pc,
            GlossaryCore.DiscountedQuantityUtilityPopulation: discounted_utility_pop,}


def compute_utility_quantities_der(quantity_name: str, years: np.ndarray, consumption: np.ndarray,
                                   energy_price: np.ndarray,
                                   population: np.ndarray, init_rate_time_pref: float,
                                   scurve_shift: float, scurve_stretch: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    args = (years, consumption, energy_price, population,
            init_rate_time_pref, scurve_shift, scurve_stretch)

    jac_consumption = jacobian(lambda *args: compute_utility_quantities(*args)[quantity_name], 1)
    jac_energy_price = jacobian(lambda *args: compute_utility_quantities(*args)[quantity_name], 2)
    jac_population = jacobian(lambda *args: compute_utility_quantities(*args)[quantity_name], 3)

    return jac_consumption(*args), jac_energy_price(*args), jac_population(*args)


def compute_utility_objective(years_range: np.ndarray, consumption: np.ndarray, energy_price: np.ndarray,
                              population: np.ndarray, init_rate_time_pref: float,
                              scurve_shift: float, scurve_stretch: float) -> float:
    consumption_pc = consumption / population
    quantity_pc = compute_quantity_pc(consumption_pc, energy_price)
    utility_pc = 1 - s_curve_function(quantity_pc, scurve_shift, scurve_stretch)
    discount_rate = compute_utility_discount_rate(years_range, years_range[0], init_rate_time_pref)
    discounted_utility_pc = utility_pc * discount_rate
    pop_ratio = population[0] / population
    discounted_utility_pop = pop_ratio * discounted_utility_pc

    return discounted_utility_pop.mean()


def compute_utility_objective_der(years: np.ndarray, consumption: np.ndarray, energy_price: np.ndarray,
                                  population: np.ndarray, init_rate_time_pref: float,
                                  scurve_shift: float, scurve_stretch: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the derivative of the utility objective function.

    :param years: Array of years
    :param consumption: Array of consumption values
    :param energy_price: Array of energy prices
    :param population: Array of population values
    :param init_rate_time_pref: Initial rate of time preference
    :param scurve_shift: S-curve shift parameter
    :param scurve_stretch: S-curve stretch parameter
    :return: Tuple of derivatives with respect to consumption, energy price, and population
    """
    d_consumption = jacobian(compute_utility_objective, 1)
    d_energy_price = jacobian(compute_utility_objective, 2)
    d_population = jacobian(compute_utility_objective, 3)

    args = (years, consumption, energy_price, population,
            init_rate_time_pref, scurve_shift, scurve_stretch)

    return d_consumption(*args), d_energy_price(*args), d_population(*args)


def compute_utility_quantities_bis(years: np.ndarray, consumption_pc: np.ndarray, energy_price: np.ndarray,
                                   population: np.ndarray, init_rate_time_pref: float,
                                   scurve_shift: float, scurve_stretch: float):
    year_start = int(years[0])
    year_end = int(years[-1])
    years_range = np.arange(year_start, year_end + 1)

    quantity_pc = compute_quantity_pc(consumption_pc, energy_price)
    utility_pc = s_curve_function(quantity_pc, scurve_shift, scurve_stretch)
    discount_rate = compute_utility_discount_rate(years_range, year_start, init_rate_time_pref)
    discounted_utility_pc = utility_pc * discount_rate
    pop_ratio = population / population[0]
    discounted_utility_pop = pop_ratio * discounted_utility_pc

    return {GlossaryCore.UtilityDiscountRate: discount_rate, GlossaryCore.UtilityQuantity: quantity_pc,
            GlossaryCore.PerCapitaUtilityQuantity: utility_pc,
            GlossaryCore.DiscountedUtilityQuantityPerCapita: discounted_utility_pc,
            GlossaryCore.DiscountedQuantityUtilityPopulation: discounted_utility_pop,}


def compute_utility_quantities_bis_der(quantity_name: str, years: np.ndarray, consumption_pc: np.ndarray,
                                       energy_price: np.ndarray,
                                       population: np.ndarray, init_rate_time_pref: float,
                                       scurve_shift: float, scurve_stretch: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    args = (years, consumption_pc, energy_price, population,
            init_rate_time_pref, scurve_shift, scurve_stretch)

    jac_consumption = jacobian(lambda *args: compute_utility_quantities_bis(*args)[quantity_name], 1)
    jac_energy_price = jacobian(lambda *args: compute_utility_quantities_bis(*args)[quantity_name], 2)
    jac_population = jacobian(lambda *args: compute_utility_quantities_bis(*args)[quantity_name], 3)

    return jac_consumption(*args), jac_energy_price(*args), jac_population(*args)


def compute_utility_objective_bis(years_range: np.ndarray, consumption_pc: np.ndarray, energy_price: np.ndarray,
                                  population: np.ndarray, init_rate_time_pref: float,
                                  scurve_shift: float, scurve_stretch: float) -> float:
    quantity_pc = compute_quantity_pc(consumption_pc, energy_price)
    utility_pc = 1 - s_curve_function(quantity_pc, scurve_shift, scurve_stretch)
    discount_rate = compute_utility_discount_rate(years_range, years_range[0], init_rate_time_pref)
    discounted_utility_pc = utility_pc * discount_rate
    pop_ratio = population[0] / population
    discounted_utility_pop = pop_ratio * discounted_utility_pc

    return discounted_utility_pop.mean()


def compute_utility_objective_bis_der(years: np.ndarray, consumption_pc: np.ndarray, energy_price: np.ndarray,
                                      population: np.ndarray, init_rate_time_pref: float,
                                      scurve_shift: float, scurve_stretch: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the derivative of the utility objective function.

    :param years: Array of years
    :param consumption_pc: Array of consumption values
    :param energy_price: Array of energy prices
    :param population: Array of population values
    :param init_rate_time_pref: Initial rate of time preference
    :param scurve_shift: S-curve shift parameter
    :param scurve_stretch: S-curve stretch parameter
    :return: Tuple of derivatives with respect to consumption, energy price, and population
    """
    d_consumption = jacobian(compute_utility_objective_bis, 1)
    d_energy_price = jacobian(compute_utility_objective_bis, 2)
    d_population = jacobian(compute_utility_objective_bis, 3)

    args = (years, consumption_pc, energy_price, population,
            init_rate_time_pref, scurve_shift, scurve_stretch)

    return d_consumption(*args), d_energy_price(*args), d_population(*args)


def compute_decreasing_gdp_obj(output_net_of_damage: np.ndarray):
    """
    decreasing net gdp obj =   Sum_i [min(Qi+1/Qi, 1) - 1] / nb_years

    Note: this objective is self normalized to [0,1], no need for reference.
    It should be minimized and not maximized !
    :return:
    :rtype:
    """
    increments = list(output_net_of_damage[1:]/output_net_of_damage[:-1])
    increments.append(0)
    increments = np.array(increments)

    increments[increments >= 1] = 1.
    increments -= 1
    increments[-1] = 0

    decreasing_gpd_obj = - np.array([np.mean(increments)])
    return decreasing_gpd_obj


def d_decreasing_gdp_obj(output_net_of_damage: np.ndarray):
    output_shift = list(output_net_of_damage[1:])
    output_shift.append(0)
    output_shift = np.array(output_shift)

    increments = list(output_net_of_damage[1:] / output_net_of_damage[:-1])
    increments.append(0)
    increments = np.array(increments)

    a = list(- output_shift / output_net_of_damage**2)
    derivative = np.diag(a) + np.diag(1/output_net_of_damage[:-1], k=1)

    derivative[increments > 1] = 0.
    for i, incr in enumerate(increments):
        if incr == 1:
            derivative[i, i+1] = 0.
    derivative = -np.mean(derivative, axis=0)

    return derivative

def s_curve_function(x: np.ndarray, shift: float, stretch: float) -> np.ndarray:
    """
    Compute the S-curve function.

    :param x: Input array
    :param shift: Shift parameter
    :param stretch: Stretch parameter
    :return: S-curve function values
    """
    y = (x - 1.0 - shift) * stretch
    s = np.exp(-y)
    return 1.0 / (1.0 + s)


def plot_s_curve(x: np.ndarray, shift: float, stretch: float, show: bool = False) -> go.Figure:
    """
    Create a Plotly plot of the S-curve transformation.

    :param x: Input array
    :param shift: Shift parameter
    :param stretch: Stretch parameter
    :return: Plotly Figure object
    """
    # Compute the S-curve transformation
    y = s_curve_function(x, shift, stretch)

    # Create the figure
    fig = go.Figure()

    # Add the S-curve trace
    fig.add_trace(go.Scatter(
        x=list(x),
        y=list(y),
        mode='lines',
        name='S-curve'
    ))

    # Update layout
    fig.update_layout(
        title=f'S-curve Transformation (Shift: {shift}, Stretch: {stretch})',
        xaxis_title='Original Data',
        yaxis_title='Transformed Data',
        xaxis=dict(range=[x.min(), x.max()]),
        yaxis=dict(range=[0, 1])
    )

    if show:
        fig.show()

    return fig
