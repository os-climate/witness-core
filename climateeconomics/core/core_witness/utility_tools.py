from typing import Tuple

import autograd.numpy as np
import plotly.graph_objects as go
from autograd import elementwise_grad, jacobian

from climateeconomics.glossarycore import GlossaryCore


def get_inputs_for_utility_all_sectors(inputs_dict: dict):
    years = inputs_dict[GlossaryCore.EconomicsDfValue][GlossaryCore.Years].to_numpy()
    population = inputs_dict[GlossaryCore.PopulationDfValue][GlossaryCore.PopulationValue].to_numpy()
    energy_price = inputs_dict[GlossaryCore.EnergyMeanPriceValue][GlossaryCore.EnergyPriceValue].to_numpy()

    return years, population, energy_price


def get_inputs_for_utility_per_sector(inputs_dict: dict, sector: str):
    consumption = inputs_dict[GlossaryCore.AllSectorsDemandDfValue][sector].to_numpy()
    energy_price_ref = inputs_dict[f"{sector}.initial_raw_energy_price"]
    init_rate_time_pref = inputs_dict[f"{sector}.init_rate_time_pref"]
    scurve_stretch = inputs_dict[f"{sector}.strech_scurve"]
    scurve_shift = inputs_dict[f"{sector}.shift_scurve"]

    return consumption, energy_price_ref, init_rate_time_pref, scurve_shift, scurve_stretch


def compute_utility_discount_rate(years_range: np.ndarray, year_start: int, time_step: int,
                                  init_rate_time_pref: float) -> np.ndarray:
    """
    Compute utility discount rate.

    :param years_range: Array of years
    :param year_start: Starting year
    :param time_step: Time step between years
    :param init_rate_time_pref: Initial rate of time preference
    :return: Array of utility discount rates

    The discount rate is calculated as:
    rr(t) = 1/((1+prstp)**(tstep*(t.val-1)))
    """
    t = ((years_range - year_start) / time_step) + 1
    u_discount_rate = 1 / ((1 + init_rate_time_pref) ** (time_step * (t - 1)))
    return u_discount_rate


def compute_utility_quantity(consumption: np.ndarray, energy_price: np.ndarray,
                             energy_price_ref: float, ) -> np.ndarray:
    """
    Compute utility per capita based on consumption and energy prices.

    :param consumption: Array of consumption values
    :param energy_price: Array of energy prices
    :param energy_price_ref: Reference energy price
    :return: Array of utility per capita values

    Consumption = Quantity (of "things" consumed") * Price ("average price of things consumed")
    We consider that the average price of things that are consumed is driven by energy price.
    """
    consumption_year_start = consumption[0]
    quantity_year_start = consumption_year_start / energy_price_ref
    quantity = consumption / energy_price
    utility_quantity = quantity / quantity_year_start
    return utility_quantity


def compute_utility_per_capita(utility_quantity: np.ndarray, population: np.array,
                               scurve_shift: float, scurve_stretch: float) -> np.ndarray:
    """
    Compute utility per capita based on consumption and energy prices.

    :param utility_quantity: Array with the utility quantity to apply the s-curve transformation
    :param scurve_shift: S-curve shift parameter
    :param scurve_stretch: S-curve stretch parameter
    :return: Array of utility per capita values transformed by s-curve
    """

    utility_pc = utility_quantity / population

    return s_curve_function(utility_pc, scurve_shift, scurve_stretch)


def compute_discounted_utility(utility: np.ndarray, discount_factor: np.ndarray) -> np.ndarray:
    """
    Compute discounted utility.

    :param utility: Array of utility values
    :param discount_factor: Array of discount factors
    :return: Array of discounted utility values

    Discounted utility quantity (year) = Utility quantity(year) * discount factor (year)
    """
    return utility * discount_factor


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
                               population: np.ndarray, energy_price_ref: float, init_rate_time_pref: float,
                               scurve_shift: float, scurve_stretch: float):
    year_start = int(years[0])
    year_end = int(years[-1])
    time_step = int(years[1]) - int(years[0])
    years_range = np.arange(year_start, year_end + 1, time_step)

    utility_quantity = compute_utility_quantity(consumption, energy_price, energy_price_ref)
    utility_pc = compute_utility_per_capita(utility_quantity, population, scurve_shift, scurve_stretch)
    discount_rate = compute_utility_discount_rate(years_range, year_start, time_step, init_rate_time_pref)
    discounted_utility_pc = compute_discounted_utility(utility_pc, discount_rate)
    discounted_utility_pop = compute_utility_population(discounted_utility_pc, population)
    utility_obj = np.mean(discounted_utility_pop)

    return {GlossaryCore.UtilityDiscountRate: discount_rate, GlossaryCore.UtilityQuantity: utility_quantity,
            GlossaryCore.PerCapitaUtilityQuantity: utility_pc,
            GlossaryCore.DiscountedUtilityQuantityPerCapita: discounted_utility_pc,
            GlossaryCore.DiscountedQuantityUtilityPopulation: discounted_utility_pop,
            GlossaryCore.UtilityObjectiveName: utility_obj}


def compute_utility_quantities_der(quantity_name: str, years: np.ndarray, consumption: np.ndarray,
                                   energy_price: np.ndarray,
                                   population: np.ndarray, energy_price_ref: float, init_rate_time_pref: float,
                                   scurve_shift: float, scurve_stretch: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    args = (years, consumption, energy_price, population, energy_price_ref,
            init_rate_time_pref, scurve_shift, scurve_stretch)

    jac_consumption = jacobian(lambda *args: compute_utility_quantities(*args)[quantity_name], 1)
    jac_energy_price = jacobian(lambda *args: compute_utility_quantities(*args)[quantity_name], 2)
    jac_population = jacobian(lambda *args: compute_utility_quantities(*args)[quantity_name], 3)

    return jac_consumption(*args), jac_energy_price(*args), jac_population(*args)


def compute_utility_objective(years: np.ndarray, consumption: np.ndarray, energy_price: np.ndarray,
                              population: np.ndarray, energy_price_ref: float, init_rate_time_pref: float,
                              scurve_shift: float, scurve_stretch: float) -> float:
    """
    Compute the utility objective function.

    :param years: Array of years
    :param consumption: Array of consumption values
    :param energy_price: Array of energy prices
    :param population: Array of population values
    :param energy_price_ref: Reference energy price
    :param init_rate_time_pref: Initial rate of time preference
    :param scurve_shift: S-curve shift parameter
    :param scurve_stretch: S-curve stretch parameter
    :return: 1.0 - Utility objective value
    """

    utility_quantities = compute_utility_quantities(years,
                                                    consumption,
                                                    energy_price,
                                                    population,
                                                    energy_price_ref,
                                                    init_rate_time_pref,
                                                    scurve_shift,
                                                    scurve_stretch)

    return 1.0 - utility_quantities[GlossaryCore.UtilityObjectiveName]


def compute_utility_objective_der(years: np.ndarray, consumption: np.ndarray, energy_price: np.ndarray,
                                  population: np.ndarray, energy_price_ref: float, init_rate_time_pref: float,
                                  scurve_shift: float, scurve_stretch: float) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the derivative of the utility objective function.

    :param years: Array of years
    :param consumption: Array of consumption values
    :param energy_price: Array of energy prices
    :param population: Array of population values
    :param energy_price_ref: Reference energy price
    :param init_rate_time_pref: Initial rate of time preference
    :param scurve_shift: S-curve shift parameter
    :param scurve_stretch: S-curve stretch parameter
    :return: Tuple of derivatives with respect to consumption, energy price, and population
    """
    d_consumption = jacobian(compute_utility_objective, argnum=1)
    d_energy_price = jacobian(compute_utility_objective, argnum=2)
    d_population = jacobian(compute_utility_objective, argnum=3)

    args = (years, consumption, energy_price, population, energy_price_ref,
            init_rate_time_pref, scurve_shift, scurve_stretch)

    return d_consumption(*args), d_energy_price(*args), d_population(*args)


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


def d_s_curve_function(x: np.ndarray, shift: float, stretch: float, use_autograd: bool = True) -> np.ndarray:
    """
    Compute the derivative of the S-curve function.

    :param x: Input array
    :param shift: Shift parameter
    :param stretch: Stretch parameter
    :param use_autograd: Whether to use autograd for computation
    :return: Derivative of S-curve function values
    """
    if use_autograd:
        grad_scurve = elementwise_grad(s_curve_function)
        return grad_scurve(x, shift, stretch)
    else:
        u_prime = stretch
        u = (x - 1.0 - shift) * stretch
        f_prime_u = np.exp(-u) / (1.0 + np.exp(-u)) ** 2.0
        return u_prime * f_prime_u


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


if __name__ == "__main__":
    test_x = np.linspace(0, 50, 10)

    test_shift = 1.0
    test_stretch = 2.0

    s_values = s_curve_function(test_x, test_shift, test_stretch)

    # ds_values = d_s_curve_function(test_x, test_shift, test_stretch)
    # print(ds_values)
    #
    # ds_values = d_s_curve_function(test_x, test_shift, test_stretch, use_autograd=False)
    # print(ds_values)

    #

    years = np.array([x + 2020 for x in range(10)])
    consumption = np.linspace(100.0, 10000.0, len(years))
    population = np.ones_like(years) * 1.0
    energy_price = np.ones_like(years) * 1.0
    energy_price_ref = 1.0
    init_rate_time_pref = 1.0

    jac = compute_utility_objective_der(years, consumption, energy_price, population, energy_price_ref,
                                        init_rate_time_pref,
                                        test_shift, test_stretch)

    print(jac)

    # jac = compute_utility_quantities_der(GlossaryCore.DiscountedUtilityQuantityPerCapita, years, consumption,
    #                                      energy_price, population, energy_price_ref,
    #                                      init_rate_time_pref,
    #                                      test_shift, test_stretch)
    #
    # print(jac)

    utility_quantity = compute_utility_quantity(consumption, energy_price, energy_price_ref)
    utility_pc = compute_utility_per_capita(consumption, energy_price, energy_price_ref, test_shift, test_stretch)

    plot_s_curve(utility_quantity, test_shift, test_stretch, show=False)
