
import numpy as np
import pandas as pd

from climateeconomics.database import DatabaseWitnessCore
from climateeconomics.glossarycore import GlossaryCore


def make_dspace_invests(dspace_dict: dict[str: list], year_start:[float], overwrite_invest_index: list[int] = []) -> pd.DataFrame:
    """
    :param dspace_dict: {variable_name: [value, lower_bnd, upper_bnd, enable_variable]}
    """
    out = {
        "variable": [],
        "value": [],
        "lower_bnd": [],
        "upper_bnd": [],
        "enable_variable": [],
        "activated_elem": [],
    }
    initial_values_first_pole = {
        'fossil.FossilSimpleTechno.fossil_FossilSimpleTechno_array_mix': DatabaseWitnessCore.InvestFossilYearStart.get_value_at_year(year_start),
        f"{GlossaryCore.clean_energy}.{GlossaryCore.CleanEnergySimpleTechno}.{GlossaryCore.clean_energy}_{GlossaryCore.CleanEnergySimpleTechno}_array_mix": DatabaseWitnessCore.InvestCleanEnergyYearStart.get_value_at_year(year_start),
        'carbon_capture.direct_air_capture.DirectAirCaptureTechno.carbon_capture_direct_air_capture_DirectAirCaptureTechno_array_mix': DatabaseWitnessCore.InvestCCUSYearStart.get_value_at_year(year_start) / 3,
        'carbon_capture.flue_gas_capture.FlueGasTechno.carbon_capture_flue_gas_capture_FlueGasTechno_array_mix': DatabaseWitnessCore.InvestCCUSYearStart.get_value_at_year(year_start) / 3,
        'carbon_storage.CarbonStorageTechno.carbon_storage_CarbonStorageTechno_array_mix': DatabaseWitnessCore.InvestCCUSYearStart.get_value_at_year(year_start) / 3,
    }

    for var, infos in dspace_dict.items():
        out['variable'].append(var)
        out['value'].append([initial_values_first_pole[var]] + [infos[0]] * (GlossaryCore.NB_POLES_COARSE - 1))
        out['lower_bnd'].append([infos[1]] * GlossaryCore.NB_POLES_COARSE)
        out['upper_bnd'].append([infos[2]] * GlossaryCore.NB_POLES_COARSE)
        out['enable_variable'].append(infos[3])
        out['activated_elem'].append([False] + [True] * (GlossaryCore.NB_POLES_COARSE - 1))

    for index in overwrite_invest_index:
        out['activated_elem'][index] = [False] * GlossaryCore.NB_POLES_COARSE
    out = pd.DataFrame(out)
    return out

def make_dspace_utilization_ratio(dspace_dict: dict[str: list]) -> pd.DataFrame:
    """
    :param dspace_dict: {variable_name: [value, lower_bnd, upper_bnd, enable_variable]}
    """
    out = {
        "variable": [],
        "value": [],
        "lower_bnd": [],
        "upper_bnd": [],
        "enable_variable": [],
        "activated_elem": [],
    }

    for var, infos in dspace_dict.items():
        out['variable'].append(var)
        out['value'].append([100.] + [infos[0]] * (GlossaryCore.NB_POLES_UTILIZATION_RATIO - 1))
        out['lower_bnd'].append([infos[1]] * (GlossaryCore.NB_POLES_UTILIZATION_RATIO))
        out['upper_bnd'].append([infos[2]] * GlossaryCore.NB_POLES_UTILIZATION_RATIO)
        out['enable_variable'].append(infos[3])
        out['activated_elem'].append([False] + [True] * (GlossaryCore.NB_POLES_UTILIZATION_RATIO - 1))

    out = pd.DataFrame(out)
    return out

def make_dspace_Ine(enable_variable: bool = False):
    value = DatabaseWitnessCore.ShareInvestNonEnergy.value
    low_bound = 19. if enable_variable else 0.99 * value
    upr_bound = 28. if enable_variable else 1.01 * value
    return pd.DataFrame({
        "variable": ["share_non_energy_invest_ctrl"],
        "value": [[value] * GlossaryCore.NB_POLES_COARSE],
        "lower_bnd": [[low_bound] * GlossaryCore.NB_POLES_COARSE],
        "upper_bnd": [[upr_bound] * GlossaryCore.NB_POLES_COARSE],
        "enable_variable": [enable_variable],
        "activated_elem": [[False] + [True] * (GlossaryCore.NB_POLES_COARSE - 1)]
    })

def get_ine_dvar_descr(year_start:[float], year_end:[float]):
    return {
        'out_name': GlossaryCore.ShareNonEnergyInvestmentsValue,
        'out_type': "dataframe",
        'key': GlossaryCore.ShareNonEnergyInvestmentsValue,
        'index': np.arange(year_start, year_end + 1),
        'index_name': GlossaryCore.Years,
        'namespace_in': GlossaryCore.NS_WITNESS,
        'namespace_out': GlossaryCore.NS_WITNESS,
    }