from typing import Any

from numpy import array
from pandas import DataFrame
from sostrades_core.study_manager.study_manager import StudyManager

from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_story_telling.usecase_7_witness_coarse_mda_gdp_model_w_damage_w_co2_tax import (
    Study as WitnessStudy,
)


class Study(StudyManager):
    """The UQ study on witness_coarse_float_invest."""

    SAMPLE_GENERATOR_NAME = "SampleGenerator"

    UQ_NAME = "UncertaintyQuantification"

    def __init__(self) -> None:
        super().__init__(__file__)
        self.witness_study = WitnessStudy(execution_engine=self.ee)

    def setup_usecase(self) -> dict[str, Any]:
        """Setup the usecase.

        Returns:
            The usecase options.
        """
        self.witness_study.study_name = f"{self.study_name}.DOE.WITNESS"
        witness_params = self.witness_study.setup_usecase()
        # print([list(d.keys()) for d in witness_params])
        # design_space = params_witness[f"{self.witness_study.study_name}.design_space"]
        # params = {
        #     f"{self.study_name}.DOE.{'.'.join(k.split('.')[2:])}": v
        #     for k, v in params_witness.items()
        # }

        # Setup the uncertain inputs/outputs
        dspace_dict = {
            "variable": ["Damage.tp_a3"],
            "value": [array([3.0])],
            "lower_bnd": [array([1.0])],
            "upper_bnd": [array([6.0])],
            "enable_variable": [True],
            "activated_elem": [[True]],
        }
        design_space_uq = DataFrame(dspace_dict)
        # design_space = concat((design_space, design_space_uq))

        input_selection = {
            "selected_input": [
                True,
                # False,
                # False,
                # False,
                # False,
                # False,
                # False,
                # True,
                # False,
            ],
            "full_name": [
                # f"WITNESS.RenewableTechnoInfo.Opex_percentage",
                # f"WITNESS.RenewableTechnoInfo.Initial_capex",
                # f"WITNESS.RenewableTechnoInfo.Energy_costs",
                # f"WITNESS.FossilTechnoInfo.Opex_percentage",
                # f"WITNESS.FossilTechnoInfo.Initial_capex",
                # f"WITNESS.FossilTechnoInfo.Energy_costs",
                # f"WITNESS.FossilTechnoInfo.CO2_from_production",
                f"WITNESS.Damage.tp_a3",
                # f"WITNESS.Temperature_change.init_temp_atmo",
            ],
        }
        input_selection = DataFrame(input_selection)
        output_selection = {
            "selected_output": [True, True, True, True, True, True, True, True],
            # "selected_output": [True],
            "full_name": [
                f"WITNESS.Indicators.mean_energy_price_2100",
                f"WITNESS.Indicators.fossil_energy_price_2100",
                f"WITNESS.Indicators.renewable_energy_price_2100",
                f"WITNESS.Indicators.total_energy_production_2100",
                f"WITNESS.Indicators.fossil_energy_production_2100",
                f"WITNESS.Indicators.renewable_energy_production_2100",
                f"WITNESS.Indicators.world_net_product_2100",
                f"WITNESS.Indicators.temperature_rise_2100",
                # f"{self.study_name}.DOE.AgricultureMix.Crop.techno_production"
            ],
        }
        output_selection = DataFrame(output_selection)

        # DOE sampling
        doe_params = {
            f"{self.study_name}.{self.SAMPLE_GENERATOR_NAME}.{key}": value
            for key, value in {
            "sampling_method": "doe_algo",
            "sampling_algo": "fullfact",
            "design_space": design_space_uq,
            "algo_options": {"n_samples": 5},
            "eval_inputs": input_selection,
            "sampling_generation_mode": "at_run_time",
        }.items()
        }
        doe_params[f"{self.study_name}.DOE.with_sample_generator"] = True
        doe_params[f"{self.study_name}.DOE.gather_outputs"] = output_selection

        # UQ
        uq_params = {
                f"{self.study_name}.{self.UQ_NAME}.eval_inputs": input_selection,
                f"{self.study_name}.{self.UQ_NAME}.design_space": design_space_uq,
                f"{self.study_name}.{self.UQ_NAME}.gather_outputs": output_selection,
            }

        return [*witness_params, doe_params, uq_params]


if "__main__" == __name__:
    usecase = Study()
    usecase.load_data()
    usecase.execution_engine.display_treeview_nodes(display_variables=False)
    usecase.run()
