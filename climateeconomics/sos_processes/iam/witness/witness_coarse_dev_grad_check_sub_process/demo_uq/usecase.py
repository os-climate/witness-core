from typing import Any

from numpy import array
from pandas import DataFrame
from sostrades_core.study_manager.study_manager import StudyManager

from climateeconomics.sos_processes.iam.witness.witness_coarse_dev_grad_check_sub_process.usecase_2_fossil_only_damage_high_tax import (
    Study as WitnessStudy,
)


class Study(StudyManager):
    """The UQ study on witness_coarse_float_invest."""

    SAMPLE_GENERATOR_NAME = "SampleGenerator"

    UQ_NAME = "UncertaintyQuantification"

    def __init__(self) -> None:
        super().__init__(__file__)
        self.witness_study = WitnessStudy(run_usecase=True, execution_engine=self.ee)

    def setup_usecase(self) -> dict[str, Any]:
        """Setup the usecase.

        Returns:
            The usecase options.
        """
        self.witness_study.study_name = f"{self.study_name}.DOE"
        params_witness = self.witness_study.setup_usecase()[0]
        params = {
            f"{self.study_name}.DOE.{'.'.join(k.split('.')[2:])}": v
            for k, v in params_witness.items()
        }

        # Setup the uncertain inputs/outputs
        dspace_dict = {
            "variable": [f"{self.study_name}.Damage.tp_a3"],
            "value": [array([3.0])],
            "lower_bnd": [array([1.0])],
            "upper_bnd": [array([6.0])],
            "enable_variable": [True],
            "activated_elem": [[True]],
        }
        design_space_uq = DataFrame(dspace_dict)
        input_selection = {
            "selected_input": [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
            ],
            "full_name": [
                f"{self.study_name}.RenewableTechnoInfo.Opex_percentage",
                f"{self.study_name}.RenewableTechnoInfo.Initial_capex",
                f"{self.study_name}.RenewableTechnoInfo.Energy_costs",
                f"{self.study_name}.FossilTechnoInfo.Opex_percentage",
                f"{self.study_name}.FossilTechnoInfo.Initial_capex",
                f"{self.study_name}.FossilTechnoInfo.Energy_costs",
                f"{self.study_name}.FossilTechnoInfo.CO2_from_production",
                f"{self.study_name}.Damage.tp_a3",
                f"{self.study_name}.Temperature_change.init_temp_atmo",
            ],
        }
        input_selection = DataFrame(input_selection)
        output_selection = {
            "selected_output": [True, True, True, True, True, True, True, True],
            "full_name": [
                "WITNESS_Eval.Indicators.mean_energy_price_2100",
                "WITNESS_Eval.Indicators.fossil_energy_price_2100",
                "WITNESS_Eval.Indicators.renewable_energy_price_2100",
                "WITNESS_Eval.Indicators.total_energy_production_2100",
                "WITNESS_Eval.Indicators.fossil_energy_production_2100",
                "WITNESS_Eval.Indicators.renewable_energy_production_2100",
                "WITNESS_Eval.Indicators.world_net_product_2100",
                "WITNESS_Eval.Indicators.temperature_rise_2100",
            ],
        }
        output_selection = DataFrame(output_selection)

        # DOE sampling
        sampling_params = {
            "sampling_method": "doe_algo",
            "sampling_algo": "fullfact",
            "design_space": design_space_uq,
            "algo_options": {"n_samples": 5},
            "eval_inputs": input_selection,
            "sampling_generation_mode": "at_run_time",
        }
        params.update({
            f"{self.study_name}.{self.SAMPLE_GENERATOR_NAME}.{key}": value
            for key, value in sampling_params.items()
        })
        params[f"{self.study_name}.DOE.with_sample_generator"] = True
        params[f"{self.study_name}.DOE.gather_outputs"] = output_selection

        # UQ
        params.update(
            {
                f"{self.study_name}.{self.UQ_NAME}.eval_inputs": input_selection,
                f"{self.study_name}.{self.UQ_NAME}.design_space": design_space_uq,
                f"{self.study_name}.{self.UQ_NAME}.gather_outputs": output_selection,
            }
        )

        return [params]


if "__main__" == __name__:
    usecase = Study()
    usecase.load_data()
    usecase.execution_engine.display_treeview_nodes(display_variables=True)
    usecase.run()
