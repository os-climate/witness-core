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
import logging
from typing import TYPE_CHECKING, Union

from sostrades_core.execution_engine.sos_wrapp import SoSWrapp

if TYPE_CHECKING:
    from climateeconomics.core.tools.differentiable_model import DifferentiableModel


class AutodifferentiedDisc(SoSWrapp):
    """Discipline which model is a DifferentiableModel"""
    coupling_inputs = []  # inputs verified during jacobian test
    coupling_outputs = []  # outputs verified during jacobian test

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)
        self.model: Union[DifferentiableModel, None] = None

    def compute_sos_jacobian(self):
        """
        Compute jacobian for each coupling variable
        """

        gradients = self.model.compute_jacobians_custom(outputs=self.coupling_outputs, inputs=self.coupling_inputs)
        for output_name in gradients:
            for output_col in gradients[output_name]:
                for input_name in gradients[output_name][output_col]:
                    for input_col, value in gradients[output_name][output_col][input_name].items():
                        self.set_partial_derivative_for_other_types(
                            (output_name, output_col),
                            (input_name, input_col),
                            value)
