from sostrades_core.study_manager.study_manager import StudyManager
from climateeconomics.core.core_world3.world3 import World3
from sostrades_core.tools.post_processing.post_processing_factory import PostProcessingFactory

from climateeconomics.tests.world3_test.test_sos_population import create_population_input
from climateeconomics.tests.world3_test.test_sos_pollution import create_pollution_input
from climateeconomics.tests.world3_test.test_sos_resource import create_resource_input
from climateeconomics.tests.world3_test.test_sos_agriculture import create_agriculture_input
from climateeconomics.tests.world3_test.test_sos_capital import create_capital_input


def fast_world3(name):
    world3 = World3()
    world3.init_world3_constants()
    world3.init_world3_variables()
    world3.set_world3_table_functions()
    world3.set_world3_delay_functions()
    world3.run_world3(fast=False)
    output = world3.__dict__
    values_dict = {}
    for k in output:
        if '_' not in k:
            values_dict[name + "." + str(k)] = output[k]
    return (values_dict)


class Study(StudyManager):

    def __init__(self, execution_engine=None, run_usecase=False):
        super().__init__(__file__, execution_engine=execution_engine, run_usecase=run_usecase)

    def setup_usecase(self):
        setup_data_list = []

        world3_input = {}

        world3_input.update(create_population_input(self.study_name))
        world3_input.update(create_capital_input(self.study_name))
        world3_input.update(create_agriculture_input(self.study_name))
        world3_input.update(create_resource_input(self.study_name))
        world3_input.update(create_pollution_input(self.study_name))

        world3_input[self.study_name + '.sub_mda_class'] = "MDAGaussSeidel"
        world3_input[self.study_name + '.n_processes'] = 4
        world3_input[self.study_name + '.tolerance'] = 1.e2
        world3_input[self.study_name + '.max_mda_iter'] = 200
        world3_input[self.study_name + '.relax_factor'] = 0.99

        setup_data_list.append(world3_input)

        return setup_data_list


if '__main__' == __name__:
    uc_cls = Study(run_usecase=True)
    uc_cls.test()
