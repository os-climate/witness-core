from sostrades_core.sos_processes.base_process_builder import BaseProcessBuilder


class ProcessBuilder(BaseProcessBuilder):

    _ontology_data = {
        "label": "WITNESS Resource Copper V0 Process",
        "description": "",
        "category": "",
        "version": "",
    }

    def get_builders(self):

        mods_dict = {
            "CopperModel": "climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_copper_resource_v0.copper_disc.CopperDisc"
        }
        builder_list = self.create_builder_list(mods_dict=mods_dict, ns_dict={"ns_public": self.ee.study_name})
        return builder_list
