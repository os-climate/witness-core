



from sos_trades_core.sos_processes.base_process_builder import BaseProcessBuilder

class ProcessBuilder(BaseProcessBuilder):

    def get_builders(self):
        
        mods_dict = {'CopperModel': 'climateeconomics.sos_wrapping.sos_wrapping_resources.sos_wrapping_copper_resource_v0.copper_disc.CopperDisc'}
        builder_list = self.create_builder_list(mods_dict=mods_dict,
                                      ns_dict=None)
        return builder_list
