from climateeconomics.sos_wrapping.sos_wrapping_witness.post_proc_sectors.emissions.post_processing_sector_breakdown_emissions import post_processings as pp_template
from climateeconomics.sos_wrapping.sos_wrapping_witness.post_proc_sectors.emissions.post_processing_sector_breakdown_emissions import post_processing_filters as ppf_template
from energy_models.glossaryenergy import GlossaryEnergy


def post_processing_filters(execution_engine, namespace):
    return ppf_template(execution_engine, namespace)

def post_processings(execution_engine, scenario_name, chart_filters=None):
    return pp_template(execution_engine, scenario_name, sector=GlossaryEnergy.SectorIndustry, chart_filters=chart_filters)
