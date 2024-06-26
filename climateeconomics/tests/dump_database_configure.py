"""
Copyright 2022 Airbus SAS
Modifications on 2024/06/07 Copyright 2024 Capgemini
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import json
from copy import deepcopy
from importlib import import_module
from os import listdir
from os.path import dirname
from tempfile import gettempdir

import numpy as np
import pandas as pd
from sostrades_core.sos_processes.processes_factory import SoSProcessFactory
from sostrades_core.study_manager.base_study_manager import BaseStudyManager

NUMERICAL_KEYS = [
    "<study_ph>.sub_mda_class",
    "<study_ph>.max_mda_iter",
    "<study_ph>.n_processes",
    "<study_ph>.chain_linearize",
    "<study_ph>.tolerance",
    "<study_ph>.use_lu_fact",
    "<study_ph>.warm_start",
    "<study_ph>.acceleration",
    "<study_ph>.warm_start_threshold",
    "<study_ph>.n_subcouplings_parallel",
    "<study_ph>.tolerance_gs",
    "<study_ph>.relax_factor",
    "<study_ph>.epsilon0",
    "<study_ph>.linear_solver_MDO",
    "<study_ph>.linear_solver_MDO_preconditioner",
    "<study_ph>.linear_solver_MDO_options",
    "<study_ph>.linear_solver_MDA",
    "<study_ph>.linear_solver_MDA_preconditioner",
    "<study_ph>.linear_solver_MDA_options",
    "<study_ph>.group_mda_disciplines",
    "<study_ph>.authorize_self_coupled_disciplines",
    "<study_ph>.linearization_mode",
    "<study_ph>.cache_type",
    "<study_ph>.cache_file_path",
]

NUMERICAL_PARAMETERS = [value.split("<study_ph>.")[1] for value in NUMERICAL_KEYS]

EXCLUDED_CLASSES = [
    "SoSCoupling",
    "ArchiBuilder",
    "SoSGatherData",
    "SoSScatterData",
    "SoSDisciplineGather",
    "SoSDisciplineScatter",
    "SoSMultiScatterBuilder",
    "SoSMultiScenario",
    "SoSOptimScenario",
    "SoSVerySimpleMultiScenario",
    "SoSSimpleMultiScenario",
    "SoSSensitivity",
    "SoSScenario",
    "SoSEval",
]


def get_all_usecases(processes_repo):
    """
    Retrieve all usecases in a repository
    :params: processes_repo, repository where to find processes
    :type: String

    :returns: List of strings: ['usecase_1','usecase_2']
    """

    process_factory = SoSProcessFactory(additional_repository_list=[processes_repo], search_python_path=False)
    process_list = process_factory.get_processes_dict()
    usecase_list = []
    for repository in process_list:
        for process in process_list[repository]:
            imported_module = import_module(".".join([repository, process]))
            process_directory = dirname(imported_module.__file__)
            # Run all usecases
            for usecase_py in listdir(process_directory):
                if usecase_py.startswith("usecase"):
                    usecase = usecase_py.replace(".py", "")
                    usecase_list.append(".".join([repository, process, usecase]))
    return ["climateeconomics.sos_processes.iam.witness.witness_dev.usecase_witness"]


def configure(usecase):
    """
    Configure a usecase and return the  treeview from the configure
    :params: usecase, usecase to configure
    :type: String
    :returns:  dm as dictionary
    """

    print(f"----- CONFIGURE  THE  USECASE {usecase} -----")
    # Instanciate Study
    imported_module = import_module(usecase)
    uc = getattr(imported_module, "Study")()
    # First step : Dump data to a temp folder
    uc.set_dump_directory(gettempdir())
    uc.load_data()
    dump_dir = uc.dump_directory
    uc.dump_data(dump_dir)

    # Set repo_name, proc_name, study_name to create BaseStudyManager
    repo_name = uc.repository_name
    proc_name = uc.process_name
    study_name = uc.study_name

    # configuration : Load Data in a new BaseStudyManager and configure study
    study_1 = BaseStudyManager(repo_name, proc_name, study_name)
    study_1.load_data(from_path=dump_dir)
    study_1.execution_engine.configure()
    # Deepcopy dm
    dm_dict_1 = deepcopy(study_1.execution_engine.get_anonimated_data_dict())

    return dm_dict_1, deepcopy(study_1.execution_engine.dm)


def dump_value_into_dict(value, data_type):
    if data_type == "array":
        return {key: key for key in value.tolist()}
    elif data_type == "dataframe":
        return value.to_dict("split")

    else:
        return value


def retrieve_input_from_dict(value, data_type):
    if data_type == "array":
        return np.array(list(value.values()))
    elif data_type == "dataframe":
        return pd.DataFrame(value["data"], columns=value["columns"], index=value["index"])
    else:
        return value


def generate_dict_from_usecase(usecase):
    dict_inputs = {}
    out_dict = {}

    dm = configure(usecase)[0]

    for key, value in dm.items():
        if (
            key not in NUMERICAL_KEYS
            and all(element not in key for element in NUMERICAL_PARAMETERS)
            and value["io_type"] == "in"
            and value["value"] is not None
        ):
            dict_inputs[key.split("<study_ph>.")[1]] = value
    all_collections = set([".".join(key.split(".")[:-1]) if "." in key else "None" for key in dict_inputs.keys()])
    conversion = {
        key: ("None", key) if "." not in key else (".".join(key.split(".")[:-1]), key.split(".")[-1])
        for key in dict_inputs.keys()
    }
    for key_1 in all_collections:
        out_dict[key_1] = {}

    for key_2 in dict_inputs.keys():
        out_dict[conversion[key_2][0]][conversion[key_2][1]] = {}

    for key in dict_inputs.keys():
        out_dict[conversion[key][0]][conversion[key][1]]["value"] = dump_value_into_dict(
            dict_inputs[key]["value"], dict_inputs[key]["type"]
        )
        out_dict[conversion[key][0]][conversion[key][1]]["datatype"] = dict_inputs[key]["type"]
        out_dict[conversion[key][0]][conversion[key][1]]["source"] = "None"

    return out_dict


def generate_dict_from_usecase_2(usecase):
    out_dict = {}
    dm = configure(usecase)[1]
    all_disciplines_from_dm = dm.disciplines_dict
    interest_disciplines = []
    for id_discipline in all_disciplines_from_dm:
        if all_disciplines_from_dm[id_discipline]["classname"] not in EXCLUDED_CLASSES:
            interest_disciplines.append(
                (
                    all_disciplines_from_dm[id_discipline]["classname"],
                    all_disciplines_from_dm[id_discipline]["reference"],
                )
            )
    # type(discipline).__base__.__name__
    # 'EnergyDiscipline'
    # type(discipline).__base__.__name__
    # 'GaseousHydrogenTechnoDiscipline'

    for discipline_name, discipline in interest_disciplines:
        disciplines_inputs = discipline.get_input_data_names()
        out_dict[discipline_name] = {}
        for input_var in disciplines_inputs:
            if dm.get_data(input_var, "value") is not None:
                short_name = input_var.split(".")[-1]
                out_dict[discipline_name][short_name] = {}
                input_description = dm.get_data(input_var)
                if "namespace" in input_description.keys():
                    out_dict[discipline_name][short_name]["namespace"] = input_description["namespace"]
                else:
                    out_dict[discipline_name][short_name]["namespace"] = "local"

                out_dict[discipline_name][short_name]["source"] = "None"
                out_dict[discipline_name][short_name]["datatype"] = input_description["type"]
                out_dict[discipline_name][short_name]["value"] = dump_value_into_dict(
                    input_description["value"], input_description["type"]
                )

    return out_dict


def generate_dict_from_usecase_3(usecase):
    out_dict = {}
    dm = configure(usecase)[1]
    all_disciplines_from_dm = dm.disciplines_dict
    interest_disciplines = []
    for id_discipline in all_disciplines_from_dm:
        if all_disciplines_from_dm[id_discipline]["classname"] not in EXCLUDED_CLASSES:
            interest_disciplines.append(
                (
                    all_disciplines_from_dm[id_discipline]["classname"],
                    all_disciplines_from_dm[id_discipline]["reference"],
                )
            )
    # type(discipline).__base__.__name__
    # 'EnergyDiscipline'
    # type(discipline).__base__.__name__
    # 'GaseousHydrogenTechnoDiscipline'
    energy_disciplines = [
        (discipline_name, discipline)
        for (discipline_name, discipline) in interest_disciplines
        if type(discipline).__base__.__name__ == "EnergyDiscipline"
    ]

    technologies_disciplines = [
        (discipline_name_2, discipline_2)
        for (discipline_name_2, discipline_2) in interest_disciplines
        if ("TechnoDiscipline" in type(discipline_2).__base__.__name__)
    ]
    for discipline_name, discipline in technologies_disciplines:
        disciplines_inputs = discipline.get_input_data_names()
        out_dict[discipline_name] = {}
        for input_var in disciplines_inputs:
            if (
                dm.get_data(input_var, "value") is not None
                and dm.get_data(input_var, "coupling") is False
                and dm.get_data(input_var, "editable") is True
                and dm.get_data(input_var, "io_type") == "in"
            ):
                short_name = input_var.split(".")[-1]
                out_dict[discipline_name][short_name] = {}
                input_description = dm.get_data(input_var)
                if "namespace" in input_description.keys():
                    out_dict[discipline_name][short_name]["namespace"] = input_description["namespace"]
                else:
                    out_dict[discipline_name][short_name]["namespace"] = "local"

                out_dict[discipline_name][short_name]["source"] = None
                out_dict[discipline_name][short_name]["datatype"] = input_description["type"]
                out_dict[discipline_name][short_name]["value"] = dump_value_into_dict(
                    input_description["value"], input_description["type"]
                )
    dictionnary_list = []
    id_discipline = 1
    for energy in out_dict.keys():
        custom_dict = {}
        custom_dict["_id"] = id_discipline
        custom_dict["techno_name"] = energy.split("Discipline")[0]
        custom_dict["region"] = "world"
        for key, value in sorted(out_dict[energy].items()):
            custom_dict[key] = value
        dictionnary_list.append(custom_dict)
        id_discipline += 1

    return dictionnary_list


def dump_json_file_from_dict(dict_to_dump):
    with open("all_witness_dev.json", "w") as outfile:
        json.dump(dict_to_dump, outfile, indent=4)
    return None


def dump_json_files_from_dict(dict_to_dump):
    for discipline in dict_to_dump.keys():
        with open(f"{discipline}.json", "w") as outfile:
            dict_one_discipline = {"_id": f"{discipline}", "characteristics": dict_to_dump[discipline]}
            json.dump(dict_one_discipline, outfile, indent=4)
    return None


def dump_json_for_database(repo):
    output_dict = {}
    usecases = get_all_usecases(repo)
    for usecase in usecases:
        output_dict.update(generate_dict_from_usecase_2(usecase))
    dump_json_file_from_dict(output_dict)


def dump_json_for_database_energy(repo):
    output_list = []
    usecases = get_all_usecases(repo)
    for usecase in usecases:
        output_list.extend(generate_dict_from_usecase_3(usecase))
    dump_json_file_from_dicts(output_list)


def dump_json_file_from_dicts(list_dict_to_dump):
    with open("all_technologies.json", "w") as outfile:
        for discipline_dict in list_dict_to_dump:
            json.dump(discipline_dict, outfile, indent=4)
            outfile.write("\n")
    return None
