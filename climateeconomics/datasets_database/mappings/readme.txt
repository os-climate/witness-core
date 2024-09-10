To describe wich dataset goes with wich data in a specific usecase, the usecase will need a dataset_mapping file.

This file can be created next to the usecase.py file For testing purpose, or in the "datasets_database\mappings" folder if the mapping is official and where everyone can retrieve it.

The dataset mapping file has the following structure:

{
   "process_module_path": "module.path.to.the.process",
    "namespace_datasets_mapping": {
        "v0|<study_ph>|*": ["repos:climateeconomics|dataset_root_data|*"],
        "v0|<study_ph>.Disc1VirtualNode|*" : ["repos:climateeconomics|dataset_disc1|*"],
        "v0|<study_ph>.Disc2VirtualNode|*": ["repos:climateeconomics|dataset_disc2|*"],
        "v0|<study_ph>.Disc1|*": ["repos:climateeconomics|dataset_all_disc|*", "repos:climateeconomics|dataset_disc1|*"],
        "v0|<study_ph>.Disc2|*": ["repos:climateeconomics|dataset_all_disc|*", "repos:climateeconomics|dataset_disc2|*"]
    }
}

The process_module_path gives the path to the process module.

The namespace_datasets_mapping gives the list of the namespace to be associated with a dataset.

A namespace information has the following format:

- V0| → is the version of the mapping (not used for now)
- <study_ph>.Disc1VirtualNode → the namespace with a study name placeholder
- |* → means that it is for all parameters of this namespace. For now you can't specify a single parameter but it will be possible in the future.

A dataset information has the following format:

- repos:sostrades_core| → identifier of the dataset connector, to identify a repository datasets connector, the repository module name is preceding by the prefix "repos:".
- dataset_disc1 → name of the dataset
- |* → means that it is for all parameters of this dataset. For now you can't specify a single parameter but it will be possible in the future.
If several dataset are specified for one namespace, if a parameter is present in several datasets, the value of the parameter will be the value of the last dataset to have the parameter value.
