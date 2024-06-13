The dataset database  (the folder "datasets_database\datasets") have to contain the parameters you want to update in your usecase. 

To create a new dataset:

- In the "datasets_database\datasets" folder, create a new folder with the name of your dataset. It has to be understandable and unique.
- Add a new file named 'descriptor.json'. This file will contain one key per parameters. If the parameter is jsonifiable, its value will be directly set in the file, if not it will be set in a csv file in the dataset folder with the prefix @type@.
- Add the csv file in the folder if needed

Example of descriptor.json file content:
{
    "a": 1,
    "b": 2,
    "b_bool": false,
    "name": "A1",
    "x":4.0,
    "x_dict":{"test1":1,"test2":2},
    "y_array":"@array@y_array.csv",
    "z_list":[1.0,2.0,3.0],
    "d":"@dataframe@d.csv"
}