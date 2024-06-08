import pandas as pd
import os
import json


class DatasetLoader:

    def __init__(self, folder_name: str = "coco"):
        self.data = self.load_data(folder_name)

    def load_data(self, folder_name: str = "coco"):
        print("Loading data..")
        # Set the data folder
        data_folder = os.path.join(folder_name)

        # Create the directory if it doesn't exist
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        # Path to the captions JSON file
        filename = os.path.join(data_folder, "annotations", "captions_train2014.json")

        # Read and parse the JSON file
        with open(filename, "r") as json_file:
            data = json.load(json_file)

        print("Data loaded succesfully!")

        return data

    def get_data_dict(self):
        return self.data

    def get_df_annotations(self):
        return pd.DataFrame(self.data["annotations"])

    def get_df_images(self):
        return pd.DataFrame(self.data["images"])
