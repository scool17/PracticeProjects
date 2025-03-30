import pandas as pd
import configparser

class Parser(configparser.ConfigParser):
    def __init__(self):
        super().__init__()
        self.read('backend/filePath.ini')
        self.netflix_data = self.get_data(self.get('Global', 'Netflix'))

    def get_data(self, file_path):
        return pd.read_csv(file_path)

