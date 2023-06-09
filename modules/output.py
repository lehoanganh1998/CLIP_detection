import os
import csv

class CSVWriter:
    def __init__(self, directory, filename, prefix=0):
        self.directory = directory
        self.filename = filename
        self.prefix = prefix

    def open(self):
        # create the full file path
        file_path = os.path.join(self.directory, self.filename)

        # check if the file already exists
        if os.path.isfile(file_path):
            # check if the file has a numbered prefix
            name, ext = os.path.splitext(self.filename)
            if name.endswith('_'):
                try:
                    # get the largest number prefix
                    prefix_list = [int(f.split('_')[0]) for f in os.listdir(self.directory) if f.startswith(name)]
                    self.prefix = max(prefix_list) + 1
                except ValueError:
                    # no numbered prefixes found, use default
                    pass

            # create a new file name with the prefix
            new_filename = f"{self.prefix}_{self.filename}"
            file_path = os.path.join(self.directory, new_filename)
        else:
            new_filename = self.filename

        # open the CSV file for writing
        with open(file_path, 'w', newline='') as csvfile:
            # create a writer object
            self.csvwriter = csv.writer(csvfile)
        return self.csvwriter