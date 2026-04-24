import csv
import os

class CSVWriter:
    def __init__(self, filepath, headers):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(headers)

    def write_row(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()