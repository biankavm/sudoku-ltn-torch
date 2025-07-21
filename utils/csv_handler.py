import csv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return list(reader)
