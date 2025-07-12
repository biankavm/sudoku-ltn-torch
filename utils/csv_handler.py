# CSV file handling for Sudoku puzzles 

import csv
 
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return list(reader)

# print(read_csv("../csvs/table1.csv"))