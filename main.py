# Main interface for Sudoku LTN solver # Testando com o table1.csv
from utils.csv_handler import read_csv
from core.sudoku_board import SudokuBoard

data = read_csv('csvs/table1.csv')
board = SudokuBoard(data)

print(board.get_board_info())
# Vai mostrar:
# - Tipo: 'aberto' 
# - Posições abertas: [(0,1), (0,3), (1,0), (1,2), (2,1), (2,3), (3,0), (3,2)]
# - Números restantes: {1: 1, 2: 4, 3: 0, 4: 4}
# - Candidatos para cada posição vazia
