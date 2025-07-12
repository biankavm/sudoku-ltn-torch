# Training data generation and processing for Sudoku LTN
import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import random
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sudoku_board import SudokuBoard

class SudokuTrainingDataGenerator:
    """
    Classe para gerar dados de treinamento para os predicates LTN do Sudoku
    a partir de arquivos CSV com tabuleiros
    """
    
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.box_size = int(np.sqrt(board_size))
        
    def load_csv_data(self, csv_path: str) -> List[SudokuBoard]:
        """
        Carrega dados do CSV e converte para objetos SudokuBoard
        
        Args:
            csv_path: caminho para o arquivo CSV
            
        Returns:
            lista de objetos SudokuBoard
        """
        try:
            # Tentar carregar como CSV padrão
            df = pd.read_csv(csv_path, header=None)
            boards = []
            
            # Assumir que cada linha do CSV é um tabuleiro
            for idx, row in df.iterrows():
                # Converter linha para matriz board_size x board_size
                board_data = np.array(row.values).reshape(self.board_size, self.board_size)
                board = SudokuBoard(board_data)
                boards.append(board)
                
            return boards
            
        except Exception as e:
            print(f"Erro ao carregar CSV: {e}")
            return []
    
    def generate_valid_cell_data(self, boards: List[SudokuBoard]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gera dados de treinamento para o predicate ValidCell
        
        Returns:
            tupla com (rows, cols, values, board_states, labels)
        """
        rows, cols, values, board_states, labels = [], [], [], [], []
        
        for board in boards:
            board_tensor = board.to_tensor()
            
            # Exemplos positivos: células já preenchidas (válidas)
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if board.board[r, c] != 0:
                        rows.append(r)
                        cols.append(c)
                        values.append(board.board[r, c])
                        board_states.append(board_tensor)
                        labels.append(1.0)  # Válido
            
            # Exemplos negativos: movimentos inválidos
            open_positions = board.get_open_positions()
            for r, c in open_positions:
                possible_values = board.get_possible_numbers(r, c)
                all_values = set(range(1, self.board_size + 1))
                invalid_values = all_values - possible_values
                
                # Adicionar alguns exemplos inválidos
                for val in list(invalid_values)[:3]:  # Máximo 3 exemplos negativos por célula
                    rows.append(r)
                    cols.append(c)
                    values.append(val)
                    board_states.append(board_tensor)
                    labels.append(0.0)  # Inválido
        
        return (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long),
            torch.tensor(values, dtype=torch.long),
            torch.stack(board_states),
            torch.tensor(labels, dtype=torch.float32)
        )
    
    def generate_constraint_data(self, boards: List[SudokuBoard]) -> Dict[str, Tuple]:
        """
        Gera dados de treinamento para os predicates de constraint (Row, Col, Box)
        
        Returns:
            dicionário com dados para cada tipo de constraint
        """
        constraint_data = {
            'row': {'indices': [], 'values': [], 'boards': [], 'labels': []},
            'col': {'indices': [], 'values': [], 'boards': [], 'labels': []},
            'box': {'row_indices': [], 'col_indices': [], 'values': [], 'boards': [], 'labels': []}
        }
        
        for board in boards:
            board_tensor = board.to_tensor()
            
            # Dados para Row Constraint
            for r in range(self.board_size):
                for val in range(1, self.board_size + 1):
                    # Verificar se valor já existe na linha
                    exists_in_row = val in board.board[r, :]
                    
                    constraint_data['row']['indices'].append(r)
                    constraint_data['row']['values'].append(val)
                    constraint_data['row']['boards'].append(board_tensor)
                    constraint_data['row']['labels'].append(0.0 if exists_in_row else 1.0)
            
            # Dados para Col Constraint
            for c in range(self.board_size):
                for val in range(1, self.board_size + 1):
                    # Verificar se valor já existe na coluna
                    exists_in_col = val in board.board[:, c]
                    
                    constraint_data['col']['indices'].append(c)
                    constraint_data['col']['values'].append(val)
                    constraint_data['col']['boards'].append(board_tensor)
                    constraint_data['col']['labels'].append(0.0 if exists_in_col else 1.0)
            
            # Dados para Box Constraint
            for box_r in range(0, self.board_size, self.box_size):
                for box_c in range(0, self.board_size, self.box_size):
                    box = board.board[box_r:box_r+self.box_size, box_c:box_c+self.box_size]
                    
                    for val in range(1, self.board_size + 1):
                        # Verificar se valor já existe no quadrante
                        exists_in_box = val in box.flatten()
                        
                        constraint_data['box']['row_indices'].append(box_r // self.box_size)
                        constraint_data['box']['col_indices'].append(box_c // self.box_size)
                        constraint_data['box']['values'].append(val)
                        constraint_data['box']['boards'].append(board_tensor)
                        constraint_data['box']['labels'].append(0.0 if exists_in_box else 1.0)
        
        # Converter para tensors
        result = {}
        for constraint_type, data in constraint_data.items():
            if constraint_type == 'box':
                result[constraint_type] = (
                    torch.tensor(data['row_indices'], dtype=torch.long),
                    torch.tensor(data['col_indices'], dtype=torch.long),
                    torch.tensor(data['values'], dtype=torch.long),
                    torch.stack(data['boards']),
                    torch.tensor(data['labels'], dtype=torch.float32)
                )
            else:
                result[constraint_type] = (
                    torch.tensor(data['indices'], dtype=torch.long),
                    torch.tensor(data['values'], dtype=torch.long),
                    torch.stack(data['boards']),
                    torch.tensor(data['labels'], dtype=torch.float32)
                )
        
        return result
    
    def generate_naked_single_data(self, boards: List[SudokuBoard]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gera dados de treinamento para o predicate NakedSingle
        """
        rows, cols, board_states, candidates_data, labels = [], [], [], [], []
        
        for board in boards:
            board_tensor = board.to_tensor()
            candidates_matrix = board.get_candidates_matrix()
            
            # Verificar cada célula vazia
            for (r, c), candidates in candidates_matrix.items():
                # Criar vetor binário de candidatos
                candidates_vector = torch.zeros(self.board_size)
                for candidate in candidates:
                    candidates_vector[candidate - 1] = 1.0
                
                rows.append(r)
                cols.append(c)
                board_states.append(board_tensor)
                candidates_data.append(candidates_vector)
                
                # Label: 1.0 se é naked single (apenas 1 candidato), 0.0 caso contrário
                labels.append(1.0 if len(candidates) == 1 else 0.0)
        
        return (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long),
            torch.stack(board_states),
            torch.stack(candidates_data),
            torch.tensor(labels, dtype=torch.float32)
        )
    
    def generate_hidden_single_data(self, boards: List[SudokuBoard]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gera dados de treinamento para o predicate HiddenSingle
        """
        rows, cols, values, board_states, unit_candidates_data, labels = [], [], [], [], [], []
        
        for board in boards:
            board_tensor = board.to_tensor()
            
            # Verificar cada célula vazia e cada valor possível
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if board.board[r, c] == 0:
                        possible_values = board.get_possible_numbers(r, c)
                        
                        for val in range(1, self.board_size + 1):
                            # Criar vetor de informações das unidades
                            unit_info = self._get_unit_candidates_info(board, r, c, val)
                            
                            rows.append(r)
                            cols.append(c)
                            values.append(val)
                            board_states.append(board_tensor)
                            unit_candidates_data.append(unit_info)
                            
                            # Verificar se é hidden single
                            is_hidden_single = self._is_hidden_single(board, r, c, val)
                            labels.append(1.0 if is_hidden_single else 0.0)
        
        return (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long),
            torch.tensor(values, dtype=torch.long),
            torch.stack(board_states),
            torch.stack(unit_candidates_data),
            torch.tensor(labels, dtype=torch.float32)
        )
    
    def _get_unit_candidates_info(self, board: SudokuBoard, row: int, col: int, value: int) -> torch.Tensor:
        """
        Gera informações sobre candidatos nas unidades (linha, coluna, quadrante)
        """
        info = torch.zeros(3 * self.board_size)  # 3 unidades x board_size posições
        
        # Informações da linha
        for c in range(self.board_size):
            if board.board[row, c] == 0:
                possible = board.get_possible_numbers(row, c)
                if value in possible:
                    info[c] = 1.0
        
        # Informações da coluna
        for r in range(self.board_size):
            if board.board[r, col] == 0:
                possible = board.get_possible_numbers(r, col)
                if value in possible:
                    info[self.board_size + r] = 1.0
        
        # Informações do quadrante
        box_row_start = (row // self.box_size) * self.box_size
        box_col_start = (col // self.box_size) * self.box_size
        
        box_idx = 0
        for r in range(box_row_start, box_row_start + self.box_size):
            for c in range(box_col_start, box_col_start + self.box_size):
                if board.board[r, c] == 0:
                    possible = board.get_possible_numbers(r, c)
                    if value in possible:
                        info[2 * self.board_size + box_idx] = 1.0
                box_idx += 1
        
        return info
    
    def _is_hidden_single(self, board: SudokuBoard, row: int, col: int, value: int) -> bool:
        """
        Verifica se uma célula-valor é um hidden single
        """
        if board.board[row, col] != 0:
            return False
        
        possible_values = board.get_possible_numbers(row, col)
        if value not in possible_values:
            return False
        
        # Verificar se é o único lugar na linha
        row_candidates = 0
        for c in range(self.board_size):
            if board.board[row, c] == 0:
                cell_candidates = board.get_possible_numbers(row, c)
                if value in cell_candidates:
                    row_candidates += 1
        
        if row_candidates == 1:
            return True
        
        # Verificar se é o único lugar na coluna
        col_candidates = 0
        for r in range(self.board_size):
            if board.board[r, col] == 0:
                cell_candidates = board.get_possible_numbers(r, col)
                if value in cell_candidates:
                    col_candidates += 1
        
        if col_candidates == 1:
            return True
        
        # Verificar se é o único lugar no quadrante
        box_row_start = (row // self.box_size) * self.box_size
        box_col_start = (col // self.box_size) * self.box_size
        
        box_candidates = 0
        for r in range(box_row_start, box_row_start + self.box_size):
            for c in range(box_col_start, box_col_start + self.box_size):
                if board.board[r, c] == 0:
                    cell_candidates = board.get_possible_numbers(r, c)
                    if value in cell_candidates:
                        box_candidates += 1
        
        return box_candidates == 1
    
    def generate_all_training_data(self, csv_path: str) -> Dict[str, Tuple]:
        """
        Gera todos os dados de treinamento a partir do CSV
        
        Args:
            csv_path: caminho para o arquivo CSV
            
        Returns:
            dicionário com todos os dados de treinamento
        """
        print(f"Carregando dados do CSV: {csv_path}")
        boards = self.load_csv_data(csv_path)
        
        if not boards:
            print("Nenhum tabuleiro carregado!")
            return {}
        
        print(f"Carregados {len(boards)} tabuleiros")
        
        training_data = {}
        
        print("Gerando dados para ValidCell...")
        training_data['valid_cell'] = self.generate_valid_cell_data(boards)
        
        print("Gerando dados para Constraints...")
        training_data['constraints'] = self.generate_constraint_data(boards)
        
        print("Gerando dados para NakedSingle...")
        training_data['naked_single'] = self.generate_naked_single_data(boards)
        
        print("Gerando dados para HiddenSingle...")
        training_data['hidden_single'] = self.generate_hidden_single_data(boards)
        
        print("Dados de treinamento gerados com sucesso!")
        return training_data 