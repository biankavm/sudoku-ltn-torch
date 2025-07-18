# Logic Tensor Network predicates for Sudoku solving
import torch
import torch.nn as nn
import ltn
import numpy as np
from typing import Tuple, List, Optional

class ValidCellModel(nn.Module):
    """
    Predicate para determinar se uma célula é válida no contexto do Sudoku.
    Suporta tamanhos dinâmicos de tabuleiro (4x4 e 9x9).
    Entrada: (row, col, value, board_state, board_size)
    Saída: grau de verdade [0,1] indicando se a célula é válida
    """
    
    def __init__(self, max_board_size: int = 9):
        super(ValidCellModel, self).__init__()
        self.max_board_size = max_board_size
        self.max_input_size = 2 + 1 + (max_board_size * max_board_size) + 1  # pos + value + board + size
        
        # Rede neural para aprender a validação de células
        self.network = nn.Sequential(
            nn.Linear(self.max_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Saída entre 0 e 1
        )
        
    def forward(self, row: torch.Tensor, col: torch.Tensor, 
                value: torch.Tensor, board: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do predicate ValidCell
        
        Args:
            row: tensor com índices das linhas
            col: tensor com índices das colunas  
            value: tensor com valores a serem validados
            board: tensor com estado atual do tabuleiro
            board_size: tensor com o tamanho do tabuleiro
            
        Returns:
            tensor com graus de verdade para cada combinação
        """
        batch_size = row.shape[0]
        
        # Normalizar posições para [0,1]
        row_norm = row.float() / (board_size.float() - 1)
        col_norm = col.float() / (board_size.float() - 1)
        
        # Normalizar valores para [0,1]
        value_norm = (value.float() - 1) / (board_size.float() - 1)
        
        # Flatten do tabuleiro para cada exemplo
        board_flat = board.view(batch_size, -1).float() / board_size.float()
        
        # Normalizar tamanho do tabuleiro
        size_norm = board_size.float() / self.max_board_size
        
        # Concatenar todas as features
        features = torch.cat([
            row_norm.unsqueeze(1),
            col_norm.unsqueeze(1), 
            value_norm.unsqueeze(1),
            board_flat,
            size_norm.unsqueeze(1)
        ], dim=1)
        
        return self.network(features).squeeze()

class RowConstraintModel(nn.Module):
    """
    Predicate para constraint de linha: cada número deve aparecer exatamente uma vez por linha
    Suporta tamanhos dinâmicos de tabuleiro.
    """
    
    def __init__(self, max_board_size: int = 9):
        super(RowConstraintModel, self).__init__()
        self.max_board_size = max_board_size
        self.max_input_size = 1 + 1 + (max_board_size * max_board_size) + 1  # row + value + board + size
        
        self.network = nn.Sequential(
            nn.Linear(self.max_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, row: torch.Tensor, value: torch.Tensor, 
                board: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """
        Verifica se o valor pode ser colocado na linha sem violar constraints
        """
        batch_size = row.shape[0]
        
        row_norm = row.float() / (board_size.float() - 1)
        value_norm = (value.float() - 1) / (board_size.float() - 1)
        board_flat = board.view(batch_size, -1).float() / board_size.float()
        size_norm = board_size.float() / self.max_board_size
        
        features = torch.cat([
            row_norm.unsqueeze(1),
            value_norm.unsqueeze(1),
            board_flat,
            size_norm.unsqueeze(1)
        ], dim=1)
        
        return self.network(features).squeeze()

class ColConstraintModel(nn.Module):
    """
    Predicate para constraint de coluna: cada número deve aparecer exatamente uma vez por coluna
    Suporta tamanhos dinâmicos de tabuleiro.
    """
    
    def __init__(self, max_board_size: int = 9):
        super(ColConstraintModel, self).__init__()
        self.max_board_size = max_board_size
        self.max_input_size = 1 + 1 + (max_board_size * max_board_size) + 1  # col + value + board + size
        
        self.network = nn.Sequential(
            nn.Linear(self.max_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, col: torch.Tensor, value: torch.Tensor, 
                board: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """
        Verifica se o valor pode ser colocado na coluna sem violar constraints
        """
        batch_size = col.shape[0]
        
        col_norm = col.float() / (board_size.float() - 1)
        value_norm = (value.float() - 1) / (board_size.float() - 1)
        board_flat = board.view(batch_size, -1).float() / board_size.float()
        size_norm = board_size.float() / self.max_board_size
        
        features = torch.cat([
            col_norm.unsqueeze(1),
            value_norm.unsqueeze(1),
            board_flat,
            size_norm.unsqueeze(1)
        ], dim=1)
        
        return self.network(features).squeeze()

class BoxConstraintModel(nn.Module):
    """
    Predicate para constraint de quadrante: cada número deve aparecer exatamente uma vez por quadrante
    Suporta tamanhos dinâmicos de tabuleiro.
    """
    
    def __init__(self, max_board_size: int = 9):
        super(BoxConstraintModel, self).__init__()
        self.max_board_size = max_board_size
        self.max_input_size = 2 + 1 + (max_board_size * max_board_size) + 1  # box_row + box_col + value + board + size
        
        self.network = nn.Sequential(
            nn.Linear(self.max_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, row: torch.Tensor, col: torch.Tensor, 
                value: torch.Tensor, board: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """
        Verifica se o valor pode ser colocado no quadrante sem violar constraints
        """
        batch_size = row.shape[0]
        
        # Calcular índices do quadrante
        box_size = torch.sqrt(board_size.float()).int()
        box_row = row // box_size
        box_col = col // box_size
        
        box_row_norm = box_row.float() / (box_size.float() - 1)
        box_col_norm = box_col.float() / (box_size.float() - 1)
        value_norm = (value.float() - 1) / (board_size.float() - 1)
        board_flat = board.view(batch_size, -1).float() / board_size.float()
        size_norm = board_size.float() / self.max_board_size
        
        features = torch.cat([
            box_row_norm.unsqueeze(1),
            box_col_norm.unsqueeze(1),
            value_norm.unsqueeze(1),
            board_flat,
            size_norm.unsqueeze(1)
        ], dim=1)
        
        return self.network(features).squeeze()

class NakedSingleModel(nn.Module):
    """
    Predicate para heurística Naked Single: se uma célula tem apenas um candidato possível
    Suporta tamanhos dinâmicos de tabuleiro.
    """
    
    def __init__(self, max_board_size: int = 9):
        super(NakedSingleModel, self).__init__()
        self.max_board_size = max_board_size
        self.max_input_size = 2 + (max_board_size * max_board_size) + max_board_size + 1  # pos + board + candidates + size
        
        self.network = nn.Sequential(
            nn.Linear(self.max_input_size, 192),
            nn.ReLU(),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid()
        )
        
    def forward(self, row: torch.Tensor, col: torch.Tensor, 
                board: torch.Tensor, candidates: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """
        Determina se uma célula é um Naked Single
        
        Args:
            row: índices das linhas
            col: índices das colunas
            board: estado do tabuleiro
            candidates: vetor binário de candidatos possíveis para a célula
            board_size: tamanho do tabuleiro
        """
        batch_size = row.shape[0]
        
        row_norm = row.float() / (board_size.float() - 1)
        col_norm = col.float() / (board_size.float() - 1)
        board_flat = board.view(batch_size, -1).float() / board_size.float()
        size_norm = board_size.float() / self.max_board_size
        
        features = torch.cat([
            row_norm.unsqueeze(1),
            col_norm.unsqueeze(1),
            board_flat,
            candidates.float(),
            size_norm.unsqueeze(1)
        ], dim=1)
        
        return self.network(features).squeeze()

class HiddenSingleModel(nn.Module):
    """
    Predicate para heurística Hidden Single: se um número aparece como candidato 
    em apenas uma célula dentro de uma unidade (linha, coluna ou quadrante)
    Suporta tamanhos dinâmicos de tabuleiro.
    """
    
    def __init__(self, max_board_size: int = 9):
        super(HiddenSingleModel, self).__init__()
        self.max_board_size = max_board_size
        self.max_input_size = 2 + 1 + (max_board_size * max_board_size) + (3 * max_board_size) + 1  # pos + value + board + unit_info + size
        
        self.network = nn.Sequential(
            nn.Linear(self.max_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, row: torch.Tensor, col: torch.Tensor, value: torch.Tensor,
                board: torch.Tensor, unit_candidates: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """
        Determina se uma célula-valor é um Hidden Single
        
        Args:
            row: índices das linhas
            col: índices das colunas
            value: valor a ser testado
            board: estado do tabuleiro
            unit_candidates: candidatos para o valor nas unidades (linha, coluna, quadrante)
            board_size: tamanho do tabuleiro
        """
        batch_size = row.shape[0]
        
        row_norm = row.float() / (board_size.float() - 1)
        col_norm = col.float() / (board_size.float() - 1)
        value_norm = (value.float() - 1) / (board_size.float() - 1)
        board_flat = board.view(batch_size, -1).float() / board_size.float()
        size_norm = board_size.float() / self.max_board_size
        
        features = torch.cat([
            row_norm.unsqueeze(1),
            col_norm.unsqueeze(1),
            value_norm.unsqueeze(1),
            board_flat,
            unit_candidates.float(),
            size_norm.unsqueeze(1)
        ], dim=1)
        
        return self.network(features).squeeze()

class SudokuPredicates:
    """
    Classe principal que encapsula todos os predicates LTN para Sudoku
    Suporta tamanhos dinâmicos de tabuleiro (4x4 e 9x9)
    """
    
    def __init__(self, max_board_size: int = 9):
        self.max_board_size = max_board_size
        
        # Instanciar todos os modelos de predicates com tamanho máximo
        self.valid_cell_model = ValidCellModel(max_board_size)
        self.row_constraint_model = RowConstraintModel(max_board_size)
        self.col_constraint_model = ColConstraintModel(max_board_size)
        self.box_constraint_model = BoxConstraintModel(max_board_size)
        self.naked_single_model = NakedSingleModel(max_board_size)
        self.hidden_single_model = HiddenSingleModel(max_board_size)
        
        # Converter para predicates LTN
        self.ValidCell = ltn.Predicate(self.valid_cell_model)
        self.RowConstraint = ltn.Predicate(self.row_constraint_model)
        self.ColConstraint = ltn.Predicate(self.col_constraint_model)
        self.BoxConstraint = ltn.Predicate(self.box_constraint_model)
        self.NakedSingle = ltn.Predicate(self.naked_single_model)
        self.HiddenSingle = ltn.Predicate(self.hidden_single_model)
        
    def get_all_predicates(self) -> dict:
        """
        Retorna dicionário com todos os predicates disponíveis
        """
        return {
            'ValidCell': self.ValidCell,
            'RowConstraint': self.RowConstraint,
            'ColConstraint': self.ColConstraint,
            'BoxConstraint': self.BoxConstraint,
            'NakedSingle': self.NakedSingle,
            'HiddenSingle': self.HiddenSingle
        }
        
    def get_model_parameters(self) -> List[torch.nn.Parameter]:
        """
        Retorna todos os parâmetros dos modelos para otimização
        """
        params = []
        params.extend(self.valid_cell_model.parameters())
        params.extend(self.row_constraint_model.parameters())
        params.extend(self.col_constraint_model.parameters())
        params.extend(self.box_constraint_model.parameters())
        params.extend(self.naked_single_model.parameters())
        params.extend(self.hidden_single_model.parameters())
        return params
    
    def valid_cell_model(self, row: torch.Tensor, col: torch.Tensor, 
                        value: torch.Tensor, board: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """Wrapper para o modelo ValidCell com tamanho dinâmico"""
        return self.valid_cell_model(row, col, value, board, board_size)
    
    def row_constraint_model(self, row: torch.Tensor, value: torch.Tensor, 
                           board: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """Wrapper para o modelo RowConstraint com tamanho dinâmico"""
        return self.row_constraint_model(row, value, board, board_size)
    
    def col_constraint_model(self, col: torch.Tensor, value: torch.Tensor, 
                           board: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """Wrapper para o modelo ColConstraint com tamanho dinâmico"""
        return self.col_constraint_model(col, value, board, board_size)
    
    def box_constraint_model(self, row: torch.Tensor, col: torch.Tensor, 
                           value: torch.Tensor, board: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """Wrapper para o modelo BoxConstraint com tamanho dinâmico"""
        return self.box_constraint_model(row, col, value, board, board_size)
    
    def naked_single_model(self, row: torch.Tensor, col: torch.Tensor, 
                          board: torch.Tensor, candidates: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """Wrapper para o modelo NakedSingle com tamanho dinâmico"""
        return self.naked_single_model(row, col, board, candidates, board_size)
    
    def hidden_single_model(self, row: torch.Tensor, col: torch.Tensor, value: torch.Tensor,
                           board: torch.Tensor, unit_candidates: torch.Tensor, board_size: torch.Tensor) -> torch.Tensor:
        """Wrapper para o modelo HiddenSingle com tamanho dinâmico"""
        return self.hidden_single_model(row, col, value, board, unit_candidates, board_size) 