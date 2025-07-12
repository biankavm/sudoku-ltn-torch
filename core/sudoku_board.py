# Sudoku board representation and basic operations
import numpy as np
import torch

class SudokuBoard:
    """
    Classe para representação e operações básicas do tabuleiro Sudoku.
    Responsável por análise imediata e determinística do estado atual.
    """
    
    def __init__(self, board_data):
        """
        Inicializa o tabuleiro com dados do CSV.
        
        Args:
            board_data: Lista de listas ou array numpy com dados do tabuleiro
        """
        if isinstance(board_data, list):
            # Converter lista de strings para inteiros
            self.board = np.array([[int(cell) for cell in row] for row in board_data], dtype=int)
        else:
            self.board = np.array(board_data, dtype=int)
        
        self.size = self.board.shape[0]  # Tamanho do tabuleiro (4x4 ou 9x9)
        self.box_size = int(np.sqrt(self.size))  # Tamanho do quadrante (2x2 ou 3x3)
        
    def is_open(self):
        """
        Verifica se o tabuleiro é 'aberto' (tem células vazias representadas por 0).
        
        Returns:
            bool: True se tem zeros, False caso contrário
        """
        return np.any(self.board == 0)
    
    def is_closed(self):
        """
        Verifica se o tabuleiro é 'fechado' (completamente preenchido).
        
        Returns:
            bool: True se não tem zeros, False caso contrário
        """
        return np.all(self.board != 0)
    
    def get_open_positions(self):
        """
        Retorna lista de posições (row, col) com células vazias.
        
        Returns:
            list: Lista de tuplas (row, col) onde há zeros
        """
        positions = np.argwhere(self.board == 0)
        return [(int(pos[0]), int(pos[1])) for pos in positions]
    
    def is_valid(self):
        """
        Verifica se o tabuleiro atual é válido (sem conflitos).
        
        Returns:
            bool: True se válido, False se há conflitos
        """
        # Verificar linhas
        for row in range(self.size):
            if not self._is_valid_unit(self.board[row]):
                return False
        
        # Verificar colunas
        for col in range(self.size):
            if not self._is_valid_unit(self.board[:, col]):
                return False
        
        # Verificar quadrantes
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                box = self.board[box_row:box_row+self.box_size, box_col:box_col+self.box_size]
                if not self._is_valid_unit(box.flatten()):
                    return False
        
        return True
    
    def _is_valid_unit(self, unit):
        """
        Verifica se uma unidade (linha, coluna ou quadrante) é válida.
        
        Args:
            unit: Array numpy representando a unidade
            
        Returns:
            bool: True se válida, False caso contrário
        """
        # Remover zeros (células vazias)
        filled_cells = unit[unit != 0]
        
        # Verificar se não há duplicatas
        return len(filled_cells) == len(np.unique(filled_cells))
    
    def find_invalid_numbers(self):
        """
        Identifica quais números estão causando invalidade no tabuleiro.
        
        Returns:
            list: Lista de dicionários com informações sobre conflitos
        """
        conflicts = []
        
        # Verificar conflitos em linhas
        for row in range(self.size):
            row_conflicts = self._find_conflicts_in_unit(self.board[row], f"linha {row}")
            conflicts.extend(row_conflicts)
        
        # Verificar conflitos em colunas
        for col in range(self.size):
            col_conflicts = self._find_conflicts_in_unit(self.board[:, col], f"coluna {col}")
            conflicts.extend(col_conflicts)
        
        # Verificar conflitos em quadrantes
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                box = self.board[box_row:box_row+self.box_size, box_col:box_col+self.box_size]
                box_name = f"quadrante ({box_row//self.box_size}, {box_col//self.box_size})"
                box_conflicts = self._find_conflicts_in_unit(box.flatten(), box_name)
                conflicts.extend(box_conflicts)
        
        return conflicts
    
    def _find_conflicts_in_unit(self, unit, unit_name):
        """
        Encontra conflitos em uma unidade específica.
        
        Args:
            unit: Array numpy representando a unidade
            unit_name: Nome da unidade para identificação
            
        Returns:
            list: Lista de conflitos encontrados
        """
        conflicts = []
        filled_cells = unit[unit != 0]
        
        # Encontrar números duplicados
        unique_numbers, counts = np.unique(filled_cells, return_counts=True)
        duplicates = unique_numbers[counts > 1]
        
        for duplicate in duplicates:
            conflicts.append({
                'numero': int(duplicate),
                'local': unit_name,
                'ocorrencias': int(counts[unique_numbers == duplicate][0])
            })
        
        return conflicts
    
    def count_remaining_numbers(self):
        """
        Conta quantos números de cada tipo ainda podem ser jogados.
        
        Returns:
            dict: Dicionário com contagem de números restantes
        """
        remaining = {}
        
        for num in range(1, self.size + 1):
            used_count = np.sum(self.board == num)
            remaining[num] = self.size - used_count
        
        return remaining
    
    def get_possible_numbers(self, row, col):
        """
        Retorna números possíveis para uma posição específica.
        
        Args:
            row: Linha da posição
            col: Coluna da posição
            
        Returns:
            set: Conjunto de números possíveis
        """
        if self.board[row, col] != 0:
            return set()  # Célula já preenchida
        
        # Números já usados na linha
        used_in_row = set(self.board[row][self.board[row] != 0])
        
        # Números já usados na coluna
        used_in_col = set(self.board[:, col][self.board[:, col] != 0])
        
        # Números já usados no quadrante
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        box = self.board[box_row:box_row+self.box_size, box_col:box_col+self.box_size]
        used_in_box = set(box.flatten()[box.flatten() != 0])
        
        # Números possíveis = todos os números - números já usados
        all_numbers = set(range(1, self.size + 1))
        used_numbers = used_in_row | used_in_col | used_in_box
        
        return all_numbers - used_numbers
    
    def get_candidates_matrix(self):
        """
        Retorna matriz de candidatos para cada célula.
        
        Returns:
            dict: Dicionário com candidatos para cada posição
        """
        candidates = {}
        
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] == 0:
                    candidates[(row, col)] = self.get_possible_numbers(row, col)
        
        return candidates
    
    def to_tensor(self):
        """
        Converte o tabuleiro para tensor PyTorch (para uso com LTN).
        
        Returns:
            torch.Tensor: Tensor do tabuleiro
        """
        return torch.tensor(self.board, dtype=torch.float32)
    
    def get_board_info(self):
        """
        Retorna informações completas sobre o tabuleiro.
        
        Returns:
            dict: Dicionário com todas as informações do tabuleiro
        """
        return {
            'tamanho': f"{self.size}x{self.size}",
            'tipo': 'aberto' if self.is_open() else 'fechado',
            'valido': self.is_valid(),
            'posicoes_abertas': self.get_open_positions(),
            'numeros_restantes': self.count_remaining_numbers(),
            'candidatos': self.get_candidates_matrix(),
            'conflitos': self.find_invalid_numbers() if not self.is_valid() else []
        }
    
    def __str__(self):
        """
        Representação string do tabuleiro.
        
        Returns:
            str: Tabuleiro formatado
        """
        result = []
        for row in self.board:
            result.append(' '.join(str(cell) if cell != 0 else '.' for cell in row))
        return '\n'.join(result)
    
    def __repr__(self):
        return f"SudokuBoard({self.size}x{self.size}, {'aberto' if self.is_open() else 'fechado'})" 