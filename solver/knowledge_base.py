# Knowledge Base for Sudoku LTN - Axioms and Rules
import torch
import ltn
from typing import Dict, List, Tuple
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.predicates import SudokuPredicates

class SudokuKnowledgeBase:
    """
    Base de conhecimento LTN para Sudoku com axiomas e regras lógicas
    Implementa as constraints básicas e heurísticas do Sudoku
    """
    
    def __init__(self, board_size: int = 9):
        self.board_size = board_size
        self.predicates = SudokuPredicates(board_size)
        
        # Operadores lógicos LTN
        self.Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
        self.Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        
    def get_basic_sudoku_axioms(self, board_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Retorna os axiomas básicos do Sudoku
        
        Args:
            board_tensor: tensor representando o estado do tabuleiro
            
        Returns:
            lista de axiomas (tensors com valores de verdade)
        """
        axioms = []
        
        # Axioma 1: Células válidas devem satisfazer todas as constraints
        # ∀r,c,v: ValidCell(r,c,v,board) → (RowConstraint(r,v,board) ∧ ColConstraint(c,v,board) ∧ BoxConstraint(r,c,v,board))
        
        # Criar tensors para todas as combinações possíveis
        rows = torch.arange(self.board_size).repeat(self.board_size * self.board_size)
        cols = torch.arange(self.board_size).repeat_interleave(self.board_size).repeat(self.board_size)
        values = torch.arange(1, self.board_size + 1).repeat_interleave(self.board_size * self.board_size)
        
        # Expandir board_tensor para match com as combinações
        batch_size = len(rows)
        boards = board_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Aplicar predicates
        valid_cells = self.predicates.ValidCell(rows, cols, values, boards)
        row_constraints = self.predicates.RowConstraint(rows, values, boards)
        col_constraints = self.predicates.ColConstraint(cols, values, boards)
        box_constraints = self.predicates.BoxConstraint(rows, cols, values, boards)
        
        # Axioma: ValidCell → (RowConstraint ∧ ColConstraint ∧ BoxConstraint)
        constraint_conjunction = self.And(
            self.And(row_constraints, col_constraints),
            box_constraints
        )
        
        axiom_1 = self.Forall(
            ltn.diag(rows, cols, values),
            self.Implies(valid_cells, constraint_conjunction)
        )
        axioms.append(axiom_1)
        
        return axioms
    
    def get_naked_single_axioms(self, board_tensor: torch.Tensor, 
                               candidates_data: Dict[Tuple[int, int], torch.Tensor]) -> List[torch.Tensor]:
        """
        Axiomas para heurística Naked Single
        
        Args:
            board_tensor: estado do tabuleiro
            candidates_data: dados de candidatos para cada célula
            
        Returns:
            lista de axiomas para Naked Single
        """
        axioms = []
        
        if not candidates_data:
            return axioms
        
        # Preparar dados
        positions = list(candidates_data.keys())
        if not positions:
            return axioms
            
        rows = torch.tensor([pos[0] for pos in positions])
        cols = torch.tensor([pos[1] for pos in positions])
        candidates_tensors = torch.stack([candidates_data[pos] for pos in positions])
        
        batch_size = len(positions)
        boards = board_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Axioma: NakedSingle(r,c,board,candidates) → ∃!v: ValidCell(r,c,v,board)
        # Se uma célula é naked single, então existe exatamente um valor válido
        
        naked_singles = self.predicates.NakedSingle(rows, cols, boards, candidates_tensors)
        
        # Para cada posição que é naked single, deve existir exatamente um valor válido
        for i, (r, c) in enumerate(positions):
            if len(positions) > i:
                # Verificar todos os valores possíveis para esta célula
                possible_values = torch.arange(1, self.board_size + 1)
                r_expanded = torch.full((self.board_size,), r)
                c_expanded = torch.full((self.board_size,), c)
                board_expanded = board_tensor.unsqueeze(0).expand(self.board_size, -1, -1)
                
                valid_values = self.predicates.ValidCell(r_expanded, c_expanded, possible_values, board_expanded)
                
                # Se é naked single, então exatamente um valor deve ser válido
                exactly_one_valid = self.Exists(
                    ltn.diag(possible_values),
                    valid_values
                )
                
                axiom = self.Implies(naked_singles[i], exactly_one_valid)
                axioms.append(axiom)
        
        return axioms
    
    def get_hidden_single_axioms(self, board_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Axiomas para heurística Hidden Single
        
        Args:
            board_tensor: estado do tabuleiro
            
        Returns:
            lista de axiomas para Hidden Single
        """
        axioms = []
        
        # Axioma: HiddenSingle(r,c,v,board,unit_info) → ValidCell(r,c,v,board)
        # Se uma célula-valor é hidden single, então deve ser válida
        
        # Gerar dados para células vazias
        empty_positions = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board_tensor[r, c] == 0:
                    empty_positions.append((r, c))
        
        if not empty_positions:
            return axioms
        
        # Para cada célula vazia e cada valor possível
        for r, c in empty_positions:
            for v in range(1, self.board_size + 1):
                # Criar unit_info dummy (seria calculado corretamente na prática)
                unit_info = torch.zeros(3 * self.board_size)
                
                r_tensor = torch.tensor([r])
                c_tensor = torch.tensor([c])
                v_tensor = torch.tensor([v])
                board_batch = board_tensor.unsqueeze(0)
                unit_info_batch = unit_info.unsqueeze(0)
                
                hidden_single = self.predicates.HiddenSingle(r_tensor, c_tensor, v_tensor, board_batch, unit_info_batch)
                valid_cell = self.predicates.ValidCell(r_tensor, c_tensor, v_tensor, board_batch)
                
                axiom = self.Implies(hidden_single, valid_cell)
                axioms.append(axiom)
        
        return axioms
    
    def get_consistency_axioms(self, board_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Axiomas de consistência para garantir que as regras básicas sejam respeitadas
        
        Args:
            board_tensor: estado do tabuleiro
            
        Returns:
            lista de axiomas de consistência
        """
        axioms = []
        
        # Axioma: Cada linha deve ter cada número exatamente uma vez
        for r in range(self.board_size):
            for v in range(1, self.board_size + 1):
                # Contar quantas vezes o valor v aparece na linha r
                row_values = board_tensor[r, :]
                count = torch.sum(row_values == v)
                
                # Se o valor já aparece na linha, RowConstraint deve ser falso para novas inserções
                if count > 0:
                    r_tensor = torch.tensor([r])
                    v_tensor = torch.tensor([v])
                    board_batch = board_tensor.unsqueeze(0)
                    
                    row_constraint = self.predicates.RowConstraint(r_tensor, v_tensor, board_batch)
                    
                    # O constraint deve ser falso (valor já existe)
                    axiom = self.Not(row_constraint)
                    axioms.append(axiom)
        
        # Axioma similar para colunas
        for c in range(self.board_size):
            for v in range(1, self.board_size + 1):
                col_values = board_tensor[:, c]
                count = torch.sum(col_values == v)
                
                if count > 0:
                    c_tensor = torch.tensor([c])
                    v_tensor = torch.tensor([v])
                    board_batch = board_tensor.unsqueeze(0)
                    
                    col_constraint = self.predicates.ColConstraint(c_tensor, v_tensor, board_batch)
                    axiom = self.Not(col_constraint)
                    axioms.append(axiom)
        
        # Axioma similar para quadrantes
        box_size = int(self.board_size ** 0.5)
        for box_r in range(0, self.board_size, box_size):
            for box_c in range(0, self.board_size, box_size):
                box = board_tensor[box_r:box_r+box_size, box_c:box_c+box_size]
                
                for v in range(1, self.board_size + 1):
                    count = torch.sum(box == v)
                    
                    if count > 0:
                        # Usar coordenadas do centro do quadrante
                        r_tensor = torch.tensor([box_r + box_size//2])
                        c_tensor = torch.tensor([box_c + box_size//2])
                        v_tensor = torch.tensor([v])
                        board_batch = board_tensor.unsqueeze(0)
                        
                        box_constraint = self.predicates.BoxConstraint(r_tensor, c_tensor, v_tensor, board_batch)
                        axiom = self.Not(box_constraint)
                        axioms.append(axiom)
        
        return axioms
    
    def get_all_axioms(self, board_tensor: torch.Tensor, 
                      candidates_data: Dict[Tuple[int, int], torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Retorna todos os axiomas da base de conhecimento
        
        Args:
            board_tensor: estado do tabuleiro
            candidates_data: dados de candidatos (opcional)
            
        Returns:
            lista com todos os axiomas
        """
        all_axioms = []
        
        # Axiomas básicos do Sudoku
        basic_axioms = self.get_basic_sudoku_axioms(board_tensor)
        all_axioms.extend(basic_axioms)
        
        # Axiomas de consistência
        consistency_axioms = self.get_consistency_axioms(board_tensor)
        all_axioms.extend(consistency_axioms)
        
        # Axiomas de heurísticas (se dados disponíveis)
        if candidates_data:
            naked_single_axioms = self.get_naked_single_axioms(board_tensor, candidates_data)
            all_axioms.extend(naked_single_axioms)
        
        hidden_single_axioms = self.get_hidden_single_axioms(board_tensor)
        all_axioms.extend(hidden_single_axioms)
        
        return all_axioms
    
    def compute_satisfaction_level(self, axioms: List[torch.Tensor]) -> torch.Tensor:
        """
        Calcula o nível de satisfação da base de conhecimento
        
        Args:
            axioms: lista de axiomas
            
        Returns:
            tensor com o nível de satisfação [0,1]
        """
        if not axioms:
            return torch.tensor(1.0)
        
        # Combinar todos os axiomas com AND
        satisfaction = axioms[0]
        for axiom in axioms[1:]:
            satisfaction = self.And(satisfaction, axiom)
        
        return satisfaction
    
    def get_loss_function(self, axioms: List[torch.Tensor]) -> torch.Tensor:
        """
        Calcula a função de perda baseada na satisfação dos axiomas
        
        Args:
            axioms: lista de axiomas
            
        Returns:
            tensor com a perda
        """
        satisfaction = self.compute_satisfaction_level(axioms)
        
        # Perda = 1 - satisfação (queremos maximizar satisfação)
        loss = 1.0 - satisfaction
        
        return loss 