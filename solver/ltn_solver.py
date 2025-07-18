# Main LTN Solver for Sudoku
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
import os

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sudoku_board import SudokuBoard
from core.memory_system import MemorySystem
from solver.predicates import SudokuPredicates
from solver.knowledge_base import SudokuKnowledgeBase
from utils.training_data import SudokuTrainingDataGenerator

class SudokuLTNSolver:
    """
    Solver principal LTN para Sudoku que integra todos os componentes:
    - Predicates neurais
    - Base de conhecimento com axiomas
    - Sistema de memória para heurísticas
    - Treinamento e inferência
    """
    
    def __init__(self, board_size: int = 9, learning_rate: float = 0.001):
        self.board_size = board_size
        self.learning_rate = learning_rate
        
        # Componentes principais
        self.predicates = SudokuPredicates(board_size)
        self.knowledge_base = SudokuKnowledgeBase(board_size)
        self.memory_system = MemorySystem()
        self.data_generator = SudokuTrainingDataGenerator(board_size)
        
        # Otimizador
        self.optimizer = optim.Adam(
            self.predicates.get_model_parameters(),
            lr=learning_rate
        )
        
        # Histórico de treinamento
        self.training_history = {
            'losses': [],
            'satisfactions': [],
            'epochs': 0
        }
        
    def train_from_csv(self, csv_path: str, epochs: int = 100, batch_size: int = 32):
        """
        Treina o solver usando dados de um arquivo CSV
        
        Args:
            csv_path: caminho para o arquivo CSV com tabuleiros
            epochs: número de épocas de treinamento
            batch_size: tamanho do batch
        """
        print(f"Iniciando treinamento com dados de {csv_path}")
        
        # Gerar dados de treinamento
        training_data = self.data_generator.generate_all_training_data(csv_path)
        
        if not training_data:
            print("Erro: Nenhum dado de treinamento gerado!")
            return
        
        print(f"Dados de treinamento gerados. Iniciando {epochs} épocas...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_satisfaction = 0.0
            num_batches = 0
            
            # Treinar ValidCell
            if 'valid_cell' in training_data:
                loss, satisfaction = self._train_valid_cell_batch(training_data['valid_cell'], batch_size)
                epoch_loss += loss
                epoch_satisfaction += satisfaction
                num_batches += 1
            
            # Treinar Constraints
            if 'constraints' in training_data:
                loss, satisfaction = self._train_constraints_batch(training_data['constraints'], batch_size)
                epoch_loss += loss
                epoch_satisfaction += satisfaction
                num_batches += 1
            
            # Treinar NakedSingle
            if 'naked_single' in training_data:
                loss, satisfaction = self._train_naked_single_batch(training_data['naked_single'], batch_size)
                epoch_loss += loss
                epoch_satisfaction += satisfaction
                num_batches += 1
            
            # Treinar HiddenSingle
            if 'hidden_single' in training_data:
                loss, satisfaction = self._train_hidden_single_batch(training_data['hidden_single'], batch_size)
                epoch_loss += loss
                epoch_satisfaction += satisfaction
                num_batches += 1
            
            # Calcular médias
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_satisfaction = epoch_satisfaction / max(num_batches, 1)
            
            # Armazenar histórico
            self.training_history['losses'].append(avg_loss)
            self.training_history['satisfactions'].append(avg_satisfaction)
            
            if epoch % 10 == 0:
                print(f"Época {epoch}: Loss = {avg_loss:.4f}, Satisfação = {avg_satisfaction:.4f}")
        
        self.training_history['epochs'] = epochs
        print(f"Treinamento concluído! Loss final: {avg_loss:.4f}")
    
    def train_with_boards(self, boards: List[SudokuBoard], epochs: int = 100, 
                         batch_size: int = 32, situation_type: str = "general", is_open_sudokus: bool = True):
        """
        Treina o solver usando uma lista de objetos SudokuBoard
        
        Args:
            boards: lista de objetos SudokuBoard
            epochs: número de épocas de treinamento
            batch_size: tamanho do batch
            situation_type: tipo de situação sendo treinada
            is_open_sudokus: True se são sudokus abertos, False se são fechados
        """
        print(f"Iniciando treinamento para '{situation_type}' com {len(boards)} tabuleiros")
        
        # Gerar dados de treinamento a partir dos boards
        training_data = self.data_generator.generate_training_data_from_boards(boards, is_open_sudokus)
        
        if not training_data:
            print("Erro: Nenhum dado de treinamento gerado!")
            return
        
        print(f"Dados de treinamento gerados. Iniciando {epochs} épocas...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_satisfaction = 0.0
            num_batches = 0
            
            # Treinar ValidCell
            if 'valid_cell' in training_data:
                loss, satisfaction = self._train_valid_cell_batch(training_data['valid_cell'], batch_size)
                epoch_loss += loss
                epoch_satisfaction += satisfaction
                num_batches += 1
            
            # Treinar Constraints
            if 'constraints' in training_data:
                loss, satisfaction = self._train_constraints_batch(training_data['constraints'], batch_size)
                epoch_loss += loss
                epoch_satisfaction += satisfaction
                num_batches += 1
            
            # Treinar NakedSingle (só se há dados)
            if 'naked_single' in training_data:
                naked_single_data = training_data['naked_single']
                if len(naked_single_data[0]) > 0:  # Verificar se há dados reais
                    loss, satisfaction = self._train_naked_single_batch(naked_single_data, batch_size)
                    epoch_loss += loss
                    epoch_satisfaction += satisfaction
                    num_batches += 1
            
            # Treinar HiddenSingle (só se há dados)
            if 'hidden_single' in training_data:
                hidden_single_data = training_data['hidden_single']
                if len(hidden_single_data[0]) > 0:  # Verificar se há dados reais
                    loss, satisfaction = self._train_hidden_single_batch(hidden_single_data, batch_size)
                    epoch_loss += loss
                    epoch_satisfaction += satisfaction
                    num_batches += 1
            
            # Calcular médias
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_satisfaction = epoch_satisfaction / max(num_batches, 1)
            
            # Armazenar histórico
            self.training_history['losses'].append(avg_loss)
            self.training_history['satisfactions'].append(avg_satisfaction)
            
            if epoch % 10 == 0:
                print(f"  Época {epoch}: Loss = {avg_loss:.4f}, Satisfação = {avg_satisfaction:.4f}")
        
        self.training_history['epochs'] += epochs
        print(f"Treinamento '{situation_type}' concluído! Loss final: {avg_loss:.4f}")
    
    def _train_valid_cell_batch(self, data: Tuple, batch_size: int) -> Tuple[float, float]:
        """
        Treina o predicate ValidCell
        """
        rows, cols, values, boards, labels = data
        total_loss = 0.0
        total_satisfaction = 0.0
        num_batches = 0
        
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i+batch_size]
            batch_cols = cols[i:i+batch_size]
            batch_values = values[i:i+batch_size]
            batch_boards = boards[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            self.optimizer.zero_grad()
            
            # Forward pass - usar o modelo diretamente em vez do predicate LTN
            predictions = self.predicates.valid_cell_model(batch_rows, batch_cols, batch_values, batch_boards)
            
            # Loss usando BCE
            loss = nn.functional.binary_cross_entropy(predictions, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_satisfaction += torch.mean(predictions).item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1), total_satisfaction / max(num_batches, 1)
    
    def _train_constraints_batch(self, data: Dict, batch_size: int) -> Tuple[float, float]:
        """
        Treina os predicates de constraints (Row, Col, Box)
        """
        total_loss = 0.0
        total_satisfaction = 0.0
        num_batches = 0
        
        # Treinar Row Constraint
        if 'row' in data:
            indices, values, boards, labels = data['row']
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_values = values[i:i+batch_size]
                batch_boards = boards[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                self.optimizer.zero_grad()
                
                predictions = self.predicates.row_constraint_model(batch_indices, batch_values, batch_boards)
                loss = nn.functional.binary_cross_entropy(predictions, batch_labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_satisfaction += torch.mean(predictions).item()
                num_batches += 1
        
        # Treinar Col Constraint
        if 'col' in data:
            indices, values, boards, labels = data['col']
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_values = values[i:i+batch_size]
                batch_boards = boards[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                self.optimizer.zero_grad()
                
                predictions = self.predicates.col_constraint_model(batch_indices, batch_values, batch_boards)
                loss = nn.functional.binary_cross_entropy(predictions, batch_labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_satisfaction += torch.mean(predictions).item()
                num_batches += 1
        
        # Treinar Box Constraint
        if 'box' in data:
            row_indices, col_indices, values, boards, labels = data['box']
            
            for i in range(0, len(row_indices), batch_size):
                batch_row_indices = row_indices[i:i+batch_size]
                batch_col_indices = col_indices[i:i+batch_size]
                batch_values = values[i:i+batch_size]
                batch_boards = boards[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                
                self.optimizer.zero_grad()
                
                predictions = self.predicates.box_constraint_model(batch_row_indices, batch_col_indices, batch_values, batch_boards)
                loss = nn.functional.binary_cross_entropy(predictions, batch_labels)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_satisfaction += torch.mean(predictions).item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1), total_satisfaction / max(num_batches, 1)
    
    def _train_naked_single_batch(self, data: Tuple, batch_size: int) -> Tuple[float, float]:
        """
        Treina o predicate NakedSingle
        """
        rows, cols, boards, candidates, labels = data
        total_loss = 0.0
        total_satisfaction = 0.0
        num_batches = 0
        
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i+batch_size]
            batch_cols = cols[i:i+batch_size]
            batch_boards = boards[i:i+batch_size]
            batch_candidates = candidates[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            self.optimizer.zero_grad()
            
            predictions = self.predicates.naked_single_model(batch_rows, batch_cols, batch_boards, batch_candidates)
            loss = nn.functional.binary_cross_entropy(predictions, batch_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_satisfaction += torch.mean(predictions).item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1), total_satisfaction / max(num_batches, 1)
    
    def _train_hidden_single_batch(self, data: Tuple, batch_size: int) -> Tuple[float, float]:
        """
        Treina o predicate HiddenSingle
        """
        rows, cols, values, boards, unit_candidates, labels = data
        total_loss = 0.0
        total_satisfaction = 0.0
        num_batches = 0
        
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i:i+batch_size]
            batch_cols = cols[i:i+batch_size]
            batch_values = values[i:i+batch_size]
            batch_boards = boards[i:i+batch_size]
            batch_unit_candidates = unit_candidates[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            self.optimizer.zero_grad()
            
            predictions = self.predicates.hidden_single_model(batch_rows, batch_cols, batch_values, batch_boards, batch_unit_candidates)
            loss = nn.functional.binary_cross_entropy(predictions, batch_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_satisfaction += torch.mean(predictions).item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1), total_satisfaction / max(num_batches, 1)
    
    def solve_sudoku(self, board: SudokuBoard, max_iterations: int = 100) -> Dict:
        """
        Resolve um tabuleiro Sudoku usando LTN e sistema de memória
        
        Args:
            board: tabuleiro Sudoku para resolver
            max_iterations: número máximo de iterações
            
        Returns:
            dicionário com resultado da resolução
        """
        print("Iniciando resolução do Sudoku com LTN...")
        
        # Análise inicial usando SudokuBoard
        initial_analysis = board.get_board_info()
        print(f"Estado inicial: {initial_analysis['tipo']}, válido: {initial_analysis['valido']}")
        
        if not initial_analysis['valido']:
            return {
                'sucesso': False,
                'motivo': 'Tabuleiro inicial inválido',
                'conflitos': initial_analysis['conflitos'],
                'board_final': board,
                'iteracoes': 0
            }
        
        if initial_analysis['tipo'] == 'fechado':
            return {
                'sucesso': True,
                'motivo': 'Tabuleiro já resolvido',
                'board_final': board,
                'iteracoes': 0
            }
        
        # Inicializar sistema de memória
        self.memory_system.clear_memory()
        
        # Processo de resolução iterativo
        current_board = SudokuBoard(board.board.copy())
        iteration = 0
        
        for iteration in range(max_iterations):
            print(f"Iteração {iteration + 1}")
            
            # Verificar se foi resolvido
            if current_board.is_closed():
                if current_board.is_valid():
                    print("Sudoku resolvido com sucesso!")
                    return {
                        'sucesso': True,
                        'motivo': 'Resolvido completamente',
                        'board_final': current_board,
                        'iteracoes': iteration + 1,
                        'memoria_usada': self.memory_system.get_memory_summary()
                    }
                else:
                    print("Tabuleiro completo mas inválido!")
                    return {
                        'sucesso': False,
                        'motivo': 'Solução inválida encontrada',
                        'board_final': current_board,
                        'iteracoes': iteration + 1
                    }
            
            # Tentar fazer um movimento usando LTN
            move_made = self._make_ltn_move(current_board)
            
            if not move_made:
                print("Nenhum movimento possível encontrado.")
                break
        
        # Resultado final
        final_analysis = current_board.get_board_info()
        return {
            'sucesso': current_board.is_closed() and current_board.is_valid(),
            'motivo': 'Máximo de iterações atingido' if iteration >= max_iterations - 1 else 'Sem movimentos possíveis',
            'board_final': current_board,
            'iteracoes': iteration + 1,
            'posicoes_restantes': len(final_analysis['posicoes_abertas']),
            'memoria_usada': self.memory_system.get_memory_summary()
        }
    
    def _make_ltn_move(self, board: SudokuBoard) -> bool:
        """
        Faz um movimento usando LTN e sistema de memória
        
        Args:
            board: tabuleiro atual
            
        Returns:
            True se um movimento foi feito, False caso contrário
        """
        board_tensor = board.to_tensor()
        
        # Tentar Naked Single primeiro
        naked_single_move = self._try_naked_single(board, board_tensor)
        if naked_single_move:
            row, col, value = naked_single_move
            board.board[row, col] = value
            self.memory_system.add_heuristic_usage('naked_single', (row, col), value)
            print(f"Naked Single: ({row}, {col}) = {value}")
            return True
        
        # Tentar Hidden Single
        hidden_single_move = self._try_hidden_single(board, board_tensor)
        if hidden_single_move:
            row, col, value = hidden_single_move
            board.board[row, col] = value
            self.memory_system.add_heuristic_usage('hidden_single', (row, col), value)
            print(f"Hidden Single: ({row}, {col}) = {value}")
            return True
        
        # Tentar movimento baseado em ValidCell com maior confiança
        valid_cell_move = self._try_valid_cell_move(board, board_tensor)
        if valid_cell_move:
            row, col, value = valid_cell_move
            board.board[row, col] = value
            self.memory_system.add_heuristic_usage('valid_cell', (row, col), value)
            print(f"Valid Cell: ({row}, {col}) = {value}")
            return True
        
        return False
    
    def _try_naked_single(self, board: SudokuBoard, board_tensor: torch.Tensor) -> Optional[Tuple[int, int, int]]:
        """
        Tenta encontrar um movimento Naked Single
        """
        candidates_matrix = board.get_candidates_matrix()
        
        for (row, col), candidates in candidates_matrix.items():
            if len(candidates) == 1:
                # Verificar com LTN
                candidates_vector = torch.zeros(self.board_size)
                for candidate in candidates:
                    candidates_vector[candidate - 1] = 1.0
                
                r_tensor = torch.tensor([row])
                c_tensor = torch.tensor([col])
                board_batch = board_tensor.unsqueeze(0)
                candidates_batch = candidates_vector.unsqueeze(0)
                
                confidence = self.predicates.naked_single_model(r_tensor, c_tensor, board_batch, candidates_batch)
                
                if confidence.item() > 0.7:  # Threshold de confiança
                    return (row, col, list(candidates)[0])
        
        return None
    
    def _try_hidden_single(self, board: SudokuBoard, board_tensor: torch.Tensor) -> Optional[Tuple[int, int, int]]:
        """
        Tenta encontrar um movimento Hidden Single
        """
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board.board[row, col] == 0:
                    possible_values = board.get_possible_numbers(row, col)
                    
                    for value in possible_values:
                        # Verificar se é hidden single usando a lógica determinística
                        if self.data_generator._is_hidden_single(board, row, col, value):
                            # Confirmar com LTN
                            unit_info = self.data_generator._get_unit_candidates_info(board, row, col, value)
                            
                            r_tensor = torch.tensor([row])
                            c_tensor = torch.tensor([col])
                            v_tensor = torch.tensor([value])
                            board_batch = board_tensor.unsqueeze(0)
                            unit_info_batch = unit_info.unsqueeze(0)
                            
                            confidence = self.predicates.hidden_single_model(r_tensor, c_tensor, v_tensor, board_batch, unit_info_batch)
                            
                            if confidence.item() > 0.7:
                                return (row, col, value)
        
        return None
    
    def _try_valid_cell_move(self, board: SudokuBoard, board_tensor: torch.Tensor) -> Optional[Tuple[int, int, int]]:
        """
        Tenta encontrar um movimento baseado no predicate ValidCell
        """
        best_move = None
        best_confidence = 0.0
        
        open_positions = board.get_open_positions()
        
        for row, col in open_positions:
            possible_values = board.get_possible_numbers(row, col)
            
            for value in possible_values:
                r_tensor = torch.tensor([row])
                c_tensor = torch.tensor([col])
                v_tensor = torch.tensor([value])
                board_batch = board_tensor.unsqueeze(0)
                
                confidence = self.predicates.valid_cell_model(r_tensor, c_tensor, v_tensor, board_batch)
                
                if confidence.item() > best_confidence:
                    best_confidence = confidence.item()
                    best_move = (row, col, value)
        
        if best_confidence > 0.8:  # Threshold alto para ValidCell
            return best_move
        
        return None
    
    def get_training_summary(self) -> Dict:
        """
        Retorna resumo do treinamento
        """
        return {
            'epochs_trained': self.training_history['epochs'],
            'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else 0.0,
            'final_satisfaction': self.training_history['satisfactions'][-1] if self.training_history['satisfactions'] else 0.0,
            'loss_history': self.training_history['losses'],
            'satisfaction_history': self.training_history['satisfactions']
        }
    
    def save_model(self, path: str):
        """
        Salva o modelo treinado
        """
        torch.save({
            'predicates_state': {
                'valid_cell': self.predicates.valid_cell_model.state_dict(),
                'row_constraint': self.predicates.row_constraint_model.state_dict(),
                'col_constraint': self.predicates.col_constraint_model.state_dict(),
                'box_constraint': self.predicates.box_constraint_model.state_dict(),
                'naked_single': self.predicates.naked_single_model.state_dict(),
                'hidden_single': self.predicates.hidden_single_model.state_dict(),
            },
            'training_history': self.training_history,
            'board_size': self.board_size
        }, path)
        print(f"Modelo salvo em {path}")
    
    def load_model(self, path: str):
        """
        Carrega um modelo treinado
        """
        checkpoint = torch.load(path)
        
        self.predicates.valid_cell_model.load_state_dict(checkpoint['predicates_state']['valid_cell'])
        self.predicates.row_constraint_model.load_state_dict(checkpoint['predicates_state']['row_constraint'])
        self.predicates.col_constraint_model.load_state_dict(checkpoint['predicates_state']['col_constraint'])
        self.predicates.box_constraint_model.load_state_dict(checkpoint['predicates_state']['box_constraint'])
        self.predicates.naked_single_model.load_state_dict(checkpoint['predicates_state']['naked_single'])
        self.predicates.hidden_single_model.load_state_dict(checkpoint['predicates_state']['hidden_single'])
        
        self.training_history = checkpoint['training_history']
        self.board_size = checkpoint['board_size']
        
        print(f"Modelo carregado de {path}") 