#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema LTN para Resolução de Sudoku - Versão Separada por Dimensão
Projeto Final - Inteligência Artificial - UFAM
Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas

Este script treina modelos especializados separadamente para Sudoku 4x4 e 9x9,
criando modelos independentes para cada dimensão.
"""

import sys
import os
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

# Adicionar o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from core.sudoku_board import SudokuBoard
from solver.ltn_solver import SudokuLTNSolver

def create_sample_sudoku():
    """
    Cria um tabuleiro Sudoku de exemplo 9x9 para demonstração
    """
    # Tabuleiro 9x9 parcialmente preenchido
    board_data = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    return SudokuBoard(board_data)

def create_sample_4x4_sudoku():
    """
    Cria um tabuleiro Sudoku de exemplo 4x4 para demonstração
    """
    # Tabuleiro 4x4 parcialmente preenchido
    board_data = [
        [1, 0, 3, 4],
        [3, 4, 0, 2],
        [0, 1, 4, 3],
        [4, 3, 2, 0]
    ]
    return SudokuBoard(board_data)

def create_unsolvable_4x4_sudoku():
    """
    Cria um sudoku 4x4 impossível para demonstração
    """
    # Sudoku 4x4 com conflito na primeira linha (dois números 1)
    board_data = [
        [1, 1, 3, 4],  # Conflito: dois números 1 na mesma linha
        [3, 4, 0, 2],
        [0, 1, 4, 3],
        [4, 3, 2, 0]
    ]
    return SudokuBoard(board_data)

def create_another_unsolvable_4x4_sudoku():
    """
    Cria outro sudoku 4x4 impossível para demonstração
    """
    # Sudoku 4x4 com conflito na primeira coluna (dois números 1)
    board_data = [
        [1, 0, 3, 4],
        [1, 4, 0, 2],  # Conflito: dois números 1 na mesma coluna
        [0, 1, 4, 3],
        [4, 3, 2, 0]
    ]
    return SudokuBoard(board_data)

def sudoku_string_to_board(sudoku_str: str) -> SudokuBoard:
    """
    Converte string de sudoku em SudokuBoard (suporta 4x4 e 9x9)
    """
    length = len(sudoku_str)
    
    if length == 16:  # 4x4 sudoku
        board_size = 4
    elif length == 81:  # 9x9 sudoku
        board_size = 9
    else:
        raise ValueError(f"String deve ter 16 caracteres (4x4) ou 81 caracteres (9x9), recebido: {length}")
    
    # Converter string para matriz
    board_data = []
    for i in range(board_size):
        row = []
        for j in range(board_size):
            char = sudoku_str[i * board_size + j]
            # Converter '.' para 0 se necessário
            row.append(0 if char == '.' else int(char))
        board_data.append(row)
    
    return SudokuBoard(board_data)

def load_sudokus_from_csv(csv_path: str, max_samples: int = 2000) -> List[SudokuBoard]:
    """
    Carrega sudokus de um arquivo CSV (suporta 4x4 e 9x9)
    """
    sudokus = []
    print(f"Carregando sudokus de {csv_path}...")
    
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            
        # Limita o número de amostras
        lines = lines[:max_samples]
        
        for i, line in enumerate(lines):
            sudoku_str = line.strip()
            # Verificar se é 4x4 (16 caracteres) ou 9x9 (81 caracteres)
            if len(sudoku_str) in [16, 81]:
                try:
                    board = sudoku_string_to_board(sudoku_str)
                    sudokus.append(board)
                except Exception as e:
                    print(f"Erro ao converter linha {i+1}: {e}")
            
            if i % 5000 == 0 and i > 0:
                print(f"  Carregados {i} sudokus...")
    
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {csv_path}")
        return []
    
    print(f"Total carregado: {len(sudokus)} sudokus")
    return sudokus

def load_sudoku_from_csv(csv_path: str) -> SudokuBoard:
    """
    Carrega um único sudoku de um arquivo CSV
    """
    print(f"Carregando sudoku de {csv_path}...")
    
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            raise ValueError("Arquivo CSV está vazio")
        
        # Pegar a primeira linha (assumindo que é o sudoku)
        sudoku_str = lines[0].strip()
        
        # Verificar se é 4x4 (16 caracteres) ou 9x9 (81 caracteres)
        if len(sudoku_str) not in [16, 81]:
            raise ValueError(f"String deve ter 16 caracteres (4x4) ou 81 caracteres (9x9), recebido: {len(sudoku_str)}")
        
        board = sudoku_string_to_board(sudoku_str)
        print(f"✅ Sudoku carregado com sucesso!")
        return board
        
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {csv_path}")
        return None
    except Exception as e:
        print(f"❌ Erro ao carregar sudoku: {e}")
        return None

def classify_closed_board(board: SudokuBoard) -> Dict:
    """
    Questão 1: Classifica um tabuleiro fechado
    """
    print("\n🔍 QUESTÃO 1: CLASSIFICANDO TABULEIRO FECHADO")
    print("=" * 60)
    
    print("📋 Tabuleiro:")
    print(board)
    
    # Verificar se é válido
    is_valid = board.is_valid()
    
    print(f"\n📊 ANÁLISE:")
    print(f"  Tamanho: {board.size}x{board.size}")
    print(f"  Tipo: Fechado (completamente preenchido)")
    print(f"  Válido: {'✅ Sim' if is_valid else '❌ Não'}")
    
    if not is_valid:
        print(f"\n🚨 CONFLITOS ENCONTRADOS:")
        conflicts = board.find_invalid_numbers()
        for conflict in conflicts:
            print(f"  - Número {conflict['numero']} aparece {conflict['ocorrencias']} vezes na {conflict['local']}")
    
    # Classificação final
    if is_valid:
        classification = "VÁLIDO"
        description = "Tabuleiro Sudoku válido e completo"
    else:
        classification = "INVÁLIDO"
        description = "Tabuleiro Sudoku com conflitos"
    
    print(f"\n🎯 CLASSIFICAÇÃO FINAL: {classification}")
    print(f"📝 Descrição: {description}")
    
    return {
        'classification': classification,
        'description': description,
        'is_valid': is_valid,
        'conflicts': board.find_invalid_numbers() if not is_valid else []
    }

def solve_open_board(board: SudokuBoard, solver: SudokuLTNSolver) -> Dict:
    """
    Questão 2: Classifica um tabuleiro aberto e sugere jogadas possíveis
    """
    print("\n🧩 QUESTÃO 2: CLASSIFICANDO TABULEIRO ABERTO")
    print("=" * 60)
    
    print("📋 Tabuleiro inicial:")
    print(board)
    
    # Análise inicial
    info = board.get_board_info()
    print(f"\n📊 ANÁLISE INICIAL:")
    print(f"  Tamanho: {board.size}x{board.size}")
    print(f"  Tipo: Aberto (com células vazias)")
    print(f"  Válido: {'✅ Sim' if info['valido'] else '❌ Não'}")
    print(f"  Posições abertas: {len(info['posicoes_abertas'])}")
    
    if not info['valido']:
        print(f"\n🚨 CONFLITOS ENCONTRADOS:")
        for conflict in info['conflitos']:
            print(f"  - Número {conflict['numero']} aparece {conflict['ocorrencias']} vezes na {conflict['local']}")
        
        return {
            'solvable': False,
            'reason': 'Tabuleiro inicial inválido',
            'conflicts': info['conflitos']
        }
    
    # Verificar candidatos para cada célula vazia
    print(f"\n🔍 ANALISANDO CANDIDATOS:")
    candidates_matrix = board.get_candidates_matrix()
    
    cells_without_candidates = []
    possible_moves = []
    
    for (row, col), candidates in candidates_matrix.items():
        print(f"  Célula ({row},{col}): candidatos = {candidates}")
        if not candidates:
            cells_without_candidates.append((row, col))
        else:
            # Adicionar o primeiro candidato como sugestão
            first_candidate = list(candidates)[0]
            possible_moves.append((row, col, first_candidate))
    
    if cells_without_candidates:
        print(f"\n❌ CÉLULAS SEM CANDIDATOS:")
        for row, col in cells_without_candidates:
            print(f"  - Célula ({row},{col}) não tem candidatos válidos")
        
        return {
            'solvable': False,
            'reason': 'Existem células sem candidatos válidos',
            'cells_without_candidates': cells_without_candidates
        }
    
    # Classificação final
    if len(possible_moves) > 0:
        print(f"\n✅ RESULTADO: TABULEIRO SOLUCIONÁVEL")
        print(f"📝 Motivo: Encontradas {len(possible_moves)} jogadas possíveis")
        
        # Mostrar todas as jogadas encontradas
        print(f"\n🎯 TODAS AS JOGADAS POSSÍVEIS ({len(possible_moves)} encontradas):")
        for i, (row, col, value) in enumerate(possible_moves, 1):
            print(f"  {i}️⃣ ({row},{col}) = {value}")
        
        return {
            'solvable': True,
            'reason': f'Encontradas {len(possible_moves)} jogadas possíveis',
            'total_moves': len(possible_moves),
            'suggested_moves': possible_moves,
            'all_moves': possible_moves
        }
    else:
        print(f"\n❌ RESULTADO: TABULEIRO NÃO SOLUCIONÁVEL")
        print(f"📝 Motivo: Nenhuma jogada válida encontrada")
        
        return {
            'solvable': False,
            'reason': 'Nenhuma jogada válida encontrada',
            'total_moves': 0,
            'suggested_moves': [],
            'all_moves': []
        }

def check_solvability(board: SudokuBoard, solver: SudokuLTNSolver) -> Dict:
    """
    Questão 3: Verifica se um tabuleiro aberto é solucionável
    """
    print("\n🔍 QUESTÃO 3: VERIFICANDO SOLUCIONABILIDADE")
    print("=" * 60)
    
    print("📋 Tabuleiro:")
    print(board)
    
    # Análise inicial
    info = board.get_board_info()
    print(f"\n📊 ANÁLISE INICIAL:")
    print(f"  Tamanho: {board.size}x{board.size}")
    print(f"  Tipo: Aberto (com células vazias)")
    print(f"  Válido: {'✅ Sim' if info['valido'] else '❌ Não'}")
    print(f"  Posições abertas: {len(info['posicoes_abertas'])}")
    
    if not info['valido']:
        print(f"\n🚨 CONFLITOS ENCONTRADOS:")
        for conflict in info['conflitos']:
            print(f"  - Número {conflict['numero']} aparece {conflict['ocorrencias']} vezes na {conflict['local']}")
        
        return {
            'solvable': False,
            'reason': 'Tabuleiro inicial inválido',
            'conflicts': info['conflitos']
        }
    
    # Verificar candidatos para cada célula vazia
    print(f"\n🔍 ANALISANDO CANDIDATOS:")
    candidates_matrix = board.get_candidates_matrix()
    
    cells_without_candidates = []
    for (row, col), candidates in candidates_matrix.items():
        print(f"  Célula ({row},{col}): candidatos = {candidates}")
        if not candidates:
            cells_without_candidates.append((row, col))
    
    if cells_without_candidates:
        print(f"\n❌ CÉLULAS SEM CANDIDATOS:")
        for row, col in cells_without_candidates:
            print(f"  - Célula ({row},{col}) não tem candidatos válidos")
        
        return {
            'solvable': False,
            'reason': 'Existem células sem candidatos válidos',
            'cells_without_candidates': cells_without_candidates
        }
    
    # Analisar possíveis jogadas usando heurísticas
    print(f"\n🎯 ANALISANDO HEURÍSTICAS:")
    
    possible_moves = []
    
    # 1. Verificar Naked Singles
    print("  🔍 Procurando Naked Singles...")
    naked_singles = []
    for (row, col), candidates in candidates_matrix.items():
        if len(candidates) == 1:
            value = list(candidates)[0]
            naked_singles.append((row, col, value, "Naked Single"))
            print(f"    ✅ Naked Single: ({row},{col}) = {value}")
    
    possible_moves.extend(naked_singles)
    
    # 2. Verificar Hidden Singles
    print("  🔍 Procurando Hidden Singles...")
    hidden_singles = []
    
    # Verificar linhas
    for row in range(board.size):
        for value in range(1, board.size + 1):
            cells_with_value = []
            for col in range(board.size):
                if board.board[row, col] == 0 and value in board.get_possible_numbers(row, col):
                    cells_with_value.append((row, col))
            
            if len(cells_with_value) == 1:
                r, c = cells_with_value[0]
                hidden_singles.append((r, c, value, f"Hidden Single (linha {row})"))
                print(f"    ✅ Hidden Single: ({r},{c}) = {value} (linha {row})")
    
    # Verificar colunas
    for col in range(board.size):
        for value in range(1, board.size + 1):
            cells_with_value = []
            for row in range(board.size):
                if board.board[row, col] == 0 and value in board.get_possible_numbers(row, col):
                    cells_with_value.append((row, col))
            
            if len(cells_with_value) == 1:
                r, c = cells_with_value[0]
                hidden_singles.append((r, c, value, f"Hidden Single (coluna {col})"))
                print(f"    ✅ Hidden Single: ({r},{c}) = {value} (coluna {col})")
    
    # Verificar quadrantes
    box_size = int(np.sqrt(board.size))
    for box_row in range(0, board.size, box_size):
        for box_col in range(0, board.size, box_size):
            for value in range(1, board.size + 1):
                cells_with_value = []
                for r in range(box_row, box_row + box_size):
                    for c in range(box_col, box_col + box_size):
                        if board.board[r, c] == 0 and value in board.get_possible_numbers(r, c):
                            cells_with_value.append((r, c))
                
                if len(cells_with_value) == 1:
                    r, c = cells_with_value[0]
                    box_idx = (box_row // box_size) * box_size + (box_col // box_size)
                    hidden_singles.append((r, c, value, f"Hidden Single (quadrante {box_idx})"))
                    print(f"    ✅ Hidden Single: ({r},{c}) = {value} (quadrante {box_idx})")
    
    possible_moves.extend(hidden_singles)
    
    # 3. Verificar movimentos baseados em ValidCell (maior confiança)
    print("  🔍 Analisando movimentos por confiança...")
    confidence_moves = []
    
    for (row, col), candidates in candidates_matrix.items():
        best_confidence = 0
        best_value = None
        
        for value in candidates:
            # Simular o movimento
            temp_board = SudokuBoard(board.board.copy())
            temp_board.board[row, col] = value
            
            # Verificar se é válido
            if temp_board.is_valid():
                # Calcular confiança baseada no número de candidatos
                confidence = 1.0 / len(candidates)  # Menos candidatos = maior confiança
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_value = value
        
        if best_value is not None:
            confidence_moves.append((row, col, best_value, f"Movimento por confiança ({best_confidence:.2f})"))
            print(f"    📊 Movimento sugerido: ({row},{col}) = {best_value} (confiança: {best_confidence:.2f})")
    
    possible_moves.extend(confidence_moves)
    
    # Remover duplicatas
    unique_moves = []
    seen_positions = set()
    for move in possible_moves:
        row, col, value, description = move
        if (row, col) not in seen_positions:
            unique_moves.append(move)
            seen_positions.add((row, col))
    
    # Tentar resolver para verificar solucionabilidade
    print(f"\n🔍 TESTANDO RESOLUÇÃO...")
    resultado = solver.solve_sudoku(board)
    
    if resultado['sucesso']:
        print(f"\n✅ RESULTADO: SOLUCIONÁVEL")
        print(f"📝 Motivo: Tabuleiro foi resolvido com sucesso")
        print(f"📋 Solução:")
        print(resultado['board_final'])
        
        return {
            'solvable': True,
            'reason': 'Tabuleiro foi resolvido com sucesso',
            'solution': resultado['board_final'],
            'heuristics_found': len(unique_moves),
            'heuristic_moves': unique_moves
        }
    else:
        print(f"\n❌ RESULTADO: NÃO SOLUCIONÁVEL")
        print(f"📝 Motivo: {resultado['motivo']}")
        
        if 'posicoes_restantes' in resultado:
            print(f"  Posições restantes: {resultado['posicoes_restantes']}")
        
        return {
            'solvable': False,
            'reason': resultado['motivo'],
            'remaining_positions': resultado.get('posicoes_restantes', 0),
            'heuristics_found': len(unique_moves),
            'heuristic_moves': unique_moves
        }

def get_training_configs_4x4(data_dir: str) -> List[Dict]:
    """
    Retorna as configurações de treinamento para 4x4
    """
    max_samples = 20000
    base_path = os.path.join(data_dir, "4x4")
    
    training_configs = [
        {
            "name": "Sudokus 4x4 Fechados Válidos",
            "file": os.path.join(base_path, "sudoku_4x4_closed_valid.csv"),
            "description": "Aprende a reconhecer sudokus 4x4 completos e corretos",
            "max_samples": max_samples
        },
        {
            "name": "Sudokus 4x4 Fechados Inválidos", 
            "file": os.path.join(base_path, "sudoku_4x4_closed_invalid.csv"),
            "description": "Aprende a identificar sudokus 4x4 completos mas incorretos",
            "max_samples": max_samples
        },
        {
            "name": "Sudokus 4x4 Abertos Solucionáveis",
            "file": os.path.join(base_path, "sudoku_4x4_open_solvable.csv"), 
            "description": "Aprende a resolver sudokus 4x4 parciais com solução",
            "max_samples": max_samples
        },
        {
            "name": "Sudokus 4x4 Abertos Impossíveis",
            "file": os.path.join(base_path, "sudoku_4x4_open_unsolvable.csv"),
            "description": "Aprende a identificar sudokus 4x4 impossíveis de resolver",
            "max_samples": max_samples
        }
    ]
    
    return training_configs

def get_training_configs_9x9(data_dir: str) -> List[Dict]:
    """
    Retorna as configurações de treinamento para 9x9
    """
    max_samples = 500
    base_path = os.path.join(data_dir, "9x9")
    
    # Verificar se existe o diretório 9x9, senão usar a raiz
    if not os.path.exists(base_path):
        base_path = data_dir
        
    training_configs = [
        {
            "name": "Sudokus 9x9 Fechados Válidos",
            "file": os.path.join(base_path, "sudoku_closed_valid.csv"),
            "description": "Aprende a reconhecer sudokus 9x9 completos e corretos",
            "max_samples": max_samples
        },
        {
            "name": "Sudokus 9x9 Fechados Inválidos", 
            "file": os.path.join(base_path, "sudoku_closed_invalid.csv"),
            "description": "Aprende a identificar sudokus 9x9 completos mas incorretos",
            "max_samples": max_samples
        },
        {
            "name": "Sudokus 9x9 Abertos Solucionáveis",
            "file": os.path.join(base_path, "sudoku_open_solvable.csv"), 
            "description": "Aprende a resolver sudokus 9x9 parciais com solução",
            "max_samples": max_samples
        },
        {
            "name": "Sudokus 9x9 Abertos Impossíveis",
            "file": os.path.join(base_path, "sudoku_open_unsolvable.csv"),
            "description": "Aprende a identificar sudokus 9x9 impossíveis de resolver",
            "max_samples": max_samples
        }
    ]
    
    return training_configs

def train_model_for_dimension(board_size: int, data_dir: str, epochs: int = 50):
    """
    Treina um modelo para uma dimensão específica (4x4 ou 9x9)
    """
    print(f"\n🎯 TREINAMENTO PARA SUDOKU {board_size}x{board_size}")
    print("=" * 60)
    
    # Obter configurações de treinamento baseadas na dimensão
    if board_size == 4:
        training_configs = get_training_configs_4x4(data_dir)
    else:
        training_configs = get_training_configs_9x9(data_dir)
    
    # Inicializar solver para a dimensão específica
    solver = SudokuLTNSolver(board_size=board_size)
    
    training_results = {}
    
    for i, config in enumerate(training_configs, 1):
        print(f"\n{i}️⃣ TREINANDO: {config['name']}")
        print(f"📝 {config['description']}")
        print("-" * 50)
        
        # Carregar dados
        csv_path = config['file']
        sudokus = load_sudokus_from_csv(csv_path, config['max_samples'])
        
        if not sudokus:
            print(f"❌ Falha ao carregar dados de {csv_path}")
            continue
        
        # Treinar para este tipo específico
        print(f"🎓 Iniciando treinamento com {len(sudokus)} amostras por {epochs} épocas...")
        start_time = time.time()
        
        try:
            # Determinar se são sudokus abertos ou fechados baseado no nome do arquivo
            is_open_sudokus = "open" in config['file']
            
            # Treinar o solver com este conjunto de dados
            solver.train_with_boards(sudokus, epochs=epochs, situation_type=config['name'], is_open_sudokus=is_open_sudokus)
            
            training_time = time.time() - start_time
            summary = solver.get_training_summary()
            
            training_results[config['name']] = {
                'samples': len(sudokus),
                'epochs': epochs,
                'training_time': training_time,
                'final_loss': summary.get('final_loss', 0),
                'final_satisfaction': summary.get('final_satisfaction', 0)
            }
            
            print(f"✅ Treinamento concluído em {training_time:.2f}s")
            print(f"📊 Loss final: {summary.get('final_loss', 0):.4f}")
            print(f"📊 Satisfação final: {summary.get('final_satisfaction', 0):.4f}")
            
        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            training_results[config['name']] = {'error': str(e)}
    
    # Salvar modelo para esta dimensão
    model_path = f"models/sudoku_ltn_{board_size}x{board_size}.pth"
    os.makedirs("models", exist_ok=True)
    
    print(f"\n💾 Salvando modelo {board_size}x{board_size} em: {model_path}")
    solver.save_model(model_path)
    
    # Criar relatório de treinamento
    report_path = f"models/training_report_{board_size}x{board_size}.txt"
    with open(report_path, 'w') as f:
        f.write(f"RELATÓRIO DE TREINAMENTO - SISTEMA LTN SUDOKU {board_size}x{board_size}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas\n")
        f.write(f"Tamanho do tabuleiro: {board_size}x{board_size}\n\n")
        
        f.write("RESULTADOS POR SITUAÇÃO:\n")
        f.write("-" * 40 + "\n")
        
        total_samples = 0
        total_time = 0
        
        for situation, results in training_results.items():
            f.write(f"\n{situation}:\n")
            if 'error' in results:
                f.write(f"  ❌ Erro: {results['error']}\n")
            else:
                f.write(f"  Amostras: {results['samples']}\n")
                f.write(f"  Épocas: {results['epochs']}\n")
                f.write(f"  Tempo: {results['training_time']:.2f}s\n")
                f.write(f"  Loss final: {results['final_loss']:.4f}\n")
                f.write(f"  Satisfação final: {results['final_satisfaction']:.4f}\n")
                
                total_samples += results['samples']
                total_time += results['training_time']
        
        f.write(f"\nRESUMO GERAL:\n")
        f.write(f"Tamanho do tabuleiro: {board_size}x{board_size}\n")
        f.write(f"Total de amostras: {total_samples}\n")
        f.write(f"Tempo total: {total_time:.2f}s\n")
        f.write(f"Modelo salvo em: {model_path}\n")
    
    print(f"📄 Relatório salvo em: {report_path}")
    return model_path, training_results

def test_model_for_dimension(board_size: int, model_path: str, data_dir: str):
    """
    Testa um modelo para uma dimensão específica
    """
    print(f"\n🧪 TESTANDO MODELO {board_size}x{board_size}")
    print("=" * 60)
    
    # Inicializar solver
    solver = SudokuLTNSolver(board_size=board_size)
    
    # Carregar o modelo
    print(f"📂 Carregando modelo: {model_path}")
    if os.path.exists(model_path):
        solver.load_model(model_path)
    else:
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    # Definir arquivos de teste baseados na dimensão
    if board_size == 4:
        data_files = [
            ("data/4x4/sudoku_4x4_open_solvable.csv", "4x4 Aberto Solucionável"),
            ("data/4x4/sudoku_4x4_open_unsolvable.csv", "4x4 Aberto Impossível"),
            ("data/4x4/sudoku_4x4_closed_valid.csv", "4x4 Fechado Válido"),
            ("data/4x4/sudoku_4x4_closed_invalid.csv", "4x4 Fechado Inválido")
        ]
    else:
        data_files = [
            ("data/9x9/sudoku_open_solvable.csv", "9x9 Aberto Solucionável"),
            ("data/9x9/sudoku_open_unsolvable.csv", "9x9 Aberto Impossível"),
            ("data/9x9/sudoku_closed_valid.csv", "9x9 Fechado Válido"),
            ("data/9x9/sudoku_closed_invalid.csv", "9x9 Fechado Inválido")
        ]
        
        # Verificar se existe o formato antigo (arquivos na raiz)
        if not os.path.exists(data_files[0][0]):
            data_files = [
                ("data/sudoku_open_solvable.csv", "9x9 Aberto Solucionável"),
                ("data/sudoku_open_unsolvable.csv", "9x9 Aberto Impossível"),
                ("data/sudoku_closed_valid.csv", "9x9 Fechado Válido"),
                ("data/sudoku_closed_invalid.csv", "9x9 Fechado Inválido")
            ]
    
    for file_path, category in data_files:
        if os.path.exists(file_path):
            print(f"\n🔍 Testando {category}...")
            
            # Carrega apenas alguns exemplos para teste
            with open(file_path, 'r') as f:
                lines = f.readlines()[:3]  # Apenas 3 exemplos
            
            for i, line in enumerate(lines):
                sudoku_str = line.strip()
                expected_length = board_size * board_size
                if len(sudoku_str) == expected_length:
                    try:
                        board = sudoku_string_to_board(sudoku_str)
                        
                        print(f"  Teste {i+1} ({category}):")
                        info = board.get_board_info()
                        print(f"    Tipo: {info['tipo']}")
                        print(f"    Válido: {info['valido']}")
                        print(f"    Posições abertas: {len(info['posicoes_abertas'])}")
                        
                        # Tentar resolver
                        resultado = solver.solve_sudoku(board)
                        print(f"    Resultado: {resultado['sucesso']} - {resultado['motivo']}")
                        
                    except Exception as e:
                        print(f"    Erro no teste {i+1}: {e}")

def main():
    """
    Função principal do sistema
    """
    parser = argparse.ArgumentParser(description='Sistema LTN para Resolução de Sudoku - Versão Separada')
    parser.add_argument('--path', type=str, help='Caminho para o arquivo CSV com o tabuleiro')
    parser.add_argument('--train-4x4', action='store_true', help='Treinar modelo para 4x4')
    parser.add_argument('--train-9x9', action='store_true', help='Treinar modelo para 9x9')
    parser.add_argument('--test-4x4', action='store_true', help='Testar modelo 4x4')
    parser.add_argument('--test-9x9', action='store_true', help='Testar modelo 9x9')
    parser.add_argument('--solve', action='store_true', help='Resolver um Sudoku')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas de treinamento')
    parser.add_argument('--data-dir', type=str, default='data', help='Diretório dos dados')
    parser.add_argument('--board-size', type=int, choices=[4, 9], help='Tamanho do tabuleiro (4 ou 9)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SISTEMA LTN PARA RESOLUÇÃO DE SUDOKU - VERSÃO SEPARADA POR DIMENSÃO")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 80)
    
    # Se foi fornecido um arquivo CSV, processar o tabuleiro
    if args.path:
        print(f"\n📁 PROCESSANDO ARQUIVO: {args.path}")
        
        # Carregar o sudoku do arquivo
        board = load_sudoku_from_csv(args.path)
        if board is None:
            print("❌ Falha ao carregar o sudoku. Encerrando...")
            return
        
        # Detectar tamanho do tabuleiro
        board_size = board.size
        print(f"🎲 Tamanho do tabuleiro detectado: {board_size}x{board_size}")
        
        # Inicializar solver com o tamanho correto
        solver = SudokuLTNSolver(board_size=board_size)
        
        # Carregar modelo se existir
        model_path = f"models/sudoku_ltn_{board_size}x{board_size}.pth"
        if os.path.exists(model_path):
            solver.load_model(model_path)
            print(f"✅ Modelo carregado: {model_path}")
        else:
            print(f"⚠️  Modelo não encontrado: {model_path}")
            print("Executando sem modelo pré-treinado...")
        
        # Verificar se o tabuleiro está aberto ou fechado
        if board.is_closed():
            print(f"\n🔒 TABULEIRO FECHADO DETECTADO")
            print("Executando Questão 1: Classificação de tabuleiro fechado")
            
            result = classify_closed_board(board)
            
        elif board.is_open():
            print(f"\n🔓 TABULEIRO ABERTO DETECTADO")
            print("Escolha a operação:")
            print("  2 - Questão 2: Classificar tabuleiro aberto e sugerir jogadas")
            print("  3 - Questão 3: Verificar se é solucionável")
            
            while True:
                try:
                    choice = input("\nDigite sua escolha (2 ou 3): ").strip()
                    if choice == "2":
                        print("\nExecutando Questão 2: Classificação de tabuleiro aberto")
                        result = solve_open_board(board, solver)
                        break
                    elif choice == "3":
                        print("\nExecutando Questão 3: Verificação de solucionabilidade")
                        result = check_solvability(board, solver)
                        break
                    else:
                        print("❌ Opção inválida. Digite 2 ou 3.")
                except KeyboardInterrupt:
                    print("\n\n❌ Operação cancelada pelo usuário.")
                    return
                except Exception as e:
                    print(f"❌ Erro na entrada: {e}")
        
        else:
            print("❌ Erro: Não foi possível determinar se o tabuleiro está aberto ou fechado.")
            return
        
        print(f"\n🎉 PROCESSAMENTO CONCLUÍDO!")
        return
    
    # Treinar modelo 4x4
    if args.train_4x4:
        print(f"\n�� TREINANDO MODELO 4x4")
        try:
            model_path, training_results = train_model_for_dimension(4, args.data_dir, args.epochs)
            print(f"\n🎉 TREINAMENTO 4x4 CONCLUÍDO!")
            print(f"Modelo salvo em: {model_path}")
        except Exception as e:
            print(f"❌ Erro durante o treinamento 4x4: {e}")
            import traceback
            traceback.print_exc()
    
    # Treinar modelo 9x9
    if args.train_9x9:
        print(f"\n🎓 TREINANDO MODELO 9x9")
        try:
            model_path, training_results = train_model_for_dimension(9, args.data_dir, args.epochs)
            print(f"\n🎉 TREINAMENTO 9x9 CONCLUÍDO!")
            print(f"Modelo salvo em: {model_path}")
        except Exception as e:
            print(f"❌ Erro durante o treinamento 9x9: {e}")
            import traceback
            traceback.print_exc()
    
    # Testar modelo 4x4
    if args.test_4x4:
        model_path = "models/sudoku_ltn_4x4.pth"
        test_model_for_dimension(4, model_path, args.data_dir)
    
    # Testar modelo 9x9
    if args.test_9x9:
        model_path = "models/sudoku_ltn_9x9.pth"
        test_model_for_dimension(9, model_path, args.data_dir)
    
    # Modo resolução
    if args.solve:
        # Detectar tamanho do tabuleiro
        if args.board_size:
            board_size = args.board_size
        else:
            print("❌ Para --solve, especifique --board-size 4 ou 9")
            return
        
        # Inicializar solver
        solver = SudokuLTNSolver(board_size=board_size)
        
        print(f"\n🧩 MODO RESOLUÇÃO {board_size}x{board_size}")
        
        # Carregar modelo se existir
        model_path = f"models/sudoku_ltn_{board_size}x{board_size}.pth"
        if os.path.exists(model_path):
            solver.load_model(model_path)
            print(f"✅ Modelo carregado: {model_path}")
        else:
            print(f"⚠️  Modelo não encontrado: {model_path}")
            print("Executando sem modelo pré-treinado...")
        
        # Criar exemplo baseado no tamanho
        if board_size == 4:
            board = create_sample_4x4_sudoku()
        else:
            board = create_sample_sudoku()
        
        print("\n📋 Sudoku para resolver:")
        print(board)
        
        # Tentar resolver
        resultado = solver.solve_sudoku(board)
        print(f"\n🎯 Resultado: {resultado['sucesso']}")
        print(f"📝 Motivo: {resultado['motivo']}")
        
        if resultado['sucesso']:
            print("\n📋 Sudoku resolvido:")
            print(board)
    
    # Se nenhum argumento foi fornecido, mostrar ajuda
    if not any([args.path, args.train_4x4, args.train_9x9, args.test_4x4, args.test_9x9, args.solve]):
        print("\n📖 USO:")
        print("  python main.py --path arquivo.csv")
        print("  python main.py --train-4x4 --epochs 30")
        print("  python main.py --train-9x9 --epochs 30")
        print("  python main.py --test-4x4")
        print("  python main.py --test-9x9")
        print("  python main.py --solve --board-size 4")
        print("  python main.py --solve --board-size 9")
        print("\n📋 EXEMPLOS:")
        print("  # Processar um tabuleiro do arquivo CSV")
        print("  python main.py --path data/meu_sudoku.csv")
        print("\n  # Treinar modelo 4x4")
        print("  python main.py --train-4x4 --epochs 50")
        print("\n  # Treinar modelo 9x9")
        print("  python main.py --train-9x9 --epochs 50")
        print("\n  # Testar modelo 4x4")
        print("  python main.py --test-4x4")
        print("\n  # Resolver sudoku 4x4")
        print("  python main.py --solve --board-size 4")

if __name__ == "__main__":
    main()
