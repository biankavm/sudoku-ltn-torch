#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema LTN para Resolução de Sudoku - Versão Integrada
Projeto Final - Inteligência Artificial - UFAM
Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas

Este script treina modelos especializados para diferentes tipos de Sudoku
e cria um modelo integrado final.
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
    # board_data = [
    #     []
    # ]
    return SudokuBoard(board_data)

def create_unsolvable_4x4_sudoku():
    """
    Cria um tabuleiro Sudoku 4x4 SEM SOLUÇÃO para demonstração
    """
    # Tabuleiro 4x4 impossível de resolver
    # Problema: Duas células vazias na mesma linha/coluna/caixa
    # que não podem ser preenchidas sem violar as regras
    board_data = [
        [1, 2, 3, 0],  # Linha 1: falta o 4
        [3, 4, 0, 2],  # Linha 2: falta o 1
        [0, 1, 2, 3],  # Linha 3: falta o 4
        [2, 3, 1, 0]   # Linha 4: falta o 4
    ]
    
    # Explicação do problema:
    # - Célula (0,3) precisa ser 4
    # - Célula (2,0) precisa ser 4  
    # - Mas ambas estão na mesma coluna (coluna 0 e 3)
    # - E na mesma caixa 2x2 (canto superior direito)
    # - IMPOSSÍVEL: não pode ter dois 4s na mesma coluna/caixa
    
    return SudokuBoard(board_data)

def create_another_unsolvable_4x4_sudoku():
    """
    Outro exemplo de Sudoku 4x4 SEM SOLUÇÃO
    """
    # Tabuleiro 4x4 com contradição nas regras
    board_data = [
        [1, 2, 0, 0],  # Linha 1: faltam 3 e 4
        [2, 1, 0, 0],  # Linha 2: faltam 3 e 4
        [0, 0, 3, 4],  # Linha 3: faltam 1 e 2
        [0, 0, 4, 3]   # Linha 4: faltam 1 e 2
    ]
    
    # Explicação do problema:
    # - Caixa superior esquerda: precisa de 3 e 4
    # - Caixa superior direita: precisa de 3 e 4
    # - Mas 3 e 4 já estão nas caixas inferiores
    # - IMPOSSÍVEL: não pode ter 3 e 4 em duas caixas diferentes
    
    return SudokuBoard(board_data)

def sudoku_string_to_board(sudoku_str: str) -> SudokuBoard:
    """
    Converte string de 81 caracteres em SudokuBoard 9x9
    """
    if len(sudoku_str) != 81:
        raise ValueError("String deve ter exatamente 81 caracteres")
    
    # Converter string para matriz 9x9
    board_data = []
    for i in range(9):
        row = []
        for j in range(9):
            char = sudoku_str[i * 9 + j]
            # Converter '.' para 0 se necessário
            row.append(0 if char == '.' else int(char))
        board_data.append(row)
    
    return SudokuBoard(board_data)

def load_sudokus_from_csv(csv_path: str, max_samples: int = 2000) -> List[SudokuBoard]:
    """
    Carrega sudokus de um arquivo CSV
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
            if len(sudoku_str) == 81:
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

def train_specialized_models(solver: SudokuLTNSolver, data_dir: str = "data", epochs: int = 50):
    """
    Treina modelos especializados para cada tipo de situação
    """
    print("\n🎯 TREINAMENTO ESPECIALIZADO POR SITUAÇÃO")
    print("=" * 60)
    
    max_samples_global = 500
    # Definir os tipos de treinamento
    training_configs = [
        {
            "name": "Sudokus Fechados Válidos",
            "file": "sudoku_closed_valid.csv",
            "description": "Aprende a reconhecer sudokus completos e corretos",
            "max_samples": max_samples_global
        },
        {
            "name": "Sudokus Fechados Inválidos", 
            "file": "sudoku_closed_invalid.csv",
            "description": "Aprende a identificar sudokus completos mas incorretos",
            "max_samples": max_samples_global
        },
        {
            "name": "Sudokus Abertos Solucionáveis",
            "file": "sudoku_open_solvable.csv", 
            "description": "Aprende a resolver sudokus parciais com solução",
            "max_samples": max_samples_global
        },
        {
            "name": "Sudokus Abertos Impossíveis",
            "file": "sudoku_open_unsolvable.csv",
            "description": "Aprende a identificar sudokus impossíveis de resolver",
            "max_samples": max_samples_global
        }
    ]
    
    training_results = {}
    
    for i, config in enumerate(training_configs, 1):
        print(f"\n{i}️⃣ TREINANDO: {config['name']}")
        print(f"📝 {config['description']}")
        print("-" * 50)
        
        # Carregar dados
        csv_path = os.path.join(data_dir, config['file'])
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
    
    return training_results

def create_integrated_model(solver: SudokuLTNSolver, training_results: Dict):
    """
    Cria um modelo integrado baseado nos treinamentos especializados
    """
    print("\n🔗 CRIANDO MODELO INTEGRADO")
    print("=" * 60)
    
    # Salvar o modelo integrado
    model_path = "models/sudoku_ltn_integrated_9x9.pth"
    os.makedirs("models", exist_ok=True)
    
    print(f"💾 Salvando modelo integrado em: {model_path}")
    solver.save_model(model_path)
    
    # Criar relatório de treinamento
    report_path = "models/training_report.txt"
    with open(report_path, 'w') as f:
        f.write("RELATÓRIO DE TREINAMENTO - SISTEMA LTN SUDOKU\n")
        f.write("=" * 60 + "\n")
        f.write(f"Data: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas\n\n")
        
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
        f.write(f"Total de amostras: {total_samples}\n")
        f.write(f"Tempo total: {total_time:.2f}s\n")
        f.write(f"Modelo salvo em: {model_path}\n")
    
    print(f"📄 Relatório salvo em: {report_path}")
    return model_path

def test_integrated_model(solver: SudokuLTNSolver, model_path: str):
    """
    Testa o modelo integrado com diferentes tipos de sudoku
    """
    print("\n🧪 TESTANDO MODELO INTEGRADO")
    print("=" * 60)
    
    # Carregar o modelo
    print(f"📂 Carregando modelo: {model_path}")
    solver.load_model(model_path)
    
    # Testar com alguns sudokus dos arquivos
    data_files = [
        ("data/sudoku_open_solvable.csv", "Aberto Solucionável"),
        ("data/sudoku_open_unsolvable.csv", "Aberto Impossível"),
        ("data/sudoku_closed_valid.csv", "Fechado Válido"),
        ("data/sudoku_closed_invalid.csv", "Fechado Inválido")
    ]
    
    for file_path, category in data_files:
        if os.path.exists(file_path):
            print(f"\n🔍 Testando {category}...")
            
            # Carrega apenas alguns exemplos para teste
            with open(file_path, 'r') as f:
                lines = f.readlines()[:3]  # Apenas 3 exemplos
            
            for i, line in enumerate(lines):
                sudoku_str = line.strip()
                if len(sudoku_str) == 81:
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
    parser = argparse.ArgumentParser(description='Sistema LTN para Resolução de Sudoku')
    parser.add_argument('--train', action='store_true', help='Treinar modelos especializados')
    parser.add_argument('--solve', action='store_true', help='Resolver um Sudoku')
    parser.add_argument('--test', action='store_true', help='Testar modelo integrado')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas de treinamento')
    parser.add_argument('--demo', action='store_true', help='Executar demonstração completa')
    parser.add_argument('--data-dir', type=str, default='data', help='Diretório dos dados')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SISTEMA LTN PARA RESOLUÇÃO DE SUDOKU - VERSÃO INTEGRADA")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 80)
    
    # Inicializar solver
    solver = SudokuLTNSolver(board_size=9)
    
    if args.demo:
        print("\n🎯 EXECUTANDO DEMONSTRAÇÃO COMPLETA")
        demo_complete_system(solver, args.data_dir, args.epochs)
        return
    
    if args.train:
        print(f"\n🎓 MODO TREINAMENTO ESPECIALIZADO")
        print(f"Épocas por situação: {args.epochs}")
        print(f"Diretório de dados: {args.data_dir}")
        
        # Treinar modelos especializados
        training_results = train_specialized_models(solver, args.data_dir, args.epochs)
        
        # Criar modelo integrado
        model_path = create_integrated_model(solver, training_results)
        
        print(f"\n🎉 TREINAMENTO CONCLUÍDO!")
        print(f"Modelo integrado salvo em: {model_path}")
        
    if args.test:
        print(f"\n🧪 MODO TESTE")
        
        # Usar modelo integrado
        model_path = "models/sudoku_ltn_integrated_9x9.pth"
        if os.path.exists(model_path):
            test_integrated_model(solver, model_path)
        else:
            print(f"❌ Modelo não encontrado: {model_path}")
            print("Execute primeiro o treinamento com --train")
    
    if args.solve:
        print(f"\n🧩 MODO RESOLUÇÃO")
        
        # Tentar carregar modelo integrado
        model_path = "models/sudoku_ltn_integrated_9x9.pth"
        if os.path.exists(model_path):
            print(f"Carregando modelo integrado: {model_path}")
            solver.load_model(model_path)
        else:
            print("⚠️  Nenhum modelo integrado encontrado. Usando modelo não treinado.")
        
        # Criar tabuleiro de exemplo (9x9, não 4x4)
        board = create_sample_sudoku()
        
        print("\n📋 TABULEIRO INICIAL:")
        print(board)
        
        # Análise inicial
        info = board.get_board_info()
        print(f"\n📊 ANÁLISE INICIAL:")
        print(f"Tipo: {info['tipo']}")
        print(f"Válido: {info['valido']}")
        print(f"Posições abertas: {len(info['posicoes_abertas'])}")
        
        if info['conflitos']:
            print(f"Conflitos encontrados: {len(info['conflitos'])}")
            for conflito in info['conflitos']:
                print(f"  - Número {conflito['numero']} em {conflito['local']}")
        
        # Resolver
        print("\n🔍 INICIANDO RESOLUÇÃO...")
        resultado = solver.solve_sudoku(board)
        
        print(f"\n📊 RESULTADO:")
        print(f"Sucesso: {resultado['sucesso']}")
        print(f"Motivo: {resultado['motivo']}")
        print(f"Iterações: {resultado['iteracoes']}")
        
        if 'posicoes_restantes' in resultado:
            print(f"Posições restantes: {resultado['posicoes_restantes']}")
        
        print("\n📋 TABULEIRO FINAL:")
        print(resultado['board_final'])
        
        if 'memoria_usada' in resultado:
            print(f"\n🧠 MEMÓRIA UTILIZADA:")
            memoria = resultado['memoria_usada']
            print(f"Heurísticas usadas: {memoria['heuristics_count']}")
            print(f"Movimentos: {memoria['moves_count']}")

def demo_complete_system(solver: SudokuLTNSolver, data_dir: str, epochs: int):
    """
    Demonstração completa do sistema
    """
    print("\n1️⃣ TREINANDO MODELOS ESPECIALIZADOS...")
    training_results = train_specialized_models(solver, data_dir, epochs)
    
    print("\n2️⃣ CRIANDO MODELO INTEGRADO...")
    model_path = create_integrated_model(solver, training_results)
    
    print("\n3️⃣ TESTANDO MODELO INTEGRADO...")
    test_integrated_model(solver, model_path)
    
    print("\n4️⃣ TESTANDO RESOLUÇÃO...")
    board = create_sample_sudoku()
    
    print("\n📋 TABULEIRO DE TESTE:")
    print(board)
    
    resultado = solver.solve_sudoku(board)
    
    print(f"\n✅ RESULTADO DA RESOLUÇÃO:")
    print(f"Sucesso: {resultado['sucesso']}")
    print(f"Iterações: {resultado['iteracoes']}")
    
    print("\n📋 TABULEIRO FINAL:")
    print(resultado['board_final'])
    
    print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    print("O sistema LTN foi treinado e testado com sucesso!")

if __name__ == "__main__":
    main()
