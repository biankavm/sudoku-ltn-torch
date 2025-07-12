#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema LTN para Resolução de Sudoku
Projeto Final - Inteligência Artificial - UFAM
Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas

Este script demonstra o uso do sistema LTN para resolver Sudoku,
incluindo treinamento dos predicates e resolução de tabuleiros.
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Adicionar o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from core.sudoku_board import SudokuBoard
from solver.ltn_solver import SudokuLTNSolver

def create_sample_sudoku():
    """
    Cria um tabuleiro Sudoku de exemplo 4x4 para demonstração
    """
    # Tabuleiro 4x4 parcialmente preenchido
    board_data = [
        [4, 0, 0, 2],
        [0, 3, 0, 1],
        [3, 0, 1, 0],
        [0, 1, 0, 3]
    ]
    return SudokuBoard(board_data)

def create_sample_csv():
    """
    Cria um arquivo CSV de exemplo com tabuleiros para treinamento
    """
    csv_path = "data/sample_training.csv"
    
    # Criar diretório se não existir
    os.makedirs("data", exist_ok=True)
    
    # Tabuleiros de exemplo 4x4
    boards = [
        # Tabuleiro 1 - parcialmente preenchido
        [1, 0, 0, 4, 0, 2, 0, 0, 0, 0, 1, 0, 4, 0, 0, 2],
        # Tabuleiro 2 - outro exemplo
        [0, 2, 3, 0, 4, 0, 0, 1, 1, 0, 0, 4, 0, 3, 2, 0],
        # Tabuleiro 3 - mais preenchido
        [1, 3, 4, 2, 2, 4, 1, 3, 3, 1, 2, 4, 4, 2, 3, 1],
        # Tabuleiro 4 - parcialmente preenchido
        [0, 0, 4, 2, 0, 4, 0, 3, 3, 0, 2, 0, 4, 2, 0, 0]
    ]
    
    # Salvar no CSV
    with open(csv_path, 'w') as f:
        for board in boards:
            f.write(','.join(map(str, board)) + '\n')
    
    print(f"Arquivo CSV de exemplo criado: {csv_path}")
    return csv_path

def main():
    """
    Função principal do sistema
    """
    parser = argparse.ArgumentParser(description='Sistema LTN para Resolução de Sudoku')
    parser.add_argument('--train', action='store_true', help='Treinar o modelo')
    parser.add_argument('--solve', action='store_true', help='Resolver um Sudoku')
    parser.add_argument('--csv', type=str, help='Caminho para arquivo CSV de treinamento')
    parser.add_argument('--board-size', type=int, default=4, help='Tamanho do tabuleiro (4 ou 9)')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de treinamento')
    parser.add_argument('--demo', action='store_true', help='Executar demonstração completa')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SISTEMA LTN PARA RESOLUÇÃO DE SUDOKU")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 60)
    
    # Inicializar solver
    solver = SudokuLTNSolver(board_size=args.board_size)
    
    if args.demo:
        print("\n🎯 EXECUTANDO DEMONSTRAÇÃO COMPLETA")
        demo_complete_system(solver)
        return
    
    if args.train:
        print(f"\n🎓 MODO TREINAMENTO (Tabuleiro {args.board_size}x{args.board_size})")
        
        # Usar CSV fornecido ou criar um de exemplo
        csv_path = args.csv
        if not csv_path:
            print("Nenhum CSV fornecido. Criando dados de exemplo...")
            csv_path = create_sample_csv()
        
        if not os.path.exists(csv_path):
            print(f"Erro: Arquivo CSV não encontrado: {csv_path}")
            return
        
        print(f"Treinando com dados de: {csv_path}")
        print(f"Épocas: {args.epochs}")
        
        # Treinar o modelo
        solver.train_from_csv(csv_path, epochs=args.epochs)
        
        # Salvar modelo treinado
        model_path = f"models/sudoku_ltn_{args.board_size}x{args.board_size}.pth"
        os.makedirs("models", exist_ok=True)
        solver.save_model(model_path)
        
        # Mostrar resumo do treinamento
        summary = solver.get_training_summary()
        print("\n📊 RESUMO DO TREINAMENTO:")
        print(f"Épocas treinadas: {summary['epochs_trained']}")
        print(f"Loss final: {summary['final_loss']:.4f}")
        print(f"Satisfação final: {summary['final_satisfaction']:.4f}")
    
    if args.solve:
        print(f"\n🧩 MODO RESOLUÇÃO (Tabuleiro {args.board_size}x{args.board_size})")
        
        # Tentar carregar modelo treinado
        model_path = f"models/sudoku_ltn_{args.board_size}x{args.board_size}.pth"
        if os.path.exists(model_path):
            print(f"Carregando modelo treinado: {model_path}")
            solver.load_model(model_path)
        else:
            print("⚠️  Nenhum modelo treinado encontrado. Usando modelo não treinado.")
        
        # Criar tabuleiro de exemplo
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

def demo_complete_system(solver):
    """
    Demonstração completa do sistema
    """
    print("\n1️⃣ CRIANDO DADOS DE TREINAMENTO...")
    csv_path = create_sample_csv()
    
    print("\n2️⃣ TREINANDO MODELO LTN...")
    solver.train_from_csv(csv_path, epochs=30)
    
    print("\n3️⃣ SALVANDO MODELO...")
    model_path = "models/demo_model.pth"
    os.makedirs("models", exist_ok=True)
    solver.save_model(model_path)
    
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
    
    # Mostrar estatísticas do treinamento
    summary = solver.get_training_summary()
    print(f"\n📊 ESTATÍSTICAS DO TREINAMENTO:")
    print(f"Loss final: {summary['final_loss']:.4f}")
    print(f"Satisfação final: {summary['final_satisfaction']:.4f}")
    
    print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    print("O sistema LTN foi treinado e testado com sucesso!")

if __name__ == "__main__":
    main()
