#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo de Uso dos Modelos LTN Separados
Projeto Final - Inteligência Artificial - UFAM
Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas

Este script demonstra como usar os modelos treinados para resolver
sudokus de diferentes dimensões.
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from core.sudoku_board import SudokuBoard
from solver.ltn_solver import SudokuLTNSolver

def exemplo_4x4():
    """
    Exemplo de uso do modelo 4x4
    """
    print("🎲 EXEMPLO 4x4")
    print("=" * 40)
    
    # Criar um sudoku 4x4 de exemplo
    sudoku_4x4 = [
        [1, 0, 3, 4],
        [3, 4, 0, 2],
        [0, 1, 4, 3],
        [4, 3, 2, 0]
    ]
    
    board = SudokuBoard(sudoku_4x4)
    print("Sudoku inicial:")
    print(board)
    
    # Inicializar solver 4x4
    solver = SudokuLTNSolver(board_size=4)
    
    # Carregar modelo se existir
    model_path = "models/sudoku_ltn_4x4.pth"
    if os.path.exists(model_path):
        solver.load_model(model_path)
        print(f"✅ Modelo 4x4 carregado")
    else:
        print(f"⚠️  Modelo 4x4 não encontrado, executando sem treinamento")
    
    # Resolver
    resultado = solver.solve_sudoku(board)
    
    print(f"\nResultado: {resultado['sucesso']}")
    print(f"Motivo: {resultado['motivo']}")
    print(f"Iterações: {resultado['iteracoes']}")
    
    if resultado['sucesso']:
        print("\nSudoku resolvido:")
        print(resultado['board_final'])

def exemplo_9x9():
    """
    Exemplo de uso do modelo 9x9
    """
    print("\n🎲 EXEMPLO 9x9")
    print("=" * 40)
    
    # Criar um sudoku 9x9 de exemplo
    sudoku_9x9 = [
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
    
    board = SudokuBoard(sudoku_9x9)
    print("Sudoku inicial:")
    print(board)
    
    # Inicializar solver 9x9
    solver = SudokuLTNSolver(board_size=9)
    
    # Carregar modelo se existir
    model_path = "models/sudoku_ltn_9x9.pth"
    if os.path.exists(model_path):
        solver.load_model(model_path)
        print(f"✅ Modelo 9x9 carregado")
    else:
        print(f"⚠️  Modelo 9x9 não encontrado, executando sem treinamento")
    
    # Resolver
    resultado = solver.solve_sudoku(board)
    
    print(f"\nResultado: {resultado['sucesso']}")
    print(f"Motivo: {resultado['motivo']}")
    print(f"Iterações: {resultado['iteracoes']}")
    
    if resultado['sucesso']:
        print("\nSudoku resolvido:")
        print(resultado['board_final'])

def exemplo_arquivo_csv():
    """
    Exemplo de uso com arquivo CSV
    """
    print("\n📁 EXEMPLO COM ARQUIVO CSV")
    print("=" * 40)
    
    # Verificar se existe o arquivo de exemplo
    exemplo_4x4_path = "exemplo_4x4_fechado_valido.csv"
    if os.path.exists(exemplo_4x4_path):
        print(f"Usando arquivo: {exemplo_4x4_path}")
        
        # Ler o arquivo
        with open(exemplo_4x4_path, 'r') as f:
            sudoku_str = f.read().strip()
        
        print(f"String do sudoku: {sudoku_str}")
        
        # Converter para SudokuBoard
        board = SudokuBoard([[int(sudoku_str[i*4 + j]) for j in range(4)] for i in range(4)])
        print("Sudoku do arquivo:")
        print(board)
        
        # Verificar se é válido
        if board.is_closed():
            print("✅ Tabuleiro fechado detectado")
            is_valid = board.is_valid()
            print(f"Válido: {is_valid}")
        else:
            print("🔓 Tabuleiro aberto detectado")
            
            # Tentar resolver
            solver = SudokuLTNSolver(board_size=4)
            model_path = "models/sudoku_ltn_4x4.pth"
            if os.path.exists(model_path):
                solver.load_model(model_path)
                print("✅ Modelo 4x4 carregado")
            
            resultado = solver.solve_sudoku(board)
            print(f"Resultado: {resultado['sucesso']}")
            print(f"Motivo: {resultado['motivo']}")
    else:
        print(f"Arquivo {exemplo_4x4_path} não encontrado")

def main():
    """
    Função principal
    """
    print("=" * 60)
    print("EXEMPLO DE USO - MODELOS LTN SEPARADOS")
    print("Projeto Final - Inteligência Artificial - UFAM")
    print("Equipe: Bianka Vasconcelos, Micael Viana, Vinicius Chagas")
    print("=" * 60)
    
    # Verificar se os modelos existem
    modelo_4x4_existe = os.path.exists("models/sudoku_ltn_4x4.pth")
    modelo_9x9_existe = os.path.exists("models/sudoku_ltn_9x9.pth")
    
    print(f"\n📁 Status dos modelos:")
    print(f"  Modelo 4x4: {'✅ Disponível' if modelo_4x4_existe else '❌ Não encontrado'}")
    print(f"  Modelo 9x9: {'✅ Disponível' if modelo_9x9_existe else '❌ Não encontrado'}")
    
    if not modelo_4x4_existe and not modelo_9x9_existe:
        print(f"\n⚠️  Nenhum modelo encontrado!")
        print(f"💡 Para treinar os modelos, execute:")
        print(f"  python train_all_models.py")
        print(f"  ou")
        print(f"  python train_4x4.py --epochs 30")
        print(f"  python train_9x9.py --epochs 30")
    
    # Executar exemplos
    exemplo_4x4()
    exemplo_9x9()
    exemplo_arquivo_csv()
    
    print(f"\n🎉 Exemplos concluídos!")
    print(f"\n💡 Para mais informações, consulte:")
    print(f"  python demo_separated_models.py")
    print(f"  python main.py --help")

if __name__ == "__main__":
    main() 