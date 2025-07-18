#!/usr/bin/env python3
"""
Script para testar sudokus 4x4 sem solução
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import create_unsolvable_4x4_sudoku, create_another_unsolvable_4x4_sudoku
from core.sudoku_board import SudokuBoard

def test_4x4_unsolvable():
    """
    Testa sudokus 4x4 sem solução
    """
    print("🧩 TESTANDO SUDOKUS 4x4 SEM SOLUÇÃO")
    print("=" * 50)
    
    # Teste 1: Primeiro exemplo
    print("\n📋 EXEMPLO 1 - Sudoku 4x4 sem solução:")
    board1 = create_unsolvable_4x4_sudoku()
    
    print("Tabuleiro inicial:")
    print(board1)
    
    print(f"\nAnálise:")
    info = board1.get_board_info()
    print(f"  Tipo: {info['tipo']}")
    print(f"  Válido: {info['valido']}")
    print(f"  Posições abertas: {len(info['posicoes_abertas'])}")
    
    # Verificar candidatos para células vazias
    print(f"\nCandidatos para células vazias:")
    candidates_matrix = board1.get_candidates_matrix()
    for (row, col), candidates in candidates_matrix.items():
        print(f"  Célula ({row},{col}): candidatos = {candidates}")
    
    # Teste 2: Segundo exemplo
    print("\n" + "="*50)
    print("📋 EXEMPLO 2 - Outro sudoku 4x4 sem solução:")
    board2 = create_another_unsolvable_4x4_sudoku()
    
    print("Tabuleiro inicial:")
    print(board2)
    
    print(f"\nAnálise:")
    info = board2.get_board_info()
    print(f"  Tipo: {info['tipo']}")
    print(f"  Válido: {info['valido']}")
    print(f"  Posições abertas: {len(info['posicoes_abertas'])}")
    
    # Verificar candidatos para células vazias
    print(f"\nCandidatos para células vazias:")
    candidates_matrix = board2.get_candidates_matrix()
    for (row, col), candidates in candidates_matrix.items():
        print(f"  Célula ({row},{col}): candidatos = {candidates}")
    
    print("\n" + "="*50)
    print("✅ Testes concluídos!")

if __name__ == "__main__":
    test_4x4_unsolvable() 