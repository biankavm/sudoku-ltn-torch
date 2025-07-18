#!/usr/bin/env python3
"""
Script para gerar sudokus abertos (com zeros) que são impossíveis de resolver.
Cria situações onde não há jogadas válidas disponíveis.
"""

import random
import csv
from typing import List, Set, Tuple, Optional

def sudoku_string_to_grid(sudoku_str: str) -> List[List[int]]:
    """Converte string de sudoku em grid 9x9"""
    return [[int(sudoku_str[i*9 + j]) for j in range(9)] for i in range(9)]

def grid_to_sudoku_string(grid: List[List[int]]) -> str:
    """Converte grid 9x9 em string de sudoku"""
    return ''.join(str(grid[i][j]) for i in range(9) for j in range(9))

def get_possible_values(grid: List[List[int]], row: int, col: int) -> Set[int]:
    """Retorna os valores possíveis para uma posição específica"""
    if grid[row][col] != 0:
        return set()
    
    used_values = set()
    
    # Valores usados na linha
    for c in range(9):
        if grid[row][c] != 0:
            used_values.add(grid[row][c])
    
    # Valores usados na coluna
    for r in range(9):
        if grid[r][col] != 0:
            used_values.add(grid[r][col])
    
    # Valores usados no quadrante 3x3
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if grid[r][c] != 0:
                used_values.add(grid[r][c])
    
    return set(range(1, 10)) - used_values

def is_valid_placement(grid: List[List[int]], row: int, col: int, num: int) -> bool:
    """Verifica se é válido colocar um número em uma posição"""
    return num in get_possible_values(grid, row, col)

def has_solution(grid: List[List[int]]) -> bool:
    """Verifica se o sudoku tem solução usando backtracking simples"""
    def solve(g):
        for i in range(9):
            for j in range(9):
                if g[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid_placement(g, i, j, num):
                            g[i][j] = num
                            if solve(g):
                                return True
                            g[i][j] = 0
                    return False
        return True
    
    # Cria uma cópia para não modificar o original
    grid_copy = [row[:] for row in grid]
    return solve(grid_copy)

def create_impossible_situation_type1(grid: List[List[int]]) -> List[List[int]]:
    """
    Tipo 1: Número precisa ser colocado mas todas as posições disponíveis 
    criam conflitos na linha ou coluna
    """
    new_grid = [row[:] for row in grid]
    
    # Escolhe um quadrante aleatório
    box_row = random.randint(0, 2)
    box_col = random.randint(0, 2)
    
    # Encontra números que ainda não estão no quadrante
    used_in_box = set()
    empty_positions = []
    
    for r in range(box_row * 3, (box_row + 1) * 3):
        for c in range(box_col * 3, (box_col + 1) * 3):
            if new_grid[r][c] != 0:
                used_in_box.add(new_grid[r][c])
            else:
                empty_positions.append((r, c))
    
    missing_numbers = set(range(1, 10)) - used_in_box
    
    if len(missing_numbers) > 0 and len(empty_positions) > 1:
        target_number = random.choice(list(missing_numbers))
        
        # Coloca o número alvo em todas as linhas e colunas das posições vazias
        # exceto em uma posição específica, criando um conflito
        for r, c in empty_positions:
            # Coloca o número alvo na linha (em outra coluna)
            for other_c in range(9):
                if other_c != c and new_grid[r][other_c] == 0:
                    new_grid[r][other_c] = target_number
                    break
            
            # Coloca o número alvo na coluna (em outra linha)
            for other_r in range(9):
                if other_r != r and new_grid[other_r][c] == 0:
                    new_grid[other_r][c] = target_number
                    break
    
    return new_grid

def create_impossible_situation_type2(grid: List[List[int]]) -> List[List[int]]:
    """
    Tipo 2: Cria uma situação onde duas células precisam dos mesmos números
    mas só há uma possibilidade para cada
    """
    new_grid = [row[:] for row in grid]
    
    # Encontra duas posições vazias na mesma linha
    for row in range(9):
        empty_cols = [col for col in range(9) if new_grid[row][col] == 0]
        if len(empty_cols) >= 2:
            col1, col2 = random.sample(empty_cols, 2)
            
            # Preenche outras posições para forçar que apenas um número seja possível
            available_numbers = list(range(1, 10))
            
            # Remove números já usados na linha
            for c in range(9):
                if new_grid[row][c] != 0 and new_grid[row][c] in available_numbers:
                    available_numbers.remove(new_grid[row][c])
            
            if len(available_numbers) >= 2:
                # Escolhe dois números que serão impossíveis de colocar
                num1, num2 = random.sample(available_numbers, 2)
                
                # Coloca num1 na coluna de col1 (em outra linha)
                for other_row in range(9):
                    if other_row != row and new_grid[other_row][col1] == 0:
                        new_grid[other_row][col1] = num1
                        break
                
                # Coloca num2 na coluna de col2 (em outra linha)
                for other_row in range(9):
                    if other_row != row and new_grid[other_row][col2] == 0:
                        new_grid[other_row][col2] = num2
                        break
                
                # Preenche o resto da linha deixando apenas essas duas posições
                remaining_numbers = [n for n in available_numbers if n not in [num1, num2]]
                fill_positions = [c for c in empty_cols if c not in [col1, col2]]
                
                for i, pos in enumerate(fill_positions[:len(remaining_numbers)]):
                    new_grid[row][pos] = remaining_numbers[i]
            
            break
    
    return new_grid

def create_impossible_situation_type3(grid: List[List[int]]) -> List[List[int]]:
    """
    Tipo 3: Força uma situação onde um quadrante não pode ser completado
    """
    new_grid = [row[:] for row in grid]
    
    # Escolhe um quadrante
    box_row = random.randint(0, 2)
    box_col = random.randint(0, 2)
    
    # Encontra posições vazias no quadrante
    empty_positions = []
    used_numbers = set()
    
    for r in range(box_row * 3, (box_row + 1) * 3):
        for c in range(box_col * 3, (box_col + 1) * 3):
            if new_grid[r][c] == 0:
                empty_positions.append((r, c))
            else:
                used_numbers.add(new_grid[r][c])
    
    missing_numbers = list(set(range(1, 10)) - used_numbers)
    
    if len(empty_positions) > 0 and len(missing_numbers) > 0:
        # Escolhe um número que será impossível de colocar
        target_number = random.choice(missing_numbers)
        
        # Coloca esse número em todas as linhas e colunas das posições vazias
        for r, c in empty_positions:
            # Tenta colocar na linha
            for other_c in range(9):
                if (other_c < box_col * 3 or other_c >= (box_col + 1) * 3) and new_grid[r][other_c] == 0:
                    new_grid[r][other_c] = target_number
                    break
            
            # Tenta colocar na coluna
            for other_r in range(9):
                if (other_r < box_row * 3 or other_r >= (box_row + 1) * 3) and new_grid[other_r][c] == 0:
                    new_grid[other_r][c] = target_number
                    break
    
    return new_grid

def remove_cells_strategically(grid: List[List[int]], target_empty: int = 40) -> List[List[int]]:
    """Remove células estrategicamente para manter a impossibilidade"""
    new_grid = [row[:] for row in grid]
    
    # Lista de posições preenchidas
    filled_positions = []
    for r in range(9):
        for c in range(9):
            if new_grid[r][c] != 0:
                filled_positions.append((r, c))
    
    # Remove células aleatoriamente, mas mantém algumas estratégicas
    random.shuffle(filled_positions)
    
    cells_to_remove = min(target_empty, len(filled_positions) - 20)  # Mantém pelo menos 20 células
    
    for i in range(cells_to_remove):
        r, c = filled_positions[i]
        new_grid[r][c] = 0
    
    return new_grid

def generate_unsolvable_sudokus(input_file: str, output_file: str, num_unsolvable: int = 5000):
    """
    Gera sudokus abertos impossíveis de resolver
    
    Args:
        input_file: Arquivo CSV com sudokus válidos
        output_file: Arquivo CSV para salvar sudokus impossíveis
        num_unsolvable: Número de sudokus impossíveis a gerar
    """
    
    # Tipos de situações impossíveis
    impossible_types = [
        ("conflict_placement", create_impossible_situation_type1),
        ("mutual_exclusion", create_impossible_situation_type2),
        ("box_blocking", create_impossible_situation_type3)
    ]
    
    # Lê sudokus válidos
    valid_sudokus = []
    print(f"Lendo sudokus válidos de {input_file}...")
    
    try:
        with open(input_file, 'r') as f:
            for line in f:
                sudoku_str = line.strip()
                if len(sudoku_str) == 81 and sudoku_str.replace('0', '').isdigit():
                    valid_sudokus.append(sudoku_str)
    except FileNotFoundError:
        print(f"Arquivo {input_file} não encontrado!")
        return
    
    print(f"Carregados {len(valid_sudokus)} sudokus válidos")
    
    # Gera sudokus impossíveis
    unsolvable_sudokus = []
    print(f"Gerando {num_unsolvable} sudokus impossíveis...")
    
    attempts = 0
    max_attempts = num_unsolvable * 10  # Máximo de tentativas
    
    while len(unsolvable_sudokus) < num_unsolvable and attempts < max_attempts:
        attempts += 1
        
        if attempts % 100 == 0:
            print(f"Tentativas: {attempts}, Gerados: {len(unsolvable_sudokus)}")
        
        # Escolhe um sudoku válido aleatório
        valid_sudoku = random.choice(valid_sudokus)
        grid = sudoku_string_to_grid(valid_sudoku)
        
        # Escolhe um tipo de situação impossível
        situation_name, situation_func = random.choice(impossible_types)
        
        try:
            # Cria situação impossível
            impossible_grid = situation_func(grid)
            
            # Remove células estrategicamente
            open_grid = remove_cells_strategically(impossible_grid, random.randint(30, 50))
            
            # Verifica se realmente é impossível
            if not has_solution(open_grid):
                # Verifica se tem pelo menos algumas células vazias
                empty_count = sum(1 for r in range(9) for c in range(9) if open_grid[r][c] == 0)
                if empty_count >= 20:  # Pelo menos 20 células vazias
                    unsolvable_sudoku = grid_to_sudoku_string(open_grid)
                    unsolvable_sudokus.append(unsolvable_sudoku)
        
        except Exception as e:
            # Em caso de erro, tenta uma abordagem mais simples
            if attempts % 50 == 0:
                print(f"Erro na tentativa {attempts}: {e}")
            
            # Abordagem simples: remove muitas células e introduz conflito
            simple_grid = [row[:] for row in grid]
            
            # Remove muitas células
            for r in range(9):
                for c in range(9):
                    if random.random() < 0.6:  # 60% de chance de remover
                        simple_grid[r][c] = 0
            
            # Introduz um conflito simples
            if simple_grid[0][0] == 0 and simple_grid[0][1] == 0:
                simple_grid[0][0] = 5
                simple_grid[0][1] = 5  # Conflito na linha
                
                unsolvable_sudoku = grid_to_sudoku_string(simple_grid)
                unsolvable_sudokus.append(unsolvable_sudoku)
    
    # Salva sudokus impossíveis
    print(f"Salvando {len(unsolvable_sudokus)} sudokus impossíveis em {output_file}...")
    
    with open(output_file, 'w') as f:
        for unsolvable_sudoku in unsolvable_sudokus:
            f.write(unsolvable_sudoku + '\n')
    
    print(f"Concluído! {len(unsolvable_sudokus)} sudokus impossíveis salvos em {output_file}")

def verify_unsolvable_sudoku(sudoku_str: str) -> dict:
    """Verifica se um sudoku é realmente impossível de resolver"""
    if len(sudoku_str) != 81:
        return {"valid": False, "reason": "invalid_length"}
    
    try:
        grid = sudoku_string_to_grid(sudoku_str)
    except:
        return {"valid": False, "reason": "invalid_format"}
    
    # Conta células vazias
    empty_count = sum(1 for r in range(9) for c in range(9) if grid[r][c] == 0)
    
    # Verifica se tem solução
    solvable = has_solution(grid)
    
    return {
        "valid": True,
        "empty_cells": empty_count,
        "solvable": solvable,
        "is_unsolvable": not solvable and empty_count > 0
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera sudokus abertos impossíveis de resolver")
    parser.add_argument("--input", default="sudoku-ltn-torch/data/9x9/sudoku_closed_valid.csv", 
                       help="Arquivo CSV com sudokus válidos")
    parser.add_argument("--output", default="sudoku-ltn-torch/data/9x9/sudoku_open_unsolvable.csv", 
                       help="Arquivo CSV para sudokus impossíveis")
    parser.add_argument("--count", type=int, default=5000, 
                       help="Número de sudokus impossíveis a gerar")
    parser.add_argument("--verify", action="store_true", 
                       help="Verifica alguns sudokus gerados")
    
    args = parser.parse_args()
    
    # Gera sudokus impossíveis
    generate_unsolvable_sudokus(args.input, args.output, args.count)
    
    # Verifica alguns exemplos se solicitado
    if args.verify:
        print("\nVerificando alguns sudokus gerados...")
        try:
            with open(args.output, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:5]):
                    sudoku_str = line.strip()
                    result = verify_unsolvable_sudoku(sudoku_str)
                    print(f"Sudoku {i+1}: {result}")
        except FileNotFoundError:
            print("Arquivo de saída não encontrado para verificação.") 