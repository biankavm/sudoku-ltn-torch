#!/usr/bin/env python3
"""
Script para gerar sudokus fechados inválidos a partir de sudokus válidos.
Introduz diferentes tipos de erros para criar exemplos de sudokus incorretos.
"""

import random
import csv
from typing import List, Set

def sudoku_string_to_grid(sudoku_str: str) -> List[List[int]]:
    """Converte string de sudoku em grid 9x9"""
    return [[int(sudoku_str[i*9 + j]) for j in range(9)] for i in range(9)]

def grid_to_sudoku_string(grid: List[List[int]]) -> str:
    """Converte grid 9x9 em string de sudoku"""
    return ''.join(str(grid[i][j]) for i in range(9) for j in range(9))

def introduce_row_duplicate(grid: List[List[int]]) -> List[List[int]]:
    """Introduz duplicata na mesma linha"""
    new_grid = [row[:] for row in grid]
    row_idx = random.randint(0, 8)
    
    # Encontra dois valores diferentes na linha
    available_positions = list(range(9))
    pos1, pos2 = random.sample(available_positions, 2)
    
    # Faz pos2 ter o mesmo valor que pos1
    new_grid[row_idx][pos2] = new_grid[row_idx][pos1]
    
    return new_grid

def introduce_column_duplicate(grid: List[List[int]]) -> List[List[int]]:
    """Introduz duplicata na mesma coluna"""
    new_grid = [row[:] for row in grid]
    col_idx = random.randint(0, 8)
    
    # Encontra duas posições diferentes na coluna
    available_positions = list(range(9))
    pos1, pos2 = random.sample(available_positions, 2)
    
    # Faz pos2 ter o mesmo valor que pos1
    new_grid[pos2][col_idx] = new_grid[pos1][col_idx]
    
    return new_grid

def introduce_box_duplicate(grid: List[List[int]]) -> List[List[int]]:
    """Introduz duplicata no mesmo quadrante 3x3"""
    new_grid = [row[:] for row in grid]
    
    # Escolhe um quadrante aleatório
    box_row = random.randint(0, 2)
    box_col = random.randint(0, 2)
    
    # Posições dentro do quadrante
    positions = [(box_row*3 + i, box_col*3 + j) for i in range(3) for j in range(3)]
    
    # Escolhe duas posições diferentes no quadrante
    pos1, pos2 = random.sample(positions, 2)
    
    # Faz pos2 ter o mesmo valor que pos1
    new_grid[pos2[0]][pos2[1]] = new_grid[pos1[0]][pos1[1]]
    
    return new_grid

def introduce_invalid_number(grid: List[List[int]]) -> List[List[int]]:
    """Introduz número inválido (0 ou > 9)"""
    new_grid = [row[:] for row in grid]
    
    row_idx = random.randint(0, 8)
    col_idx = random.randint(0, 8)
    
    # Escolhe um número inválido
    invalid_numbers = [0, 10, 11, 12]
    new_grid[row_idx][col_idx] = random.choice(invalid_numbers)
    
    return new_grid

def introduce_multiple_errors(grid: List[List[int]]) -> List[List[int]]:
    """Introduz múltiplos erros no mesmo sudoku"""
    new_grid = [row[:] for row in grid]
    
    # Aplica 2-3 erros diferentes
    num_errors = random.randint(2, 3)
    error_functions = [
        introduce_row_duplicate,
        introduce_column_duplicate,
        introduce_box_duplicate
    ]
    
    for _ in range(num_errors):
        error_func = random.choice(error_functions)
        new_grid = error_func(new_grid)
    
    return new_grid

def generate_invalid_sudokus(input_file: str, output_file: str, num_invalid: int = 10000):
    """
    Gera sudokus inválidos a partir de sudokus válidos
    
    Args:
        input_file: Arquivo CSV com sudokus válidos
        output_file: Arquivo CSV para salvar sudokus inválidos
        num_invalid: Número de sudokus inválidos a gerar
    """
    
    # Tipos de erros a introduzir
    error_types = [
        ("row_duplicate", introduce_row_duplicate),
        ("column_duplicate", introduce_column_duplicate),
        ("box_duplicate", introduce_box_duplicate),
        ("invalid_number", introduce_invalid_number),
        ("multiple_errors", introduce_multiple_errors)
    ]
    
    # Lê sudokus válidos
    valid_sudokus = []
    print(f"Lendo sudokus válidos de {input_file}...")
    
    with open(input_file, 'r') as f:
        for line in f:
            sudoku_str = line.strip()
            if len(sudoku_str) == 81 and sudoku_str.isdigit():
                valid_sudokus.append(sudoku_str)
    
    print(f"Carregados {len(valid_sudokus)} sudokus válidos")
    
    # Gera sudokus inválidos
    invalid_sudokus = []
    print(f"Gerando {num_invalid} sudokus inválidos...")
    
    for i in range(num_invalid):
        if i % 1000 == 0:
            print(f"Progresso: {i}/{num_invalid}")
        
        # Escolhe um sudoku válido aleatório
        valid_sudoku = random.choice(valid_sudokus)
        grid = sudoku_string_to_grid(valid_sudoku)
        
        # Escolhe um tipo de erro aleatório
        error_name, error_func = random.choice(error_types)
        
        # Aplica o erro
        try:
            invalid_grid = error_func(grid)
            invalid_sudoku = grid_to_sudoku_string(invalid_grid)
            invalid_sudokus.append(invalid_sudoku)
        except Exception as e:
            print(f"Erro ao gerar sudoku inválido {i}: {e}")
            # Se falhar, usa o sudoku original com um erro simples
            grid[0][0] = grid[0][1]  # Duplicata simples
            invalid_sudoku = grid_to_sudoku_string(grid)
            invalid_sudokus.append(invalid_sudoku)
    
    # Salva sudokus inválidos
    print(f"Salvando {len(invalid_sudokus)} sudokus inválidos em {output_file}...")
    
    with open(output_file, 'w') as f:
        for invalid_sudoku in invalid_sudokus:
            f.write(invalid_sudoku + '\n')
    
    print(f"Concluído! {len(invalid_sudokus)} sudokus inválidos salvos em {output_file}")

def verify_invalid_sudoku(sudoku_str: str) -> List[str]:
    """Verifica que tipos de erros um sudoku possui"""
    if len(sudoku_str) != 81:
        return ["invalid_length"]
    
    try:
        grid = sudoku_string_to_grid(sudoku_str)
    except:
        return ["invalid_format"]
    
    errors = []
    
    # Verifica números inválidos
    for i in range(9):
        for j in range(9):
            if grid[i][j] < 1 or grid[i][j] > 9:
                errors.append("invalid_number")
                break
        if "invalid_number" in errors:
            break
    
    # Verifica duplicatas em linhas
    for i in range(9):
        row = grid[i]
        if len(set(row)) != len(row):
            errors.append("row_duplicate")
            break
    
    # Verifica duplicatas em colunas
    for j in range(9):
        col = [grid[i][j] for i in range(9)]
        if len(set(col)) != len(col):
            errors.append("column_duplicate")
            break
    
    # Verifica duplicatas em quadrantes
    for box_row in range(3):
        for box_col in range(3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(grid[box_row*3 + i][box_col*3 + j])
            if len(set(box)) != len(box):
                errors.append("box_duplicate")
                break
        if "box_duplicate" in errors:
            break
    
    return errors if errors else ["unknown_error"]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera sudokus fechados inválidos")
    parser.add_argument("--input", default="../data/9x9/sudoku_closed_valid.csv", 
                       help="Arquivo CSV com sudokus válidos")
    parser.add_argument("--output", default="../data/9x9/sudoku_closed_invalid.csv", 
                       help="Arquivo CSV para sudokus inválidos")
    parser.add_argument("--count", type=int, default=10000, 
                       help="Número de sudokus inválidos a gerar")
    parser.add_argument("--verify", action="store_true", 
                       help="Verifica alguns sudokus gerados")
    
    args = parser.parse_args()
    
    # Gera sudokus inválidos
    generate_invalid_sudokus(args.input, args.output, args.count)
    
    # Verifica alguns exemplos se solicitado
    if args.verify:
        print("\nVerificando alguns sudokus gerados...")
        with open(args.output, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:5]):
                sudoku_str = line.strip()
                errors = verify_invalid_sudoku(sudoku_str)
                print(f"Sudoku {i+1}: {errors}") 