#!/usr/bin/env python3
"""
Script para gerar sudokus 4x4 fechados com solução válida.
Gera tabuleiros completos que seguem todas as regras do Sudoku.
"""

import random
import csv
from typing import List, Set

def is_valid_4x4_sudoku(grid: List[List[int]]) -> bool:
    """
    Verifica se um sudoku 4x4 é válido
    """
    # Verificar linhas
    for row in grid:
        if len(set(row)) != 4:
            return False
    
    # Verificar colunas
    for col in range(4):
        column = [grid[row][col] for row in range(4)]
        if len(set(column)) != 4:
            return False
    
    # Verificar caixas 2x2
    for box_row in range(0, 4, 2):
        for box_col in range(0, 4, 2):
            box = []
            for i in range(2):
                for j in range(2):
                    box.append(grid[box_row + i][box_col + j])
            if len(set(box)) != 4:
                return False
    
    return True

def generate_valid_4x4_sudoku() -> List[List[int]]:
    """
    Gera um sudoku 4x4 válido usando backtracking com aleatorização
    """
    def solve(grid: List[List[int]]) -> bool:
        for row in range(4):
            for col in range(4):
                if grid[row][col] == 0:
                    # Tenta números de 1 a 4 em ordem aleatória
                    numbers = list(range(1, 5))
                    random.shuffle(numbers)
                    for num in numbers:
                        if is_valid_placement(grid, row, col, num):
                            grid[row][col] = num
                            if solve(grid):
                                return True
                            grid[row][col] = 0
                    return False
        return True
    
    def is_valid_placement(grid: List[List[int]], row: int, col: int, num: int) -> bool:
        # Verificar linha
        for c in range(4):
            if grid[row][c] == num:
                return False
        
        # Verificar coluna
        for r in range(4):
            if grid[r][col] == num:
                return False
        
        # Verificar caixa 2x2
        box_row, box_col = 2 * (row // 2), 2 * (col // 2)
        for r in range(box_row, box_row + 2):
            for c in range(box_col, box_col + 2):
                if grid[r][c] == num:
                    return False
        
        return True
    
    # Inicializar grid vazio
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Resolver o sudoku
    if solve(grid):
        return grid
    else:
        # Fallback: criar um sudoku válido manualmente
        return [
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ]

def generate_valid_4x4_sudoku_alternative() -> List[List[int]]:
    """
    Método alternativo: gera sudokus válidos usando permutações
    """
    # Padrões base de sudokus 4x4 válidos
    base_patterns = [
        # Padrão 1
        [
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ],
        # Padrão 2
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
            [2, 1, 4, 3],
            [3, 4, 1, 2]
        ],
        # Padrão 3
        [
            [1, 3, 2, 4],
            [2, 4, 1, 3],
            [3, 1, 4, 2],
            [4, 2, 3, 1]
        ],
        # Padrão 4
        [
            [1, 4, 2, 3],
            [2, 3, 1, 4],
            [3, 2, 4, 1],
            [4, 1, 3, 2]
        ]
    ]
    
    # Escolher um padrão base aleatório
    base = random.choice(base_patterns)
    
    # Aplicar transformações aleatórias
    transformations = random.randint(0, 3)
    
    for _ in range(transformations):
        transform_type = random.randint(1, 4)
        
        if transform_type == 1:
            # Trocar duas linhas
            row1, row2 = random.sample(range(4), 2)
            base[row1], base[row2] = base[row2], base[row1]
            
        elif transform_type == 2:
            # Trocar duas colunas
            col1, col2 = random.sample(range(4), 2)
            for row in range(4):
                base[row][col1], base[row][col2] = base[row][col2], base[row][col1]
                
        elif transform_type == 3:
            # Trocar dois números
            num1, num2 = random.sample(range(1, 5), 2)
            for row in range(4):
                for col in range(4):
                    if base[row][col] == num1:
                        base[row][col] = num2
                    elif base[row][col] == num2:
                        base[row][col] = num1
                        
        elif transform_type == 4:
            # Rotacionar 90 graus
            base = list(zip(*base[::-1]))
            base = [list(row) for row in base]
    
    return base

def grid_to_string(grid: List[List[int]]) -> str:
    """Converte grid 4x4 em string de 16 caracteres"""
    return ''.join(str(grid[i][j]) for i in range(4) for j in range(4))

def print_grid(grid: List[List[int]]):
    """Imprime o grid 4x4 de forma legível"""
    for i, row in enumerate(grid):
        if i == 2:
            print("--+--")
        
        row_str = ""
        for j, cell in enumerate(row):
            if j == 2:
                row_str += "| "
            row_str += f"{cell} "
        
        print(row_str)

def generate_4x4_closed_valid_sudokus(output_file: str, count: int = 1000):
    """
    Gera sudokus 4x4 fechados válidos
    
    Args:
        output_file: Arquivo CSV para salvar os sudokus
        count: Número de sudokus a gerar
    """
    print(f"Gerando {count} sudokus 4x4 fechados válidos...")
    
    sudokus = []
    
    for i in range(count):
        if i % 100 == 0:
            print(f"Progresso: {i}/{count}")
        
        try:
            # Escolher método de geração aleatoriamente
            if random.random() < 0.7:  # 70% das vezes usa backtracking
                grid = generate_valid_4x4_sudoku()
            else:  # 30% das vezes usa método alternativo
                grid = generate_valid_4x4_sudoku_alternative()
            
            # Verificar se é válido
            if is_valid_4x4_sudoku(grid):
                sudoku_str = grid_to_string(grid)
                sudokus.append(sudoku_str)
            else:
                print(f"Erro: Sudoku {i} não é válido")
                # Fallback
                fallback_grid = generate_valid_4x4_sudoku_alternative()
                sudoku_str = grid_to_string(fallback_grid)
                sudokus.append(sudoku_str)
                
        except Exception as e:
            print(f"Erro ao gerar sudoku {i}: {e}")
            # Fallback: sudoku válido padrão
            fallback_grid = [
                [1, 2, 3, 4],
                [3, 4, 1, 2],
                [2, 1, 4, 3],
                [4, 3, 2, 1]
            ]
            sudoku_str = grid_to_string(fallback_grid)
            sudokus.append(sudoku_str)
    
    # Salvar no arquivo CSV
    print(f"Salvando {len(sudokus)} sudokus em {output_file}...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for sudoku in sudokus:
            writer.writerow([sudoku])
    
    print(f"Concluído! {len(sudokus)} sudokus 4x4 fechados válidos salvos em {output_file}")

def demo_example():
    """Demonstra um exemplo de sudoku 4x4 válido"""
    print("Exemplo de sudoku 4x4 fechado válido:")
    print("="*40)
    
    grid = generate_valid_4x4_sudoku()
    print_grid(grid)
    
    print(f"\nVálido: {is_valid_4x4_sudoku(grid)}")
    print(f"String: {grid_to_string(grid)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera sudokus 4x4 fechados válidos")
    parser.add_argument("--output", default="../data/4x4/sudoku_4x4_closed_valid.csv", 
                       help="Arquivo CSV para sudokus válidos")
    parser.add_argument("--count", type=int, default=1000, 
                       help="Número de sudokus a gerar")
    parser.add_argument("--demo", action="store_true", 
                       help="Mostra exemplo de sudoku válido")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_example()
        print("\n" + "="*50 + "\n")
    
    # Gerar sudokus válidos
    generate_4x4_closed_valid_sudokus(args.output, args.count)
    
    # Mostrar alguns exemplos
    print("\nPrimeiros 3 sudokus gerados:")
    try:
        with open(args.output, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                sudoku_str = row[0]
                print(f"\nSudoku {i+1}:")
                grid = [[int(sudoku_str[r*4 + c]) for c in range(4)] for r in range(4)]
                print_grid(grid)
                print(f"Válido: {is_valid_4x4_sudoku(grid)}")
    except FileNotFoundError:
        print("Arquivo não encontrado para mostrar exemplos.") 