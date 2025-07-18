#!/usr/bin/env python3
"""
Script para gerar sudokus 4x4 fechados com solução inválida.
Gera tabuleiros completos que violam as regras do Sudoku.
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

def generate_invalid_4x4_sudoku() -> List[List[int]]:
    """
    Gera um sudoku 4x4 inválido introduzindo conflitos
    """
    # Começar com um sudoku válido
    valid_grid = [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1]
    ]
    
    # Escolher um tipo de conflito aleatório
    conflict_type = random.randint(1, 3)
    
    if conflict_type == 1:
        # Conflito na linha: duplicar um número na mesma linha
        row = random.randint(0, 3)
        col1, col2 = random.sample(range(4), 2)
        number = random.randint(1, 4)
        valid_grid[row][col1] = number
        valid_grid[row][col2] = number
        
    elif conflict_type == 2:
        # Conflito na coluna: duplicar um número na mesma coluna
        col = random.randint(0, 3)
        row1, row2 = random.sample(range(4), 2)
        number = random.randint(1, 4)
        valid_grid[row1][col] = number
        valid_grid[row2][col] = number
        
    else:
        # Conflito na caixa 2x2: duplicar um número na mesma caixa
        box_row = random.randint(0, 1) * 2
        box_col = random.randint(0, 1) * 2
        number = random.randint(1, 4)
        
        # Escolher duas posições na mesma caixa
        positions = []
        for i in range(2):
            for j in range(2):
                positions.append((box_row + i, box_col + j))
        
        pos1, pos2 = random.sample(positions, 2)
        valid_grid[pos1[0]][pos1[1]] = number
        valid_grid[pos2[0]][pos2[1]] = number
    
    return valid_grid

def create_invalid_by_missing_numbers() -> List[List[int]]:
    """
    Cria um sudoku inválido faltando números (não tem todos os números de 1 a 4)
    """
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Preencher parcialmente, deixando alguns números faltando
    numbers_used = set()
    
    for row in range(4):
        for col in range(4):
            if random.random() < 0.8:  # 80% de chance de preencher
                available = [n for n in range(1, 5) if n not in numbers_used]
                if available:
                    number = random.choice(available)
                    grid[row][col] = number
                    numbers_used.add(number)
    
    # Garantir que pelo menos um número está faltando
    if len(numbers_used) == 4:
        # Remover um número aleatório
        number_to_remove = random.choice(list(numbers_used))
        for row in range(4):
            for col in range(4):
                if grid[row][col] == number_to_remove:
                    grid[row][col] = 0
                    break
    
    return grid

def create_invalid_by_wrong_numbers() -> List[List[int]]:
    """
    Cria um sudoku inválido com números fora do intervalo 1-4
    """
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Preencher com números válidos primeiro
    for row in range(4):
        for col in range(4):
            grid[row][col] = random.randint(1, 4)
    
    # Introduzir números inválidos (5-9)
    num_invalid = random.randint(1, 3)
    for _ in range(num_invalid):
        row = random.randint(0, 3)
        col = random.randint(0, 3)
        grid[row][col] = random.randint(5, 9)
    
    return grid

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

def generate_4x4_closed_invalid_sudokus(output_file: str, count: int = 1000):
    """
    Gera sudokus 4x4 fechados inválidos
    
    Args:
        output_file: Arquivo CSV para salvar os sudokus
        count: Número de sudokus a gerar
    """
    print(f"Gerando {count} sudokus 4x4 fechados inválidos...")
    
    sudokus = []
    
    # Tipos de geradores de sudokus inválidos
    generators = [
        generate_invalid_4x4_sudoku,
        create_invalid_by_missing_numbers,
        create_invalid_by_wrong_numbers
    ]
    
    for i in range(count):
        if i % 100 == 0:
            print(f"Progresso: {i}/{count}")
        
        try:
            # Escolher um gerador aleatório
            generator = random.choice(generators)
            grid = generator()
            
            # Verificar se é realmente inválido
            if not is_valid_4x4_sudoku(grid):
                sudoku_str = grid_to_string(grid)
                sudokus.append(sudoku_str)
            else:
                # Forçar invalidade se necessário
                grid[0][0] = grid[0][1]  # Conflito direto
                sudoku_str = grid_to_string(grid)
                sudokus.append(sudoku_str)
                
        except Exception as e:
            print(f"Erro ao gerar sudoku {i}: {e}")
            # Fallback: sudoku inválido padrão
            fallback_grid = [
                [1, 1, 3, 4],  # Conflito na primeira linha
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
    
    print(f"Concluído! {len(sudokus)} sudokus 4x4 fechados inválidos salvos em {output_file}")

def demo_example():
    """Demonstra exemplos de sudokus 4x4 inválidos"""
    print("Exemplos de sudokus 4x4 fechados inválidos:")
    print("="*50)
    
    print("\n1. Conflito na linha:")
    grid1 = generate_invalid_4x4_sudoku()
    print_grid(grid1)
    print(f"Válido: {is_valid_4x4_sudoku(grid1)}")
    
    print("\n2. Números faltando:")
    grid2 = create_invalid_by_missing_numbers()
    print_grid(grid2)
    print(f"Válido: {is_valid_4x4_sudoku(grid2)}")
    
    print("\n3. Números inválidos:")
    grid3 = create_invalid_by_wrong_numbers()
    print_grid(grid3)
    print(f"Válido: {is_valid_4x4_sudoku(grid3)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera sudokus 4x4 fechados inválidos")
    parser.add_argument("--output", default="../data/4x4/sudoku_4x4_closed_invalid.csv", 
                       help="Arquivo CSV para sudokus inválidos")
    parser.add_argument("--count", type=int, default=1000, 
                       help="Número de sudokus a gerar")
    parser.add_argument("--demo", action="store_true", 
                       help="Mostra exemplos de sudokus inválidos")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_example()
        print("\n" + "="*50 + "\n")
    
    # Gerar sudokus inválidos
    generate_4x4_closed_invalid_sudokus(args.output, args.count)
    
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