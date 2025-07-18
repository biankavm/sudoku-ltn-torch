#!/usr/bin/env python3
"""
Script para gerar sudokus 4x4 abertos com solução possível.
Gera tabuleiros parciais que podem ser resolvidos.
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
        numbers = [x for x in row if x != 0]
        if len(set(numbers)) != len(numbers):
            return False
    
    # Verificar colunas
    for col in range(4):
        column = [grid[row][col] for row in range(4) if grid[row][col] != 0]
        if len(set(column)) != len(column):
            return False
    
    # Verificar caixas 2x2
    for box_row in range(0, 4, 2):
        for box_col in range(0, 4, 2):
            box = []
            for i in range(2):
                for j in range(2):
                    if grid[box_row + i][box_col + j] != 0:
                        box.append(grid[box_row + i][box_col + j])
            if len(set(box)) != len(box):
                return False
    
    return True

def has_solution_4x4(grid: List[List[int]]) -> bool:
    """
    Verifica se um sudoku 4x4 tem solução usando backtracking
    """
    def solve(g):
        for i in range(4):
            for j in range(4):
                if g[i][j] == 0:
                    for num in range(1, 5):
                        if is_valid_placement(g, i, j, num):
                            g[i][j] = num
                            if solve(g):
                                return True
                            g[i][j] = 0
                    return False
        return True
    
    def is_valid_placement(g, row, col, num):
        # Verificar linha
        for c in range(4):
            if g[row][c] == num:
                return False
        
        # Verificar coluna
        for r in range(4):
            if g[r][col] == num:
                return False
        
        # Verificar caixa 2x2
        box_row, box_col = 2 * (row // 2), 2 * (col // 2)
        for r in range(box_row, box_row + 2):
            for c in range(box_col, box_col + 2):
                if g[r][c] == num:
                    return False
        
        return True
    
    # Cria uma cópia para não modificar o original
    grid_copy = [row[:] for row in grid]
    return solve(grid_copy)

def generate_valid_4x4_sudoku() -> List[List[int]]:
    """
    Gera um sudoku 4x4 válido usando backtracking
    """
    def solve(grid: List[List[int]]) -> bool:
        for row in range(4):
            for col in range(4):
                if grid[row][col] == 0:
                    # Tenta números de 1 a 4
                    for num in range(1, 5):
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

def remove_cells_randomly(grid: List[List[int]], num_to_remove: int) -> List[List[int]]:
    """
    Remove células aleatoriamente de um sudoku válido
    """
    new_grid = [row[:] for row in grid]
    
    # Lista de posições preenchidas
    filled_positions = []
    for r in range(4):
        for c in range(4):
            if new_grid[r][c] != 0:
                filled_positions.append((r, c))
    
    # Remove células aleatoriamente
    if len(filled_positions) > num_to_remove:
        positions_to_remove = random.sample(filled_positions, num_to_remove)
        for r, c in positions_to_remove:
            new_grid[r][c] = 0
    
    return new_grid

def remove_cells_strategically(grid: List[List[int]], num_to_remove: int) -> List[List[int]]:
    """
    Remove células estrategicamente para manter a solvabilidade
    """
    new_grid = [row[:] for row in grid]
    
    # Lista de posições preenchidas
    filled_positions = []
    for r in range(4):
        for c in range(4):
            if new_grid[r][c] != 0:
                filled_positions.append((r, c))
    
    # Remove células uma por uma, verificando se ainda tem solução
    removed_count = 0
    random.shuffle(filled_positions)
    
    for r, c in filled_positions:
        if removed_count >= num_to_remove:
            break
        
        # Salva o valor atual
        current_value = new_grid[r][c]
        new_grid[r][c] = 0
        
        # Verifica se ainda tem solução
        if has_solution_4x4(new_grid):
            removed_count += 1
        else:
            # Restaura o valor se não tem solução
            new_grid[r][c] = current_value
    
    return new_grid

def create_solvable_by_pattern() -> List[List[int]]:
    """
    Cria um sudoku solucionável usando padrões conhecidos
    """
    # Padrões de sudokus 4x4 solucionáveis
    patterns = [
        # Padrão 1: Canto superior esquerdo preenchido
        [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        # Padrão 2: Linha superior preenchida
        [
            [1, 2, 3, 4],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ],
        # Padrão 3: Diagonal principal
        [
            [1, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [0, 0, 0, 4]
        ],
        # Padrão 4: Quadrantes alternados
        [
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 1, 2],
            [0, 0, 3, 4]
        ]
    ]
    
    return random.choice(patterns)

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
            if cell == 0:
                row_str += ". "
            else:
                row_str += f"{cell} "
        
        print(row_str)

def generate_4x4_open_solvable_sudokus(output_file: str, count: int = 1000):
    """
    Gera sudokus 4x4 abertos com solução possível
    
    Args:
        output_file: Arquivo CSV para salvar os sudokus
        count: Número de sudokus a gerar
    """
    print(f"Gerando {count} sudokus 4x4 abertos com solução possível...")
    
    sudokus = []
    
    for i in range(count):
        if i % 100 == 0:
            print(f"Progresso: {i}/{count}")
        
        try:
            # Escolher método de geração
            method = random.randint(1, 3)
            
            if method == 1:
                # Método 1: Gerar sudoku válido e remover células aleatoriamente
                valid_grid = generate_valid_4x4_sudoku()
                num_to_remove = random.randint(4, 8)  # Remove 4-8 células
                grid = remove_cells_randomly(valid_grid, num_to_remove)
                
            elif method == 2:
                # Método 2: Gerar sudoku válido e remover células estrategicamente
                valid_grid = generate_valid_4x4_sudoku()
                num_to_remove = random.randint(4, 8)  # Remove 4-8 células
                grid = remove_cells_strategically(valid_grid, num_to_remove)
                
            else:
                # Método 3: Usar padrões conhecidos
                grid = create_solvable_by_pattern()
                # Adicionar algumas células aleatoriamente
                for r in range(4):
                    for c in range(4):
                        if grid[r][c] == 0 and random.random() < 0.3:
                            grid[r][c] = random.randint(1, 4)
            
            # Verificar se é válido e tem solução
            if is_valid_4x4_sudoku(grid) and has_solution_4x4(grid):
                sudoku_str = grid_to_string(grid)
                sudokus.append(sudoku_str)
            else:
                # Fallback: sudoku solucionável padrão
                fallback_grid = [
                    [1, 2, 0, 0],
                    [3, 4, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]
                sudoku_str = grid_to_string(fallback_grid)
                sudokus.append(sudoku_str)
                
        except Exception as e:
            print(f"Erro ao gerar sudoku {i}: {e}")
            # Fallback: sudoku solucionável padrão
            fallback_grid = [
                [1, 2, 0, 0],
                [3, 4, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
            sudoku_str = grid_to_string(fallback_grid)
            sudokus.append(sudoku_str)
    
    # Salvar no arquivo CSV
    print(f"Salvando {len(sudokus)} sudokus em {output_file}...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for sudoku in sudokus:
            writer.writerow([sudoku])
    
    print(f"Concluído! {len(sudokus)} sudokus 4x4 abertos com solução possível salvos em {output_file}")

def demo_example():
    """Demonstra exemplos de sudokus 4x4 abertos com solução possível"""
    print("Exemplos de sudokus 4x4 abertos com solução possível:")
    print("="*50)
    
    print("\n1. Remoção aleatória:")
    valid_grid = generate_valid_4x4_sudoku()
    grid1 = remove_cells_randomly(valid_grid, 6)
    print_grid(grid1)
    print(f"Válido: {is_valid_4x4_sudoku(grid1)}")
    print(f"Tem solução: {has_solution_4x4(grid1)}")
    
    print("\n2. Remoção estratégica:")
    valid_grid = generate_valid_4x4_sudoku()
    grid2 = remove_cells_strategically(valid_grid, 6)
    print_grid(grid2)
    print(f"Válido: {is_valid_4x4_sudoku(grid2)}")
    print(f"Tem solução: {has_solution_4x4(grid2)}")
    
    print("\n3. Padrão conhecido:")
    grid3 = create_solvable_by_pattern()
    print_grid(grid3)
    print(f"Válido: {is_valid_4x4_sudoku(grid3)}")
    print(f"Tem solução: {has_solution_4x4(grid3)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera sudokus 4x4 abertos com solução possível")
    parser.add_argument("--output", default="../data/4x4/sudoku_4x4_open_solvable.csv", 
                       help="Arquivo CSV para sudokus com solução")
    parser.add_argument("--count", type=int, default=1000, 
                       help="Número de sudokus a gerar")
    parser.add_argument("--demo", action="store_true", 
                       help="Mostra exemplos de sudokus com solução")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_example()
        print("\n" + "="*50 + "\n")
    
    # Gerar sudokus com solução
    generate_4x4_open_solvable_sudokus(args.output, args.count)
    
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
                print(f"Tem solução: {has_solution_4x4(grid)}")
    except FileNotFoundError:
        print("Arquivo não encontrado para mostrar exemplos.") 