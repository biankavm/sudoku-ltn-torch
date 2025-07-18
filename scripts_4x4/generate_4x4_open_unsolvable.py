#!/usr/bin/env python3
"""
Script para gerar sudokus 4x4 abertos sem solução.
Gera tabuleiros parciais que são impossíveis de resolver.
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

def create_simple_unsolvable() -> List[List[int]]:
    """
    Cria um sudoku 4x4 impossível com conflito direto
    """
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Escolhe uma linha e coloca o mesmo número duas vezes
    row = random.randint(0, 3)
    col1, col2 = random.sample(range(4), 2)
    number = random.randint(1, 4)
    
    grid[row][col1] = number
    grid[row][col2] = number  # Conflito na linha
    
    # Preenche algumas outras células aleatoriamente
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0 and random.random() < 0.2:
                available = [n for n in range(1, 5) if n != number]
                if available:
                    grid[r][c] = random.choice(available)
    
    return grid

def create_unsolvable_by_blocking() -> List[List[int]]:
    """
    Cria um sudoku 4x4 impossível bloqueando todas as posições possíveis de um número
    """
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Escolhe um número para bloquear
    blocked_number = random.randint(1, 4)
    
    # Escolhe uma linha onde o número será impossível
    target_row = random.randint(0, 3)
    
    # Coloca o número bloqueado em todas as colunas desta linha, exceto uma
    empty_position = random.randint(0, 3)
    
    for col in range(4):
        if col != empty_position:
            grid[target_row][col] = blocked_number
    
    # Agora coloca o número bloqueado na coluna da posição vazia (em outra linha)
    other_rows = [r for r in range(4) if r != target_row]
    if other_rows:
        other_row = random.choice(other_rows)
        grid[other_row][empty_position] = blocked_number
    
    # Preenche outras células aleatoriamente
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0 and random.random() < 0.3:
                available = [n for n in range(1, 5) if n != blocked_number]
                if available:
                    grid[r][c] = random.choice(available)
    
    return grid

def create_unsolvable_by_quadrant_conflict() -> List[List[int]]:
    """
    Cria um sudoku 4x4 impossível com conflito no quadrante
    """
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Escolhe um quadrante
    box_row = random.randint(0, 1) * 2
    box_col = random.randint(0, 1) * 2
    
    # Escolhe um número para tornar impossível no quadrante
    impossible_number = random.randint(1, 4)
    
    # Preenche parcialmente o quadrante
    numbers = [n for n in range(1, 5) if n != impossible_number]
    random.shuffle(numbers)
    
    # Coloca 3 números no quadrante, deixando 1 posição vazia
    positions = []
    for r in range(box_row, box_row + 2):
        for c in range(box_col, box_col + 2):
            positions.append((r, c))
    
    # Preenche 3 posições
    for i in range(3):
        r, c = positions[i]
        grid[r][c] = numbers[i]
    
    # Para a posição vazia, coloca o número impossível em sua linha e coluna
    empty_position = positions[3]
    r, c = empty_position
    
    # Coloca o número impossível na linha (fora do quadrante)
    for other_c in range(4):
        if other_c < box_col or other_c >= box_col + 2:
            if grid[r][other_c] == 0:
                grid[r][other_c] = impossible_number
                break
    
    # Coloca o número impossível na coluna (fora do quadrante)
    for other_r in range(4):
        if other_r < box_row or other_r >= box_row + 2:
            if grid[other_r][c] == 0:
                grid[other_r][c] = impossible_number
                break
    
    return grid

def create_unsolvable_by_mutual_exclusion() -> List[List[int]]:
    """
    Cria um sudoku 4x4 impossível com exclusão mútua
    """
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Encontra duas posições vazias na mesma linha
    row = random.randint(0, 3)
    col1, col2 = random.sample(range(4), 2)
    
    # Preenche outras posições para forçar que apenas um número seja possível
    available_numbers = list(range(1, 5))
    
    # Remove números já usados na linha
    for c in range(4):
        if grid[row][c] != 0 and grid[row][c] in available_numbers:
            available_numbers.remove(grid[row][c])
    
    if len(available_numbers) >= 2:
        # Escolhe dois números que serão impossíveis de colocar
        num1, num2 = random.sample(available_numbers, 2)
        
        # Coloca num1 na coluna de col1 (em outra linha)
        for other_row in range(4):
            if other_row != row and grid[other_row][col1] == 0:
                grid[other_row][col1] = num1
                break
        
        # Coloca num2 na coluna de col2 (em outra linha)
        for other_row in range(4):
            if other_row != row and grid[other_row][col2] == 0:
                grid[other_row][col2] = num2
                break
        
        # Preenche o resto da linha deixando apenas essas duas posições
        remaining_numbers = [n for n in available_numbers if n not in [num1, num2]]
        fill_positions = [c for c in range(4) if c not in [col1, col2]]
        
        for i, pos in enumerate(fill_positions[:len(remaining_numbers)]):
            grid[row][pos] = remaining_numbers[i]
    
    return grid

def create_unsolvable_by_column_conflict() -> List[List[int]]:
    """
    Cria um sudoku 4x4 impossível com conflito na coluna
    """
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Escolhe uma coluna e coloca o mesmo número duas vezes
    col = random.randint(0, 3)
    row1, row2 = random.sample(range(4), 2)
    number = random.randint(1, 4)
    
    grid[row1][col] = number
    grid[row2][col] = number  # Conflito na coluna
    
    # Preenche algumas outras células aleatoriamente
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0 and random.random() < 0.2:
                available = [n for n in range(1, 5) if n != number]
                if available:
                    grid[r][c] = random.choice(available)
    
    return grid

def create_unsolvable_by_box_conflict() -> List[List[int]]:
    """
    Cria um sudoku 4x4 impossível com conflito na caixa 2x2
    """
    grid = [[0 for _ in range(4)] for _ in range(4)]
    
    # Escolhe uma caixa 2x2
    box_row = random.randint(0, 1) * 2
    box_col = random.randint(0, 1) * 2
    
    # Escolhe um número para duplicar na caixa
    number = random.randint(1, 4)
    
    # Coloca o número duas vezes na mesma caixa
    positions = []
    for r in range(box_row, box_row + 2):
        for c in range(box_col, box_col + 2):
            positions.append((r, c))
    
    pos1, pos2 = random.sample(positions, 2)
    grid[pos1[0]][pos1[1]] = number
    grid[pos2[0]][pos2[1]] = number  # Conflito na caixa
    
    # Preenche algumas outras células aleatoriamente
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0 and random.random() < 0.2:
                available = [n for n in range(1, 5) if n != number]
                if available:
                    grid[r][c] = random.choice(available)
    
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
            if cell == 0:
                row_str += ". "
            else:
                row_str += f"{cell} "
        
        print(row_str)

def generate_4x4_open_unsolvable_sudokus(output_file: str, count: int = 1000):
    """
    Gera sudokus 4x4 abertos sem solução
    
    Args:
        output_file: Arquivo CSV para salvar os sudokus
        count: Número de sudokus a gerar
    """
    print(f"Gerando {count} sudokus 4x4 abertos sem solução...")
    
    sudokus = set()  # Usar set para evitar duplicatas automaticamente
    
    # Tipos de geradores de sudokus impossíveis
    generators = [
        create_simple_unsolvable,
        create_unsolvable_by_blocking,
        create_unsolvable_by_quadrant_conflict,
        create_unsolvable_by_mutual_exclusion,
        create_unsolvable_by_column_conflict,
        create_unsolvable_by_box_conflict
    ]
    
    attempts = 0
    max_attempts = min(count * 2, 10000)  # Limitar tentativas da Fase 1
    
    print("Fase 1: Gerando sudokus impossíveis com métodos complexos...")
    
    # Tentar métodos complexos por um tempo limitado
    while len(sudokus) < count and attempts < max_attempts:
        attempts += 1
        
        if attempts % 1000 == 0:
            print(f"Progresso: {len(sudokus)}/{count} (tentativas: {attempts})")
            
            # Se após 5000 tentativas não gerou pelo menos 10% do objetivo, pular para Fase 2
            if attempts >= 5000 and len(sudokus) < count * 0.1:
                print(f"Fase 1 ineficiente ({len(sudokus)}/{count}), pulando para Fase 2...")
                break
        
        try:
            # Escolher um gerador aleatório
            generator = random.choice(generators)
            grid = generator()
            
            # Verificar se é realmente impossível
            if not has_solution_4x4(grid) and is_valid_4x4_sudoku(grid):
                sudoku_str = grid_to_string(grid)
                sudokus.add(sudoku_str)
                
        except Exception as e:
            if attempts % 1000 == 0:
                print(f"Erro ao gerar sudoku {attempts}: {e}")
            continue
    
    print(f"Fase 1 concluída: {len(sudokus)} sudokus gerados")
    
    # Fase 2: Método simples para completar a quantidade
    print("Fase 2: Completando com método simples...")
    
    while len(sudokus) < count:
        # Método simples: criar conflitos diretos garantidos
        grid = [[0 for _ in range(4)] for _ in range(4)]
        
        # Escolhe tipo de conflito
        conflict_type = random.choice(['row', 'column', 'box'])
        
        if conflict_type == 'row':
            # Conflito na linha
            row = random.randint(0, 3)
            col1, col2 = random.sample(range(4), 2)
            number = random.randint(1, 4)
            grid[row][col1] = number
            grid[row][col2] = number
            
        elif conflict_type == 'column':
            # Conflito na coluna
            col = random.randint(0, 3)
            row1, row2 = random.sample(range(4), 2)
            number = random.randint(1, 4)
            grid[row1][col] = number
            grid[row2][col] = number
            
        else:  # box
            # Conflito na caixa 2x2
            box_row = random.randint(0, 1) * 2
            box_col = random.randint(0, 1) * 2
            number = random.randint(1, 4)
            
            # Coloca o número duas vezes na mesma caixa
            positions = []
            for r in range(box_row, box_row + 2):
                for c in range(box_col, box_col + 2):
                    positions.append((r, c))
            
            pos1, pos2 = random.sample(positions, 2)
            grid[pos1[0]][pos1[1]] = number
            grid[pos2[0]][pos2[1]] = number
        
        # Preenche algumas outras células aleatoriamente
        for r in range(4):
            for c in range(4):
                if grid[r][c] == 0 and random.random() < 0.3:
                    grid[r][c] = random.randint(1, 4)
        
        sudoku_str = grid_to_string(grid)
        sudokus.add(sudoku_str)
        
        if len(sudokus) % 1000 == 0:
            print(f"Progresso Fase 2: {len(sudokus)}/{count}")
    
    # Converter set para lista
    sudokus_list = list(sudokus)
    
    # Salvar no arquivo CSV
    print(f"Salvando {len(sudokus_list)} sudokus em {output_file}...")
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for sudoku in sudokus_list:
            writer.writerow([sudoku])
    
    print(f"Concluído! {len(sudokus_list)} sudokus 4x4 abertos sem solução salvos em {output_file}")
    print(f"Total de tentativas: {attempts}")

def demo_example():
    """Demonstra exemplos de sudokus 4x4 abertos sem solução"""
    print("Exemplos de sudokus 4x4 abertos sem solução:")
    print("="*50)
    
    print("\n1. Conflito simples na linha:")
    grid1 = create_simple_unsolvable()
    print_grid(grid1)
    print(f"Válido: {is_valid_4x4_sudoku(grid1)}")
    print(f"Tem solução: {has_solution_4x4(grid1)}")
    
    print("\n2. Bloqueio de número:")
    grid2 = create_unsolvable_by_blocking()
    print_grid(grid2)
    print(f"Válido: {is_valid_4x4_sudoku(grid2)}")
    print(f"Tem solução: {has_solution_4x4(grid2)}")
    
    print("\n3. Conflito no quadrante:")
    grid3 = create_unsolvable_by_quadrant_conflict()
    print_grid(grid3)
    print(f"Válido: {is_valid_4x4_sudoku(grid3)}")
    print(f"Tem solução: {has_solution_4x4(grid3)}")
    
    print("\n4. Conflito na coluna:")
    grid4 = create_unsolvable_by_column_conflict()
    print_grid(grid4)
    print(f"Válido: {is_valid_4x4_sudoku(grid4)}")
    print(f"Tem solução: {has_solution_4x4(grid4)}")
    
    print("\n5. Conflito na caixa:")
    grid5 = create_unsolvable_by_box_conflict()
    print_grid(grid5)
    print(f"Válido: {is_valid_4x4_sudoku(grid5)}")
    print(f"Tem solução: {has_solution_4x4(grid5)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera sudokus 4x4 abertos sem solução")
    parser.add_argument("--output", default="../data/4x4/sudoku_4x4_open_unsolvable.csv", 
                       help="Arquivo CSV para sudokus sem solução")
    parser.add_argument("--count", type=int, default=1000, 
                       help="Número de sudokus a gerar")
    parser.add_argument("--demo", action="store_true", 
                       help="Mostra exemplos de sudokus sem solução")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_example()
        print("\n" + "="*50 + "\n")
    
    # Gerar sudokus sem solução
    generate_4x4_open_unsolvable_sudokus(args.output, args.count)
    
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