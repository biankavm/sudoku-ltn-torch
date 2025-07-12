#!/usr/bin/env python3
"""
Script simples e eficiente para gerar sudokus abertos impossíveis.
Cria situações garantidamente impossíveis baseadas no exemplo fornecido.
"""

import random
from typing import List

def create_simple_impossible_sudoku() -> List[List[int]]:
    """
    Cria um sudoku 9x9 impossível usando uma estratégia simples e garantida
    Baseado no exemplo 4x4: coloca números de forma que criem conflitos inevitáveis
    """
    grid = [[0 for _ in range(9)] for _ in range(9)]
    
    # Estratégia 1: Cria conflito direto - mesmo número em linha
    # Exemplo: linha 0 tem o número 5 em duas posições, mas com células vazias entre elas
    target_number = random.randint(1, 9)
    target_row = random.randint(0, 8)
    
    # Coloca o número alvo em duas posições da mesma linha
    pos1 = random.randint(0, 3)
    pos2 = random.randint(5, 8)
    
    grid[target_row][pos1] = target_number
    grid[target_row][pos2] = target_number
    
    # Deixa uma célula vazia entre elas que precisaria do mesmo número
    empty_pos = random.randint(pos1 + 1, pos2 - 1)
    grid[target_row][empty_pos] = 0
    
    # Preenche outras células aleatoriamente mas sem conflitos óbvios
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0 and random.random() < 0.3:  # 30% de chance
                # Escolhe um número que não conflite imediatamente
                available = list(range(1, 10))
                if target_number in available:
                    available.remove(target_number)
                
                if available:
                    grid[r][c] = random.choice(available)
    
    return grid

def create_impossible_by_blocking() -> List[List[int]]:
    """
    Cria impossibilidade bloqueando todas as posições possíveis de um número
    """
    grid = [[0 for _ in range(9)] for _ in range(9)]
    
    # Escolhe um número para bloquear
    blocked_number = random.randint(1, 9)
    
    # Escolhe uma linha onde o número será impossível
    target_row = random.randint(0, 8)
    
    # Coloca o número bloqueado em todas as colunas desta linha, exceto algumas
    empty_positions = random.sample(range(9), 3)  # 3 posições vazias
    
    for col in range(9):
        if col not in empty_positions:
            grid[target_row][col] = blocked_number
    
    # Agora coloca o número bloqueado nas colunas das posições vazias
    for col in empty_positions:
        # Escolhe uma linha diferente para colocar o número
        other_rows = [r for r in range(9) if r != target_row]
        if other_rows:
            other_row = random.choice(other_rows)
            grid[other_row][col] = blocked_number
    
    # Preenche outras células aleatoriamente
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0 and random.random() < 0.25:
                available = [n for n in range(1, 10) if n != blocked_number]
                if available:
                    grid[r][c] = random.choice(available)
    
    return grid

def create_quadrant_impossible() -> List[List[int]]:
    """
    Cria impossibilidade em um quadrante 3x3
    """
    grid = [[0 for _ in range(9)] for _ in range(9)]
    
    # Escolhe um quadrante
    box_row = random.randint(0, 2)
    box_col = random.randint(0, 2)
    
    # Escolhe um número para tornar impossível no quadrante
    impossible_number = random.randint(1, 9)
    
    # Preenche parcialmente o quadrante
    numbers = [n for n in range(1, 10) if n != impossible_number]
    random.shuffle(numbers)
    
    # Coloca 6 números no quadrante, deixando 3 posições vazias
    positions = []
    for r in range(box_row * 3, (box_row + 1) * 3):
        for c in range(box_col * 3, (box_col + 1) * 3):
            positions.append((r, c))
    
    # Preenche 6 posições
    for i in range(6):
        r, c = positions[i]
        grid[r][c] = numbers[i]
    
    # Para as 3 posições vazias, coloca o número impossível em suas linhas/colunas
    empty_positions = positions[6:]
    for r, c in empty_positions:
        # Coloca o número impossível na linha (fora do quadrante)
        for other_c in range(9):
            if other_c < box_col * 3 or other_c >= (box_col + 1) * 3:
                if grid[r][other_c] == 0:
                    grid[r][other_c] = impossible_number
                    break
        
        # Coloca o número impossível na coluna (fora do quadrante)
        for other_r in range(9):
            if other_r < box_row * 3 or other_r >= (box_row + 1) * 3:
                if grid[other_r][c] == 0:
                    grid[other_r][c] = impossible_number
                    break
    
    return grid

def grid_to_string(grid: List[List[int]]) -> str:
    """Converte grid em string de 81 caracteres"""
    return ''.join(str(grid[i][j]) for i in range(9) for j in range(9))

def print_grid(grid: List[List[int]]):
    """Imprime o grid de forma legível"""
    for i, row in enumerate(grid):
        if i % 3 == 0 and i != 0:
            print("------+-------+------")
        
        row_str = ""
        for j, cell in enumerate(row):
            if j % 3 == 0 and j != 0:
                row_str += "| "
            
            if cell == 0:
                row_str += ". "
            else:
                row_str += f"{cell} "
        
        print(row_str)

def generate_unsolvable_sudokus_simple(output_file: str, count: int = 1000):
    """
    Gera sudokus impossíveis usando estratégias simples e garantidas
    """
    print(f"Gerando {count} sudokus impossíveis...")
    
    generators = [
        create_simple_impossible_sudoku,
        create_impossible_by_blocking,
        create_quadrant_impossible
    ]
    
    sudokus = []
    
    for i in range(count):
        if i % 100 == 0:
            print(f"Progresso: {i}/{count}")
        
        # Escolhe um gerador aleatório
        generator = random.choice(generators)
        
        try:
            grid = generator()
            sudoku_str = grid_to_string(grid)
            sudokus.append(sudoku_str)
        except Exception as e:
            print(f"Erro ao gerar sudoku {i}: {e}")
            # Fallback: cria um conflito muito simples
            simple_grid = [[0 for _ in range(9)] for _ in range(9)]
            simple_grid[0][0] = 5
            simple_grid[0][1] = 5  # Conflito direto na linha
            simple_grid[1][0] = 5  # Conflito direto na coluna
            sudoku_str = grid_to_string(simple_grid)
            sudokus.append(sudoku_str)
    
    # Salva os sudokus
    print(f"Salvando {len(sudokus)} sudokus em {output_file}...")
    
    with open(output_file, 'w') as f:
        for sudoku in sudokus:
            f.write(sudoku + '\n')
    
    print(f"Concluído! {len(sudokus)} sudokus impossíveis salvos em {output_file}")

def demo_example():
    """Demonstra um exemplo de sudoku impossível gerado"""
    print("Exemplo de sudoku impossível gerado:")
    print("="*40)
    
    grid = create_simple_impossible_sudoku()
    print_grid(grid)
    
    print("\nAnálise:")
    print("- Este sudoku tem conflitos que tornam impossível resolver")
    print("- Números duplicados em linhas/colunas criam situações sem solução")
    print("- Baseado no princípio do exemplo 4x4 fornecido")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera sudokus abertos impossíveis (versão simples)")
    parser.add_argument("--output", default="../data/sudoku_open_unsolvable.csv", 
                       help="Arquivo CSV para sudokus impossíveis")
    parser.add_argument("--count", type=int, default=5000, 
                       help="Número de sudokus impossíveis a gerar")
    parser.add_argument("--demo", action="store_true", 
                       help="Mostra exemplo de sudoku impossível")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_example()
        print("\n" + "="*50 + "\n")
    
    # Gera sudokus impossíveis
    generate_unsolvable_sudokus_simple(args.output, args.count)
    
    # Mostra alguns exemplos
    print("\nPrimeiros 3 sudokus gerados:")
    try:
        with open(args.output, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:3]):
                sudoku_str = line.strip()
                print(f"\nSudoku {i+1}:")
                grid = [[int(sudoku_str[r*9 + c]) for c in range(9)] for r in range(9)]
                print_grid(grid)
                print()
    except FileNotFoundError:
        print("Arquivo não encontrado para mostrar exemplos.") 