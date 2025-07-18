#!/usr/bin/env python3
"""
Script principal para gerar todos os tipos de sudokus 4x4.
Executa os 4 scripts de geração em sequência.
"""

import os
import sys
import subprocess
import argparse

def run_script(script_name: str, output_file: str, count: int, demo: bool = False):
    """
    Executa um script de geração de sudokus 4x4
    """
    print(f"\n{'='*60}")
    print(f"EXECUTANDO: {script_name}")
    print(f"{'='*60}")
    
    # Construir comando
    cmd = [
        sys.executable, script_name,
        "--output", output_file,
        "--count", str(count)
    ]
    
    if demo:
        cmd.append("--demo")
    
    try:
        # Executar script
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_name))
        
        if result.returncode == 0:
            print(f"✅ {script_name} executado com sucesso!")
            print(result.stdout)
        else:
            print(f"❌ Erro ao executar {script_name}:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Erro ao executar {script_name}: {e}")
        return False
    
    return True

def main():
    """
    Função principal que executa todos os scripts
    """
    parser = argparse.ArgumentParser(description="Gera todos os tipos de sudokus 4x4")
    parser.add_argument("--output-dir", default="../data", 
                       help="Diretório para salvar os arquivos CSV")
    parser.add_argument("--count", type=int, default=1000, 
                       help="Número de sudokus a gerar por tipo")
    parser.add_argument("--demo", action="store_true", 
                       help="Mostra exemplos de cada tipo")
    
    args = parser.parse_args()
    
    # Configurações
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(args.output_dir)
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Lista de scripts e seus arquivos de saída
    scripts_config = [
        {
            "script": "generate_4x4_closed_valid.py",
            "output": "sudoku_4x4_closed_valid.csv",
            "description": "Sudokus 4x4 fechados válidos"
        },
        {
            "script": "generate_4x4_closed_invalid.py", 
            "output": "sudoku_4x4_closed_invalid.csv",
            "description": "Sudokus 4x4 fechados inválidos"
        },
        {
            "script": "generate_4x4_open_solvable.py",
            "output": "sudoku_4x4_open_solvable.csv", 
            "description": "Sudokus 4x4 abertos com solução"
        },
        {
            "script": "generate_4x4_open_unsolvable.py",
            "output": "sudoku_4x4_open_unsolvable.csv",
            "description": "Sudokus 4x4 abertos sem solução"
        }
    ]
    
    print("🧩 GERADOR DE SUDOKUS 4x4 - TODOS OS TIPOS")
    print("="*60)
    print(f"Diretório de saída: {output_dir}")
    print(f"Sudokus por tipo: {args.count}")
    print(f"Modo demo: {args.demo}")
    print("="*60)
    
    # Executar cada script
    success_count = 0
    total_scripts = len(scripts_config)
    
    for config in scripts_config:
        script_path = os.path.join(scripts_dir, config["script"])
        output_path = os.path.join(output_dir, config["output"])
        
        print(f"\n📋 {config['description']}")
        print(f"Script: {config['script']}")
        print(f"Saída: {config['output']}")
        
        if run_script(script_path, output_path, args.count, args.demo):
            success_count += 1
        else:
            print(f"⚠️  Falha ao executar {config['script']}")
    
    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO DA EXECUÇÃO")
    print(f"{'='*60}")
    print(f"Scripts executados com sucesso: {success_count}/{total_scripts}")
    
    if success_count == total_scripts:
        print("✅ Todos os scripts foram executados com sucesso!")
        print(f"📁 Arquivos gerados em: {output_dir}")
        
        # Listar arquivos gerados
        print("\n📋 Arquivos gerados:")
        for config in scripts_config:
            output_path = os.path.join(output_dir, config["output"])
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                print(f"  ✅ {config['output']} ({size} bytes)")
            else:
                print(f"  ❌ {config['output']} (não encontrado)")
    else:
        print("❌ Alguns scripts falharam. Verifique os erros acima.")
    
    print(f"\n🎯 Geração de sudokus 4x4 concluída!")

if __name__ == "__main__":
    main() 