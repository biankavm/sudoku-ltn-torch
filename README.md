# Sistema LTN para Resolução de Sudoku - Versão Separada por Dimensão

## Equipe

- **Bianka Vasconcelos**
- **Micael Viana**
- **Vinicius Chagas**

## Objetivo

Desenvolver uma solução em LTN Torch que resolva o problema do Sudoku com modelos especializados separados para cada dimensão (4x4 e 9x9).

## Características Principais

- **Modelos Separados**: Treinamento independente para Sudoku 4x4 e 9x9
- **Heurísticas Adaptativas**: Estratégias específicas para cada dimensão
- **Base de Conhecimento LTN**: Axiomas e regras lógicas do Sudoku
- **Sistema de Memória**: Rastreamento de heurísticas aplicadas

## Estrutura do Projeto

```
sudoku-ltn-torch/
├── main.py                 # Script principal com funcionalidades integradas
├── train_4x4.py           # Script específico para treinar modelo 4x4
├── train_9x9.py           # Script específico para treinar modelo 9x9
├── test_4x4.py            # Script específico para testar modelo 4x4
├── test_9x9.py            # Script específico para testar modelo 9x9
├── demo_separated_models.py # Demonstração dos modelos separados
├── core/                  # Componentes principais
│   ├── sudoku_board.py    # Representação do tabuleiro
│   └── memory_system.py   # Sistema de memória
├── solver/                # Solver LTN
│   ├── ltn_solver.py      # Solver principal
│   ├── predicates.py      # Predicates neurais
│   ├── knowledge_base.py  # Base de conhecimento
│   └── heuristics.py      # Heurísticas de resolução
├── utils/                 # Utilitários
│   └── training_data.py   # Geração de dados de treinamento
├── data/                  # Dados de treinamento
│   ├── 4x4/              # Dados para Sudoku 4x4
│   └── 9x9/              # Dados para Sudoku 9x9
└── models/                # Modelos treinados
    ├── sudoku_ltn_4x4.pth # Modelo 4x4
    └── sudoku_ltn_9x9.pth # Modelo 9x9
```

## Dependências

- torch
- LTN
- numpy
- argparse

## Uso

### Treinamento dos Modelos

#### Treinar Modelo 4x4

```bash
python train_4x4.py --epochs 50
```

#### Treinar Modelo 9x9

```bash
python train_9x9.py --epochs 50
```

### Teste dos Modelos

#### Testar Modelo 4x4

```bash
python test_4x4.py
```

#### Testar Modelo 9x9

```bash
python test_9x9.py
```

### Demonstração

```bash
python demo_separated_models.py
```

### Uso Geral (main.py)

#### Processar arquivo CSV

```bash
python main.py --path data/meu_sudoku.csv
```

#### Treinar modelos separados

```bash
python main.py --train-4x4 --epochs 50
python main.py --train-9x9 --epochs 50
```

#### Testar modelos

```bash
python main.py --test-4x4
python main.py --test-9x9
```

#### Resolver sudoku

```bash
python main.py --solve --board-size 4
python main.py --solve --board-size 9
```

## Heurísticas Implementadas

### Básicas

- **Naked Single**: Célula com apenas um candidato possível
- **Hidden Single**: Número que aparece como candidato em apenas uma célula da unidade

### Adaptação por Dimensão

- **4x4**: Heurísticas otimizadas para tabuleiros menores
- **9x9**: Heurísticas completas para tabuleiros padrão

## Base de Conhecimento LTN

O sistema utiliza axiomas lógicos para:

- Constraints básicas do Sudoku (linha, coluna, quadrante)
- Regras de consistência
- Aplicação de heurísticas

## Sistema de Memória

Rastreia:

- Heurísticas aplicadas
- Histórico de movimentos
- Efetividade das estratégias
- Estatísticas de performance

## Referências

- [Dataset Sudoku - Kaggle](https://www.kaggle.com/datasets/bryanpark/sudoku?resource=download)
- [LTN (Logic Tensor Networks)](https://github.com/logictensornetworks/logictensornetworks)
