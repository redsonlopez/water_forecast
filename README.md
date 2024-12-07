# Predição de Custos com Água - PBH

Este projeto visa analisar os dados de consumo de água da prefeitura, com o objetivo de prever custos futuros com base nos registros a partir do ano de 2022. 

---

## Estrutura do Projeto

A organização dos diretórios segue o modelo abaixo:

```plaintext
water_prediction/
├── data/
│   ├── raw/           # Dados brutos (registros originais a partir de 2022)
│   ├── processed/     # Dados processados para análises e modelagem
│   └── external/      # Dados de fontes externas (se houver)
├── notebooks/
│   ├── eda/           # Notebooks de análise exploratória de dados
│   ├── modeling/      # Notebooks relacionados à modelagem preditiva
│   └── visualization/ # Notebooks para visualizações e relatórios
├── scripts/
│   ├── preprocessing/ # Scripts para limpeza e preparação dos dados
│   ├── models/        # Scripts de treinamento, avaliação e predição
│   └── utils/         # Funções auxiliares
├── results/
│   ├── figures/       # Gráficos e visualizações
│   └── models/        # Modelos treinados
├── reports/
│   ├── pdfs/          # Relatórios finais ou intermediários
│   └── markdown/      # Relatórios e documentações em Markdown
├── tests/             # Scripts de teste para os modelos e pipelines
├── requirements.txt   # Dependências do Python (para uso com pip)
├── .gitignore         # Arquivos e pastas a serem ignorados pelo Git
└── README.md          # Documentação inicial do projeto
