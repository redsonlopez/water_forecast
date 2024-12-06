# Projeto de Predição para Custo com Água

# Estrutura do Projeto

```plaintext
project/
├── data/
│   ├── raw/           # Dados brutos (registros originais de 2022, 2023, 2024)
│   ├── processed/     # Dados processados para análises e modelagem
│   └── external/      # Dados de fontes externas (se houver)
├── notebooks/
│   ├── EDA/           # Notebooks de análise exploratória de dados
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
├── environment.yml    # Configuração do ambiente (Anaconda/Miniconda)
├── requirements.txt   # Dependências do Python (para uso com pip)
├── .gitignore         # Arquivos e pastas a serem ignorados pelo Git
└── README.md          # Documentação inicial do projeto
