import pandas as pd

def importando_dados():
    """
    Carrega e prepara a base de dados.
    """
    data = pd.read_csv("data/processed/final_data_test.csv")
    # Conversão de colunas relevantes
    data['DATA_VENCIMENTO'] = pd.to_datetime(data['DATA_VENCIMENTO'], errors='coerce')
    data['MATRICULA'] = data['MATRICULA'].astype(str)
    return data
