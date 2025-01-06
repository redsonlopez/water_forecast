import pandas as pd

def importando_dados(filepath):
    """
    Carrega e prepara a base de dados.
    """
    data = pd.read_csv(filepath)
    # Conversão de colunas relevantes
    data['DATA_VENCIMENTO'] = pd.to_datetime(data['DATA_VENCIMENTO'], errors='coerce')
    data['MATRICULA'] = data['MATRICULA'].astype(str)
    return data
