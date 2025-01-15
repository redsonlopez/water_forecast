import pandas as pd

def filtrar_matricula(data, matricula):
    """
    Filtra os dados de acordo com a matrícula fornecida.
    
    Args:
        data (pd.DataFrame): Conjunto de dados carregado.
        matricula (str): Matrícula a ser filtrada.
    
    Returns:
        pd.DataFrame: Dados filtrados para a matrícula específica.
    """
    return data[data['MATRICULA'] == matricula]

def gerar_serie(filtered_data):
    """
    Gera uma série temporal com os dados filtrados, incluindo a coluna de data formatada.
    
    Args:
        filtered_data (pd.DataFrame): Dados filtrados pela matrícula.
    
    Returns:
        pd.DataFrame: Série temporal com a coluna 'DATA_VENCIMENTO_FORMATADA'.
    """
    # Criando a coluna de data formatada (como 'YYYY-MM-DD')
    filtered_data['DATA_VENCIMENTO_FORMATADA'] = filtered_data['MES_VENCIMENTO'].astype(str) + '/' + filtered_data['ANO_VENCIMENTO'].astype(str)
    
    # Seleciona as colunas necessárias para a série temporal
    return filtered_data
