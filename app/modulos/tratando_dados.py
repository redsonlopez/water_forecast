def filtrar_matricula(data, matricula):
    """
    Filtra os dados por matrícula.
    """
    return data[data['MATRICULA'] == matricula]

def gerar_serie(data):
    """
    Prepara a série temporal de consumo.
    """
    series = data[['DATA_VENCIMENTO', 'VALOR_FATURA']].set_index('DATA_VENCIMENTO')
    return series.sort_index()
