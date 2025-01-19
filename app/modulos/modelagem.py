# modulos/modelagem.py
import joblib
import pandas as pd

# Carregar o modelo treinado
modelo_carregado = joblib.load("../scripts/models/base_line.joblib")

def prever_series(series):
    """
    Função para gerar a previsão de fatura para o próximo mês, usando o modelo carregado.
    
    Args:
        series (pd.DataFrame): Dados da série temporal para a matrícula.
    
    Returns:
        tuple: Valor previsto para o próximo mês e erro da previsão (caso tenha erro calculado).
    """
    # Gerar resumo dos dados
    dados_resumo = series.mean(numeric_only=True)
    dados_resumo['BAIRRO'] = series.iloc[-1]['BAIRRO']
    dados_resumo['ANO_VENCIMENTO'] = series.iloc[-1]['ANO_VENCIMENTO'] + (series.iloc[-1]['MES_VENCIMENTO'] // 12)
    dados_resumo['MES_VENCIMENTO'] = (series.iloc[-1]['MES_VENCIMENTO'] % 12) + 1

    # Preparar entrada para a previsão
    X_entrada = pd.DataFrame([dados_resumo.drop(['VALOR_FATURA'])])

    # Gerar previsão
    previsao = modelo_carregado.predict(X_entrada)[0]
    
    # Erro da previsão (pode ser calculado com base em algum critério)
    # Neste exemplo, vamos apenas retornar None, mas vamos modificar 
    erro = None

    return previsao, erro

