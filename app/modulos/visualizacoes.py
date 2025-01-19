import plotly.express as px
import pandas as pd 
import plotly.graph_objects as go

def plotar_serie_temporal(series):
    """
    Função para criar o gráfico da série temporal de valores da fatura.
    
    Args:
        series (pd.DataFrame): Série temporal com os dados de fatura.
    
    Returns:
        plotly.graph_objs._figure.Figure: Gráfico da série temporal.
    """
    fig = px.line(series, x='DATA_VENCIMENTO_FORMATADA', y='VALOR_FATURA', title='Série Temporal de Faturas')
    return fig


def plotar_serie_e_previsao(matricula, dados_matricula, valor_previsto):
    """
    Função para plotar os valores reais da série temporal de uma matrícula, mostrando as datas das faturas no eixo X
    e o valor futuro previsto já calculado externamente.
    
    Args:
        matricula (int): Matrícula para a qual será feita a previsão.
        dados_matricula (pd.DataFrame): Conjunto de dados original contendo todas as informações referente à matrícula inserida pelo usuário.
        valor_previsto (float): Valor já previsto para o próximo mês (calculado externamente).
    """
    import plotly.graph_objects as go  # Certifique-se de importar aqui, caso não tenha sido importado globalmente.

    # Ordenar os dados pela data de vencimento para consistência na série temporal
    dados_matricula = dados_matricula.sort_values(by=['ANO_VENCIMENTO', 'MES_VENCIMENTO'])

    # Criar coluna de datas legíveis
    dados_matricula['DATA_VENCIMENTO_FORMATADA'] = (
        dados_matricula['MES_VENCIMENTO'].astype(str) + '/' + dados_matricula['ANO_VENCIMENTO'].astype(str)
    )

    # Adicionar a próxima data prevista no eixo X
    proximo_mes = (dados_matricula.iloc[-1]['MES_VENCIMENTO'] % 12) + 1
    proximo_ano = dados_matricula.iloc[-1]['ANO_VENCIMENTO'] + (dados_matricula.iloc[-1]['MES_VENCIMENTO'] // 12)
    data_proxima_fatura = f"{proximo_mes}/{proximo_ano}"

    # Criar o gráfico com Plotly
    fig = go.Figure()

    # Adicionar linha para os valores reais
    fig.add_trace(go.Scatter(
        x=dados_matricula['DATA_VENCIMENTO_FORMATADA'], 
        y=dados_matricula['VALOR_FATURA'], 
        mode='lines+markers', 
        name='Valor Real', 
        line=dict(color='blue')
    ))

    # Adicionar o ponto previsto
    fig.add_trace(go.Scatter(
        x=[data_proxima_fatura], 
        y=[valor_previsto], 
        mode='markers', 
        name='Valor Previsto', 
        marker=dict(color='red', size=10)
    ))

    # Configurar o layout do gráfico
    fig.update_layout(
        title=f'Série Temporal de Valores Reais e Previsão para Matrícula: {matricula}',
        xaxis_title='Data de Vencimento',
        yaxis_title='Valor da Fatura (R$)',
        xaxis=dict(tickangle=45),
        showlegend=True
    )

    return fig



