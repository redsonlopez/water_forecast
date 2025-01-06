import plotly.express as px

def plotar_serie_temporal(series):
    """
    Plota a série temporal.
    """
    fig = px.line(series, x=series.index, y=series.columns[0], title="Valor Fatura")
    return fig
