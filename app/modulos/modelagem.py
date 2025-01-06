def prever_series(series):
    """
    Realiza a previsão da série temporal.
    """
    from statsmodels.tsa.arima.model import ARIMA
    import pandas as pd
    import numpy as np

    # Verificação básica: a série deve ter pelo menos 2 valores
    if len(series) < 2:
        return None, None  # Retorna vazio se não puder prever

    # Modelagem
    model = ARIMA(series, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)  # Previsão de 1 passo à frente
    
    # Acessando o valor previsto
    forecast_value = forecast.iloc[0] if isinstance(forecast, pd.Series) else forecast[0]

    # Erro da previsão
    error = np.sqrt(model_fit.mse)
    return forecast_value, error
