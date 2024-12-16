#%%
# Bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

# Carregar os dados
data = pd.read_csv("../../data/processed/processed_water.csv",\
                   parse_dates=["DATA_VENCIMENTO"])

# Configurar o índice para a data
data = data.set_index('DATA_VENCIMENTO').to_period('D')

# Selecionar a coluna de interesse
valor_fatura = data['VALOR_FATURA']

# Visualizar os valores ao longo do tempo
ax = valor_fatura.plot(title="Custo de Consumo de Água",\
                       ylabel="Valor da Fatura (R$)", style="o")
plt.show()

#%%
# Visualização com Média Móvel
###

# Calcular a média móvel (janela de 3 meses para suavização)
trend = valor_fatura.rolling(window=3, center=True, min_periods=1).mean()

# Plotar a série original e a tendência
ax = valor_fatura.plot(alpha=0.5, title="Custo de Consumo de Água com Tendência")
trend.plot(ax=ax, linewidth=3, label="Tendência", color='C0')
ax.set(ylabel="Valor da Fatura (R$)")
ax.legend()
plt.show()

#%%
# Modelagem da Tendência
###

# Criar o conjunto de dados para a modelagem
dp = DeterministicProcess(index=valor_fatura.index, order=3)  # Modelo cúbico
X = dp.in_sample()  # Conjunto de entrada para os dados existentes
y = valor_fatura  # Variável alvo

# Treinar o modelo
model = LinearRegression()
model.fit(X, y)

# Predição dos valores ajustados
y_pred = pd.Series(model.predict(X), index=X.index)

# Criar conjunto de dados para previsão
X_fore = dp.out_of_sample(steps=90)

# Fazer a previsão
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

# Visualizar os resultados
ax = valor_fatura.plot(alpha=0.5, title="Custo de Consumo de Água com Previsão", ylabel="Valor da Fatura (R$)")
y_pred.plot(ax=ax, linewidth=3, label="Tendência", color='C0')
y_fore.plot(ax=ax, linewidth=3, label="Previsão", color='C3')
ax.legend()
plt.show()

#%%
X_fore.head(10)