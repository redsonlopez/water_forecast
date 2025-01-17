# %% 
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestRegressor  
from sklearn.pipeline import Pipeline
from pre_processing import importar_dados, preprocessar_dados
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from scipy.stats import randint
import joblib

# %% 
# Utilizando as funções criadas para importar e pré-processar os dados
# A função 'importar_dados' divide os dados em treino, validação e teste.
# A função 'preprocessar_dados' aplica transformações (como escalonamento e codificação) nos dados.
X_train, X_val, X_test, y_train, y_val, y_test = importar_dados() 
preprocessor = preprocessar_dados()

# %%
# Criando o modelo de regressão RandomForestRegressor
# O 'random_state=42' garante que os resultados sejam reprodutíveis.
model = RandomForestRegressor(random_state=42)

# %% 
# Pipeline
# O pipeline permite integrar o pré-processamento e o modelo em um único fluxo de trabalho.
# Isso facilita a execução do pré-processamento e a aplicação do modelo de forma sequencial e correta.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Primeiro, aplica o pré-processamento aos dados.
    ('regressor', model)  # Depois, treina o modelo RandomForest.
])

# %% 
# Definindo os parâmetros a serem testados durante a busca de hiperparâmetros.
# Estamos utilizando uma distribuição aleatória para parâmetros como 'n_estimators' e 'max_depth'.
param_distributions = {
    'n_estimators': randint(100, 1000),  # Número de árvores a serem usadas no modelo.
    'max_depth': randint(10, 50),  # Profundidade máxima de cada árvore.
    'min_samples_split': randint(2, 20),  # Número mínimo de amostras para dividir um nó.
    'min_samples_leaf': randint(1, 10),  # Número mínimo de amostras em um nó folha.
    'max_features': ['sqrt', 'log2', None],  # Como selecionar as features para dividir os nós.
}

# RandomizedSearchCV realiza uma busca aleatória nos parâmetros definidos acima.
# Ele testa combinações aleatórias e avalia qual combinação tem o melhor desempenho.
random_search = RandomizedSearchCV(
    pipeline,  # O pipeline com o pré-processamento e o modelo.
    param_distributions,  # O espaço de busca para os hiperparâmetros.
    n_iter=10,  # Número de iterações para testar combinações aleatórias.
    cv=3,  # Número de divisões (folds) para validação cruzada durante a busca.
    scoring='neg_mean_absolute_error',  # A métrica a ser otimizada (erro absoluto médio negativo).
    n_jobs=-1,  # Utiliza todos os núcleos de processamento disponíveis para acelerar a busca.
    random_state=42  # Garante a reprodutibilidade da busca aleatória.
)

# Realiza a busca de hiperparâmetros ajustando o modelo aos dados de treinamento.
random_search.fit(X_train, y_train)
# Exibe os melhores parâmetros encontrados durante a busca.
print("Melhores hiperparâmetros:", random_search.best_params_)

# %% 
# Após a busca de hiperparâmetros, obtemos o modelo final treinado com os melhores parâmetros.
random_forest = random_search.best_estimator_

# %% [markdown]
# ### Avaliando Resultado dos modelos
#%%
# Avalia o modelo com a validação cruzada
validacao_cruzada = KFold(n_splits=10, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(random_forest, X_val, y_val, cv=validacao_cruzada)
acuracia_media_rf = cross_val_scores.mean()
print(cross_val_scores)
print("Acurácia média do Random Forest:", acuracia_media_rf)

# Faz previsões com o pipeline ajustado
y_pred = random_forest.predict(X_val)

# %% [markdown]
# 1. Mean Absolute Error (MAE)
# O MAE calcula o erro médio absoluto entre as previsões e os valores reais. Ele é útil para entender o erro médio sem considerar a direção (positivo ou negativo).
#%%
from sklearn.metrics import mean_absolute_error

# y_val: valores reais, y_pred: valores previstos
mae = mean_absolute_error(y_val, y_pred)

# Exibindo o valor do MAE
print(f"Mean Absolute Error (MAE): {mae}")


# %% [markdown]
# 2. Mean Squared Error (MSE)
# O MSE calcula o erro quadrado médio. Ele penaliza mais os erros grandes, pois a diferença entre o valor real e o previsto é elevada ao quadrado
#%%
from sklearn.metrics import mean_squared_error

# y_val: valores reais, y_pred: valores previstos
mse = mean_squared_error(y_val, y_pred)

# Exibindo o valor do MSE
print(f"Mean Squared Error (MSE): {mse}")

# %% [markdown]
# 3. Root Mean Squared Error (RMSE)
# O RMSE é a raiz quadrada do MSE. Ele traz a medida do erro para a mesma escala dos dados originais, o que torna mais fácil de interpretar.

#%%
import numpy as np
# Calculando o RMSE como a raiz quadrada do MSE
rmse = np.sqrt(mse)

# Exibindo o valor do RMSE
print(f"Root Mean Squared Error (RMSE): {rmse}")


# %% [markdown]
# 4. Coefficient of Determination (R²)
# O R² (ou coeficiente de determinação) mede a proporção da variação nos dados que o modelo é capaz de explicar. Um valor de 1 significa explicação perfeita, enquanto um valor de 0 significa que o modelo não explicou nada além da média.
#%%
from sklearn.metrics import r2_score

# y_val: valores reais, y_pred: valores previstos
r2 = r2_score(y_val, y_pred)

# Exibindo o valor do R²
print(f"Coefficient of Determination (R²): {r2}")





# %% [markdown]
# 5. Explained Variance Score
# A Expained Variance Score calcula a variação explicada pelo modelo. Um valor próximo de 1 significa que o modelo explicou a maior parte da variação dos dados.
#%%
from sklearn.metrics import explained_variance_score

# y_val: valores reais, y_pred: valores previstos
explained_variance = explained_variance_score(y_val, y_pred)

# Exibindo o valor do Explained Variance Score
print(f"Explained Variance Score: {explained_variance}")

# %% 
# Após treinar o modelo e validá-lo, podemos salvar o modelo final em um arquivo para uso posterior.
# O arquivo será salvo com o nome 'random_forest.joblib' para fácil carregamento e uso no futuro.
joblib.dump(random_forest, 'random_forest.joblib')