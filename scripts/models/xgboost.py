# %% 
import pandas as pd 
import numpy as np 
import xgboost as xgb  # Importando o XGBoost
from sklearn.pipeline import Pipeline
from pre_processing import importar_dados, preprocessar_dados
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from scipy.stats import randint
import joblib

# %% 
# Utilizando as funções criadas para importar e pré-processar os dados
X_train, X_val, X_test, y_train, y_val, y_test = importar_dados() 
preprocessor = preprocessar_dados()

# %%
# Criando o modelo de regressão XGBoost
# O 'random_state=42' garante que os resultados sejam reprodutíveis.
model = xgb.XGBRegressor(random_state=42)

# %% 
# Pipeline
# O pipeline permite integrar o pré-processamento e o modelo em um único fluxo de trabalho.
# Isso facilita a execução do pré-processamento e a aplicação do modelo de forma sequencial e correta.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Primeiro, aplica o pré-processamento aos dados.
    ('regressor', model)  # Depois, treina o modelo XGBoost.
])

# %% 
# Definindo os parâmetros a serem testados durante a busca de hiperparâmetros.
# Estamos utilizando uma distribuição aleatória para parâmetros como 'n_estimators' e 'max_depth'.
param_distributions = {
    'n_estimators': randint(100, 1000),  # Número de árvores a serem usadas no modelo.
    'max_depth': randint(3, 15),  # Profundidade máxima de cada árvore.
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Taxa de aprendizado do modelo.
    'subsample': [0.8, 0.9, 1.0],  # Proporção de amostras usadas para cada árvore.
    'colsample_bytree': [0.8, 0.9, 1.0],  # Proporção de features usadas para cada árvore.
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
xgboost_model = random_search.best_estimator_

# %% 
# Criando a validação cruzada para avaliar o desempenho do modelo.
# A validação cruzada divide o conjunto de dados em várias "dobras" (folds) para avaliar o modelo em diferentes subconjuntos.
validacao_cruzada = KFold(n_splits=10, shuffle=True, random_state=42)

# %% 
# Avalia o modelo com a validação cruzada utilizando o conjunto de validação (X_val, y_val).
# Isso nos dá uma estimativa da performance do modelo em dados desconhecidos.
cross_val_score(xgboost_model, X_val, y_val, cv=validacao_cruzada)

# %% 
# Calcula a média da acurácia das dobras de validação cruzada.
# A média é uma boa medida do desempenho geral do modelo.
acuracia_media_xgb = cross_val_score(xgboost_model, X_val, y_val, cv=validacao_cruzada).mean()
print("Acurácia média do XGBoost:", acuracia_media_xgb)

# %% 
# Após treinar o modelo e validá-lo, podemos salvar o modelo final em um arquivo para uso posterior.
# O arquivo será salvo com o nome 'xgboost_model.joblib' para fácil carregamento e uso no futuro.
joblib.dump(xgboost_model, 'xgboost_model.joblib')
