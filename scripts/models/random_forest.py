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

# %% 
# Criando a validação cruzada para avaliar o desempenho do modelo.
# A validação cruzada divide o conjunto de dados em várias "dobras" (folds) para avaliar o modelo em diferentes subconjuntos.
validacao_cruzada = KFold(n_splits=10, shuffle=True, random_state=42)

# %% 
# Avalia o modelo com a validação cruzada utilizando o conjunto de validação (X_val, y_val).
# Isso nos dá uma estimativa da performance do modelo em dados desconhecidos.
cross_val_score(random_forest, X_val, y_val, cv=validacao_cruzada)

# %% 
# Calcula a média da acurácia das dobras de validação cruzada.
# A média é uma boa medida do desempenho geral do modelo.
acuracia_media_rf = cross_val_score(random_forest, X_val, y_val, cv=validacao_cruzada).mean()
print("Acurácia média do RandomForest:", acuracia_media_rf)

# %% 
# Após treinar o modelo e avaliá-lo, podemos salvar o modelo final em um arquivo para uso posterior.
# O arquivo será salvo com o nome 'base_line.joblib' para fácil carregamento e uso no futuro.
joblib.dump(random_forest, 'base_line.joblib')
