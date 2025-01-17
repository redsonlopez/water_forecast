# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 


def importar_dados(): 
    """
    Importa os dados para o projeto

    Returns:
        data: dataframe pandas com os dados para o modelo
    """
    # Carregar os dados
    X_train= pd.read_csv("Train&TestData\\X_train.csv")
    X_val= pd.read_csv("Train&TestData\\X_val.csv")
    X_test= pd.read_csv("Train&TestData\\X_test.csv")
    y_train= pd.read_csv("Train&TestData\\y_train.csv")
    y_val= pd.read_csv("Train&TestData\\y_val.csv")
    y_test= pd.read_csv("Train&TestData\\y_test.csv")
    
    return  X_train, X_val, X_test, y_train, y_val, y_test


def separar_base(data): 
    """Separa a base entre treino validação, treino e teste 

    Args:
        data (dataframe): dataframe pandas com os dados 

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = data.drop(columns=["VALOR_FATURA"])
    y = data["VALOR_FATURA"]

    # treino+validação e teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # treino+validação em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    # A validação será 20% (0.25 * 0.8) dos dados totais.

    print("Tamanho do treino:", X_train.shape)
    print("Tamanho da validação:", X_val.shape)
    print("Tamanho do teste:", X_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
def preprocessar_dados():

    """
    Transformer para pré-processamento dos dados. Recebe um dataframe como parametro e aplica as transformações
    listadas. 
    
    """    
        
    # Selecionando variaveis numericas, categoricas e variaveis que devem ser removidas para evitar data leakage
    numerical_features = ['VALOR_FATURA']
    categorical_features = ['BAIRRO']
    ordinal_features = ['ANO_VENCIMENTO', 'MES_VENCIMENTO']
    index = ['DATA_FATURA']
    drop_features = ['Unnamed: 0', 'MATRICULA', 'VOLUME_FATURA_AGUA',
                    'VOLUME_FATURA_ESGOTO', 'VOLUME_MEDIDO_AGUA', 'VOLUME_MEDIDO_ESGOTO']
    
    # Pré-processamento
    # Configura um transformador para aplicar diferentes preprocessamentos às colunas:
    # - StandardScaler: Normaliza valores numéricos (para que todos tenham média 0 e desvio padrão 1).
    # - OneHotEncoder: Converte valores categóricos em formato binário.
    preprocessor = ColumnTransformer(
        transformers=[
            # Aplica escalonamento aos dados numéricos.
            ('num', StandardScaler(), numerical_features),
            # Aplica codificação one-hot aos dados categóricos.
            ('cat', OneHotEncoder(), categorical_features),
            # Não vamos fazer nada com as variaveis ordinais porque elas já estão no formato esperado
            # Removendo colunas para evitar o data leakage
            ('drop', "drop", drop_features)
        
        ]

    )

    
    return preprocessor


