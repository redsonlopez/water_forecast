#%% 
import pandas as pd 
import numpy as np 

pd.set_option('display.max_rows', None)

#%% 
data = pd.read_csv("..\\..\\data\\processed\\processed_water.csv")
# %%
data.info()


#%% 
# Deixando no typo correto 
data[['VALOR_FATURA','VOLUME_FATURA_AGUA','VOLUME_FATURA_ESGOTO', 'VOLUME_MEDIDO_AGUA', 'VOLUME_MEDIDO_ESGOTO']] = data[['VALOR_FATURA','VOLUME_FATURA_AGUA','VOLUME_FATURA_ESGOTO', 'VOLUME_MEDIDO_AGUA', 'VOLUME_MEDIDO_ESGOTO']].astype('float64')
data[['BAIRRO', 'MATRICULA']] = data[['BAIRRO', 'MATRICULA']].astype('category')
data['DATA_VENCIMENTO'] = pd.to_datetime(data['DATA_VENCIMENTO'])
# %%
# Verificando os valores em todas as colunas
colunas = data.columns
for col in colunas: 
    print(f"Valores da coluna {col}")
    print()
    print(data[col].value_counts())
    print("-"*40)

# %%
# Removendo todas as faturas que estão zeradas:
data_filtrada = data[data['VALOR_FATURA'] != 0]
data.head()
# %%

# Extraindo Ano e Mês de vencimento das faturas

data['ANO_VENCIMENTO'] = data['DATA_VENCIMENTO'].dt.year
data['MES_VENCIMENTO'] = data['DATA_VENCIMENTO'].dt.month

# %% 
data.head()
