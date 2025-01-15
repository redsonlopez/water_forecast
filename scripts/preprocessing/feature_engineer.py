#%% 
import pandas as pd 
import numpy as np 

pd.set_option('display.max_rows', None)

#%% 
data = pd.read_csv("..\\..\\data\\processed\\processed_test.csv")
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
data = data[data['VALOR_FATURA'] != 0]
data.head()
# %%

# Extraindo Ano e Mês de vencimento das faturas

data['ANO_VENCIMENTO'] = data['DATA_VENCIMENTO'].dt.year
data['MES_VENCIMENTO'] = data['DATA_VENCIMENTO'].dt.month

# %% 
data.head()

#%% [markdown]
# > Como Veremos abaixo, existem diversos valores onde todos os valores correspondentes 
# > ao volume de **AGUA** e **ESGOTO** estão zerados, porém temos valores para as faturas nas mesmas instâncias
# > oque pode caracterizar inconsistencias nos dados. 
# > Se possível, é recomendavel validar este tipo de inconsistencia com um especialista do domínio em questão.  
# %%
 
data_zero = data[
    (data['VOLUME_FATURA_AGUA'] == 0) &
    (data['VOLUME_MEDIDO_AGUA'] == 0) &
    (data['VOLUME_FATURA_ESGOTO'] == 0) &
    (data['VOLUME_MEDIDO_ESGOTO'] == 0)]

data_zero.head()
# %%
data_zero['VALOR_FATURA'].value_counts()
# %%
data.to_csv("..\\..\\data\\processed\\final_data_test.csv")

# %%
