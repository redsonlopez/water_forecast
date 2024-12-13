import pandas as pd

columns= [
        "MATRICULA",
        "VALOR",
        "VOL_FATURA_AGUA",
        "VOL_FATURA_ESGOTO",
        "VOL_MEDIDO_AGUA",
        "VOL_MEDIDO_ESGOTO",
        "CLIENTE",
        "HIDROMETRO",
        "DATA_VENCIMENTO",
        "TIPO_LOGRADOURO",
        "NOME_LOGRADOURO",
        "NUM_IMOVEL",
        "COMPLEMENTO_",
        "INF_COMPLEMENTO",
        "BAIRRO",
        "residuo"
]

files= []
for year in range(2022, 2025):
    for month in range(1, 13):
        if year == 2024 and month == 12:
            break
        files.append(f"{year}{month:02d}_PBH.txt")

file_list= []
for file in files:
        df= pd.read_csv(f"../../data/raw/{file}", sep=";", encoding="latin1", header=None)
        file_list.append(df)

df= pd.concat(file_list, ignore_index=True)
df.columns= columns

#df.to_csv("../../data/processed/water.csv", index=False, encoding="utf-8")

print(df)