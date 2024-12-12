import pandas as pd

files= "202201_PBH.txt  202206_PBH.txt  202211_PBH.txt  202304_PBH.txt  202309_PBH.txt  202402_PBH.txt  202407_PBH.txt  202202_PBH.txt  202207_PBH.txt  202212_PBH.txt  202305_PBH.txt  202310_PBH.txt  202403_PBH.txt  202408_PBH.txt  202203_PBH.txt  202208_PBH.txt  202301_PBH.txt  202306_PBH.txt  202311_PBH.txt  202404_PBH.txt  202409_PBH.txt  202204_PBH.txt  202209_PBH.txt  202302_PBH.txt  202307_PBH.txt  202312_PBH.txt  202405_PBH.txt  202410_PBH.txt  202205_PBH.txt  202210_PBH.txt  202303_PBH.txt  202308_PBH.txt  202401_PBH.txt  202406_PBH.txt  202411_PBH.txt".split()

file_list= []
for file in files:
	df= pd.read_csv(f"~/Projects/water_prediction/data/raw/{file}")
	file_list.append(df)

print(file_list)

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


