# streamlit_app.py
import streamlit as st
from modulos.importando_dados import importando_dados
from modulos.tratando_dados import filtrar_matricula, gerar_serie
from modulos.visualizacoes import plotar_serie_temporal, plotar_serie_e_previsao
from modulos.modelagem import prever_series
import joblib

# Carregar o modelo treinado
modelo_carregado = joblib.load("scripts/models/base_line.joblib")

# Configurações do app
st.title("Gestão de Consumo de Água - Prefeitura")


data = importando_dados()

# Verifica se o arquivo foi carregado
if data is not None:
    # Carrega os dados do arquivo utilizando a função importando_dados
    

    # Seleciona a matrícula com base nos dados disponíveis
    matriculas_disponiveis = data['MATRICULA'].unique()
    # Campo de pesquisa
    termo_pesquisa = st.text_input("Pesquise uma matrícula:")

    #Filtrar as opções de matrícula com base no termo de pesquisa
    opcoes_filtradas = [mat for mat in matriculas_disponiveis if termo_pesquisa in mat]

    # Selectbox com as opções filtradas
    matricula = st.selectbox("Selecione a matrícula:", opcoes_filtradas)

    st.write(f"Matrícula selecionada: {matricula}")

    if matricula:
        # Filtra os dados com base na matrícula fornecida pelo usuário
        filtered_data = filtrar_matricula(data, matricula)
        
        # Gera a série temporal com os dados filtrados
        series = gerar_serie(filtered_data)

        # Verifica se a série gerada não está vazia
        if not series.empty:
            # Plota a série temporal utilizando a função plotar_serie_temporal
            st.plotly_chart(plotar_serie_temporal(series), use_container_width=True)

            # Realiza a previsão da série temporal utilizando o modelo de previsão
            forecast, error = prever_series(series)

            # Plota a série temporal com previsão
            fig = plotar_serie_e_previsao(matricula, filtered_data, forecast)
            st.plotly_chart(fig, use_container_width=True)

            # Exibe a previsão e o erro, se disponível
            if forecast is not None:
                st.write(f"**Previsão para o próximo mês:** R$ {forecast:.2f}")
            if error is not None:
                st.write(f"**Erro da previsão:** {error:.2f}")
        else:
            st.warning("Não há dados suficientes para gerar a série temporal desta matrícula.")
else:
    st.info("Verifique os arquivos importados no código fonte.")
