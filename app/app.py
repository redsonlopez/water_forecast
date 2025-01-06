import streamlit as st
from modulos.importando_dados import importando_dados
from modulos.tratando_dados import filtrar_matricula, gerar_serie
from modulos.visualizacoes import plotar_serie_temporal
from modulos.modelagem import prever_series

# Configurações do app
st.title("Gestão de Consumo de Água - Prefeitura")

# Upload de arquivo
uploaded_file = st.file_uploader("Carregue a base de dados", type=["csv"])

if uploaded_file:
    # Carregar dados
    data = importando_dados(uploaded_file)

    # Entrada do usuário
    matricula = st.text_input("Insira a matrícula:")
    
    if matricula:
        # Processamento
        filtered_data = filtrar_matricula(data, matricula)
        series = gerar_serie(filtered_data)

        if not series.empty:
            # Visualização
            st.plotly_chart(plotar_serie_temporal(series))
            
            # Previsão
            forecast, error = prever_series(series)

            if forecast is not None:
                st.write(f"Previsão para o próximo mês: {forecast:.2f}")
                st.write(f"Erro da previsão: {error:.2f}")
            else:
                st.warning("Não foi possível gerar a previsão. Verifique os dados da série temporal.")
        else:
            st.warning("Matrícula não encontrada.")
