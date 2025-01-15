# streamlit_app.py
import streamlit as st
from modulos.importando_dados import importando_dados
from modulos.tratando_dados import filtrar_matricula, gerar_serie
from modulos.visualizacoes import plotar_serie_temporal, plotar_serie_e_previsao
from modulos.modelagem import prever_series
import joblib
# Carregar o modelo treinado
modelo_carregado = joblib.load(r'C:\Users\maype\Desktop\projetos\water_forecast\scripts\models\modelo_fatura.joblib')

# Configurações do app
st.title("Gestão de Consumo de Água - Prefeitura")

# Solicita o upload do arquivo CSV contendo a base de dados
uploaded_file = st.file_uploader("Carregue a base de dados", type=["csv"])

# Verifica se o arquivo foi carregado
if uploaded_file:
    # Carrega os dados do arquivo utilizando a função importando_dados
    data = importando_dados(uploaded_file)

    # Solicita ao usuário a matrícula para filtrar os dados
    matricula = st.text_input("Insira a matrícula:")

    if matricula:
        # Filtra os dados com base na matrícula fornecida pelo usuário
        filtered_data = filtrar_matricula(data, matricula)
        
        # Gera a série temporal com os dados filtrados
        series = gerar_serie(filtered_data)

        # Verifica se a série gerada não está vazia
        if not series.empty:
            # Plota a série temporal utilizando a função plotar_serie_temporal
            st.plotly_chart(plotar_serie_temporal(series))

            # Realiza a previsão da série temporal utilizando o modelo de previsão
            forecast, error = prever_series(series)

           
            # Exibe a previsão para o próximo mês, caso ela seja gerada com sucesso
            if forecast is not None:
                st.write(f"Previsão para o próximo mês: R$ {forecast:.2f}")

                # Exibe o erro da previsão, caso tenha sido calculado
                if error is not None:
                    st.write(f"Erro da previsão: {error:.2f}")

                # Exibe o gráfico com a previsão
                st.plotly_chart(plotar_serie_e_previsao(int(matricula), data, modelo_carregado))
            else:
                # Exibe um alerta caso não seja possível gerar a previsão
                st.warning("Não foi possível gerar a previsão. Verifique os dados da série temporal.")
        else:
            # Exibe um alerta caso a matrícula não seja encontrada nos dados
            st.warning("Matrícula não encontrada.")
