# streamlit_app.py
import streamlit as st
from modulos.importando_dados import importando_dados
from modulos.tratando_dados import filtrar_matricula, gerar_serie
from modulos.visualizacoes import plotar_serie_e_previsao
from modulos.modelagem import prever_series
import joblib

st.set_page_config(
    page_title="Consumo de Água - PBH",
    layout="wide",
)

st.sidebar.header("Prefeitura de Belo Horizonte")
view_option = st.sidebar.radio('Escolha o tipo de exibição', ('Introdução', 'Previsão'))

# Carregar o modelo treinado
modelo_carregado = joblib.load("scripts/models/base_line.joblib")

# Configurações do app
st.title("Previsão do Consumo de Água")

data = importando_dados()

if view_option == 'Introdução':
    st.header("Introdução ao Painel de Previsão de Custos do Consumo de Água")
    st.markdown("""
Bem-vindo ao **Painel de Previsão de Custos do Consumo de Água**! Este painel foi desenvolvido para oferecer uma análise aprofundada e previsões detalhadas sobre os custos relacionados ao consumo de água na Prefeitura de Belo Horizonte. Utilizando técnicas avançadas de análise de dados e visualização interativa, o objetivo é apoiar a tomada de decisões estratégicas e o planejamento eficiente de recursos.

### Contexto dos Dados:
Os dados utilizados neste painel foram coletados a partir de registros históricos do consumo de água em diferentes unidades da prefeitura. Com base nesses dados, foram aplicados modelos preditivos para estimar os custos futuros e identificar tendências de consumo.

### Funcionalidades do Painel:
- **Previsões de Custos**: Visualize as projeções de custos futuros com base nos padrões históricos de consumo.
- **Análise Comparativa**: Compare os custos de diferentes períodos para identificar variações e possíveis pontos de economia.
- **Gráfico Interativo**: Explore o gráfico de linhas para visualizar padrões e tendências de forma clara e intuitiva.

### Utilização do Painel:
- Utilize o filtro disponível na barra lateral para selecionar a matrícula da unidade de interesse.
- Analise o gráfico para identificar oportunidades para redução de custos.

Este painel foi criado com o propósito de otimizar a gestão dos recursos públicos. Esperamos que ele seja uma ferramenta valiosa para o planejamento estratégico e a sustentabilidade das operações da prefeitura.
""")

elif view_option == 'Previsão':

    st.sidebar.header("Filtro")

    # Verifica se o arquivo foi carregado
    if data is not None:
        # Carrega os dados do arquivo utilizando a função importando_dados
        
        # Seleciona a matrícula com base nos dados disponíveis
        matriculas_disponiveis = data['MATRICULA'].unique()

        #Filtrar as opções de matrícula com base no termo de pesquisa
        opcoes_filtradas = [mat for mat in matriculas_disponiveis]

        # Selectbox com as opções filtradas
        matricula = st.sidebar.selectbox("Selecione ou pesquise uma matrícula:", opcoes_filtradas)

        if matricula:
            # Filtra os dados com base na matrícula fornecida pelo usuário
            filtered_data = filtrar_matricula(data, matricula)
            
            # Gera a série temporal com os dados filtrados
            series = gerar_serie(filtered_data)

            # Verifica se a série gerada não está vazia
            if not series.empty:

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
