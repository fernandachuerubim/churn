import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
import joblib
import plotly.express as px

st.set_page_config(page_title='Crédito de Clientes', layout='wide')

st.subheader('Conheça abaixo o 1º Painel Machine Learning e o 2º Painel Dashboard', text_alignment='center')

st.markdown("""
<style>
/* Container das abas */
div[data-testid="stTabs"] > div {
    width: 100%;
}

/* Lista das abas */
div[data-testid="stTabs"] [role="tablist"] {
    width: 100%;
    display: flex;
}

/* Cada aba */
div[data-testid="stTabs"] [role="tab"] {
    flex: 1 1 0 !important;
    width: 100% !important;
    max-width: none !important;
    justify-content: center !important;
    font-size: 30px !important;
    min-height: 60px !important;
    padding: 15px 30px !important;
}

/* Texto dentro da aba */
div[data-testid="stTabs"] [role="tab"] p {
    font-size: 30px !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["1º Painel - Machine Learning", "2º Painel - Dashboard"])

load_dotenv()

@st.cache_resource
def load_data():
    conexao = create_engine(
        os.getenv('DATABASE_URL')
    )

    query = 'SELECT * FROM bank_churners'
    df = pd.read_sql(query, conexao)
    return df

@st.cache_resource() #armazena o resultado da função
def load_model():
    modelo = joblib.load('Modelo/churn.joblib')
    return modelo

df = load_data()

with tab1:
    st.header('📊Previsão de Churn dos Clientes de Cartão de Crédito')

    st.subheader('Selecione as opções para rodar a Previsão do Modelo abaixo:')

    with st.form('predicao_form', border=False):
        idade = st.slider(
            label='Idade',
            min_value=18,
            max_value=70
        )

        dependentes = st.slider(
            label='Número de Dependentes',
            max_value=12
        )

        relacionamento = st.slider(
            label='Período de Relacionamento',
            max_value=60
        )

        total_produtos=st.slider(
            label='Número Total de Produtos',
            max_value=10
        )

        meses_inativo = st.slider(
            label='Número de Meses de Inatividade',
            max_value=12
        )

        numero_contatos = st.slider(
            label='Número de Contatos',
            max_value=10
        )

        credito = st.slider(
            label='Limite de Crédito',
            max_value=35_000,
            step=500
        )

        saldo_rotativo = st.slider(
            label='Saldo Rotativo',
            max_value=25_000,
            step=500
        )

        linha_credito = st.slider(
            label='Linha de Crédito',
            max_value=35_000,
            step=500
        )

        var_transacao = st.number_input(
            label='Variação no Valor das Transações',
            min_value=0.0,
            max_value=3.0,
            step=0.5
        )

        valor_total_trans = st.slider(
            label='Valor total das transações',
            min_value=500,
            max_value=20_000,
            step=500
        )

        qtd_trans = st.slider(
            label='Quantidade de Transações',
            max_value=140
        )

        var_num_trans = st.number_input(
            label='Variação no Número de Transações',
            min_value=0.0,
            max_value=4.0,
            step=0.5
        )

        taxa_utilizacao = st.number_input(
            label='Taxa Média de Utilização do Cartão',
            min_value=0.0,
            max_value=1.0,
            step=0.1
        )

        sexo = st.selectbox(
            label='Sexo',
            options=df['gender'].unique()
        )

        education = st.selectbox(
            label='Nível de Educação',
            options=df['education_level'].unique()
        )

        estado_civil = st.selectbox(
            label='Estado Civil',
            options=df['marital_status'].unique()
        )

        categoria_renda = st.selectbox(
            label='Categoria de Renda',
            options=df['income_category'].unique()
        )

        categoria_cartao = st.selectbox(
            label='Categoria Cartão',
            options=df['card_category'].unique()
        )


        st.markdown("""
            <style>
            button[kind="secondaryFormSubmit"] p {
            font-size: 23px !important;
            }
            </style>
            """, unsafe_allow_html=True)
        
        prever = st.form_submit_button('⏳Rodar Previsão 👈 clique aqui', use_container_width=True)

        if prever:
            dados = pd.DataFrame({
                'customer_age': [idade],
                'dependent_count': [dependentes],
                'months_on_book': [relacionamento],
                'total_relationship_count': [total_produtos],
                'months_inactive_12_mon': [meses_inativo],
                'contacts_count_12_mon': [numero_contatos],
                'credit_limit': [credito],
                'total_revolving_bal': [saldo_rotativo],
                'avg_open_to_buy': [linha_credito],
                'total_amt_chng_q4_q1': [var_transacao],
                'total_trans_amt': [valor_total_trans],
                'total_trans_ct': [qtd_trans],
                'total_ct_chng_q4_q1': [var_num_trans],
                'avg_utilization_ratio': [taxa_utilizacao],
                'gender': [sexo],
                'education_level': [education],
                'marital_status': [estado_civil],
                'income_category': [categoria_renda],
                'card_category': [categoria_cartao]
            })

            modelo = load_model()

            pred = modelo.predict(dados)
            pred_prob = modelo.predict_proba(dados)

            # Classes que o modelo conhece
            classes = modelo.classes_

            # Probabilidade de cada classe
            probabilidades = dict(zip(classes, pred_prob[0]))

            # Classe prevista
            resultado = pred[0]

            st.write("### Resultado da previsão")
            st.write(f"Classe prevista: **{resultado}**")

            st.write("### Probabilidades")

            for classe, probabilidade in probabilidades.items():
                st.write(f"**{classe}: {probabilidade:.2%}**")

                if classe == resultado:
                    churn = probabilidade

            if pred == 'Existing Customer':
                st.write(
                    f"🟢 Resultado Não é churn / Baixo risco: o Modelo Estatístico chamado Regressão Logística não identificou sinais relevantes de abandono com {churn:.2%} da classe Existing Customer."
                )
            else:
                st.write(
                    f"🔴 Resultado de Churn / Alto risco: o Modelo Estatístico chamado Regressão Logística identificou características e comportamentos associados a clientes com maior probabilidade de abandono com {churn:.2%} da classe Attrited Customer."
                )
    
with tab2:
    aux = df['gender'].value_counts().reset_index()

    fig = px.pie(
        aux,
        values='count',
        names='gender',
        title='Clientes de Cartão de Crédito por Gênero'
    )

    fig.update_layout(title_x=0.4)

    st.plotly_chart(fig)

    aux = (df[['education_level', 'gender']]
           .groupby(['education_level', 'gender'])
           .size()
           .reset_index(name='count')
    )
    
    fig = px.bar(
        aux,
        x='education_level',
        y='count',
        color='gender',
        barmode='group',
        title='Clientes de Cartão de Crédito por Nível de Educação',
        labels={'education_level': 'Nível de Educação', 'count': 'Quantidade'}
    )

    fig.update_layout(title_x=0.4)

    st.plotly_chart(fig)

    aux = (df[['marital_status', 'gender']]
           .groupby(['marital_status', 'gender'])
           .size()
           .reset_index(name='count')
    )
    
    fig = px.bar(
        aux,
        x='marital_status',
        y='count',
        color='gender',
        barmode='group',
        title='Clientes de Cartão de Crédito por Estado Civil e Gênero',
        labels={'education_level': 'Estado Civil', 'count': 'Quantidade'}
    )

    fig.update_layout(title_x=0.4)

    st.plotly_chart(fig)

    aux = (df[['income_category', 'gender']]
           .groupby(['income_category', 'gender'])
           .size()
           .reset_index(name='count')
    )
    
    fig = px.bar(
        aux,
        x='income_category',
        y='count',
        color='gender',
        barmode='group',
        title='Clientes de Cartão de Crédito por Faixa de Renda e Gênero',
        labels={'income_category': 'Faixa de Renda', 'count': 'Quantidade'}
    )

    fig.update_layout(title_x=0.4)

    st.plotly_chart(fig)

    aux = (df[['card_category', 'gender']]
           .groupby(['card_category', 'gender'])
           .size()
           .reset_index(name='count')
    )
    
    fig = px.bar(
        aux,
        x='card_category',
        y='count',
        color='gender',
        barmode='group',
        title='Clientes por Categoria de Cartão de Crédito e Gênero',
        labels={'card_category': 'Categoria do Cartão', 'count': 'Quantidade'}
    )

    fig.update_layout(title_x=0.4)

    st.plotly_chart(fig)








        