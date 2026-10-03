import pandas as pd
import json
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()

print('Carregando bankchurners')

bank_churners = pd.read_csv('dataset/BankChurners.csv')

# removendo colunas que não serão utilizadas
bank_churners = bank_churners.drop(
    columns=[
        'CLIENTNUM',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'
    ]
)

# deixando os nomes das colunas em minúsculo
bank_churners.columns = bank_churners.columns.str.lower()

conexao = create_engine(
    os.getenv("DATABASE_URL")
)

print('Salvando bank_churners')

bank_churners.to_sql(
    'bank_churners',
    conexao,
    if_exists='replace',
    index=False
)

print('bank_churners salvo com sucesso!')

