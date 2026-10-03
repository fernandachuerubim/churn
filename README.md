# Credit Customers — Previsão de Churn

Projeto de Machine Learning para prever o **churn** (abandono) de clientes de cartão de
crédito, usando o dataset [BankChurners](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
do Kaggle e o resultado no streamlit pelo link https://fc-churn.streamlit.app/.

A solução é composta por três partes:

| Parte | Arquivo | Descrição |
| --- | --- | --- |
| Carga no banco | `main.py` | Lê o CSV, limpa as colunas e grava a tabela `bank_churners` no PostgreSQL |
| Estudo do modelo | `notebook/churn.ipynb` | Análise exploratória, pré-processamento e comparação de 3 algoritmos com `GridSearchCV` |
| Aplicação | `home.py` | App Streamlit com formulário de previsão e dashboard dos clientes |

## Stack

- **Python 3.14** gerenciado com [uv](https://docs.astral.sh/uv/)
- **pandas** / **SQLAlchemy** / **psycopg2** — carga e leitura dos dados
- **scikit-learn** — pré-processamento, treinamento e métricas
- **joblib** — persistência do modelo (`modelo/churn.joblib`)
- **Streamlit** + **Plotly** — interface web e gráficos
- **matplotlib** / **seaborn** — visualizações no notebook

## Estrutura do projeto

```
credit_customers/
├── dataset/
│   └── BankChurners.csv     # dataset original (10.127 linhas x 23 colunas)
├── modelo/
│   └── churn.joblib         # pipeline treinado (sklearn Pipeline + joblib)
├── notebook/
│   └── churn.ipynb          # EDA e experimentos de modelagem
├── .env                     # variáveis de conexão com o banco (não versionado)
├── dicionario.json          # dicionário de dados das variáveis
├── home.py                  # app Streamlit
├── main.py                  # carga do CSV para o PostgreSQL
├── pyproject.toml           # dependências do projeto
└── uv.lock                  # lockfile das dependências
```

## Configuração

### 1. Dependências

```bash
uv sync
```

### 2. Variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto com a conexão do banco:

```env
DATABASE_URL=postgresql+psycopg2://USUARIO:SENHA@HOST:PORTA/NOME_DO_BANCO
```

### 3. Carga dos dados

`main.py` lê `dataset/BankChurners.csv`, remove as duas colunas do classificador
Naive Bayes (variáveis alvo vazadas pelo próprio modelo), normaliza os nomes das
colunas para minúsculas e cria/substitui a tabela `bank_churners`:

```bash
uv run python main.py
```

## Modelo

O estudo no notebook compara três algoritmos com validação cruzada (`cv=3`) e
`GridSearchCV`, usando `f1_weighted` como métrica de seleção:

| Modelo | Hiperparâmetros testados |
| --- | --- |
| Decision Tree | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Logistic Regression | `C`, `solver`, `max_iter` |
| KNN | `n_neighbors`, `weights` |

O pipeline salvo em `modelo/churn.joblib` é o da **Regressão Logística** (`C=10`,
`solver='liblinear'`) e é compuesto por:

1. `ColumnTransformer` — `MinMaxScaler` para as 14 variáveis numéricas e
   `OneHotEncoder` para as 5 categóricas;
2. `LogisticRegression` — classificador binário entre `Existing Customer` e
   `Attrited Customer`.

### Reexecutando o treinamento

```bash
uv run jupyter lab notebook/churn.ipynb
```

O notebook lê o dataset em `../dataset/BankChurners.csv` e grava o modelo em
`../modelo/churn.joblib`. **A ordem das células importa:** a célula que faz
`joblib.dump` da Regressão Logística precisa ser executada por último, caso
contrário o modelo salvo será outro.

## Aplicação

```bash
uv run streamlit run home.py
```

O app tem duas abas:

- **1º Painel — Machine Learning**: formulário com as 19 variáveis de entrada
  (sliders, campos numéricos e selects) que retorna a classe prevista e a
  probabilidade de cada uma — `🟢` baixo risco / `🔴` alto risco de churn.
- **2º Painel — Dashboard**: gráficos da distribuição de clientes por gênero,
  nível de educação, estado civil, faixa de renda e categoria de cartão.

> Os caminhos de `dataset/` e `modelo/` são relativos: execute `main.py` e
> `streamlit run home.py` sempre a partir da raiz do projeto.

## Dados

Fonte: [Kaggle — Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

As 23 colunas originais cobrem dados demográficos (`customer_age`, `gender`,
`education_level`, `marital_status`, `income_category`), do produto
(`card_category`), o relacionamento com o banco (`months_on_book`,
`total_relationship_count`) e o comportamento de uso do cartão
(`months_inactive_12_mon`, `contacts_count_12_mon`, `credit_limit`,
`total_revolving_bal`, `total_trans_amt`, entre outras).

A descrição completa de cada campo está em [`dicionario.json`](dicionario.json).
O alvo é `attrition_flag`, que assume os valores `Existing Customer` e
`Attrited Customer`.
