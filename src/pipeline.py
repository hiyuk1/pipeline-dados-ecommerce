"""
Pipeline ETL para análise de dados de E-commerce.
Realiza ingestão, limpeza, análise exploratória, segmentação de clientes (RFM) e carga no MySQL.
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Define diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / 'dataset'
OUTPUT_DIR = ROOT_DIR / 'output'

# Carrega variáveis de ambiente do arquivo .env
load_dotenv(ROOT_DIR / '.env')

DB_USER = os.getenv('DB_USER')
DB_PASS = os.getenv('DB_PASS')
DB_HOST = os.getenv('DB_HOST')
DB_NAME = os.getenv('DB_NAME')

CONN_STRING = f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306'


def create_database_connection():
    """Cria conexão com o MySQL e garante que o banco de dados existe."""
    engine_temp = create_engine(CONN_STRING)
    with engine_temp.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
    return create_engine(f'{CONN_STRING}/{DB_NAME}')


def extract_and_clean_data(filepath: str) -> pd.DataFrame:
    """Extrai dados do CSV e realiza limpeza básica."""
    df = pd.read_csv(filepath, encoding='latin1')
    
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    return df


def generate_sales_chart(df: pd.DataFrame, output_path: str):
    """Gera gráfico de vendas por país."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.head(100), x='TotalAmount', y='Country', estimator=sum)
    plt.title('Vendas por País (Amostra)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica segmentação RFM (Recency, Frequency, Monetary) usando K-Means."""
    snapshot_date = df['InvoiceDate'].max() + pd.to_timedelta(1, 'D')
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'count',
        'TotalAmount': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalAmount': 'Monetary'
    })

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    rfm['Perfil_Cliente'] = rfm['Cluster'].map({0: 'Bronze', 1: 'Prata', 2: 'Ouro'})
    
    return rfm


def load_to_database(engine, df: pd.DataFrame, rfm: pd.DataFrame):
    """Carrega os dados processados no banco MySQL."""
    df.to_sql('fact_sales', engine, if_exists='replace', index=False, chunksize=1000)
    rfm.reset_index().to_sql('dim_customer_segmentation', engine, if_exists='replace', index=False)


def pipeline_completo():
    """Executa o pipeline ETL completo."""
    print(">>> 1. CONEXÃO COM BANCO DE DADOS...")
    engine = create_database_connection()
    print(f"Banco '{DB_NAME}' conectado com sucesso.")

    print("\n>>> 2. EXTRAÇÃO E LIMPEZA DOS DADOS...")
    df = extract_and_clean_data(DATA_DIR / 'data.csv')
    print(f"Dados processados: {len(df)} registros.")

    print("\n>>> 3. ANÁLISE EXPLORATÓRIA (EDA)...")
    generate_sales_chart(df, OUTPUT_DIR / 'eda_vendas_pais.png')
    print("Gráfico 'output/eda_vendas_pais.png' gerado.")

    print("\n>>> 4. SEGMENTAÇÃO DE CLIENTES (K-MEANS)...")
    rfm = segment_customers(df)
    print("Modelo de clusterização aplicado com sucesso.")

    print("\n>>> 5. CARGA NO MYSQL...")
    load_to_database(engine, df, rfm)
    print("Tabelas 'fact_sales' e 'dim_customer_segmentation' carregadas.")

    print("\n>>> PIPELINE FINALIZADO COM SUCESSO! <<<")
    print(f"Conecte o Power BI ao banco '{DB_NAME}' para visualização.")

if __name__ == "__main__":
    pipeline_completo()