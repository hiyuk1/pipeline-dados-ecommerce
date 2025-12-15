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
from sklearn.metrics import silhouette_score

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
    rfm['Perfil_Cliente'] = rfm['Cluster'].map({0: 'Prata', 1: 'Bronze', 2: 'Ouro'})
    
    return rfm


def elbow_method(rfm_data: pd.DataFrame, output_path: str = None):
    """
    Aplica o Método do Cotovelo para encontrar o K ideal.
    
    Testa K de 2 a 10 e gera gráfico com inércia e Silhouette Score.
    Retorna DataFrame com os valores de K e métricas.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    print("\n>>> Calculando Método do Cotovelo (K=2 a K=10)...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(rfm_scaled, kmeans.labels_))
        print(f"K={k}: Inércia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
    
    # Cria DataFrame com resultados
    elbow_df = pd.DataFrame({
        'K': list(K_range),
        'Inertia': inertias,
        'Silhouette_Score': silhouette_scores
    })
    
    # Gera gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfico 1: Inércia (Elbow)
    ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=3, color='r', linestyle='--', label='K=3 (escolhido)')
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inércia')
    ax1.set_title('Método do Cotovelo - Inércia')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfico 2: Silhouette Score
    ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=3, color='r', linestyle='--', label='K=3 (escolhido)')
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score por K')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Gráfico salvo em: {output_path}")
    plt.close()
    
    return elbow_df


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

    print("\n>>> 4.1. VALIDAÇÃO COM MÉTODO DO COTOVELO...")
    elbow_results = elbow_method(rfm[['Recency', 'Frequency', 'Monetary']], OUTPUT_DIR / 'elbow_method.png')
    print("\nResultados do Método do Cotovelo:")
    print(elbow_results.to_string(index=False))

    print("\n>>> 5. CARGA NO MYSQL...")
    load_to_database(engine, df, rfm)
    print("Tabelas 'fact_sales' e 'dim_customer_segmentation' carregadas.")

    print("\n>>> PIPELINE FINALIZADO COM SUCESSO! <<<")
    print(f"Conecte o Power BI ao banco '{DB_NAME}' para visualização.")

if __name__ == "__main__":
    pipeline_completo()