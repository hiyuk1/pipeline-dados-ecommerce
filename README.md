# Pipeline ETL - E-commerce Analytics

Pipeline de dados para análise de e-commerce com segmentação de clientes usando Machine Learning.

## 🚀 Tecnologias

- **Python** - Linguagem principal
- **Pandas** - Manipulação e limpeza de dados
- **Matplotlib/Seaborn** - Visualização de dados
- **Scikit-learn** - Machine Learning (K-Means clustering)
- **SQLAlchemy/PyMySQL** - Conexão com banco de dados
- **MySQL** - Banco de dados relacional
- **Power BI** - Dashboard de visualização

## 📊 O que o pipeline faz?

1. **Extração** - Lê dados de vendas de um arquivo CSV
2. **Limpeza** - Remove valores nulos e inconsistentes
3. **Análise Exploratória** - Gera gráficos de vendas por país
4. **Segmentação RFM** - Classifica clientes em Bronze, Prata e Ouro usando K-Means
5. **Carga** - Envia os dados processados para o MySQL

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/hiyuk1/pipeline-dados-ecommerce.git
cd pipeline-dados-ecommerce
```

2. Crie um ambiente virtual e ative:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas credenciais do MySQL
```

5. Execute o pipeline:
```bash
python src/pipeline.py
```

## 📁 Estrutura do Projeto

```
pipeline/
├── dataset/
│   └── data.csv              # Dados de vendas (fonte)
├── output/
│   ├── eda_vendas_pais.png   # Gráfico gerado
│   └── graficos.pbix         # Dashboard Power BI
├── src/
│   └── pipeline.py           # Script principal do ETL
├── .env                      # Credenciais (não commitado)
├── .env.example              # Template de credenciais
├── .gitignore                # Arquivos ignorados pelo git
├── requirements.txt          # Dependências do projeto
└── README.md                 # Documentação
```

## 📈 Resultados

- **397.884** registros processados
- **3 clusters** de clientes identificados (Bronze, Prata, Ouro)
- Tabelas geradas no MySQL: `fact_sales` e `dim_customer_segmentation`

## 👤 Autor

Desenvolvido por [hiyuk1](https://github.com/hiyuk1)
