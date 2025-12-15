import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

ROOT_DIR = Path('.')
DATA_DIR = ROOT_DIR / 'dataset'
OUTPUT_DIR = ROOT_DIR / 'output'

# Load data
df = pd.read_csv(DATA_DIR / 'data.csv', encoding='latin1')
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate RFM
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

# Normalize
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

print(">>> MÉTODO DO COTOVELO (K=2 a K=10)...")
print("=" * 60)
print(f"{'K':^3} | {'Inércia':^15} | {'Silhouette Score':^18}")
print("=" * 60)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    inertia = kmeans.inertia_
    sil_score = silhouette_score(rfm_scaled, kmeans.labels_)
    inertias.append(inertia)
    silhouette_scores.append(sil_score)
    print(f"{k:^3} | {inertia:^15.2f} | {sil_score:^18.4f}")

print("=" * 60)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=3, color='r', linestyle='--', linewidth=2, label='K=3 (escolhido)')
ax1.set_xlabel('Número de Clusters (K)', fontsize=12)
ax1.set_ylabel('Inércia', fontsize=12)
ax1.set_title('Método do Cotovelo - Inércia', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=3, color='r', linestyle='--', linewidth=2, label='K=3 (escolhido)')
ax2.set_xlabel('Número de Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score por K', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'elbow_method.png', dpi=300)
print(f"\nGráfico salvo em: output/elbow_method.png ✓")
