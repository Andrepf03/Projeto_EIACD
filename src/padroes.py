import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Garantir que a pasta de gráficos existe
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# Garantir que a pasta de destino existe
os.makedirs('data/processed', exist_ok=True)

# 1. Carregar o dataset limpo
df = pd.read_csv('data/processed/dataset_dieta_limpo.csv')

# 2. Selecionar características numéricas relevantes do paciente e do resultado para agrupar
# Vamos tentar agrupar os pacientes pelo seu perfil físico/comportamental e resultado
features_para_cluster = ['age', 'baseline_weight_kg', 'sleep_hours', 'mean_adherence_pct', 'weight_change_kg_6m']

X = df[features_para_cluster]

# 3. Normalizar as características (Obrigatório para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Método do Cotovelo (Elbow Method) para encontrar o K ideal
wcss = []  # Within-Cluster Sum of Squares
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Desenhar o gráfico do Cotovelo
plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, 'ro-', linewidth=2)
plt.title('Método do Cotovelo (Elbow Method) para Escolha de Clusters')
plt.xlabel('Número de Clusters (K)')
plt.ylabel('WCSS (Inércia)')
plt.grid(True)
plt.savefig('graficos/06_metodo_cotovelo.png')
plt.close()

print("--- FASE 4: ANÁLISE DE CLUSTERING ---")
print("Gráfico do Cotovelo gerado em 'graficos/06_metodo_cotovelo.png'.")
print("Escolhe-se o 'K' onde a curva dobra de forma brusca.")

# K=3 identificado como ponto de inflexão pelo Método do Cotovelo
k_ideal = 3
kmeans_final = KMeans(n_clusters=k_ideal, random_state=42, n_init=10)
df['cluster_paciente'] = kmeans_final.fit_predict(X_scaled)  # atribui o label do cluster a cada linha

# 5. Caracterizar os Clusters obtidos
perfil_clusters = df.groupby('cluster_paciente')[features_para_cluster].mean()
print("\nPerfil Médio de cada Cluster (Grupo de Pacientes):")
print(perfil_clusters)

# Guardar o dataset com a coluna do cluster em data/processed/ para ser lido pela fase seguinte
df.to_csv('data/processed/dataset_dieta_com_clusters.csv', index=False)
print("\nNovo ficheiro 'data/processed/dataset_dieta_com_clusters.csv' guardado com sucesso!")

# Gráfico para visualizar a separação dos clusters (Adesão vs Perda de Peso)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='mean_adherence_pct', y='weight_change_kg_6m', hue='cluster_paciente', palette='viridis', alpha=0.7)
plt.title('Segmentação de Pacientes: Adesão vs. Perda de Peso')
plt.xlabel('Adesão Média (%)')
plt.ylabel('Alteração de Peso (kg)')
plt.legend(title='Cluster')
plt.savefig('graficos/07_visualizacao_clusters.png')
plt.close()
