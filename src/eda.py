import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Criar uma pasta para guardar os gráficos (para manter o projeto organizado)
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# 2. Carregar o dataset limpo
df = pd.read_csv('data/processed/dataset_dieta_limpo.csv')

# Definir o estilo visual dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("--- A INICIAR A ANÁLISE EXPLORATÓRIA DE DADOS (EDA) ---")

# ==========================================
# PERGUNTA 1: O sexo do paciente ou o tipo de dieta influenciam o sucesso (perda de peso)?
# ==========================================
# Gráfico A: Distribuição da Perda de Peso por Sexo
plt.figure()
sns.boxplot(data=df, x='sex', y='weight_change_kg_6m', palette='Set2')
plt.title('Distribuição da Alteração de Peso por Sexo (6 Meses)')
plt.xlabel('Sexo')
plt.ylabel('Alteração de Peso (kg) - Valores negativos indicam perda')
plt.savefig('graficos/01_perda_peso_por_sexo.png')
plt.close()

# Gráfico B: Distribuição da Perda de Peso por Tipo de Dieta
plt.figure()
sns.boxplot(data=df, x='diet_type', y='weight_change_kg_6m', palette='Set3')
plt.xticks(rotation=15)
plt.title('Eficácia do Peso por Tipo de Dieta')
plt.xlabel('Tipo de Dieta')
plt.ylabel('Alteração de Peso (kg)')
plt.savefig('graficos/02_perda_peso_por_tipo_dieta.png')
plt.close()

# ==========================================
# PERGUNTA 2: O perfil/experiência do nutricionista é relevante?
# ==========================================
# Gráfico C: Correlação entre Experiência do Nutricionista e a Adesão do Paciente
plt.figure()
sns.scatterplot(data=df, x='years_experience', y='mean_adherence_pct', alpha=0.5, color='purple')
plt.title('Experiência do Nutricionista vs. Adesão Média do Paciente')
plt.xlabel('Anos de Experiência do Nutricionista')
plt.ylabel('Adesão Média do Paciente (%)')
plt.savefig('graficos/03_experiencia_vs_adesao.png')
plt.close()

# Gráfico D: Abordagem do Nutricionista vs Alteração de Peso
plt.figure()
sns.barplot(data=df, x='approach', y='weight_change_kg_6m', estimator=lambda x: sum(x)/len(x), palette='coolwarm')
plt.title('Média de Alteração de Peso por Abordagem do Nutricionista')
plt.xlabel('Abordagem')
plt.ylabel('Média de Alteração de Peso (kg)')
plt.savefig('graficos/04_abordagem_vs_peso.png')
plt.close()

# ==========================================
# PERGUNTA 3: Existem variáveis altamente correlacionadas?
# ==========================================
# Vamos selecionar apenas as colunas numéricas mais importantes para a matriz de correlação
colunas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
# Remover IDs se existirem nas colunas numéricas para não poluir o gráfico
colunas_numericas = [c for c in colunas_numericas if 'id' not in c.lower() and 'index' not in c.lower()]

plt.figure(figsize=(12, 10))
matriz_correlacao = df[colunas_numericas].corr()
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlação das Variáveis Numéricas')
plt.tight_layout()
plt.savefig('graficos/05_matriz_correlacao.png')
plt.close()

print("\n--- EDA CONCLUÍDA ---")
print("Todos os gráficos foram guardados com sucesso na pasta 'graficos/'!")

# Imprimir algumas estatísticas textuais importantes para o relatório
print("\nEstatísticas Rápidas de Perda de Peso por Tipo de Dieta:")
print(df.groupby('diet_type')['weight_change_kg_6m'].mean())