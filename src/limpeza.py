import pandas as pd
import numpy as np


df = pd.read_csv('data/processed/dataset_dieta_completo.csv')
print(f"Dimensões antes da limpeza: {df.shape}")

# ==========================================
# PASSO 1: Remoção de Características Redundantes
# ==========================================
# Vamos remover colunas que não acrescentam informação nova
colunas_para_remover = ['bmi_redundant', 'experience_years']
df.drop(columns=colunas_para_remover, errors='ignore', inplace=True)
print("-> Colunas redundantes removidas.")

# ==========================================
# PASSO 2: Padronização de Variáveis Categóricas
# ==========================================
# Limpar espaços em branco e colocar tudo em minúsculas na abordagem do nutricionista
if 'approach' in df.columns:
    df['approach'] = df['approach'].astype(str).str.strip().str.lower()
    # Se existirem valores que ficaram como 'nan' devido à conversão, voltamos a colocá-los como NaN real
    df['approach'] = df['approach'].replace('nan', np.nan)
print("-> Categorias de texto padronizadas (removidos espaços e maiúsculas).")

# ==========================================
# PASSO 3: Tratamento de Outliers
# ==========================================
# O nutricionista N008 tem 99 anos de experiência. Vamos substituir valores irreais (> 60 anos) pela mediana da experiência
mediana_experiencia = df[df['years_experience'] <= 60]['years_experience'].median()
df.loc[df['years_experience'] > 60, 'years_experience'] = mediana_experiencia
print("-> Outliers na experiência dos nutricionistas corrigidos.")

# ==========================================
# PASSO 4: Tratamento de Valores em Falta (Missing Values)
# ==========================================
print("\nValores em falta detetados por coluna antes da imputação:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Estratégia de Imputação:
# - Variáveis numéricas: Substituir pela MEDIANA (mais segura contra outliers do que a média)
colunas_numericas_com_nulos = [
    'fiber_target_g', 'sodium_limit_mg', 'years_experience',
    'sleep_hours', 'motivation_score', 'motivation_score_program',
    'mean_adherence_pct', 'adherence_ratio', 'weight_change_kg_6m', 'age'
]

for col in colunas_numericas_com_nulos:
    if col in df.columns:
        mediana_col = df[col].median()
        df[col] = df[col].fillna(mediana_col)

# - Variáveis categóricas: Substituir pela MODA (o valor mais frequente) ou criar classe 'unknown'
if 'approach' in df.columns:
    moda_approach = df['approach'].mode()[0]
    df['approach'] = df['approach'].fillna(moda_approach)

if 'specialty' in df.columns:
    moda_specialty = df['specialty'].mode()[0]
    df['specialty'] = df['specialty'].fillna(moda_specialty)

print("\n-> Valores em falta tratados com sucesso!")
print(f"Valores nulos restantes: {df.isnull().sum().sum()}")

# ==========================================
# 5. Guardar o Dataset Limpo
# ==========================================
print(f"\nDimensões após a limpeza: {df.shape}")
df.to_csv('data/processed/dataset_dieta_limpo.csv', index=False)
print("Ficheiro 'data/processed/dataset_dieta_limpo.csv' guardado com sucesso!")