import pandas as pd

df_pacientes = pd.read_csv('data/raw/patients.csv')
df_planos = pd.read_csv('data/raw/diets.csv')
df_nutricionistas = pd.read_csv('data/raw/nutritionists.csv')
df_resultados = pd.read_csv('data/raw/outcomes.csv')

print("Dimensões iniciais:")
print(f"Pacientes: {df_pacientes.shape}")
print(f"Planos (Dietas): {df_planos.shape}")
print(f"Nutricionistas: {df_nutricionistas.shape}")
print(f"Resultados: {df_resultados.shape}\n")

# 2. Iniciar a integração a partir da tabela central (df_resultados)
# Juntar os Resultados com os Planos de Dieta através de 'diet_id'
df_integrado = pd.merge(df_resultados, df_planos, on='diet_id', how='inner')

# Juntar o resultado anterior com os Pacientes através de 'patient_id'
df_integrado = pd.merge(df_integrado, df_pacientes, on='patient_id', how='inner')

# Juntar com os Nutricionistas através de 'nutritionist_id'
df_final = pd.merge(df_integrado, df_nutricionistas, on='nutritionist_id', how='inner')

print("Dimensões do dataset unificado:")
print(df_final.shape)

df_final.to_csv('data/processed/dataset_dieta_completo.csv', index=False)
print("\nFicheiro 'data/processed/dataset_dieta_completo.csv' gerado com sucesso!")