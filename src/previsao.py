import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Garantir pasta de gráficos
if not os.path.exists('graficos'):
    os.makedirs('graficos')

# 1. Carregar o dataset que contém a coluna de clusters criada na Fase 4
df = pd.read_csv('data/processed/dataset_dieta_com_clusters.csv')

# 2. Definição da Target (Variável Alvo) e Features (Características Preditivas)
target = 'weight_change_kg_6m'

# Vamos selecionar as colunas mais importantes, excluindo IDs e datas
features_numericas = [
    'age', 'height_cm', 'baseline_weight_kg', 'sleep_hours', 
    'motivation_score', 'motivation_score_program', 'mean_adherence_pct', 
    'adherence_ratio', 'years_experience', 'cluster_paciente'
]
features_categoricas = ['sex', 'smoker', 'diet_type', 'approach', 'specialty']

X = df[features_numericas + features_categoricas]
y = df[target]

# 3. Divisão em Treino e Teste (80% para treino, 20% para teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pré-processamento Automatizado usando Pipeline e ColumnTransformer
# - Numéricas recebem Normalização (StandardScaler)
# - Categóricas recebem codificação binária (OneHotEncoder)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_numericas),
        ('cat', OneHotEncoder(handle_unknown='ignore'), features_categoricas)
    ])

# 5. Definição dos dois Modelos em Pipelines completos
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', LinearRegression())])

pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10))])

# 6. Treinar e Avaliar os Modelos
modelos = {
    'Regressão Linear': pipeline_lr,
    'Random Forest Regressor': pipeline_rf
}

print("--- FASE 5: TREINO E AVALIAÇÃO DE MODELOS DE PREVISÃO ---")

resultados = {}

for nome, pipeline in modelos.items():
    # Treinar o modelo
    pipeline.fit(X_train, y_train)
    
    # Prever no conjunto de teste
    y_pred = pipeline.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    resultados[nome] = {'RMSE': rmse, 'R2': r2}
    
    print(f"\n {nome}:")
    print(f"   -> Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f} kg")
    print(f"   -> Coeficiente de Determinação (R²): {r2:.4f}")
    
    # Gráfico de Valores Reais vs. Valores Previstos
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred, alpha=0.4, color='teal' if 'Linear' in nome else 'coral')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'{nome}: Reais vs. Previstos')
    plt.xlabel('Valores Reais (kg)')
    plt.ylabel('Valores Previstos (kg)')
    plt.tight_layout()
    plt.savefig(f"graficos/08_reais_vs_previstos_{nome.lower().replace(' ', '_')}.png")
    plt.close()

print("\nModelos guardados e gráficos comparativos gerados na pasta 'graficos/'.")