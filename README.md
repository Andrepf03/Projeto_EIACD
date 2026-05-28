# EIACD — Projeto Completo: Previsão de Sucesso Nutricional

Projeto completo desenvolvido para a unidade curricular de **Elementos de Inteligência Artificial e Ciência de Dados** da Licenciatura em Inteligência Artificial e Ciência de Dados na **Universidade da Beira Interior (UBI)**.

## Objetivos
- **Integração de dados:** Unificação de múltiplos ficheiros relacionais fragmentados;
- **Limpeza e processamento:** Resolução de valores em falta, remoção de redundâncias e correção de outliers;
- **Análise exploratória:** Extração de estatísticas e cruzamento de variáveis clínicas/comportamentais;
- **Clustering:** Segmentação não supervisionada de perfis de pacientes via algoritmo K-Means;
- **Modelos supervisionados:** Desenvolvimento e treino de modelos preditivos (Linear Regression e Random Forest);
- **Interpretabilidade:** Discussão analítica e crítica sobre o comportamento dos modelos obtidos.

## Estrutura
```text
├── data/
│   ├── raw/                  # Datasets originais do enunciado
│   └── processed/            # Datasets intermédios e finais tratados
├── graficos/                 # Artefactos visuais gerados automaticamente
├── src/                      # Código-fonte reutilizável (.py) organizado por fases
│   ├── integração.py
│   ├── limpeza.py
│   ├── eda.py
│   ├── padroes.py
│   └── previsao.py
├── .gitignore                # Ficheiros ignorados pelo versionamento Git
├── README.md                 # Documentação e resumo macro do projeto
└── requirements.txt          # Especificação das bibliotecas e dependências Python

Como correr:

Criar ambiente virtual (no terminal da raiz do projeto):
    Bash:
    python -m venv .venv

Ativar ambiente virtual:
    Windows: .\.venv\Scripts\activate
    Linux/macOS: source .venv/bin/activate

Instalar dependências:
    Bash:
    pip install -r requirements.txt
    
Executar os scripts na ordem recomendada:
    Bash:
    python src/integração.py
    python src/limpeza.py
    python src/eda.py
    python src/padroes.py
    python src/previsao.py