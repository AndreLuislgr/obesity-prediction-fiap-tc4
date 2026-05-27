[README.md](https://github.com/user-attachments/files/28291392/README.md)
# 🏥 Sistema Preditivo de Obesidade — Modelo Comportamental
**Tech Challenge Fase 4 — FIAP Pós-Tech Data Analytics**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://obesityproject-w2l9d7ffhjkepi5mbnjqso.streamlit.app/)

---

## 📋 Sobre o Projeto

Sistema inteligente de apoio à decisão médica para classificação do nível de obesidade, desenvolvido com Machine Learning. O diferencial desta solução é o **modelo comportamental** — capaz de prever o risco de obesidade **sem utilizar Peso e Altura**, baseando-se apenas em hábitos alimentares, estilo de vida e dados demográficos.

> 💡 Isso permite aplicar o modelo em **triagens sem equipamentos de medição**, ampliando o alcance clínico da solução.

---

## 🎯 Resultados do Modelo

| Métrica | Resultado | Requisito |
|---------|-----------|-----------|
| Acurácia (Teste) | **~80%** | > 75% ✅ |
| Validação Cruzada (5-fold) | **~81% ± 2%** | — ✅ |
| Tamanho do modelo | **~3MB** | < 25MB ✅ |

---

## 🔬 Modelo Comportamental vs Tradicional

| | Com Peso e Altura | Sem Peso e Altura |
|--|--|--|
| Acurácia | >97% | ~80% |
| O modelo aprende | IMC (tautologia) | Padrões comportamentais reais |
| Aplicação clínica | Redundante | Triagem sem equipamentos |

---

## 🏗️ Pipeline de Machine Learning

```
Obesity.csv (2111 registros, 17 colunas)
    │
    ▼
1. Remoção de Weight e Height (data leakage)
    │
    ▼
2. EDA — Análise Exploratória
    │
    ▼
3. Feature Engineering
   ├── age_group  → faixa etária (teen / young_adult / adult / senior)
   ├── inactive   → flag de sedentarismo total (FAF = 0)
   └── risk_score → score de risco composto (0–3)
    │
    ▼
4. Pré-processamento (ColumnTransformer)
   ├── StandardScaler    → variáveis numéricas
   └── OneHotEncoder     → variáveis categóricas
    │
    ▼
5. Modelo: Random Forest (100 estimadores, compress=3)
    │
    ▼
6. Avaliação
   ├── Acurácia, F1, Precision, Recall
   ├── Matriz de Confusão
   └── Validação Cruzada Estratificada (5-fold)
    │
    ▼
7. Deploy — Streamlit Cloud
```

---

## 📁 Estrutura do Projeto

```
obesity-behavioral-predictor/
├── app.py                   # Aplicação Streamlit (sistema preditivo + dashboard)
├── pipeline_obesity.ipynb   # Notebook com a pipeline completa de ML
├── Obesity.csv              # Dataset original
├── model.pkl                # Modelo treinado (Random Forest Pipeline ~3MB)
├── classes.json             # Labels das 7 classes
├── feature_importance.json  # Importância das features
├── model_info.json          # Métricas do modelo
├── requirements.txt         # Dependências
├── .streamlit/
│   └── config.toml          # Tema light do Streamlit
└── README.md
```

---

## 📊 Funcionalidades do App

### 🔮 Sistema Preditivo
- Formulário interativo com dados comportamentais e demográficos
- Diagnóstico preditivo em tempo real (7 categorias)
- Score de risco comportamental (0–3)
- Probabilidades por classe com visualização gráfica
- Recomendações clínicas personalizadas
- Identificação automática de fatores de risco

### 📊 Dashboard Analítico
- KPIs executivos de prevalência
- Distribuição por nível de obesidade
- Risk Score médio por categoria
- Histórico familiar × nível de obesidade
- Frequência de atividade física por categoria
- Importância das features do modelo
- Insights acionáveis para a equipe médica

---

## 🚀 Como Rodar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Rode a aplicação
streamlit run app.py
```

---

## 🌐 Deploy no Streamlit Cloud

1. Faça push de todos os arquivos para o GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório
4. Configure: `Main file path = app.py`
5. Clique em **Deploy!**

---

## 🔬 Features Utilizadas

| Feature | Tipo | Descrição |
|---------|------|-----------|
| Gender | Categórica | Gênero biológico |
| Age | Numérica | Idade em anos |
| family_history | Binária | Histórico familiar de excesso de peso |
| FAVC | Binária | Consumo frequente de alimentos calóricos |
| FCVC | Ordinal (1–3) | Frequência de vegetais nas refeições |
| NCP | Ordinal (1–4) | Número de refeições principais/dia |
| CAEC | Categórica | Consumo entre refeições |
| SMOKE | Binária | Tabagismo |
| CH2O | Ordinal (1–3) | Consumo diário de água |
| SCC | Binária | Monitora calorias |
| FAF | Ordinal (0–3) | Frequência de atividade física |
| TUE | Ordinal (0–2) | Tempo em dispositivos eletrônicos |
| CALC | Categórica | Consumo de álcool |
| MTRANS | Categórica | Meio de transporte habitual |
| **age_group** | Categórica (eng.) | **Faixa etária** |
| **inactive** | Binária (eng.) | **Flag de sedentarismo total** |
| **risk_score** | Ordinal (eng.) | **Score de risco comportamental** |

---

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|-----------|-----------|
| Linguagem | Python 3.10+ |
| ML | scikit-learn |
| Dados | Pandas, NumPy |
| Visualização | Plotly |
| Interface | Streamlit |
| Serialização | Joblib |
| Deploy | Streamlit Cloud |
| Versionamento | GitHub |

---

## ⚠️ Disclaimer

Este sistema é uma ferramenta de **apoio à decisão médica**. O diagnóstico final é responsabilidade exclusiva do profissional de saúde. O modelo foi treinado em dados sintéticos/augmentados e deve ser validado clinicamente antes de uso em ambiente real.

---

**FIAP — Pós-Tech Data Analytics | Tech Challenge Fase 4**
