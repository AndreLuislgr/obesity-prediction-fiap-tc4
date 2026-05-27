"""
Tech Challenge — Predição de Obesidade (SEM Peso e Altura)
Modelo comportamental — apenas hábitos e dados demográficos
FIAP Pós-Tech Data Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model.pkl"))

@st.cache_data
def load_json(name):
    with open(os.path.join(BASE, name)) as f:
        return json.load(f)

model      = load_model()
classes    = load_json("classes.json")
model_info = load_json("model_info.json")
feat_imp   = load_json("feature_importance.json")

# ── Labels e cores ────────────────────────────────────────────────────────────
CLASS_PT = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight":       "Peso Normal",
    "Overweight_Level_I":  "Sobrepeso Grau I",
    "Overweight_Level_II": "Sobrepeso Grau II",
    "Obesity_Type_I":      "Obesidade Grau I",
    "Obesity_Type_II":     "Obesidade Grau II",
    "Obesity_Type_III":    "Obesidade Grau III",
}
CLASS_COLOR = {
    "Insufficient_Weight": "#3b82f6",
    "Normal_Weight":       "#22c55e",
    "Overweight_Level_I":  "#facc15",
    "Overweight_Level_II": "#f97316",
    "Obesity_Type_I":      "#ef4444",
    "Obesity_Type_II":     "#dc2626",
    "Obesity_Type_III":    "#991b1b",
}
CLASS_RISK = {
    "Insufficient_Weight": "⚠️ Baixo Peso",
    "Normal_Weight":       "✅ Saudável",
    "Overweight_Level_I":  "🟡 Atenção",
    "Overweight_Level_II": "🟠 Alerta",
    "Obesity_Type_I":      "🔴 Risco Moderado",
    "Obesity_Type_II":     "🔴 Risco Alto",
    "Obesity_Type_III":    "🚨 Risco Muito Alto",
}
RECOMENDACOES = {
    "Insufficient_Weight": [
        "Aumentar aporte calórico com alimentos nutritivos",
        "Consultar nutricionista para plano alimentar individualizado",
        "Avaliar causas subjacentes com exames laboratoriais",
        "Monitorar peso semanalmente",
    ],
    "Normal_Weight": [
        "Manter hábitos alimentares atuais",
        "Continuar praticando atividade física regularmente",
        "Revisão médica anual preventiva",
        "Monitorar histórico familiar de doenças crônicas",
    ],
    "Overweight_Level_I": [
        "Reduzir consumo de alimentos ultra-processados",
        "Iniciar ou intensificar atividade física (≥150 min/semana)",
        "Aumentar ingestão de vegetais e fibras",
        "Monitorar pressão arterial e glicemia",
    ],
    "Overweight_Level_II": [
        "Acompanhamento multidisciplinar (médico + nutricionista)",
        "Estabelecer metas de perda de peso (0,5–1 kg/semana)",
        "Reduzir consumo de álcool e alimentos calóricos",
        "Solicitar exames: perfil lipídico e glicemia",
    ],
    "Obesity_Type_I": [
        "Encaminhar para acompanhamento médico especializado",
        "Programa estruturado de atividade física",
        "Avaliação de comorbidades (DM2, hipertensão)",
        "Terapia cognitivo-comportamental para mudança de hábitos",
    ],
    "Obesity_Type_II": [
        "Avaliação clínica urgente para comorbidades graves",
        "Considerar tratamento farmacológico (decisão médica)",
        "Programa intensivo de mudança de estilo de vida",
        "Monitoramento cardiológico e metabólico frequente",
    ],
    "Obesity_Type_III": [
        "🚨 Encaminhamento imediato para especialista em obesidade",
        "Avaliar elegibilidade para cirurgia bariátrica",
        "Suporte psicológico e nutricional intensivo",
        "Monitoramento contínuo de risco cardiovascular",
    ],
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8fafc; }
.metric-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08); text-align: center;
}
.metric-card .value { font-size: 1.8rem; font-weight: 700; }
.metric-card .label { font-size: .82rem; color: #64748b; margin-top: 4px; }
.result-box { border-radius: 14px; padding: 24px 28px; margin: 16px 0; border-left: 6px solid; }
.prob-bar-wrap { background: #e2e8f0; border-radius: 8px; height: 16px; margin: 4px 0; }
.prob-bar { border-radius: 8px; height: 16px; }
.banner {
    background: linear-gradient(135deg, #1e3a5f, #2e75b6);
    color: white; border-radius: 12px; padding: 16px 24px; margin-bottom: 20px;
    font-size: .9rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/hospital.png", width=60)
    st.title("Sistema Preditivo")
    st.caption("Obesidade — FIAP Tech Challenge")
    st.divider()
    page = st.radio("Navegação", [
        "🔮 Sistema Preditivo",
        "📊 Dashboard Analítico",
        "📋 Sobre o Modelo"
    ], label_visibility="collapsed")
    st.divider()
    st.metric("Acurácia (Teste)",    f"{model_info['test_accuracy']*100:.1f}%")
    st.metric("Validação Cruzada",   f"{model_info['cv_mean']*100:.1f}%")
    st.caption("Random Forest — sem Peso e Altura")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — SISTEMA PREDITIVO
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔮 Sistema Preditivo":
    st.title("🔮 Sistema Preditivo de Obesidade")
    st.markdown("Preencha os dados comportamentais e demográficos do paciente.")

    # Banner explicativo
    st.markdown("""
    <div class="banner">
        🧬 <strong>Modelo Comportamental</strong> — Este preditor classifica o risco de obesidade
        <strong>sem utilizar Peso e Altura</strong>, baseando-se apenas em hábitos alimentares,
        estilo de vida e dados demográficos. Útil para triagem em ambientes sem equipamentos de medição.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    with st.form("form_paciente"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👤 Dados Pessoais")
            gender = st.selectbox("Gênero", ["Male","Female"],
                                  format_func=lambda x: "Masculino" if x=="Male" else "Feminino")
            age    = st.slider("Idade (anos)", 14, 80, 25)
            family = st.selectbox("Histórico familiar de excesso de peso?", ["yes","no"],
                                  format_func=lambda x: "Sim" if x=="yes" else "Não")
            smoke  = st.selectbox("Fuma?", ["no","yes"],
                                  format_func=lambda x: "Não" if x=="no" else "Sim")
            mtrans = st.selectbox("Meio de transporte habitual",
                                  ["Public_Transportation","Automobile","Walking","Motorbike","Bike"],
                                  format_func={"Public_Transportation":"Transporte Público",
                                               "Automobile":"Carro","Walking":"A pé",
                                               "Motorbike":"Moto","Bike":"Bicicleta"}.get)

        with col2:
            st.subheader("🍎 Hábitos Alimentares")
            favc = st.selectbox("Come alimentos calóricos com frequência?", ["yes","no"],
                                format_func=lambda x: "Sim" if x=="yes" else "Não")
            fcvc = st.select_slider("Frequência de vegetais nas refeições",
                                    options=[1,2,3],
                                    format_func={1:"Raramente",2:"Às vezes",3:"Sempre"}.get)
            ncp  = st.select_slider("Nº de refeições principais/dia",
                                    options=[1,2,3,4],
                                    format_func={1:"1 refeição",2:"2 refeições",
                                                 3:"3 refeições",4:"4+ refeições"}.get, value=3)
            caec = st.selectbox("Come entre as refeições?",
                                ["no","Sometimes","Frequently","Always"],
                                format_func={"no":"Não","Sometimes":"Às vezes",
                                             "Frequently":"Frequentemente","Always":"Sempre"}.get,
                                index=1)
            ch2o = st.select_slider("Consumo diário de água",
                                    options=[1,2,3],
                                    format_func={1:"< 1 L/dia",2:"1–2 L/dia",3:"> 2 L/dia"}.get,
                                    value=2)
            calc = st.selectbox("Frequência de consumo de álcool?",
                                ["no","Sometimes","Frequently","Always"],
                                format_func={"no":"Não","Sometimes":"Às vezes",
                                             "Frequently":"Frequentemente","Always":"Sempre"}.get,
                                index=1)

        with col3:
            st.subheader("🏃 Estilo de Vida")
            scc = st.selectbox("Monitora calorias ingeridas?", ["no","yes"],
                               format_func=lambda x: "Não" if x=="no" else "Sim")
            faf = st.select_slider("Frequência de atividade física",
                                   options=[0,1,2,3],
                                   format_func={0:"Nenhuma",1:"1–2×/sem",
                                                2:"3–4×/sem",3:"5+/sem"}.get)
            tue = st.select_slider("Tempo com dispositivos eletrônicos",
                                   options=[0,1,2],
                                   format_func={0:"0–2 h/dia",1:"3–5 h/dia",2:"> 5 h/dia"}.get)

        submitted = st.form_submit_button("🔍 Analisar Paciente",
                                          use_container_width=True, type="primary")

    if submitted:
        age_grp    = pd.cut([age], bins=[0,18,30,45,100],
                            labels=['teen','young_adult','adult','senior'])[0]
        inactive   = int(faf == 0)
        risk_score = int(favc=="yes") + inactive + int(family=="yes")

        input_df = pd.DataFrame([{
            "Gender": gender, "Age": float(age),
            "family_history": family, "FAVC": favc, "FCVC": fcvc, "NCP": ncp,
            "CAEC": caec, "SMOKE": smoke, "CH2O": ch2o, "SCC": scc,
            "FAF": faf, "TUE": tue, "CALC": calc, "MTRANS": mtrans,
            "age_group": age_grp, "inactive": inactive, "risk_score": risk_score
        }])

        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        color      = CLASS_COLOR[prediction]
        label_pt   = CLASS_PT[prediction]
        conf       = max(proba) * 100

        st.divider()
        st.subheader("📋 Resultado do Diagnóstico Preditivo")

        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(f'<div class="metric-card"><div class="value" style="color:{color}">{label_pt}</div><div class="label">Diagnóstico Previsto</div></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="metric-card"><div class="value">{CLASS_RISK[prediction]}</div><div class="label">Nível de Risco</div></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="metric-card"><div class="value">{risk_score}/3</div><div class="label">Score de Risco Comportamental</div></div>', unsafe_allow_html=True)
        r4.markdown(f'<div class="metric-card"><div class="value">{conf:.0f}%</div><div class="label">Confiança do Modelo</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-box" style="background:{color}18; border-color:{color}">
            <h3 style="color:{color}; margin:0">{CLASS_RISK[prediction]} — {label_pt}</h3>
            <p style="margin:8px 0 0; color:#374151">
                Diagnóstico baseado em dados comportamentais e demográficos, <strong>sem uso de Peso e Altura</strong>.
                Confiança do modelo: <strong>{conf:.0f}%</strong>.
            </p>
        </div>""", unsafe_allow_html=True)

        col_prob, col_rec = st.columns(2)
        with col_prob:
            st.markdown("**Probabilidades por Classe**")
            for i in np.argsort(proba)[::-1]:
                cls = classes[i]
                p   = proba[i]
                clr = CLASS_COLOR[cls]
                lbl = CLASS_PT[cls]
                bw  = int(p * 100)
                st.markdown(f"""
                <div style="margin:5px 0">
                    <div style="display:flex;justify-content:space-between;font-size:.82rem">
                        <span>{'<b>' if cls==prediction else ''}{lbl}{'</b>' if cls==prediction else ''}</span>
                        <span style="color:{clr};font-weight:{'700' if cls==prediction else '400'}">{p*100:.1f}%</span>
                    </div>
                    <div class="prob-bar-wrap">
                        <div class="prob-bar" style="width:{bw}%;background:{clr}"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col_rec:
            st.markdown("**🩺 Recomendações Clínicas**")
            for rec in RECOMENDACOES[prediction]:
                st.markdown(f"• {rec}")
            st.info("⚠️ Este sistema é um apoio à decisão. O diagnóstico final é responsabilidade do médico.")

        st.markdown("**📌 Fatores de Risco Identificados**")
        factors = []
        if family == "yes":  factors.append("🔴 Histórico familiar de excesso de peso")
        if favc == "yes":    factors.append("🟠 Consumo frequente de alimentos calóricos")
        if faf == 0:         factors.append("🔴 Sedentarismo total — sem atividade física")
        elif faf == 1:       factors.append("🟡 Baixa frequência de atividade física")
        if smoke == "yes":   factors.append("🔴 Tabagismo")
        if calc in ["Frequently","Always"]: factors.append("🟠 Consumo frequente de álcool")
        if ch2o == 1:        factors.append("🟡 Baixo consumo de água (< 1 L/dia)")
        if tue == 2:         factors.append("🟡 Alto tempo em dispositivos eletrônicos")
        if caec == "Always": factors.append("🟠 Come entre refeições com frequência alta")
        if not factors:      factors.append("✅ Nenhum fator de risco crítico identificado")
        cols_f = st.columns(2)
        for i, f in enumerate(factors):
            cols_f[i % 2].markdown(f)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — DASHBOARD ANALÍTICO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard Analítico":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("📊 Dashboard Analítico — Estudo de Obesidade")
    st.markdown("Insights comportamentais para a equipe médica — **sem variáveis antropométricas**.")
    st.divider()

    df = pd.read_csv(os.path.join(BASE, "Obesity.csv"))
    for col in ['FCVC','NCP','CH2O','FAF','TUE']:
        df[col] = df[col].round().astype(int)
    df['age_group']  = pd.cut(df['Age'], bins=[0,18,30,45,100], labels=['teen','young_adult','adult','senior'])
    df['inactive']   = (df['FAF'] == 0).astype(int)
    df['risk_score'] = ((df['FAVC']=='yes').astype(int) + df['inactive'] + (df['family_history']=='yes').astype(int))

    total     = len(df)
    obeso     = df['Obesity'].str.startswith('Obesity').sum()
    sobrepeso = df['Obesity'].str.startswith('Overweight').sum()
    normal    = (df['Obesity'] == 'Normal_Weight').sum()
    sed       = (df['FAF'] == 0).sum()

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.markdown(f'<div class="metric-card"><div class="value">{total}</div><div class="label">Total Pacientes</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="metric-card"><div class="value" style="color:#ef4444">{obeso}</div><div class="label">Obesidade ({obeso/total*100:.0f}%)</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="metric-card"><div class="value" style="color:#f97316">{sobrepeso}</div><div class="label">Sobrepeso ({sobrepeso/total*100:.0f}%)</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="metric-card"><div class="value" style="color:#22c55e">{normal}</div><div class="label">Peso Normal ({normal/total*100:.0f}%)</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="metric-card"><div class="value" style="color:#f97316">{sed}</div><div class="label">Sedentários ({sed/total*100:.0f}%)</div></div>', unsafe_allow_html=True)

    st.divider()
    order   = ["Insufficient_Weight","Normal_Weight","Overweight_Level_I","Overweight_Level_II","Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]
    labels  = [CLASS_PT[o] for o in order]
    colors  = [CLASS_COLOR[o] for o in order]
    counts  = df['Obesity'].value_counts().reindex(order)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribuição por Nível de Obesidade")
        fig = go.Figure(go.Bar(x=labels, y=counts.values, marker_color=colors,
                               text=counts.values, textposition='outside'))
        fig.update_layout(height=320, margin=dict(t=20,b=0), xaxis_tickangle=-30,
                          plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Risk Score Médio por Categoria")
        rs_class = df.groupby('Obesity')['risk_score'].mean().reindex(order)
        fig2 = go.Figure(go.Bar(x=labels, y=rs_class.values.round(2),
                                marker_color=colors, text=rs_class.values.round(2),
                                textposition='outside'))
        fig2.update_layout(height=320, margin=dict(t=20,b=0), xaxis_tickangle=-30,
                           plot_bgcolor='white', paper_bgcolor='white',
                           yaxis_title="Score médio (0–3)")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Histórico Familiar × Nível de Obesidade")
        cross = pd.crosstab(df['Obesity'], df['family_history']).reindex(order)
        cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name='Sem histórico', x=labels, y=cross_pct['no'].round(1), marker_color='#94a3b8'))
        fig3.add_trace(go.Bar(name='Com histórico', x=labels, y=cross_pct['yes'].round(1), marker_color='#ef4444'))
        fig3.update_layout(barmode='stack', height=320, margin=dict(t=20,b=0),
                           xaxis_tickangle=-30, plot_bgcolor='white', paper_bgcolor='white',
                           legend=dict(orientation='h', y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Atividade Física Média por Categoria")
        faf_class = df.groupby('Obesity')['FAF'].mean().reindex(order)
        colors_faf = ['#22c55e' if v>=1.5 else '#f97316' if v>=0.8 else '#ef4444' for v in faf_class.values]
        fig4 = go.Figure(go.Bar(x=labels, y=faf_class.values.round(2),
                                marker_color=colors_faf, text=faf_class.values.round(2),
                                textposition='outside'))
        fig4.update_layout(height=320, margin=dict(t=20,b=0), xaxis_tickangle=-30,
                           plot_bgcolor='white', paper_bgcolor='white',
                           yaxis_title="Frequência média (0–3)")
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Importância das Features — Modelo Comportamental")
        fi_items = sorted(feat_imp.items(), key=lambda x: x[1])[-15:]
        fig5 = go.Figure(go.Bar(x=[v*100 for _,v in fi_items], y=[k for k,_ in fi_items],
                                orientation='h', marker_color='#6366f1',
                                text=[f"{v*100:.1f}%" for _,v in fi_items], textposition='outside'))
        fig5.update_layout(height=380, margin=dict(t=20,l=160,r=60),
                           plot_bgcolor='white', paper_bgcolor='white',
                           xaxis_title="Importância (%)")
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        st.subheader("Distribuição de Idade por Categoria")
        df_plot = df.copy()
        df_plot['Categoria'] = df_plot['Obesity'].map(CLASS_PT)
        fig6 = px.box(df_plot, x='Categoria', y='Age', color='Gender',
                      color_discrete_map={'Male':'#6366f1','Female':'#ec4899'},
                      category_orders={'Categoria': labels})
        fig6.update_layout(height=380, margin=dict(t=20,b=0), xaxis_tickangle=-30,
                           plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(fig6, use_container_width=True)

    st.divider()
    st.subheader("💡 Insights para a Equipe Médica")
    i1, i2, i3 = st.columns(3)
    with i1:
        fam_ob = (df[df['Obesity'].str.startswith('Obesity')]['family_history']=='yes').mean()*100
        st.markdown(f"**🧬 Histórico Familiar**\n\nEntre pacientes obesos, **{fam_ob:.0f}%** têm histórico familiar positivo. É o fator comportamental mais preditivo mesmo sem dados de peso.")
    with i2:
        faf_ob = df[df['Obesity'].str.startswith('Obesity')]['FAF'].mean()
        faf_no = df[df['Obesity']=='Normal_Weight']['FAF'].mean()
        st.markdown(f"**🏃 Atividade Física**\n\nPeso normal: **{faf_no:.1f}×/sem** vs Obesidade: **{faf_ob:.1f}×/sem**. Sedentarismo é o fator modificável mais impactante.")
    with i3:
        pct = ((df['Obesity'].str.startswith('Obesity'))|(df['Obesity'].str.startswith('Overweight'))).mean()*100
        st.markdown(f"**📊 Prevalência**\n\n**{pct:.0f}%** da amostra está em sobrepeso ou obesidade. Com apenas dados comportamentais, o modelo atinge **{model_info['test_accuracy']*100:.0f}%** de acurácia.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — SOBRE O MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Sobre o Modelo":
    st.title("📋 Sobre o Modelo")

    st.markdown("""
    ## 🧬 Modelo Comportamental — Sem Peso e Altura

    Este modelo foi desenvolvido para demonstrar que é possível prever o nível de
    obesidade **sem utilizar medidas antropométricas diretas** (peso e altura),
    baseando-se apenas em hábitos alimentares, estilo de vida e dados demográficos.

    ### Por que remover Peso e Altura?
    | Com Peso e Altura | Sem Peso e Altura |
    |---|---|
    | Acurácia >97% | Acurácia ~79% |
    | IMC quase determina o target | Modelo aprende hábitos reais |
    | Tautologia clínica | Aplicável em triagem sem equipamentos |

    ### Pipeline
    1. **Remoção** de Weight e Height
    2. **Arredondamento** de variáveis ordinais (FCVC, NCP, CH2O, FAF, TUE)
    3. **Feature Engineering**: age_group, inactive, risk_score
    4. **Pré-processamento**: StandardScaler + OneHotEncoder via ColumnTransformer
    5. **Modelo**: Random Forest 300 estimadores, class_weight='balanced'
    6. **Avaliação**: Holdout 80/20 estratificado + CV 5-fold
    """)

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia (Teste)",  f"{model_info['test_accuracy']*100:.2f}%")
    c2.metric("CV Média (5-fold)", f"{model_info['cv_mean']*100:.2f}%")
    c3.metric("CV Std",            f"±{model_info['cv_std']*100:.2f}%")

    st.info("⚠️ Este sistema é uma ferramenta de apoio à decisão médica. O diagnóstico final é responsabilidade do profissional de saúde.")
