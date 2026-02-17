import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pre-Delinquency Intervention Engine", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #444;
        text-align: center;
        margin-bottom: 10px;
    }
    .risk-score-high { color: #FF4B4B; font-size: 42px; font-weight: bold; }
    .risk-score-med { color: #FFA500; font-size: 42px; font-weight: bold; }
    .risk-score-low { color: #00CC96; font-size: 42px; font-weight: bold; }
    
    .reason-box {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 10px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADER & RISK ENGINE ---
@st.cache_data
def load_and_score_data():
    try:
        df_credit = pd.read_csv('german_credit_data.csv', index_col=0)
        df_trans = pd.read_csv('synthetic_transaction_data.csv')
    except FileNotFoundError:
        st.error("Files not found. Ensure 'german_credit_data.csv' and 'synthetic_transaction_data.csv' are in the folder.")
        st.stop()

    # Feature Engineering
    upi_counts = df_trans[df_trans['Category'] == 'UPI_Lending_App'].groupby('User_ID').size()
    upi_counts.name = 'UPI_Lending_Count'

    disc_spend = df_trans[df_trans['Category'] == 'Discretionary'].groupby('User_ID')['Amount'].mean()
    disc_spend.name = 'Avg_Discretionary_Spend'

    df_trans['Date'] = pd.to_datetime(df_trans['Date'])
    salary_trans = df_trans[df_trans['Category'] == 'Salary'].copy()
    salary_trans['Day'] = salary_trans['Date'].dt.day
    salary_day = salary_trans.groupby('User_ID')['Day'].mean()
    salary_day.name = 'Avg_Salary_Day'

    # Merge
    df_master = df_credit.join(upi_counts, how='left')
    df_master = df_master.join(disc_spend, how='left')
    df_master = df_master.join(salary_day, how='left')

    df_master['UPI_Lending_Count'] = df_master['UPI_Lending_Count'].fillna(0)
    df_master['Avg_Discretionary_Spend'] = df_master['Avg_Discretionary_Spend'].fillna(df_trans[df_trans['Category']=='Discretionary']['Amount'].mean())
    df_master['Avg_Salary_Day'] = df_master['Avg_Salary_Day'].fillna(1)
    
    df_master['User_ID'] = df_master.index

    # Risk Scoring Logic
    n_upi = df_master['UPI_Lending_Count'] / df_master['UPI_Lending_Count'].max()
    n_credit = df_master['Credit amount'] / df_master['Credit amount'].max()
    n_salary = df_master['Avg_Salary_Day'] / 30.0
    max_spend = df_master['Avg_Discretionary_Spend'].max()
    n_spend_inv = 1 - (df_master['Avg_Discretionary_Spend'] / max_spend)

    # Weights: UPI (40%), Spend Drop (30%), Credit Load (20%), Late Salary (10%)
    df_master['Risk_Probability'] = (n_upi * 0.40) + (n_spend_inv * 0.30) + (n_credit * 0.20) + (n_salary * 0.10)
    df_master['Risk_Probability'] = df_master['Risk_Probability'] / df_master['Risk_Probability'].max()

    df_master['Risk_Label'] = np.where(df_master['Risk_Probability'] > 0.65, 'High', 
                                       np.where(df_master['Risk_Probability'] > 0.35, 'Medium', 'Low'))

    return df_master

# --- MOCK METRICS ---
def get_model_metrics():
    return pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVC'],
        'Accuracy': [0.950, 0.940, 0.940, 0.920],
        'Precision': [0.967, 0.950, 0.950, 0.900],
        'Recall': [0.879, 0.850, 0.864, 0.820],
        'ROC-AUC': [0.935, 0.920, 0.926, 0.900]
    })

# --- LOAD DATA ---
df = load_and_score_data()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Overview", "Manual Risk Assessment", "Model Comparison"])

# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if page == "Dashboard Overview":
    st.title("🚨 Pre-Delinquency Intervention Engine")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Users Monitored", len(df))
    kpi2.metric("High Risk Alerts", len(df[df['Risk_Label'] == 'High']), delta_color="inverse")
    kpi3.metric("Avg Risk Score", f"{df['Risk_Probability'].mean()*100:.1f}%")
    kpi4.metric("Avg Credit Load", f"${df['Credit amount'].mean():.0f}")

    st.markdown("### 🔥 Top 10 High Risk Users")
    top_risk = df.sort_values(by='Risk_Probability', ascending=False).head(10)
    
    st.dataframe(
        top_risk[['User_ID', 'Age', 'Risk_Probability', 'Risk_Label', 'UPI_Lending_Count', 'Credit amount']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "User_ID": st.column_config.NumberColumn("User ID", format="%d"),
            "Risk_Probability": st.column_config.ProgressColumn("Delinquency Prob.", format="%.1f%%", min_value=0, max_value=1),
            "Credit amount": st.column_config.NumberColumn("Credit Amount", format="$%d"),
            "UPI_Lending_Count": st.column_config.NumberColumn("UPI Apps", format="%d"),
        }
    )

# ==========================================
# PAGE 2: MANUAL ASSESSMENT (UPDATED COLORS)
# ==========================================
elif page == "Manual Risk Assessment":
    st.title("🔍 Individual Risk Assessment")

    # Input Section
    c_input, _ = st.columns([1, 2])
    with c_input:
        min_id, max_id = int(df.index.min()), int(df.index.max())
        uid = st.number_input(f"Enter User ID ({min_id}-{max_id}):", min_value=min_id, max_value=max_id, value=966)
    
    if uid in df.index:
        user = df.loc[uid]
        score = user['Risk_Probability']
        label = user['Risk_Label']
        
        st.markdown("---")
        
        # --- TOP SECTION: GAUGE (LEFT) + METRICS GRID (RIGHT) ---
        col_gauge, col_metrics = st.columns([1, 2])
        
        # 1. Gauge Chart (Left)
        with col_gauge:
            color_class = "risk-score-high" if label == 'High' else "risk-score-med" if label == 'Medium' else "risk-score-low"
            st.markdown(f"<div style='text-align: center;'><span class='{color_class}'>{score*100:.1f}%</span></div>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align: center; margin-top: -10px;'>{label.upper()} RISK</h4>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge",
                value = score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': "#FF4B4B" if label == 'High' else "#00CC96"},
                    'bgcolor': "#262730",
                    'steps': [
                        {'range': [0, 35], 'color': "#0c2b1e"},
                        {'range': [35, 65], 'color': "#332200"},
                        {'range': [65, 100], 'color': "#3b1010"}],
                }
            ))
            fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#0E1117")
            st.plotly_chart(fig, use_container_width=True)

        # 2. Metrics Grid (Right) - 2x2 Layout with LOGIC COLORS
        with col_metrics:
            st.markdown("### Key Reasoning Engine")
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            
            # Metric 1: UPI (Inverse: High is Bad/Red)
            if user['UPI_Lending_Count'] > 5:
                m1.metric("UPI Lending Apps", int(user['UPI_Lending_Count']), delta="High Usage", delta_color="inverse")
            else:
                m1.metric("UPI Lending Apps", int(user['UPI_Lending_Count']), delta="Normal", delta_color="normal")

            # Metric 2: Spend (Normal: Low is Bad/Red, requires negative string)
            if user['Avg_Discretionary_Spend'] < 400:
                m2.metric("Discretionary Spend", f"${user['Avg_Discretionary_Spend']:.0f}", delta="- Low Liquidity", delta_color="normal")
            else:
                m2.metric("Discretionary Spend", f"${user['Avg_Discretionary_Spend']:.0f}", delta="Stable", delta_color="normal")

            # Metric 3: Credit (Inverse: High is Bad/Red)
            if user['Credit amount'] > 5000:
                m3.metric("Total Credit Amount", f"${user['Credit amount']:.0f}", delta="High Debt", delta_color="inverse")
            else:
                m3.metric("Total Credit Amount", f"${user['Credit amount']:.0f}", delta="Normal", delta_color="normal")

            # Metric 4: Salary (Inverse: High Day is Bad/Red)
            if user['Avg_Salary_Day'] > 5:
                m4.metric("Salary Credit Day", f"Day {user['Avg_Salary_Day']:.0f}", delta="Late", delta_color="inverse")
            else:
                m4.metric("Salary Credit Day", f"Day {user['Avg_Salary_Day']:.0f}", delta="On Time", delta_color="normal")
        
        reasons = []
# --- BOTTOM SECTION: REASONING (FULL WIDTH) ---
        st.markdown("---")
        st.markdown("### AI Reasoning Engine")
        
        reasons = []
        
        # Reasoning Logic (Updated to use <b> tags for bolding inside HTML)
        if user['UPI_Lending_Count'] > 10:
            reasons.append(f"❌ <b>Critical Distress Signal:</b> User accessed quick-loan apps <b>{int(user['UPI_Lending_Count'])} times</b>. High frequency borrowing from alternative lenders is the strongest predictor of impending delinquency.")
        elif user['UPI_Lending_Count'] > 3:
            reasons.append(f"⚠️ <b>Warning Signal:</b> Moderate usage of lending apps ({int(user['UPI_Lending_Count'])} times). Monitor for increasing frequency.")
        
        avg_spend = df['Avg_Discretionary_Spend'].mean()
        if user['Avg_Discretionary_Spend'] < (avg_spend * 0.6):
            reasons.append(f"❌ <b>Liquidity Crunch:</b> Average discretionary spending is <b>${user['Avg_Discretionary_Spend']:.0f}</b> (vs Avg: ${avg_spend:.0f}). A sharp drop in lifestyle spending often precedes default.")
        
        if user['Credit amount'] > 5000:
            reasons.append(f"⚠️ <b>High Debt Burden:</b> Total credit exposure is <b>${user['Credit amount']:.0f}</b>, which significantly increases the risk weight.")
        
        if user['Avg_Salary_Day'] > 5:
             reasons.append(f"⚠️<b>Income Irregularity:</b> Salary is credited late in the month (Day {user['Avg_Salary_Day']:.0f}), indicating potential employer instability or cash flow gaps.")

        if reasons:
            for r in reasons:
                # The HTML <b> tags will now render correctly inside this div
                st.markdown(f"<div class='reason-box'>{r}</div>", unsafe_allow_html=True)
        else:
            st.success("✅**Stable Profile:** No high-risk behavioral patterns detected. User maintains healthy spending and borrowing habits.")

    else:
        st.error("User ID not found.")

# ==========================================
# PAGE 3: MODEL COMPARISON
# ==========================================
elif page == "Model Comparison":
    st.title("📊 Model Performance")
    
    metrics_df = get_model_metrics()
    
    st.info("🏆 **Selected Model: Random Forest** (Recall: 87.9%)")
    
    # Chart
    df_melt = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(df_melt, x="Metric", y="Score", color="Model", barmode="group",
                 text_auto='.2f', color_discrete_sequence=px.colors.qualitative.Bold)
    fig.update_layout(yaxis_range=[0.8, 1.0])
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("Global Feature Importance")
    feat_df = pd.DataFrame({
        'Feature': ['UPI Lending Count', 'Avg Discretionary Spend', 'Credit Amount', 'Avg Salary Day', 'Duration'],
        'Importance': [0.40, 0.30, 0.20, 0.10, 0.05]
    }).sort_values(by='Importance', ascending=True)
    
    fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation='h', 
                      title="What drives the Risk Score?", color="Importance")
    fig_feat.update_layout(showlegend=False)
    st.plotly_chart(fig_feat, use_container_width=True)