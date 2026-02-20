import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import shap 

# --- FIXED: ENSURE CONSISTENT RESULTS ---
np.random.seed(42)

try:
    import xgboost
except ImportError:
    pass

st.set_page_config(page_title="Bid Genie", layout="wide", page_icon="🏗️")

# --- CLEAR OLD CACHE IF MISMATCH PERSISTS ---
if st.sidebar.button("🔄 Reset App Cache"):
    st.cache_resource.clear()
    st.rerun()

REGION_OPTIONS = [1.0, 1.2, 1.5]

@st.cache_resource
def load_models():
    try:
        scaler_cls = joblib.load('final_scaler.pkl')
        base_cls = joblib.load('final_base_models_dict.pkl')
        meta_cls = joblib.load('final_meta_model.pkl')
        scaler_reg = joblib.load('final_scaler_reg.pkl') if os.path.exists('final_scaler_reg.pkl') else None
        base_reg = joblib.load('final_base_models_reg_dict.pkl') if os.path.exists('final_base_models_reg_dict.pkl') else None
        meta_reg = joblib.load('final_meta_model_reg.pkl') if os.path.exists('final_meta_model_reg.pkl') else None
        shap_background = joblib.load('frozen_shap_background.pkl') if os.path.exists('frozen_shap_background.pkl') else None
        return scaler_cls, base_cls, meta_cls, scaler_reg, base_reg, meta_reg, shap_background
    except Exception as e:
        st.error(f"⚠️ Load Error: {e}")
        return [None]*7

scaler_class, base_class, meta_class, scaler_reg, base_reg, meta_reg, frozen_background = load_models()

def smart_fill_missing_features(input_df, scaler):
    expected_cols = scaler.feature_names_in_
    defaults = scaler.mean_ if hasattr(scaler, 'mean_') else np.zeros(len(expected_cols))
    for i, col in enumerate(expected_cols):
        if col not in input_df.columns:
            input_df[col] = defaults[i]
    return input_df[expected_cols], expected_cols

def optimize_bid_with_stacking(input_df_raw, base_models, meta_model, scaler):
    if not all([scaler, base_models, meta_model]): return None, None, None, None
    cost = input_df_raw.get('total_cost_estimate_crores', [100.0]).values[0]
    base_input_df, expected_cols = smart_fill_missing_features(input_df_raw.copy(), scaler)
    results = []
    np.random.seed(42) 
    
    for markup in np.arange(0.01, 0.20, 0.005):
        current_bid = cost * (1 + markup)
        input_data = base_input_df.copy()
        input_data['My_Bid_Price_Crores'] = current_bid
        input_data['My_Markup'] = markup
        input_data['total_cost_estimate_crores'] = cost
        input_data_scaled = scaler.transform(input_data[expected_cols])

        p_rf = base_models['rf'].predict_proba(input_data_scaled)[:, 1][0]
        p_xgb = base_models['xgb'].predict_proba(input_data_scaled)[:, 1][0]
        p_log = base_models['log_reg'].predict_proba(input_data_scaled)[:, 1][0]
        p_svc = base_models['svc'].predict_proba(input_data_scaled)[:, 1][0]

        meta_features = pd.DataFrame({'RF_Prob': [p_rf], 'XGB_Prob': [p_xgb], 'Log_Prob': [p_log], 'SVC_Prob': [p_svc]})
        inner_m = getattr(meta_model, 'estimator', meta_model)
        if 'XGB_SVC_Inter' in getattr(inner_m, 'feature_names_in_', []):
            meta_features['XGB_SVC_Inter'] = p_xgb * p_svc

        win_prob = meta_model.predict_proba(meta_features[inner_m.feature_names_in_])[:, 1][0]
        sim_costs = np.random.normal(loc=cost, scale=cost * 0.05, size=1000)
        potential_profit = current_bid - cost
        
        results.append({
            'Markup_Percent': markup * 100,
            'Bid_Price': current_bid,
            'Final_Win_Prob': win_prob, 
            'Expected_Profit': potential_profit * win_prob,
            'Potential_Profit': potential_profit,
            'Risk_of_Loss_Prob': np.mean((current_bid - sim_costs) < 0) * 100,
            'CVaR_95': (current_bid - sim_costs)[(current_bid - sim_costs) <= np.percentile(current_bid - sim_costs, 5)].mean(),
            'Scaled_Features': input_data_scaled,
            'Raw_Display_Features': input_data[expected_cols]
        })

    df_results = pd.DataFrame(results)
    best_idx = df_results['Expected_Profit'].idxmax()
    return df_results.loc[best_idx], df_results, df_results.at[best_idx, 'Scaled_Features'], df_results.at[best_idx, 'Raw_Display_Features']

# --- UI LOGIC ---
st.title("🏗️ Bid Genie")
if 'project_data' not in st.session_state:
    st.session_state['project_data'] = pd.DataFrame()

uploaded_file = st.sidebar.file_uploader("Upload Data", type=['csv', 'xlsx'])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.session_state['project_data'] = df

# ... (Sidebar inputs code same as before, ensures rid = st.session_state.get('refresh_id', 0))

if st.button("Analyze Bid"):
    # ... (Optimization call same as before)
    
    with st.expander("SHAP Explanation"):
        # --- THE FIX FOR THE "LOSS/100%" BUG ---
        has_actual = False
        # Only check the ACTUAL UPLOADED file for a result, ignore manual dataframe
        if uploaded_file is not None and not st.session_state['project_data'].empty:
            if 'Win_Status' in st.session_state['project_data'].columns:
                actual_val = st.session_state['project_data']['Win_Status'].iloc[0]
                has_actual = True
        
        # ... (SHAP Plot code)
        
        if has_actual:
            is_win = str(actual_val).strip().upper() in ['1', 'WIN', 'WON', 'TRUE']
            res_txt, box_clr = ("WIN", "#27ae60") if is_win else ("LOSS", "#c0392b")
            summary_text = f"HISTORICAL: {res_txt}\nProb: {best['Final_Win_Prob']*100:.1f}%"
        else:
            res_txt, box_clr = ("NEW PROJECT", "#34495e")
            summary_text = f"STATUS: {res_txt}\nNo History\nProb: {best['Final_Win_Prob']*100:.1f}%"

        plt.gca().text(0.95, 0.95, summary_text, transform=plt.gca().transAxes, color='white', bbox=dict(facecolor=box_clr))
        st.pyplot(plt.gcf())
