import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import shap 

# --- GLOBAL RANDOM SEED ---
np.random.seed(42)

try:
    import xgboost
except ImportError:
    pass

st.set_page_config(page_title="Bid Genie", layout="wide", page_icon="🏗️")
REGION_OPTIONS = [1.0, 1.2, 1.5]

# ==========================================
# 1. INTELLIGENT MODEL LOADING
# ==========================================
@st.cache_resource
def load_models():
    """Loads the serialized model files from the repository."""
    try:
        # These names MUST match the files in your GitHub repo
        scaler_cls = joblib.load('final_scaler.pkl')
        base_cls = joblib.load('final_base_models_dict.pkl')
        meta_cls = joblib.load('final_meta_model.pkl')
    except Exception as e:
        st.error(f"CRITICAL ERROR: Could not load Classifier models. {e}")
        return None, None, None, None, None, None

    scaler_reg, base_reg, meta_reg = None, None, None
    if os.path.exists('final_scaler_reg.pkl'):
        try:
            scaler_reg = joblib.load('final_scaler_reg.pkl')
            base_reg = joblib.load('final_base_models_reg_dict.pkl')
            meta_reg = joblib.load('final_meta_model_reg.pkl')
        except: pass
    
    return scaler_cls, base_cls, meta_cls, scaler_reg, base_reg, meta_reg

scaler_class, base_class, meta_class, scaler_reg, base_reg, meta_reg = load_models()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def smart_fill_missing_features(input_df, scaler):
    """Fills missing columns using the median values from training state."""
    expected_cols = scaler.feature_names_in_
    
    if hasattr(scaler, 'center_'): defaults = scaler.center_
    elif hasattr(scaler, 'mean_'): defaults = scaler.mean_
    else: defaults = np.zeros(len(expected_cols))

    for i, col in enumerate(expected_cols):
        if col not in input_df.columns:
            # Case-insensitive match check
            found = False
            for input_col in input_df.columns:
                if input_col.lower().strip() == col.lower().strip():
                    input_df[col] = input_df[input_col]
                    found = True
                    break
            if not found:
                input_df[col] = defaults[i]
            
    return input_df, expected_cols

def predict_ai_cost(input_df, scaler, base_models, meta_model):
    """Predicts Project Cost using the Regressor model stack."""
    if not all([scaler, base_models, meta_model]): return None
    expected_reg = scaler.feature_names_in_
    input_filled, _ = smart_fill_missing_features(input_df.copy(), scaler)
    # Force alignment for regression
    input_aligned = input_filled.reindex(columns=expected_reg).fillna(0)
    X_scaled = scaler.transform(input_aligned)
    try:
        p_rf = base_models['rf'].predict(X_scaled)
        p_xgb = base_models['xgb'].predict(X_scaled)
        p_ridge = base_models['ridge'].predict(X_scaled)
        p_svr = base_models['svr'].predict(X_scaled)
    except: return None

    meta_features = pd.DataFrame({'RF': p_rf, 'XGB': p_xgb, 'SVR': p_svr, 'Ridge': p_ridge})
    inner_m = getattr(meta_model, 'estimator', getattr(meta_model, 'base_estimator', meta_model))
    if hasattr(inner_m, 'feature_names_in_'): meta_features = meta_features[inner_m.feature_names_in_]
    return meta_model.predict(meta_features)[0]

@st.cache_resource
def get_system_explainer(_base_models, _meta_model, _scaler):
    """Wraps the stacking system for SHAP interpretation."""
    def full_system_predict(X_scaled_array):
        p_rf = _base_models['rf'].predict_proba(X_scaled_array)[:, 1]
        p_xgb = _base_models['xgb'].predict_proba(X_scaled_array)[:, 1]
        p_log = _base_models['log_reg'].predict_proba(X_scaled_array)[:, 1]
        p_svc = _base_models['svc'].predict_proba(X_scaled_array)[:, 1]
        meta_features = pd.DataFrame({'RF_Prob': p_rf, 'XGB_Prob': p_xgb, 'Log_Prob': p_log, 'SVC_Prob': p_svc})
        inner_m = getattr(_meta_model, 'estimator', getattr(_meta_model, 'base_estimator', _meta_model))
        if hasattr(inner_m, 'feature_names_in_'): meta_features = meta_features[inner_m.feature_names_in_]
        return _meta_model.predict_proba(meta_features)[:, 1]

    if _scaler is None: return None
    n_features = len(_scaler.feature_names_in_)
    background = np.zeros((1, n_features)) 
    return shap.KernelExplainer(full_system_predict, background)

if scaler_class and base_class and meta_class:
    system_explainer = get_system_explainer(base_class, meta_class, scaler_class)

def optimize_bid_with_stacking(input_df_raw, base_models, meta_model, scaler):
    """The Final Corrected Simulation Engine that fixes the 16.3% error."""
    if not all([scaler, base_models, meta_model]): return None, None, None, None
    
    # --- CORE FIX: Use your verified feature list ---
    expected_cols = scaler.feature_names_in_
    
    cost = 100.0
    if 'total_cost_estimate_crores' in input_df_raw.columns:
        cost = input_df_raw['total_cost_estimate_crores'].values[0]
    elif 'Estimated_Cost' in input_df_raw.columns:
        cost = input_df_raw['Estimated_Cost'].values[0]

    base_input_df, _ = smart_fill_missing_features(input_df_raw.copy(), scaler)
    
    possible_markups = np.arange(0.01, 0.30, 0.005) 
    results = []
    
    for markup in possible_markups:
        current_bid = cost * (1 + markup)
        input_data = base_input_df.copy()
        
        # --- CAUSE CORRECTION: Exact Name assignment ---
        input_data['My_Bid_Price_Crores'] = current_bid
        input_data['My_Markup'] = markup
        input_data['total_cost_estimate_crores'] = cost
        
        # --- THE FIX: Force strict column sequence ---
        input_data_final = input_data.reindex(columns=expected_cols).fillna(0)
        input_data_scaled = scaler.transform(input_data_final)

        try:
            p_rf  = base_models['rf'].predict_proba(input_data_scaled)[:, 1][0]
            p_xgb = base_models['xgb'].predict_proba(input_data_scaled)[:, 1][0]
            p_log = base_models['log_reg'].predict_proba(input_data_scaled)[:, 1][0]
            p_svc = base_models['svc'].predict_proba(input_data_scaled)[:, 1][0]
        except: return None, None, None, None

        meta_features = pd.DataFrame({'RF_Prob': [p_rf], 'XGB_Prob': [p_xgb], 'Log_Prob': [p_log], 'SVC_Prob': [p_svc]})
        inner_m = getattr(meta_model, 'estimator', getattr(meta_model, 'base_estimator', meta_model))
        if hasattr(inner_m, 'feature_names_in_'):
            meta_features = meta_features[inner_m.feature_names_in_]

        win_prob = meta_model.predict_proba(meta_features)[:, 1][0] 			
        
        # Monte Carlo Simulation Logic
        np.random.seed(42) 
        sim_costs = np.random.normal(loc=cost, scale=cost * 0.05, size=1000)
        sim_profits = current_bid - sim_costs
        risk_prob = np.mean(sim_profits < 0)
        
        results.append({
            'Markup_Percent': markup * 100,
            'Bid_Price': current_bid,
            'Final_Win_Prob': win_prob, 
            'Expected_Profit': (current_bid - cost) * win_prob,
            'Risk_of_Loss_Prob': risk_prob * 100,
            'Lower_Bound': np.percentile(sim_profits, 2.5) * win_prob,
            'Upper_Bound': np.percentile(sim_profits, 97.5) * win_prob,
            'Scaled_Features': input_data_scaled,
            'Raw_Display_Features': input_data_final,
            'DEBUG_RF': p_rf, 'DEBUG_XGB': p_xgb, 'DEBUG_SVC': p_svc, 'DEBUG_LOG': p_log
        })

    df_results = pd.DataFrame(results)
    best_idx = df_results['Expected_Profit'].idxmax()
    return df_results.loc[best_idx], df_results, df_results.at[best_idx, 'Scaled_Features'], df_results.at[best_idx, 'Raw_Display_Features']

# ==========================================
# 3. UI SECTION
# ==========================================
st.title("🏗️ Bid Genie: Construction Bid Optimizer")

# Session Persistence
if 'refresh_id' not in st.session_state: st.session_state['refresh_id'] = 0
if 'project_data' not in st.session_state: st.session_state['project_data'] = pd.DataFrame()
if 'cost_val' not in st.session_state: st.session_state['cost_val'] = 480.7
if 'markup_val' not in st.session_state: st.session_state['markup_val'] = 10.75
if 'comp_val' not in st.session_state: st.session_state['comp_val'] = 4
if 'dur_val' not in st.session_state: st.session_state['dur_val'] = 954
if 'tech_val' not in st.session_state: st.session_state['tech_val'] = 88
if 'reg_idx' not in st.session_state: st.session_state['reg_idx'] = 0

def on_file_change():
    u = st.session_state['file_uploader_widget']
    if u:
        try:
            df = pd.read_csv(u) if u.name.endswith('.csv') else pd.read_excel(u)
            df.columns = df.columns.str.strip()
            st.session_state['project_data'] = df
            st.session_state['refresh_id'] += 1
        except Exception as e: st.error(f"Error: {e}")

st.sidebar.header("📂 Data Source")
st.sidebar.file_uploader("Upload Project Data", type=['csv', 'xlsx', 'xls'], key='file_uploader_widget', on_change=on_file_change)

st.sidebar.markdown("---")
st.sidebar.header("📝 Project Parameters")
rid = st.session_state['refresh_id']
input_cost = st.sidebar.number_input("Estimated Cost (Cr)", 1.0, 10000.0, value=st.session_state['cost_val'], key=f"cost_{rid}")
manual_markup = st.sidebar.number_input("Actual Markup (%)", 0.0, 100.0, value=st.session_state['markup_val'], key=f"markup_{rid}")
reg_index = st.sidebar.selectbox("Region Cost Index", options=REGION_OPTIONS, index=st.session_state['reg_idx'], key=f"reg_{rid}")
duration = st.sidebar.number_input("Duration (Days)", 30, 3000, value=st.session_state['dur_val'], key=f"dur_{rid}")
competitors = st.sidebar.number_input("Competitors", 1, 50, value=st.session_state['comp_val'], key=f"comp_{rid}")
tech_score = st.sidebar.slider("Tech Score", 0, 100, value=st.session_state['tech_val'], key=f"tech_{rid}")

st.session_state.update({'cost_val': input_cost, 'markup_val': manual_markup, 'comp_val': competitors, 'dur_val': duration, 'tech_val': tech_score})

# Prepare Dataframe
if not st.session_state['project_data'].empty: f_df = st.session_state['project_data'].copy()
else: f_df = pd.DataFrame([{}])

f_df['total_cost_estimate_crores'] = input_cost
f_df['My_Markup'] = manual_markup / 100.0
f_df['regional_cost_index'] = reg_index
f_df['time_for_completion_days'] = duration
f_df['No_of_Competitors'] = competitors
f_df['Technical_Score'] = tech_score

if st.button(" Analyze Bid"):
    if not scaler_class: st.error("Models not found in repo.")
    else:
        with st.spinner('Synchronizing and calculating...'):
            best, df_sim, best_scaled, best_raw = optimize_bid_with_stacking(f_df, base_class, meta_class, scaler_class)

        if best is not None:
            wp = best['Final_Win_Prob']
            st.markdown(f"<h2 style='color:{'#27ae60' if wp >= 0.3 else '#e74c3c'}'>{'GO FOR BID' if wp >= 0.3 else 'NO-BID'}</h2>", unsafe_allow_html=True)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Markup", f"{best['Markup_Percent']:.2f}%")
            c2.metric("Bid Price", f"₹{best['Bid_Price']:.2f} Cr")
            c3.metric("Win Prob", f"{wp*100:.1f}%")
            c4.metric("Exp. Profit", f"₹{best['Expected_Profit']:.2f} Cr")
            
            t1, t2, t3 = st.tabs(["Strategy Curve", "Risk Metrics", "SHAP Waterfall"])
            with t1:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_sim['Markup_Percent'], df_sim['Expected_Profit'], color='green', lw=2, label="Profit")
                ax2 = ax.twinx(); ax2.plot(df_sim['Markup_Percent'], df_sim['Final_Win_Prob'], color='blue', ls='--', label="Win Prob")
                st.pyplot(fig); st.dataframe(df_sim[['Markup_Percent', 'Bid_Price', 'Expected_Profit', 'Final_Win_Prob']])
            with t2:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_sim['Markup_Percent'], df_sim['Risk_of_Loss_Prob'], color='red', lw=2); st.pyplot(fig)
            with t3:
                shap_vals = system_explainer.shap_values(best_scaled, nsamples=50)
                sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                exp = shap.Explanation(values=sv, base_values=system_explainer.expected_value, data=best_raw.iloc[0].values, feature_names=scaler_class.feature_names_in_)
                fig = plt.figure(); shap.plots.waterfall(exp, show=False); st.pyplot(fig)
