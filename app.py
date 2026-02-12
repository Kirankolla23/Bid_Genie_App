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
    try:
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
    expected_cols = scaler.feature_names_in_
    
    # Use Median (center_) for missing values
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
            
    return input_df[expected_cols], expected_cols

def predict_ai_cost(input_df, scaler, base_models, meta_model):
    if not all([scaler, base_models, meta_model]): return None
    input_filled, cols = smart_fill_missing_features(input_df.copy(), scaler)
    X_scaled = scaler.transform(input_filled)
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
    def full_system_predict(X_scaled_array):
        p_rf = _base_models['rf'].predict_proba(X_scaled_array)[:, 1]
        p_xgb = _base_models['xgb'].predict_proba(X_scaled_array)[:, 1]
        p_log = _base_models['log_reg'].predict_proba(X_scaled_array)[:, 1]
        p_svc = _base_models['svc'].predict_proba(X_scaled_array)[:, 1]
        meta_features = pd.DataFrame({'RF_Prob': p_rf, 'XGB_Prob': p_xgb, 'Log_Prob': p_log, 'SVC_Prob': p_svc})
        inner_m = getattr(_meta_model, 'estimator', getattr(_meta_model, 'base_estimator', _meta_model))
        if hasattr(inner_m, 'feature_names_in_') and 'XGB_SVC_Inter' in inner_m.feature_names_in_:
             meta_features['XGB_SVC_Inter'] = p_xgb * p_svc
        if hasattr(inner_m, 'feature_names_in_'): meta_features = meta_features[inner_m.feature_names_in_]
        return _meta_model.predict_proba(meta_features)[:, 1]

    if _scaler is None: return None
    n_features = len(_scaler.feature_names_in_)
    background = np.zeros((1, n_features)) 
    return shap.KernelExplainer(full_system_predict, background)

if scaler_class and base_class and meta_class:
    system_explainer = get_system_explainer(base_class, meta_class, scaler_class)

def optimize_bid_with_stacking(input_df_raw, base_models, meta_model, scaler):
    if not all([scaler, base_models, meta_model]): return None, None, None, None
    
    # 1. Determine Cost
    cost = 100.0
    if 'total_cost_estimate_crores' in input_df_raw.columns:
        cost = input_df_raw['total_cost_estimate_crores'].values[0]
    elif 'Estimated_Cost' in input_df_raw.columns:
        cost = input_df_raw['Estimated_Cost'].values[0]

    # 2. Fill Missing Columns
    base_input_df, expected_cols = smart_fill_missing_features(input_df_raw.copy(), scaler)
    
    possible_markups = np.arange(0.01, 0.20, 0.005) 
    results = []
    
    # --- IDENTIFY TARGET COLUMNS ---
    target_markup_cols = [c for c in expected_cols if 'markup' in c.lower() or 'pct' in c.lower()]
    target_bid_cols = [c for c in expected_cols if 'bid' in c.lower() and 'price' in c.lower()]
    target_cost_cols = [c for c in expected_cols if 'cost' in c.lower()]

    for markup in possible_markups:
        current_bid = cost * (1 + markup)
        input_data = base_input_df.copy()
        
        # --- THE FIX: INTELLIGENT SCALING UPDATES ---
        
        # 1. Update Markup Columns (Handle Decimals vs Percent)
        for col in target_markup_cols:
            if 'pct' in col.lower() or 'percent' in col.lower():
                 input_data[col] = markup * 100.0  # Send 19.5
            else:
                 input_data[col] = markup          # Send 0.195
        
        # 2. Update Bid Price Columns
        for col in target_bid_cols:
            input_data[col] = current_bid
            
        # 3. Update Cost Columns
        for col in target_cost_cols:
            input_data[col] = cost
        
        # Fallback Hard Updates (Explicit Logic)
        input_data['My_Markup'] = markup
        input_data['My_Bid_Price_Crores'] = current_bid
        input_data['total_cost_estimate_crores'] = cost
        
        # Extra safety for known variations
        if 'actual_markup_pct' in input_data.columns: input_data['actual_markup_pct'] = markup * 100.0
        if 'Markup_Percent' in input_data.columns: input_data['Markup_Percent'] = markup * 100.0

        input_data = input_data[expected_cols]
        input_data_scaled = scaler.transform(input_data)

        try:
            p_rf  = base_models['rf'].predict_proba(input_data_scaled)[:, 1][0]
            p_xgb = base_models['xgb'].predict_proba(input_data_scaled)[:, 1][0]
            p_log = base_models['log_reg'].predict_proba(input_data_scaled)[:, 1][0]
            p_svc = base_models['svc'].predict_proba(input_data_scaled)[:, 1][0]
        except: return None, None, None, None

        meta_features = pd.DataFrame({'RF_Prob': [p_rf], 'XGB_Prob': [p_xgb], 'Log_Prob': [p_log], 'SVC_Prob': [p_svc]})
        inner_m = getattr(meta_model, 'estimator', getattr(meta_model, 'base_estimator', meta_model))
        if hasattr(inner_m, 'feature_names_in_') and 'XGB_SVC_Inter' in inner_m.feature_names_in_:
            meta_features['XGB_SVC_Inter'] = p_xgb * p_svc
        if hasattr(inner_m, 'feature_names_in_'): meta_features = meta_features[inner_m.feature_names_in_]

        win_prob = meta_model.predict_proba(meta_features)[:, 1][0] 			
        
        # 95% Confidence Interval Logic
        np.random.seed(42) 
        simulated_costs = np.random.normal(loc=cost, scale=cost * 0.05, size=1000)
        simulated_profits_if_won = current_bid - simulated_costs
        risk_prob = np.mean(simulated_profits_if_won < 0)
        lower_bound_sim = np.percentile(simulated_profits_if_won, 2.5) * win_prob
        upper_bound_sim = np.percentile(simulated_profits_if_won, 97.5) * win_prob
        
        results.append({
            'Markup_Percent': markup * 100,
            'Bid_Price': current_bid,
            'Final_Win_Prob': win_prob, 
            'Expected_Profit': (current_bid - cost) * win_prob,
            'Risk_of_Loss_Prob': risk_prob * 100,
            'Lower_Bound': lower_bound_sim,
            'Upper_Bound': upper_bound_sim,
            'Scaled_Features': input_data_scaled,
            'Raw_Display_Features': input_data[expected_cols],
            'DEBUG_RF': p_rf, 'DEBUG_XGB': p_xgb, 'DEBUG_SVC': p_svc, 'DEBUG_LOG': p_log
        })

    df_results = pd.DataFrame(results)
    best_idx = df_results['Expected_Profit'].idxmax()
    return df_results.loc[best_idx], df_results, df_results.at[best_idx, 'Scaled_Features'], df_results.at[best_idx, 'Raw_Display_Features']

# ==========================================
# UI SECTION
# ==========================================
st.title("🏗️ Bid Genie: Construction Bid Optimizer")

default_keys = {
    'cost_val': 100.0, 'markup_val': 0.0, 'dur_val': 730, 'comp_val': 5, 
    'tech_val': 85, 'reg_idx': 0, 'refresh_id': 0, 'project_data': pd.DataFrame()
}
for key, val in default_keys.items():
    if key not in st.session_state: st.session_state[key] = val

def process_upload():
    uploaded = st.session_state['file_uploader_widget']
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'): df = pd.read_csv(uploaded)
            else: df = pd.read_excel(uploaded)
            df.columns = df.columns.str.strip()
            st.session_state['project_data'] = df
            
            # --- Update Sidebar Variables ---
            if 'total_cost_estimate_crores' in df.columns: st.session_state['cost_val'] = float(df['total_cost_estimate_crores'].iloc[0])
            elif 'Estimated_Cost' in df.columns: st.session_state['cost_val'] = float(df['Estimated_Cost'].iloc[0])
            if 'My_Markup' in df.columns: st.session_state['markup_val'] = float(df['My_Markup'].iloc[0]) * 100
            elif 'Actual_Markup' in df.columns: st.session_state['markup_val'] = float(df['Actual_Markup'].iloc[0])
            if 'No_of_Competitors' in df.columns: st.session_state['comp_val'] = int(df['No_of_Competitors'].iloc[0])
            if 'time_for_completion_days' in df.columns: st.session_state['dur_val'] = int(df['time_for_completion_days'].iloc[0])
            
            # --- THE FIX: Read Technical Score ---
            if 'Technical_Score' in df.columns: st.session_state['tech_val'] = int(df['Technical_Score'].iloc[0])

            if 'regional_cost_index' in df.columns:
                val = float(df['regional_cost_index'].iloc[0])
                closest_idx = 0
                min_diff = 999
                for i, opt in enumerate(REGION_OPTIONS):
                    if abs(val - opt) < min_diff: min_diff = abs(val - opt); closest_idx = i
                st.session_state['reg_idx'] = closest_idx
            st.session_state['refresh_id'] += 1
        except Exception as e: st.error(f"Error parsing file: {e}")

st.sidebar.header("📂 Data Source")
st.sidebar.file_uploader("Upload Excel/CSV (Model Ready)", type=['csv', 'xlsx', 'xls'], key='file_uploader_widget', on_change=process_upload)
if not st.session_state['project_data'].empty:
    with st.sidebar.expander("🔍 Input Preview", expanded=False): st.dataframe(st.session_state['project_data'].head(1).T)

st.sidebar.markdown("---")
st.sidebar.header("📝 Project Parameters")
rid = st.session_state['refresh_id']
input_cost = st.sidebar.number_input("Estimated Cost (Cr)", 1.0, 10000.0, step=1.0, value=st.session_state['cost_val'], key=f"cost_{rid}")
manual_markup = st.sidebar.number_input("Actual Markup (%)", 0.0, 100.0, step=0.1, value=st.session_state['markup_val'], key=f"markup_{rid}")
reg_index_selected = st.sidebar.selectbox("Region Cost Index", options=REGION_OPTIONS, index=st.session_state['reg_idx'], key=f"reg_{rid}")
st.session_state['reg_idx'] = REGION_OPTIONS.index(reg_index_selected)
duration = st.sidebar.number_input("Duration (Days)", 30, 3000, value=st.session_state['dur_val'], key=f"dur_{rid}")
competitors = st.sidebar.number_input("Competitors", 1, 50, value=st.session_state['comp_val'], key=f"comp_{rid}")
tech_score = st.sidebar.slider("Tech Score", 0, 100, value=st.session_state['tech_val'], key=f"tech_{rid}")

st.session_state['cost_val'] = input_cost
st.session_state['markup_val'] = manual_markup
st.session_state['comp_val'] = competitors
st.session_state['dur_val'] = duration
st.session_state['tech_val'] = tech_score

if not st.session_state['project_data'].empty: final_input_df = st.session_state['project_data'].copy()
else: final_input_df = pd.DataFrame([{}])

def get_val_adv(col, default):
    if not final_input_df.empty and col in final_input_df.columns: return float(final_input_df[col].iloc[0])
    return float(default)

final_input_df['total_cost_estimate_crores'] = input_cost
final_input_df['My_Markup'] = manual_markup / 100.0
final_input_df['regional_cost_index'] = reg_index_selected
final_input_df['time_for_completion_days'] = duration
final_input_df['No_of_Competitors'] = competitors
final_input_df['Technical_Score'] = tech_score

with st.sidebar.expander("🔧 Technical Details"):
    c1, c2 = st.columns(2)
    check_rail = c1.number_input("Check Rail", 0.0, 1000.0, get_val_adv('total_check_rail_quantity_mt', 0.0))
    dlp_days = c2.number_input("DLP", 0, 2000, int(get_val_adv('dlp_period_days', 365)))
    c3, c4 = st.columns(2)
    main_turnouts = c3.number_input("Main Turnouts", 0, 100, int(get_val_adv('mainline_turnouts', 15)))
    depot_turnouts = c4.number_input("Depot Turnouts", 0, 100, int(get_val_adv('depot_turnouts', 0)))
    c5, c6 = st.columns(2)
    ug_km = c5.number_input("Underground (km)", 0.0, 200.0, get_val_adv('underground_tkm', 0.0))
    el_km = c6.number_input("Elevated (km)", 0.0, 200.0, get_val_adv('elevated_tkm', 0.0))
    final_input_df['total_check_rail_quantity_mt'] = check_rail
    final_input_df['dlp_period_days'] = dlp_days
    final_input_df['mainline_turnouts'] = main_turnouts
    final_input_df['depot_turnouts'] = depot_turnouts
    final_input_df['underground_tkm'] = ug_km
    final_input_df['elevated_tkm'] = el_km
    for c in ['depot_included', 'project_delivery_method']: final_input_df[c] = get_val_adv(c, 0)

st.sidebar.markdown("---")
if st.sidebar.button(" Estimate Cost with AI"):
    if scaler_reg is None: st.sidebar.error("Regressor models missing!")
    else:
        estimated_ai_cost = predict_ai_cost(final_input_df, scaler_reg, base_reg, meta_reg)
        if estimated_ai_cost:
             st.success(f"AI Estimated Cost: ₹{estimated_ai_cost:.2f} Cr")
             st.session_state['cost_val'] = float(estimated_ai_cost)
             st.session_state['refresh_id'] += 1
             st.rerun()

if st.button(" Analyze Bid"):
    if not scaler_class: st.error("Models not loaded.")
    else:
        with st.spinner('Optimizing...'):
            best, df_sim, best_scaled, best_raw = optimize_bid_with_stacking(final_input_df, base_class, meta_class, scaler_class)

        if best is not None:
            win_p = best['Final_Win_Prob']
            color = "#27ae60" if win_p >= 0.30 and best['Expected_Profit'] > 0 else "#e74c3c"
            msg = "GO FOR BID" if color == "#27ae60" else " NO-BID"

            st.markdown(f"<h2 style='color:{color}'>{msg}</h2>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Markup", f"{best['Markup_Percent']:.2f}%")
            c2.metric("Bid Price", f"₹{best['Bid_Price']:.2f} Cr")
            c3.metric("Win Prob", f"{win_p*100:.1f}%")
            c4.metric("Exp. Profit", f"₹{best['Expected_Profit']:.2f} Cr")
            
            with st.expander(" Debug: Why this probability?", expanded=False):
                st.write("**Base Model Probabilities:**")
                cols = st.columns(4)
                cols[0].metric("RF", f"{best['DEBUG_RF']:.4f}")
                cols[1].metric("XGB", f"{best['DEBUG_XGB']:.4f}")
                cols[2].metric("SVC", f"{best['DEBUG_SVC']:.4f}")
                cols[3].metric("LogReg", f"{best['DEBUG_LOG']:.4f}")

            t1, t2, t3 = st.tabs(["Markup - Win Prob.", "Risk Analysis", "SHAP"])
            with t1:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_sim['Markup_Percent'], df_sim['Expected_Profit'], color='green', lw=3, label='Exp. Profit')
                ax.set_ylabel("Profit (Cr)", color='green')
                ax.axvline(manual_markup, color='black', ls=':', label='Actual Markup')
                ax.fill_between(df_sim['Markup_Percent'], df_sim['Lower_Bound'], df_sim['Upper_Bound'], color='green', alpha=0.15, label='95% Confidence Interval')
                ax2 = ax.twinx()
                ax2.plot(df_sim['Markup_Percent'], df_sim['Final_Win_Prob'], color='blue', ls='--', label='Win Prob')
                ax2.set_ylabel("Win Probability", color='blue')
                optimal_m = best['Markup_Percent']
                ax.plot(optimal_m, best['Expected_Profit'], 'ro', markersize=8, zorder=10, label='Optimal Bid')
                ax.annotate(f"Markup: {optimal_m:.1f}%\nProfit: ₹{best['Expected_Profit']:.1f}Cr", xy=(optimal_m, best['Expected_Profit']), xytext=(optimal_m+1, best['Expected_Profit']), arrowprops=dict(facecolor='black', arrowstyle='->'))
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='upper right')
                st.pyplot(fig)
                st.markdown("### 📊 Strategy Data Table")
                st.dataframe(df_sim[['Markup_Percent', 'Bid_Price', 'Expected_Profit', 'Final_Win_Prob']].style.background_gradient(cmap='Greens', subset=['Expected_Profit']))
            with t2:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_sim['Markup_Percent'], df_sim['Expected_Profit'], color='green', label='Exp. Profit')
                ax.set_ylabel("Profit (Cr)", color='green')
                ax2 = ax.twinx()
                ax2.plot(df_sim['Markup_Percent'], df_sim['Risk_of_Loss_Prob'], color='red', label='Risk of Loss')
                ax2.set_ylabel("Risk %", color='red')
                risk_at_optimal = best['Risk_of_Loss_Prob']
                ax2.plot(optimal_m, risk_at_optimal, 'ro', markersize=8, zorder=10, label='Optimal Risk')
                ax2.annotate(f"Risk: {risk_at_optimal:.1f}%", xy=(optimal_m, risk_at_optimal), xytext=(optimal_m+1, risk_at_optimal+5), arrowprops=dict(facecolor='black', arrowstyle='->'))
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc='center right')
                st.pyplot(fig)
                st.markdown("### ⚠️ Risk Analysis Data")
                st.dataframe(df_sim[['Markup_Percent', 'Risk_of_Loss_Prob', 'Expected_Profit']].style.background_gradient(cmap='Reds', subset=['Risk_of_Loss_Prob']))
            with t3:
                shap_vals = system_explainer.shap_values(best_scaled, nsamples=50)
                if isinstance(shap_vals, list): sv = shap_vals[1][0]
                elif len(shap_vals.shape)==3: sv = shap_vals[0,:,1]
                else: sv = shap_vals[0]
                ev = system_explainer.expected_value
                bv = ev[1] if isinstance(ev, (list, np.ndarray)) and len(ev) > 1 else (ev[0] if isinstance(ev, (list, np.ndarray)) else ev)
                exp = shap.Explanation(values=sv, base_values=bv, data=best_raw.iloc[0].values, feature_names=scaler_class.feature_names_in_)
                fig = plt.figure(figsize=(10,6))
                shap.plots.waterfall(exp, show=False, max_display=12)
                legend_text = (r"$\bf{E[f(x)]}$: Avg Win Prob (Train)" + "\n" + r"$\bf{f(x)}$: Pred Win Prob" + "\n" + r"$\bf{Threshold}$: 0.390")
                plt.gca().text(0.02, 0.02, legend_text, transform=plt.gca().transAxes, fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.9))
                if 'Result' in st.session_state['project_data'].columns: actual_res = st.session_state['project_data']['Result'].iloc[0]
                else: actual_res = 0 
                res_txt = "WIN" if actual_res == 1 else "LOSS"
                box_clr = "#27ae60" if res_txt == "WIN" else "#c0392b"
                idx_closest = (df_sim['Markup_Percent'] - manual_markup).abs().idxmin()
                actual_win_p = df_sim.loc[idx_closest, 'Final_Win_Prob']
                summary_text = f"ACTUAL RESULT: {res_txt}\nActual Markup: {manual_markup:.2f}%\nActual Win Prob: {actual_win_p*100:.1f}%"
                plt.gca().text(0.98, 0.98, summary_text, transform=plt.gca().transAxes, fontsize=10, fontweight='bold', ha='right', va='top', color='white', bbox=dict(facecolor=box_clr, alpha=0.9, pad=0.5, edgecolor='white'))
                st.pyplot(fig)
            
            st.download_button("📥 Download Report", df_sim.to_csv().encode('utf-8'), "bid_report.csv")
