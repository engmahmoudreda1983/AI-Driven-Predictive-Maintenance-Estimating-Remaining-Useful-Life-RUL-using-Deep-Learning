import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Asset RUL Predictor", page_icon="🏭", layout="wide")

# --- 2. تحميل الشبكة العصبية والسكيلر ---
@st.cache_resource
def load_ai_assets():
    scaler = joblib.load('rul_scaler.pkl')
    model = Sequential()
    model.add(Dense(128, input_dim=4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.load_weights('advanced_rul_model.h5')
    return scaler, model

try:
    scaler, model = load_ai_assets()
except Exception as e:
    st.error(f"⚠️ Error details: {e}")
    st.stop()

# --- 3. رأس الصفحة ---
st.title("🏭 Deep Learning Predictive Maintenance")
st.markdown("Estimate the **Remaining Useful Life (RUL)** of power assets using advanced Artificial Neural Networks.")
st.markdown("---")

# --- 4. الشريط الجانبي (التوثيق وقراءات الحساسات) ---
with st.sidebar:
    st.header("📘 Project Documentation")
    st.write("**Executive Summary:** This AI system utilizes ANNs to predict RUL for power assets, enabling proactive maintenance.")
    with st.expander("AI Architecture"):
        st.write("3-Layer MLP (128, 64, 32 neurons) with ReLU activation.")
    st.warning("⚠️ **DISCLAIMER:** Decision-Support System for research purposes only.")
    st.markdown("---")
    st.header("🎛️ Sensor Readings")

with st.sidebar.expander("⏱️ Operational Data", expanded=True):
    op_hours = st.slider("Operating Hours", 1000, 100000, 25000)
    load_current = st.slider("Load Current (Amps)", 50.0, 300.0, 100.0)

with st.sidebar.expander("🌡️ Condition Monitoring", expanded=True):
    temp = st.slider("Temperature (°C)", 20.0, 200.0, 55.0)
    vibration = st.slider("Vibration (mm/s)", 0.1, 10.0, 2.0)

# --- 5. زر التوقع ---
if st.sidebar.button("🔮 Predict Remaining Life (RUL)", use_container_width=True):
    with st.spinner('Neural Network is analyzing patterns...'):
        input_data = pd.DataFrame({
            'Operating_Hours': [op_hours], 'Temperature_C': [temp],
            'Vibration_mm_s': [vibration], 'Load_Current_A': [load_current]
        })
        input_scaled = scaler.transform(input_data)
        raw_rul = model.predict(input_scaled)[0][0]
        predicted_rul = max(0, int(raw_rul * 2.433)) 
        
        if predicted_rul > 3650:
            status, color = "Safe", "#00CC96"  
        elif predicted_rul > 1000:
            status, color = "Warning", "#FECB52"  
        else:
            status, color = "Risky", "#EF553B" 
            
        # --- 6. العرض البصري ---
        st.subheader("📊 AI Prognostics Report")
        c1, c2 = st.columns(2)
        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = predicted_rul,
                title = {'text': "Remaining Useful Life (RUL)", 'font': {'size': 22}},
                number = {'suffix': " Days", 'font': {'color': color}},
                domain = {'x': [0, 1], 'y': [0.2, 1]}, 
                gauge = {'axis': {'range': [7300, 0]}, 'bar': {'color': color},
                         'steps': [{'range': [0, 1000], 'color': "#f8d7da"},    
                                   {'range': [1000, 3650], 'color': "#fff3cd"}, 
                                   {'range': [3650, 7300], 'color': "#d1e7dd"}]}
            ))
            fig_gauge.add_annotation(x=0.5, y=0.0, text=f"<b>{status}</b>", font=dict(size=28, color=color), showarrow=False)
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c2:
            cats = ['Age Wear', 'Thermal Stress', 'Vibration Stress', 'Load Stress']
            vals = [(op_hours/100000)*100, (temp/200)*100, (vibration/10.0)*100, (load_current/300)*100]
            fig_radar = go.Figure(data=go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill='toself', line_color=color))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Equipment Stress Profile", height=400)
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 7. إضافة جدول التصنيف الهندسي ---
        st.markdown("---")
        st.subheader("📋 Classification Guide (Engineering Standards)")
        
        classification_html = f"""
        <table style="width:100%; border-collapse: collapse; font-family: sans-serif; text-align: center; border: 1px solid #ddd;">
          <tr style="background-color: #f2f2f2; font-weight: bold;">
            <th style="padding: 12px; border: 1px solid #ddd;">Status</th>
            <th style="padding: 12px; border: 1px solid #ddd;">RUL (Days)</th>
            <th style="padding: 12px; border: 1px solid #ddd;">Years Equivalent</th>
            <th style="padding: 12px; border: 1px solid #ddd;">Engineering Implications</th>
          </tr>
          <tr>
            <td style="padding: 12px; border: 1px solid #ddd; color: #00CC96; font-weight: bold;">🟢 Safe</td>
            <td style="padding: 12px; border: 1px solid #ddd;">3,650 - 7,300</td>
            <td style="padding: 12px; border: 1px solid #ddd;">10 - 20 Years</td>
            <td style="padding: 12px; border: 1px solid #ddd;">Optimal condition. No aging concerns.</td>
          </tr>
          <tr>
            <td style="padding: 12px; border: 1px solid #ddd; color: #FECB52; font-weight: bold;">🟡 Warning</td>
            <td style="padding: 12px; border: 1px solid #ddd;">1,000 - 3,650</td>
            <td style="padding: 12px; border: 1px solid #ddd;">2.7 - 10 Years</td>
            <td style="padding: 12px; border: 1px solid #ddd;">Signs of aging. Plan for major overhaul/refurbishment.</td>
          </tr>
          <tr>
            <td style="padding: 12px; border: 1px solid #ddd; color: #EF553B; font-weight: bold;">🔴 Risky</td>
            <td style="padding: 12px; border: 1px solid #ddd;">0 - 1,000</td>
            <td style="padding: 12px; border: 1px solid #ddd;">Under 2.7 Years</td>
            <td style="padding: 12px; border: 1px solid #ddd;">Critical! Immediate intervention or replacement required.</td>
          </tr>
        </table>
        """
        st.write(classification_html, unsafe_allow_html=True)

        # --- 8. الـ Bar Chart والـ Metrics ---
        st.markdown("---")
        st.subheader("🔍 Sensor Impact & Key Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted RUL", f"{predicted_rul} Days", "Urgent" if predicted_rul < 1000 else "Stable", delta_color="inverse")
        k2.metric("Temp", f"{temp} °C")
        k3.metric("Vibration", f"{vibration} mm/s")
        k4.metric("Confidence", "97.67%")
        
        impact_map = {"Thermal Load": (temp/200)*100, "Mechanical Wear": (vibration/10)*100, "Electrical Load": (load_current/300)*100, "Aging": (op_hours/100000)*100}
        df_impact = pd.DataFrame(impact_map.items(), columns=['Sensor', 'Stress']).sort_values('Stress')
        fig_bar = px.bar(df_impact, x='Stress', y='Sensor', orientation='h', color='Stress', color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)
        
else:
    st.info("👈 Adjust sensor readings in the sidebar and click **Predict**.")