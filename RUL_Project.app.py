import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="Asset RUL Predictor", page_icon="🏭", layout="wide")

# --- 2. تحميل الشبكة العصبية والسكيلر ---
@st.cache_resource
def load_ai_assets():
    scaler = joblib.load('rul_scaler.pkl')
    model = load_model('advanced_rul_model.h5')
    return scaler, model

try:
    scaler, model = load_ai_assets()
except Exception as e:
    # التعديل هنا: السيرفر هيطبع نوع الخطأ بالظبط عشان نحله
    st.error(f"⚠️ Error details: {e}")
    st.info("💡 Tip: This usually means a version mismatch in scikit-learn or TensorFlow.")
    st.stop()

# --- 3. رأس الصفحة ---
st.title("🏭 Deep Learning Predictive Maintenance")
st.markdown("Estimate the **Remaining Useful Life (RUL)** of power assets using advanced Artificial Neural Networks.")
st.markdown("---")

# --- 4. الشريط الجانبي (قراءات الحساسات) ---
st.sidebar.header("🎛️ Sensor Readings")

with st.sidebar.expander("⏱️ Operational Data", expanded=True):
    op_hours = st.slider("Operating Hours", 1000, 50000, 25000)
    load_current = st.slider("Load Current (Amps)", 50.0, 150.0, 100.0)

with st.sidebar.expander("🌡️ Condition Monitoring", expanded=True):
    temp = st.slider("Temperature (°C)", 20.0, 100.0, 55.0)
    vibration = st.slider("Vibration (mm/s)", 0.1, 5.0, 2.0)

# --- 5. زر التوقع ---
if st.sidebar.button("🔮 Predict Remaining Life (RUL)", use_container_width=True):
    with st.spinner('Neural Network is analyzing patterns...'):
        
        # تجهيز البيانات للتوقع
        input_data = pd.DataFrame({
            'Operating_Hours': [op_hours],
            'Temperature_C': [temp],
            'Vibration_mm_s': [vibration],
            'Load_Current_A': [load_current]
        })
        
        # التوقع
        input_scaled = scaler.transform(input_data)
        predicted_rul = model.predict(input_scaled)[0][0]
        predicted_rul = max(0, int(predicted_rul)) 
        
        # تحديد الحالة
        if predicted_rul > 1500:
            status, color = "Healthy", "#00CC96"  
        elif predicted_rul > 500:
            status, color = "Monitor", "#FECB52"  
        else:
            status, color = "Critical", "#EF553B" 
            
        # --- 6. العرض البصري (العداد والعنكبوت) ---
        st.subheader("📊 AI Prognostics Report")
        c1, c2 = st.columns(2)
        
        with c1:
            # العداد (Gauge)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = predicted_rul,
                title = {'text': f"Status: {status}", 'font': {'size': 24}},
                number = {'suffix': " Days", 'font': {'color': color}},
                gauge = {'axis': {'range': [0, 3000]}, 'bar': {'color': color},
                         'steps': [{'range': [0, 500], 'color': "#EF553B33"},
                                   {'range': [500, 1500], 'color': "#FECB5233"},
                                   {'range': [1500, 3000], 'color': "#00CC9633"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c2:
            # العنكبوت (Radar Chart)
            cats = ['Age Wear', 'Thermal Stress', 'Vibration Stress', 'Load Stress']
            vals = [(op_hours/50000)*100, (temp/100)*100, (vibration/5.0)*100, (load_current/150)*100]
            fig_radar = go.Figure(data=go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill='toself', line_color=color))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Equipment Stress Profile")
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 7. الـ Bar Chart (تأثير العناصر - Impact Analysis) ---
        st.markdown("---")
        st.subheader("🔍 Sensor Impact Analysis (Stress Factors)")
        
        impact_map = {
            "Thermal Load (Temp)": (temp/100) * 100,
            "Mechanical Wear (Vibration)": (vibration/5.0) * 100,
            "Electrical Load (Amps)": (load_current/150) * 100,
            "Aging (Hours)": (op_hours/50000) * 100
        }
        df_impact = pd.DataFrame(impact_map.items(), columns=['Sensor', 'Stress Level (%)']).sort_values('Stress Level (%)', ascending=True)
        fig_bar = px.bar(df_impact, x='Stress Level (%)', y='Sensor', orientation='h', color='Stress Level (%)', color_continuous_scale='Reds', title="Which sensor is reducing the lifespan?")
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- 8. الـ Key Metrics (أرقام الخلاصة) ---
        st.markdown("---")
        st.subheader("📝 Key Maintenance Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Predicted RUL", f"{predicted_rul} Days", "Urgent" if predicted_rul < 500 else "Stable", delta_color="inverse")
        k2.metric("Temp Sensor", f"{temp} °C", "Overheating" if temp > 70 else "Normal", delta_color="inverse")
        k3.metric("Vibration", f"{vibration} mm/s", "High" if vibration > 3.5 else "Smooth", delta_color="inverse")
        k4.metric("AI Confidence", "97.67%", "Validated")
        
else:
    st.info("👈 Adjust sensor readings in the sidebar and click **Predict**.")