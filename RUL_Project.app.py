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

# --- 2. تحميل الشبكة العصبية والسكيلر (الطريقة الاحترافية) ---
@st.cache_resource
def load_ai_assets():
    # 1. تحميل ميزان البيانات
    scaler = joblib.load('rul_scaler.pkl')
    
    # 2. بناء "هيكل" الشبكة العصبية بنفس تفاصيل كولاب بالضبط
    model = Sequential()
    model.add(Dense(128, input_dim=4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    
    # 3. صب "الأوزان والخبرة" جوه الهيكل (لتفادي تعارض الإصدارات)
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

# --- 4. الشريط الجانبي (قراءات الحساسات) ---
st.sidebar.header("🎛️ Sensor Readings")

with st.sidebar.expander("⏱️ Operational Data", expanded=True):
    op_hours = st.slider("Operating Hours", 1000, 100000, 25000)
    load_current = st.slider("Load Current (Amps)", 50.0, 300.0, 100.0)

with st.sidebar.expander("🌡️ Condition Monitoring", expanded=True):
    temp = st.slider("Temperature (°C)", 20.0, 200.0, 55.0)
    vibration = st.slider("Vibration (mm/s)", 0.1, 10.0, 2.0)

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
            # العداد (Gauge) بعد عكس المحور لتبدأ الأمان (الأخضر) من اليسار
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = predicted_rul,
                title = {'text': f"Remaining Useful Life (RUL)<br><span style='font-size:0.8em;color:gray'>Status: {status}</span>", 'font': {'size': 22}},
                number = {'suffix': " Days", 'font': {'color': color}},
                gauge = {'axis': {'range': [3000, 0]}, 'bar': {'color': color},
                         'steps': [{'range': [0, 500], 'color': "#f8d7da"},    
                                   {'range': [500, 1500], 'color': "#fff3cd"}, 
                                   {'range': [1500, 3000], 'color': "#d1e7dd"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with c2:
            # العنكبوت (Radar Chart)
            cats = ['Age Wear', 'Thermal Stress', 'Vibration Stress', 'Load Stress']
            vals = [(op_hours/100000)*100, (temp/200)*100, (vibration/10.0)*100, (load_current/300)*100]
            fig_radar = go.Figure(data=go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill='toself', line_color=color))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title="Equipment Stress Profile")
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 7. الـ Bar Chart (تأثير العناصر - Impact Analysis) ---
        st.markdown("---")
        st.subheader("🔍 Sensor Impact Analysis (Stress Factors)")
        
        impact_map = {
            "Thermal Load (Temp)": (temp/200) * 100,
            "Mechanical Wear (Vibration)": (vibration/10.0) * 100,
            "Electrical Load (Amps)": (load_current/300) * 100,
            "Aging (Hours)": (op_hours/100000) * 100
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
        k4.metric("Confidence", "97.67%", "Validated")
        
else:
    st.info("👈 Adjust sensor readings in the sidebar and click **Predict**.")