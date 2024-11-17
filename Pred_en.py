import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import altair as alt
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Predicción de Consumo Eléctrico - LSTM", layout="wide")

# Función para crear secuencias de datos para el entrenamiento
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Función para cargar y preprocesar datos
def load_data(file):
    try:
        # Cargar el archivo CSV
        df = pd.read_csv(file, index_col=0)
        
        # Convertir la columna Datetime a datetime
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Eliminar filas con valores faltantes
        df = df.dropna()
        
        # Ordenar por fecha
        df = df.sort_values('Datetime')
        
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

# Función para crear y entrenar el modelo LSTM
def create_model(seq_length):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Función para realizar predicciones
def predict_future(model, last_sequence, n_steps, scaler):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        # Realizar predicción
        prediction = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)
        predictions.append(prediction[0, 0])
        
        # Actualizar la secuencia
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = prediction
    
    # Desescalar predicciones
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Título de la aplicación
st.title("🔮 Predicción de Consumo Eléctrico con LSTM")

# Sidebar para configuración
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])

if uploaded_file is not None:
    # Cargar datos
    df = load_data(uploaded_file)
    
    if df is not None:
        # Parámetros de la red neuronal
        st.sidebar.header("Parámetros del Modelo")
        seq_length = st.sidebar.slider("Longitud de secuencia (horas)", 
                                     min_value=1, max_value=48, value=24)
        prediction_hours = st.sidebar.slider("Horas a predecir", 
                                           min_value=1, max_value=72, value=24)
        
        # Botón de entrenamiento
        train_button = st.sidebar.button("Entrenar Modelo")
        
        # Mostrar datos originales
        st.subheader("Vista previa de los datos")
        st.write(df.head())
        
        if train_button:
            with st.spinner('Entrenando el modelo...'):
                # Preparar datos
                data = df['Kwh'].values.reshape(-1, 1)
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data)
                
                # Crear secuencias
                X, y = create_sequences(data_scaled, seq_length)
                
                # Dividir datos en entrenamiento y validación
                train_size = int(len(X) * 0.8)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]
                
                # Crear y entrenar modelo
                model = create_model(seq_length)
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0
                )
                
                # Realizar predicciones
                last_sequence = data_scaled[-seq_length:]
                predictions = predict_future(model, last_sequence, prediction_hours, scaler)
                
                # Crear fechas para las predicciones
                last_date = df['Datetime'].iloc[-1]
                future_dates = [last_date + timedelta(hours=i+1) for i in range(len(predictions))]
                
                # Crear DataFrame con predicciones
                predictions_df = pd.DataFrame({
                    'Datetime': future_dates,
                    'Kwh_Predicted': predictions
                })
                
                # Visualización de resultados
                st.header("📈 Resultados de la Predicción")
                
                # Preparar datos para la visualización
                historical_data = df[['Datetime', 'Kwh']].copy()
                historical_data['Tipo'] = 'Histórico'
                predictions_df['Tipo'] = 'Predicción'
                
                # Combinar datos históricos y predicciones
                viz_data = pd.concat([
                    historical_data.rename(columns={'Kwh': 'Valor'}),
                    predictions_df.rename(columns={'Kwh_Predicted': 'Valor'})
                ])
                
                # Crear gráfico
                chart = alt.Chart(viz_data).mark_line().encode(
                    x=alt.X('Datetime:T', title='Fecha y Hora'),
                    y=alt.Y('Valor:Q', title='Consumo (kWh)'),
                    color=alt.Color('Tipo:N', 
                                  scale=alt.Scale(domain=['Histórico', 'Predicción'],
                                                range=['#1f77b4', '#ff7f0e']))
                ).properties(
                    width=800,
                    height=400
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
                
                # Mostrar métricas de rendimiento
                st.header("📊 Métricas del Modelo")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Error de Entrenamiento (MSE)", 
                             f"{history.history['loss'][-1]:.4f}")
                with col2:
                    st.metric("Error de Validación (MSE)", 
                             f"{history.history['val_loss'][-1]:.4f}")
                
                # Descargar predicciones
                st.header("💾 Descargar Predicciones")
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Descargar predicciones como CSV",
                    data=csv,
                    file_name="predicciones.csv",
                    mime="text/csv"
                )
                
else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")
    st.markdown("""
    El archivo CSV debe contener las siguientes columnas:
    - `Datetime`: Fecha y hora de la medición
    - `Kwh`: Consumo eléctrico en kilovatios-hora
    
    Formato esperado:
    ```
    ,Datetime,Kwh
    0,2024-11-01 00:00:00,0.14625
    1,2024-11-01 01:00:00,0.12281
    ...
    ```
    """)

# Agregar información sobre el uso
st.sidebar.markdown("""
---
### Información de Uso
- Ajusta la longitud de secuencia según el patrón temporal que quieras capturar
- Define cuántas horas hacia el futuro quieres predecir
- El modelo LSTM aprenderá patrones en los datos históricos
- Las predicciones se muestran en naranja en el gráfico
""")
