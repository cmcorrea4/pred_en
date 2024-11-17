import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import altair as alt
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Predicción de Consumo Eléctrico", layout="wide")

# Función para crear secuencias de datos para el entrenamiento
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

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

# Función para realizar predicciones
def predict_future(model, last_sequence, n_steps, scaler):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        # Preparar la secuencia para la predicción
        X = current_sequence.reshape(1, -1)
        
        # Realizar predicción
        prediction = model.predict(X)
        predictions.append(prediction[0])
        
        # Actualizar la secuencia
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = prediction
    
    # Desescalar predicciones
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Título de la aplicación
st.title("🔮 Predicción de Consumo Eléctrico")

# Sidebar para configuración
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])

if uploaded_file is not None:
    # Cargar datos
    df = load_data(uploaded_file)
    
    if df is not None:
        # Parámetros del modelo
        st.sidebar.header("Parámetros del Modelo")
        seq_length = st.sidebar.slider("Longitud de secuencia (horas)", 
                                     min_value=1, max_value=48, value=24)
        prediction_hours = st.sidebar.slider("Horas a predecir", 
                                           min_value=1, max_value=72, value=24)
        hidden_layers = st.sidebar.slider("Capas ocultas", 
                                        min_value=1, max_value=3, value=2)
        neurons = st.sidebar.slider("Neuronas por capa", 
                                  min_value=10, max_value=100, value=50)
        
        # Botón de entrenamiento
        train_button = st.sidebar.button("Entrenar Modelo")
        
        # Mostrar datos originales
        st.subheader("Vista previa de los datos")
        st.write(df.head())
        
        if train_button:
            with st.spinner('Entrenando el modelo...'):
                try:
                    # Preparar datos
                    data = df['Kwh'].values.reshape(-1, 1)
                    scaler = MinMaxScaler()
                    data_scaled = scaler.fit_transform(data)
                    
                    # Crear secuencias
                    X, y = create_sequences(data_scaled, seq_length)
                    X = X.reshape(X.shape[0], -1)  # Aplanar para MLPRegressor
                    
                    # Dividir datos en entrenamiento y validación
                    train_size = int(len(X) * 0.8)
                    X_train, X_val = X[:train_size], X[train_size:]
                    y_train, y_val = y[:train_size], y[train_size:]
                    
                    # Crear y entrenar modelo
                    hidden_layer_sizes = tuple([neurons] * hidden_layers)
                    model = MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=1000,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    
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
                    train_score = model.score(X_train, y_train)
                    val_score = model.score(X_val, y_val)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score (Entrenamiento)", f"{train_score:.4f}")
                    with col2:
                        st.metric("R² Score (Validación)", f"{val_score:.4f}")
                    
                    # Descargar predicciones
                    st.header("💾 Descargar Predicciones")
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar predicciones como CSV",
                        data=csv,
                        file_name="predicciones.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {str(e)}")
                    st.info("Intenta ajustar los parámetros del modelo o verificar los datos de entrada.")
                
else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")
    st.markdown("""
    El archivo CSV debe contener las siguientes columnas:
    - `Datetime`: Fecha y hora de la medición
    - `Kwh`: Consumo eléctrico en kilovatios-hora
    """)

# Agregar información sobre el uso
st.sidebar.markdown("""
---
### Información de Uso
- Ajusta la longitud de secuencia según el patrón temporal
- Define cuántas horas hacia el futuro predecir
- Configura la arquitectura de la red neuronal
- Las predicciones se muestran en naranja en el gráfico
""")
