import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import altair as alt
from datetime import datetime, timedelta

# Configuración de la página
st.set_page_config(page_title="Predicción de Consumo Eléctrico - LSTM", layout="wide")

# Definición del modelo LSTM
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Función para normalizar datos
def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std

# Función para desnormalizar datos
def denormalize_data(data, mean, std):
    return data * std + mean

# Función para crear secuencias
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq.reshape(-1, 1))  # Reshape para mantener la dimensionalidad correcta
        targets.append(target)
    return torch.FloatTensor(np.array(sequences)), torch.FloatTensor(targets)

# Función para cargar datos
def load_data(file):
    try:
        df = pd.read_csv(file, index_col=0)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df = df.dropna()
        df = df.sort_values('Datetime')
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

# Función para realizar predicciones
def predict_future(model, last_sequence, n_steps, mean, std):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Preparar input
            x = current_sequence.unsqueeze(0)  # Añadir dimensión de batch
            
            # Predecir siguiente valor
            pred = model(x)
            predictions.append(pred.item())
            
            # Actualizar secuencia
            current_sequence = torch.roll(current_sequence, -1, dims=0)
            current_sequence[-1] = pred.squeeze()
    
    # Desnormalizar predicciones
    predictions = denormalize_data(np.array(predictions), mean, std)
    return predictions

# Título de la aplicación
st.title("🔮 Predicción de Consumo Eléctrico con LSTM")

# Configuración en la barra lateral
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.sidebar.header("Parámetros del Modelo")
        seq_length = st.sidebar.slider("Longitud de secuencia (horas)", 
                                     min_value=1, max_value=48, value=24)
        prediction_hours = st.sidebar.slider("Horas a predecir", 
                                           min_value=1, max_value=72, value=24)
        hidden_size = st.sidebar.slider("Tamaño capa oculta", 
                                      min_value=10, max_value=100, value=50)
        epochs = st.sidebar.slider("Épocas de entrenamiento", 
                                 min_value=10, max_value=200, value=50)
        
        # Botón de entrenamiento
        train_button = st.sidebar.button("Entrenar Modelo")
        
        # Mostrar datos originales
        st.subheader("Vista previa de los datos")
        st.write(df.head())
        
        if train_button:
            with st.spinner('Entrenando el modelo...'):
                try:
                    # Preparar datos
                    data = df['Kwh'].values
                    data_normalized, mean, std = normalize_data(data)
                    
                    # Crear secuencias
                    X, y = create_sequences(data_normalized, seq_length)
                    
                    # Crear y entrenar modelo
                    model = LSTMPredictor(hidden_size=hidden_size)
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters())
                    
                    # Entrenamiento
                    progress_bar = st.progress(0)
                    train_losses = []
                    
                    for epoch in range(epochs):
                        model.train()
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model(X)
                        loss = criterion(outputs.squeeze(), y)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                        
                        train_losses.append(loss.item())
                        progress_bar.progress((epoch + 1) / epochs)
                    
                    # Realizar predicciones
                    last_sequence = torch.FloatTensor(data_normalized[-seq_length:]).reshape(-1, 1)
                    predictions = predict_future(model, last_sequence, prediction_hours, mean, std)
                    
                    # Crear fechas para predicciones
                    last_date = df['Datetime'].iloc[-1]
                    future_dates = [last_date + timedelta(hours=i+1) for i in range(len(predictions))]
                    
                    # DataFrame de predicciones
                    predictions_df = pd.DataFrame({
                        'Datetime': future_dates,
                        'Kwh_Predicted': predictions
                    })
                    
                    # Visualización
                    st.header("📈 Resultados de la Predicción")
                    
                    # Preparar datos para visualización
                    historical_data = df[['Datetime', 'Kwh']].copy()
                    historical_data['Tipo'] = 'Histórico'
                    predictions_df['Tipo'] = 'Predicción'
                    
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
                    
                    # Mostrar pérdida de entrenamiento
                    st.header("📊 Métricas del Modelo")
                    final_loss = train_losses[-1]
                    st.metric("Error de Entrenamiento (MSE)", f"{final_loss:.6f}")
                    
                    # Descargar predicciones
                    st.header("💾 Descargar Predicciones")
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar predicciones como CSV",
                        data=csv,
                        file_name="predicciones.csv",
                        mime="text/csv"
                    )
                    
                    # Mostrar tabla de predicciones
                    st.header("📋 Tabla de Predicciones")
                    # Formatear la columna Datetime para mejor visualización
                    predictions_df['Datetime'] = predictions_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    # Renombrar columnas para mejor claridad
                    predictions_display = predictions_df.copy()
                    predictions_display.columns = ['Fecha y Hora', 'Consumo Predicho (kWh)']
                    # Mostrar el DataFrame con las predicciones
                    st.dataframe(predictions_display.style.format({
                        'Consumo Predicho (kWh)': '{:.4f}'
                    }))
                    
                    # Mostrar estadísticas de las predicciones
                    st.subheader("📊 Estadísticas de las Predicciones")
                    stats_df = pd.DataFrame({
                        'Métrica': ['Consumo Mínimo', 'Consumo Máximo', 'Consumo Promedio'],
                        'Valor (kWh)': [
                            f"{predictions_df['Kwh_Predicted'].min():.4f}",
                            f"{predictions_df['Kwh_Predicted'].max():.4f}",
                            f"{predictions_df['Kwh_Predicted'].mean():.4f}"
                        ]
                    })
                    st.dataframe(stats_df)
                    
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {str(e)}")
                    st.info("Intenta ajustar los parámetros del modelo o verificar los datos.")
else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar el análisis.")
    st.markdown("""
    El archivo CSV debe contener las siguientes columnas:
    - `Datetime`: Fecha y hora de la medición
    - `Kwh`: Consumo eléctrico en kilovatios-hora
    """)

# Información de uso
st.sidebar.markdown("""
---
### Información de Uso
- Ajusta la longitud de secuencia según el patrón temporal
- Define cuántas horas hacia el futuro predecir
- Modifica el tamaño de la capa oculta para ajustar la complejidad del modelo
- Aumenta las épocas de entrenamiento para mejor precisión
""")
