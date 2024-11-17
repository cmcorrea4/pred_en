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
                    
                    # Tabla de predicciones futuras
                    st.header("📋 Predicciones de Consumo Futuro")
                    # Crear una tabla solo con las predicciones futuras
                    predictions_table = pd.DataFrame({
                        'Fecha y Hora': pd.to_datetime(future_dates).strftime('%Y-%m-%d %H:%M:%S'),
                        'Predicción (kWh)': predictions.round(4)
                    })
                    st.dataframe(predictions_table)
                    
                    # Descargar predicciones
                    st.header("💾 Descargar Predicciones")
                    csv = predictions_table.to_csv(index=False)
                    st.download_button(
                        label="Descargar predicciones como CSV",
                        data=csv,
                        file_name="predicciones.csv",
                        mime="text/csv"
                    )
                    
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
