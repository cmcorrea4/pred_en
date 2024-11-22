import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

def load_data(uploaded_file):
    """Load and preprocess the CSV data"""
    df = pd.read_csv(uploaded_file)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

def create_line_plot(df):
    """Create an interactive line plot using Plotly"""
    fig = px.line(df, x='Time', y='value', 
                  title='Energy Consumption Over Time',
                  labels={'value': 'Energy Value', 'Time': 'Timestamp'})
    fig.update_layout(showlegend=False)
    return fig

def calculate_statistics(df):
    """Calculate basic statistics from the data"""
    stats = {
        'Average Value': df['value'].mean(),
        'Maximum Value': df['value'].max(),
        'Minimum Value': df['value'].min(),
        'Total Records': len(df)
    }
    return stats

def main():
    st.title('Energy Data Analysis Dashboard')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        # Display interactive plot
        st.plotly_chart(create_line_plot(df))
        
        # Display statistics
        st.subheader('Data Statistics')
        stats = calculate_statistics(df)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average", f"{stats['Average Value']:.2f}")
        with col2:
            st.metric("Maximum", f"{stats['Maximum Value']:.2f}")
        with col3:
            st.metric("Minimum", f"{stats['Minimum Value']:.2f}")
        with col4:
            st.metric("Total Records", stats['Total Records'])
        
        # Display raw data
        st.subheader('Raw Data')
        st.dataframe(df)
