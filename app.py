
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="DC Bike Rentals Dashboard", layout="wide")

st.title("🚲 Washington D.C. Bike Rentals Dashboard")

st.markdown("Interactive dashboard summarizing bike rental patterns based on weather and time.")

# File uploader
uploaded_file = st.file_uploader("Upload train.csv from Kaggle Bike Sharing Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Feature engineering
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()

    df['day_period'] = pd.cut(
        df['hour'],
        bins=[-1, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )

    # Sidebar filters
    st.sidebar.header("Filters")
    year = st.sidebar.multiselect("Select Year", df['year'].unique(), default=df['year'].unique())
    season = st.sidebar.multiselect("Select Season", sorted(df['season'].unique()), default=sorted(df['season'].unique()))
    workingday = st.sidebar.selectbox("Working Day", ["All", 0, 1])

    filtered_df = df[df['year'].isin(year) & df['season'].isin(season)]
    if workingday != "All":
        filtered_df = filtered_df[filtered_df['workingday'] == workingday]

    # Plot 1: Rentals by hour
    st.subheader("Mean Rentals by Hour")
    hourly = filtered_df.groupby('hour')['count'].mean()
    fig, ax = plt.subplots()
    hourly.plot(ax=ax)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean Rentals")
    st.pyplot(fig)

    # Plot 2: Rentals by month
    st.subheader("Mean Rentals by Month")
    monthly = filtered_df.groupby('month')['count'].mean()
    fig, ax = plt.subplots()
    monthly.plot(kind='bar', ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Rentals")
    st.pyplot(fig)

    # Plot 3: Weather impact
    st.subheader("Weather vs Mean Rentals")
    weather = filtered_df.groupby('weather')['count'].mean()
    fig, ax = plt.subplots()
    weather.plot(kind='bar', ax=ax)
    ax.set_xlabel("Weather Category")
    ax.set_ylabel("Mean Rentals")
    st.pyplot(fig)

    # Plot 4: Working vs Non-working days
    st.subheader("Working Day vs Non-Working Day")
    work = filtered_df.groupby('workingday')['count'].mean()
    fig, ax = plt.subplots()
    work.plot(kind='bar', ax=ax)
    ax.set_xticklabels(["Non-working", "Working"], rotation=0)
    ax.set_ylabel("Mean Rentals")
    st.pyplot(fig)

    # Plot 5: Day period analysis
    st.subheader("Rentals by Period of the Day")
    period = filtered_df.groupby('day_period')['count'].mean()
    fig, ax = plt.subplots()
    period.plot(kind='bar', ax=ax)
    ax.set_ylabel("Mean Rentals")
    st.pyplot(fig)

else:
    st.info("Please upload the train.csv file to view the dashboard.")
