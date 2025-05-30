# penang_prop_insight_app.py

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(layout="wide")
st.title("🏘️ PenangPropInsight: ML-Powered Property Investment Analysis")

# 📥 Load Data
@st.cache_data
def load_data():
    df = pd.read_excel("houses_data.xlsx", sheet_name="malaysia_house_price_data_2025")
    df = df[df['State'].str.lower() == 'penang'].copy()
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("📍 Filters")
property_type = st.sidebar.multiselect("Select Property Type", options=df["Type"].unique(), default=df["Type"].unique())
selected_df = df[df["Type"].isin(property_type)]

st.subheader("🔍 Dataset Overview")
st.dataframe(selected_df.head(10))

# --------- 📊 EDA Visuals ---------
st.markdown("## 📈 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Townships by Transactions")
    top_townships = selected_df.sort_values(by='Transactions', ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(data=top_townships, x="Transactions", y="Township", palette="viridis", ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("### Average Median Price by Type")
    avg_price = selected_df.groupby('Type')['Median_Price'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=avg_price.values, y=avg_price.index, palette="magma", ax=ax)
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Top Areas by Median PSF")
    avg_psf = selected_df.groupby('Area')['Median_PSF'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=avg_psf.values, y=avg_psf.index, palette="coolwarm", ax=ax)
    st.pyplot(fig)

with col4:
    st.markdown("### Distribution of Tenure")
    tenure_counts = selected_df['Tenure'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=tenure_counts.values, y=tenure_counts.index, palette="Set2", ax=ax)
    st.pyplot(fig)

# --------- 🤖 ML Clustering ---------
st.markdown("## 🧠 ML Clustering: Investment Zone Detection")

# Prepare ML data
ml_data = selected_df[['Median_Price', 'Median_PSF', 'Transactions']].dropna()
scaler = StandardScaler()
scaled = scaler.fit_transform(ml_data)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled)
selected_df.loc[ml_data.index, 'Cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled)

# Plot
fig, ax = plt.subplots()
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=clusters, palette='viridis', s=80, ax=ax)
plt.title("🏘️ Property Clusters in Penang (PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
st.pyplot(fig)

# Cluster Stats
st.markdown("### 🔎 Cluster Summary")
st.dataframe(selected_df.groupby("Cluster")[['Median_Price', 'Median_PSF', 'Transactions']].mean().round(2))