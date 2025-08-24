# penang_prop_insight_app.py

import streamlit as st
import pandas as pd
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
from datetime import datetime

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

# Property Type Filter
property_type = st.sidebar.multiselect("Select Property Type", options=df["Type"].unique(), default=df["Type"].unique())

# Area/Township Filter
available_townships = sorted(df["Township"].unique().tolist())
selected_townships = st.sidebar.multiselect("Select Townships", options=available_townships, default=[])

# Price Range Filter
min_price = int(df["Median_Price"].min())
max_price = int(df["Median_Price"].max())
price_range = st.sidebar.slider("Price Range (RM)", min_price, max_price, (min_price, max_price), format="RM %d")

# Tenure Filter
tenures = sorted(df["Tenure"].unique().tolist())
selected_tenures = st.sidebar.multiselect("Select Tenure", options=tenures, default=tenures)

# Apply all filters
selected_df = df[
    df["Type"].isin(property_type) &
    (df["Median_Price"] >= price_range[0]) & 
    (df["Median_Price"] <= price_range[1]) &
    df["Tenure"].isin(selected_tenures)
]

# Apply township filter only if something is selected
if selected_townships:
    selected_df = selected_df[selected_df["Township"].isin(selected_townships)]

st.subheader("🔍 Dataset Overview")
# Format currency columns with RM
display_df = selected_df.copy()
display_df['Median_Price'] = display_df['Median_Price'].apply(lambda x: f"RM {x:,.2f}")
display_df['Median_PSF'] = display_df['Median_PSF'].apply(lambda x: f"RM {x:,.2f}")
st.dataframe(display_df.head(10))

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
    # Format x-axis labels with RM
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'RM {int(x):,}'))
    ax.set_xlabel("Average Price (RM)")
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Top Areas by Median PSF")
    avg_psf = selected_df.groupby('Area')['Median_PSF'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=avg_psf.values, y=avg_psf.index, palette="coolwarm", ax=ax)
    # Format x-axis labels with RM
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'RM {int(x):,}'))
    ax.set_xlabel("Price per Square Foot (RM)")
    st.pyplot(fig)

with col4:
    st.markdown("### Distribution of Tenure")
    tenure_counts = selected_df['Tenure'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=tenure_counts.values, y=tenure_counts.index, palette="Set2", ax=ax)
    st.pyplot(fig)

# --------- 🗺️ Interactive Map ---------
st.markdown("## 🗺️ Interactive Property Map")

# Check if our dataframe has coordinates
if 'Latitude' not in selected_df.columns or 'Longitude' not in selected_df.columns:
    # If we don't have coordinates, we'll create sample ones for demonstration
    # In a real app, you'd use geocoding services or real data
    import numpy as np
    
    # Create random coordinates around Penang
    # These are approximate coordinates for Penang
    penang_lat, penang_lon = 5.4141, 100.3288
    
    # Add sample coordinates for demonstration
    if 'Latitude' not in selected_df.columns:
        selected_df['Latitude'] = penang_lat + (np.random.rand(len(selected_df)) - 0.5) * 0.1
    if 'Longitude' not in selected_df.columns:
        selected_df['Longitude'] = penang_lon + (np.random.rand(len(selected_df)) - 0.5) * 0.1

# Create map
try:
    import folium
    from streamlit_folium import folium_static
    
    # Center the map on the mean of coordinates
    center_lat = selected_df['Latitude'].mean()
    center_lon = selected_df['Longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add markers for each property
    for idx, row in selected_df.iterrows():
        # Create popup text with property details
        popup_text = f"""
        <b>Township:</b> {row['Township']}<br>
        <b>Type:</b> {row['Type']}<br>
        <b>Price:</b> RM {row['Median_Price']:,.2f}<br>
        <b>PSF:</b> RM {row['Median_PSF']:,.2f}<br>
        <b>Tenure:</b> {row['Tenure']}
        """
        
        # Color markers based on price (higher = red, lower = green)
        price_percentile = (row['Median_Price'] - selected_df['Median_Price'].min()) / (selected_df['Median_Price'].max() - selected_df['Median_Price'].min())
        if price_percentile < 0.33:
            color = 'green'
        elif price_percentile < 0.66:
            color = 'orange'
        else:
            color = 'red'
        
        # Add marker
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{row['Township']} - {row['Type']}",
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    # Display the map
    folium_static(m)
    
except ImportError:
    st.warning("Please install folium and streamlit-folium to enable the interactive map. Run: `pip install folium streamlit-folium`")
except Exception as e:
    st.error(f"Error creating map: {e}")

# --------- 💰 ROI Calculator ---------
st.markdown("## 💰 Investment ROI Calculator")

col_calc1, col_calc2 = st.columns(2)

with col_calc1:
    st.markdown("### Property Investment Details")
    
    # Investment inputs
    property_value = st.number_input("Property Purchase Price (RM)", min_value=100000, max_value=10000000, value=500000, step=50000)
    down_payment_percent = st.slider("Down Payment (%)", min_value=10, max_value=90, value=10)
    interest_rate = st.slider("Loan Interest Rate (%)", min_value=1.0, max_value=10.0, value=4.5, step=0.1)
    loan_tenure = st.slider("Loan Tenure (Years)", min_value=5, max_value=35, value=30)
    
    # Rental income
    monthly_rental = st.number_input("Expected Monthly Rental (RM)", min_value=0, max_value=50000, value=1500, step=100)
    occupancy_rate = st.slider("Expected Occupancy Rate (%)", min_value=50, max_value=100, value=90)
    
    # Expenses
    annual_property_tax = st.number_input("Annual Property Tax (RM)", min_value=0, max_value=50000, value=500, step=100)
    annual_insurance = st.number_input("Annual Insurance (RM)", min_value=0, max_value=10000, value=300, step=100)
    annual_maintenance = st.number_input("Annual Maintenance (RM)", min_value=0, max_value=50000, value=1200, step=100)
    
    # Appreciation
    annual_appreciation = st.slider("Expected Annual Appreciation (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    
with col_calc2:
    st.markdown("### ROI Analysis")
    
    # Calculate mortgage details
    loan_amount = property_value * (1 - down_payment_percent / 100)
    monthly_rate = interest_rate / 100 / 12
    num_payments = loan_tenure * 12
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    
    # Calculate annual returns
    effective_monthly_rental = monthly_rental * (occupancy_rate / 100)
    annual_rental_income = effective_monthly_rental * 12
    annual_mortgage_payment = monthly_payment * 12
    total_annual_expenses = annual_property_tax + annual_insurance + annual_maintenance + annual_mortgage_payment
    annual_cash_flow = annual_rental_income - total_annual_expenses
    
    # Calculate ROI
    cash_invested = property_value * (down_payment_percent / 100)
    cash_on_cash_roi = (annual_cash_flow / cash_invested) * 100
    
    # Property value after loan term
    future_property_value = property_value * ((1 + annual_appreciation / 100) ** loan_tenure)
    total_appreciation = future_property_value - property_value
    total_equity = future_property_value - loan_amount
    
    # Display results
    st.markdown(f"**Monthly Mortgage Payment:** RM {monthly_payment:,.2f}")
    st.markdown(f"**Annual Cash Flow:** RM {annual_cash_flow:,.2f}")
    st.markdown(f"**Cash-on-Cash ROI:** {cash_on_cash_roi:.2f}%")
    
    # Conditional formatting based on ROI
    if cash_on_cash_roi < 0:
        st.error("⚠️ Negative cash flow investment. Consider adjusting parameters.")
    elif cash_on_cash_roi < 3:
        st.warning("⚠️ Low ROI. May not beat inflation.")
    elif cash_on_cash_roi > 8:
        st.success("✅ High ROI! Better than many investment options.")
    else:
        st.info("ℹ️ Moderate ROI. Compare with other investments.")
    
    st.markdown("### Long-term Projections")
    st.markdown(f"**Future Property Value (after {loan_tenure} years):** RM {future_property_value:,.2f}")
    st.markdown(f"**Total Appreciation:** RM {total_appreciation:,.2f}")
    st.markdown(f"**Total Equity:** RM {total_equity:,.2f}")
    
    # Create a simple bar chart comparing returns
    fig, ax = plt.subplots()
    data = [cash_on_cash_roi, annual_appreciation]
    labels = ['Cash-on-Cash ROI', 'Property Appreciation']
    colors = ['#1f77b4', '#ff7f0e']
    ax.bar(labels, data, color=colors)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Annual Return Comparison')
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
cluster_summary = selected_df.groupby("Cluster")[['Median_Price', 'Median_PSF', 'Transactions']].mean().round(2)
# Format the dataframe for display
display_cluster_summary = cluster_summary.copy()
display_cluster_summary['Median_Price'] = display_cluster_summary['Median_Price'].apply(lambda x: f"RM {x:,.2f}")
display_cluster_summary['Median_PSF'] = display_cluster_summary['Median_PSF'].apply(lambda x: f"RM {x:,.2f}")
st.dataframe(display_cluster_summary)

# --------- ⏱️ Data Freshness ---------
st.markdown("## ⏱️ Data Freshness")

# Get the data modification time
file_path = "houses_data.xlsx"
if os.path.exists(file_path):
    mod_time = os.path.getmtime(file_path)
    last_updated = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    st.info(f"📅 Data last updated: {last_updated}")
else:
    st.warning("⚠️ Could not determine when data was last updated.")

# --------- 📊 Export Data ---------
st.markdown("## 📊 Export Data")

export_col1, export_col2 = st.columns(2)

with export_col1:
    st.markdown("### Export Filtered Data")
    export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "JSON"])
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    @st.cache_data
    def convert_df_to_excel(df):
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='PenangPropData')
        return output.getvalue()
    
    @st.cache_data
    def convert_df_to_json(df):
        return df.to_json(orient='records', indent=4).encode('utf-8')
    
    if st.button("Generate Export File"):
        if export_format == "CSV":
            data = convert_df_to_csv(selected_df)
            file_name = "penang_property_data.csv"
            mime = "text/csv"
        elif export_format == "Excel":
            data = convert_df_to_excel(selected_df)
            file_name = "penang_property_data.xlsx"
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        else:  # JSON
            data = convert_df_to_json(selected_df)
            file_name = "penang_property_data.json"
            mime = "application/json"
        
        st.download_button(
            label=f"Download as {export_format}",
            data=data,
            file_name=file_name,
            mime=mime,
        )

with export_col2:
    st.markdown("### Export Investment Analysis")
    
    if st.button("Generate Investment Report"):
        report_data = {
            "report_date": datetime.now().strftime('%Y-%m-%d'),
            "filter_criteria": {
                "property_types": property_type,
                "price_range": f"RM {price_range[0]:,} - RM {price_range[1]:,}",
                "tenures": selected_tenures
            },
            "property_count": len(selected_df),
            "price_stats": {
                "min_price": f"RM {selected_df['Median_Price'].min():,.2f}",
                "max_price": f"RM {selected_df['Median_Price'].max():,.2f}",
                "avg_price": f"RM {selected_df['Median_Price'].mean():,.2f}",
                "median_price": f"RM {selected_df['Median_Price'].median():,.2f}"
            },
            "top_areas": top_townships['Township'].tolist(),
            "roi_calculation": {
                "property_value": f"RM {property_value:,.2f}",
                "monthly_payment": f"RM {monthly_payment:,.2f}",
                "annual_cash_flow": f"RM {annual_cash_flow:,.2f}",
                "cash_on_cash_roi": f"{cash_on_cash_roi:.2f}%",
                "future_value": f"RM {future_property_value:,.2f}"
            }
        }
        
        import json
        report_json = json.dumps(report_data, indent=4).encode('utf-8')
        
        st.download_button(
            label="Download Investment Report (JSON)",
            data=report_json,
            file_name="penang_investment_report.json",
            mime="application/json",
        )
        
        # Display a sample of what's in the report
        st.markdown("### Report Preview:")
        st.json(report_data)

# Add a footer
st.markdown("---")
st.markdown("*PenangPropInsight is a machine learning powered tool for property investment analysis in Penang.*")
st.markdown("*© 2025 | Made with ❤️ using Streamlit and Python*")