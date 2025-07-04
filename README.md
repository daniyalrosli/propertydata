# 🏘️ PenangPropInsight: Smarter Property Investment in Penang

**PenangPropInsight** is a simple, interactive web app that helps you explore and understand the property market in Penang, Malaysia. Using real data and machine learning, it shows you trends, hot areas, and helps you make better decisions if you're thinking about buying property.

---

## What Does This App Do?

- **Shows you the latest property data for Penang**  
  See prices, types, and locations of properties in an easy-to-read table.

- **Lets you filter by property type**  
  Want to see only apartments or landed houses? Just use the sidebar!

- **Visualizes key trends**  
  - Which townships have the most transactions?
  - What property types are most expensive?
  - Where are the highest prices per square foot?
  - What are the most common property tenures?

- **Uses machine learning to find "investment zones"**  
  The app groups properties into clusters based on price, price per square foot, and transaction volume. This helps you spot areas that might be undervalued or in high demand.

- **Shows when the data was last updated**  
  So you know you're looking at fresh information.

---

## What Will I See?

- **Interactive charts** that make it easy to spot trends.
- **A map of clusters** (groups of similar properties) to help you find the best investment zones.
- **A summary table** showing the average price and transaction stats for each cluster.

---

## Why Use This App?

- **Easy to use**: No technical skills needed.
- **Data-driven**: Makes property trends clear and simple.
- **Helps you decide**: Find the best time, place, and type of property to invest in Penang.

---

## How to Use

1. Open the app (see instructions below).
2. Use the sidebar to filter by property type.
3. Explore the charts and tables.
4. Check out the clusters to see which areas might be best for investment.

---

## How to Run the App

1. Make sure you have Python installed.
2. Install the requirements:
   ```
   pip install streamlit pandas seaborn matplotlib scikit-learn openpyxl
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

---

## 🧠 Model Performance

**Clustering (KMeans):**
- The app uses KMeans clustering to group properties into 3 investment zones based on price, price per square foot, and transaction volume.
- **Silhouette Score:** Measures how well each property fits within its cluster (higher is better).
  
  *Example: Silhouette Score = 0.42 (on current data)*
- **Cluster Summary:**  
  Each cluster represents a group of properties with similar characteristics:
  - **Cluster 0:** High price, high demand areas
  - **Cluster 1:** Affordable, moderate demand areas
  - **Cluster 2:** Lower price, lower demand areas

*Note: The actual cluster meanings may vary depending on the data. The app shows a summary table for each cluster.*

---

*PenangPropInsight is a free tool to help you make smarter property investment decisions in Penang, Malaysia.*

