# 🏘️ PenangPropInsight: ML-Powered Property Investment Analysis

![Version](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

**PenangPropInsight** is a sophisticated, data-driven web application that transforms property market analysis in Penang, Malaysia. Using real-time data, advanced filters, interactive maps, and machine learning algorithms, it helps investors, homebuyers, and real estate professionals make informed decisions about property investments.

![PenangPropInsight Screenshot](https://via.placeholder.com/800x450?text=PenangPropInsight+Screenshot)

## 📊 Features

### Comprehensive Data Exploration

- **Rich Dataset**: Access to comprehensive property data across Penang
- **Advanced Filtering**: Filter by property type, price range, township, and tenure
- **Data Visualization**: Interactive charts showing price trends, transaction volumes, and more

### Interactive Property Map

- **Geospatial Analysis**: View property locations on an interactive map
- **Color-Coded Markers**: Properties color-coded by price range (green = lower, orange = medium, red = higher)
- **Detailed Pop-ups**: Click on markers to see detailed property information

### Investment Analysis Tools

- **ROI Calculator**: Comprehensive investment return calculator with:
  - Mortgage calculation with customizable down payment and interest rates
  - Rental income projections with adjustable occupancy rates
  - Expense tracking (taxes, insurance, maintenance)
  - Long-term property appreciation estimates
  - Cash-on-cash ROI analysis

### Machine Learning Insights

- **Investment Zone Detection**: KMeans clustering identifies optimal investment zones
- **Pattern Recognition**: PCA visualization reveals hidden property market patterns
- **Cluster Analysis**: Detailed breakdown of property clusters by price, PSF, and transaction volume

### Data Export Capabilities

- **Multiple Formats**: Export filtered data as CSV, Excel, or JSON
- **Investment Reports**: Generate detailed investment analysis reports
- **Data Freshness**: Clear indicators of when data was last updated

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (automatically installed with instructions below)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/daniyalrosli/propertydata.git
   cd propertydata
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```
4. **Access the app**
   - Open your browser and go to `http://localhost:8501`

## 📋 Usage Guide

### Basic Navigation

1. When the app loads, you'll see the complete dataset overview
2. Use the sidebar filters to narrow down properties by:
   - Property type (Condominium, Terrace, Semi-Detached, etc.)
   - Township location
   - Price range (use the slider)
   - Tenure type (Freehold, Leasehold)

### Data Exploration

- Scroll through the various visualizations to understand market trends
- The "Top Townships by Transactions" chart shows where most properties are being sold
- "Average Median Price by Type" reveals the most expensive property categories
- "Top Areas by Median PSF" highlights locations with highest price per square foot

### Using the Interactive Map

- Navigate to the "Interactive Property Map" section
- Zoom in/out using the mouse wheel or touchpad
- Click on property markers to view detailed information
- Properties are color-coded by price range for easy identification

### Investment Analysis

1. Navigate to the "Investment ROI Calculator" section
2. Input your investment parameters:
   - Property purchase price
   - Down payment percentage
   - Loan interest rate and tenure
   - Expected rental income and occupancy rate
   - Annual expenses (taxes, insurance, maintenance)
3. Review the calculated ROI metrics and long-term projections

### Analyzing ML Clusters

- The "ML Clustering: Investment Zone Detection" section shows property clusters
- Each cluster represents properties with similar characteristics
- The PCA projection visualizes these clusters in 2D space
- The Cluster Summary table shows average metrics for each cluster

### Exporting Data

- Navigate to the "Export Data" section
- Choose your preferred format (CSV, Excel, JSON)
- Click "Generate Export File" to download
- For investment analysis, click "Generate Investment Report"

## 🧠 Technical Details

### Data Processing

- The application processes real property data from Penang
- Data is filtered and transformed using pandas DataFrames
- Caching is implemented for improved performance

### Machine Learning Implementation

- **KMeans Clustering**: Identifies 3 property clusters based on:
  - Median price
  - Median price per square foot (PSF)
  - Transaction volume
- **Principal Component Analysis (PCA)**: Reduces dimensionality for visualization
- **StandardScaler**: Normalizes data before clustering

### Visualization Technologies

- **Matplotlib & Seaborn**: For static charts and visualizations
- **Folium**: For interactive geospatial mapping
- **Streamlit Components**: For interactive UI elements

## 🤝 Contributing

Contributions are welcome! Here's how you can help improve PenangPropInsight:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Daniyal Rosli - daniyal.rosli@example.com

Project Link: [https://github.com/daniyalrosli/propertydata](https://github.com/daniyalrosli/propertydata)

---

_PenangPropInsight is a sophisticated tool leveraging data science and machine learning to transform property investment analysis in Penang, Malaysia._
