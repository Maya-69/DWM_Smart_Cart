# 🛒 SmartCart - AI-Powered Recommendation System

An intelligent e-commerce recommendation system using **Machine Learning** to provide personalized product recommendations. Features **Apriori Algorithm** for association rules, **K-Means** for customer segmentation, and **Naive Bayes** for demographic prediction.

**🎓 Academic Project:** Data Warehouse & Mining (DWM)  
**📊 Dataset:** 38,765 real grocery transactions  
**🤖 ML Models:** Apriori, K-Means, Naive Bayes, Pattern Recognition

---

## ✨ Features

- 🤖 **AI-Powered Recommendations** - Apriori algorithm with 4,588 association rules
- � **Customer Segmentation** - K-Means clustering (Budget/Regular/Premium shoppers)
- � **Age Group Prediction** - Naive Bayes classifier based on shopping patterns
- �️ **Meal Intent Detection** - Pattern recognition for meal predictions
- 🌳 **Interactive Visualization** - Multi-level hierarchical recommendation graph
- � **Real-Time Insights** - Confidence scores, lift metrics, and decision trees
- 🏛️ **Data Warehouse** - Star schema with OLAP operations
- � **Analytics Dashboard** - Comprehensive visualizations and statistics

---

## 🎯 Key Algorithms

### Apriori Algorithm
- **Support**: Frequency of itemset in transactions
- **Confidence**: Probability of rule being true
- **Lift**: How much more likely items are bought together vs independently

### K-Means Clustering (Enhanced)
- **Elbow Method**: Automatically finds optimal number of clusters
- **Silhouette Analysis**: Validates cluster quality (score: -1 to 1)
- **Customer Segmentation**: Groups customers by spending patterns
- **Smart Naming**: Budget, Regular, Premium, VIP segments

### RFM Segmentation
- **Recency**: How recently customer purchased
- **Frequency**: How often customer purchases
- **Monetary**: How much customer spends
- **Segments**: Champions, Loyal, Potential Loyalists, At Risk, Needs Attention

### Data Warehouse Features
- **Star Schema**: Fact tables + Dimension tables (Customer, Time, Product)
- **ETL Metadata**: Tracks load history, records loaded/rejected, duration
- **Data Quality**: Automated validation (null checks, duplicates, orphaned records)
- **Incremental Loads**: Only load new/changed data
- **OLAP Operations**: Roll-up, drill-down, slice, dice
- **Aggregations**: Pre-computed summaries for fast queries

### What's Cooking? - Meal Intent Prediction
- **Pattern Recognition**: Matches cart items against 10 predefined meal patterns
- **Keyword Matching**: Identifies ingredient combinations (bread + butter + cheese = sandwich)
- **Confidence Scoring**: Calculates match percentage for each meal type
- **Multi-Prediction**: Returns top 3 predictions with confidence levels

---

## 📝 Dataset

**Source:** [Groceries Market Basket Dataset](https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset)  
**Records:** 38,765 transactions  
**Items:** 169 unique products  
**Time Period:** 30 days of grocery store data

---

## 🏗️ Tech Stack

### Frontend
- **React** - UI framework
- **Lucide Icons** - Modern icon library
- **CSS3** - Custom styling

### Backend
- **Python 3.x** - Core language
- **Flask** - REST API server
- **SQLite** - Database
- **Pandas** - Data manipulation
- **mlxtend** - Apriori algorithm implementation
- **NumPy** - Numerical computing
- **NetworkX** - Graph analysis and layout algorithms
- **Plotly** - Interactive visualizations with animations
- **Scikit-learn** - K-Means clustering, silhouette analysis
- **Matplotlib/Seaborn** - Static visualizations

---

## 📂 Project Structure

<pre>
smartcart/
├── src/
│   ├── App.js                 # Main React component
│   ├── App.css                # Styling
│   └── index.js               # React entry point
├── backend/
│   ├── server.py              # Flask API server (Enhanced)
│   ├── ml_models.py           # ML algorithms (Apriori, K-Means, RFM)
│   ├── database.py            # SQLite operations
│   ├── data_warehouse.py      # Data processing (Enhanced ETL)
│   ├── load_dataset.py        # Dataset loader
│   ├── generate_graphs.py     # Graph generation (Enhanced)
│   ├── initialize.py          # Setup script
│   ├── test_enhanced_features.py  # Test suite (NEW)
│   └── graphs/                # Generated visualizations
│       ├── *.png              # Static graphs
│       └── *.html             # Interactive graphs
├── Groceries_dataset.csv      # Training data (38,000+ transactions)
├── requirements.txt           # Python dependencies
└── README.md
</pre>

---

## 🚀 Setup & Installation (Fresh PC)

### Prerequisites
- **Node.js** (v14+) - [Download](https://nodejs.org/)
- **Python** 3.8+ - [Download](https://www.python.org/downloads/)
- **Git** - [Download](https://git-scm.com/)

---

### Step 1: Clone Repository
```bash
git clone https://github.com/Maya-69/DWM_Smart_Cart.git
cd DWM_Smart_Cart
```

---

### Step 2: Backend Setup

#### Navigate to Backend
```bash
cd backend
```

#### Install Python Dependencies
```bash 
pip install -r requirements.txt
```

#### Initialize Database & Train ML Models
This will:
- Load the grocery dataset (38,765 transactions)
- Train Apriori model for product associations
- Generate 4,588 association rules
- Setup SQLite database with star schema
- Train K-Means clustering model
- Create data warehouse tables

```bash
python initialize.py
```

#### Start Backend Server (Port 5000)
```bash
python server.py
```

Backend should now be running at `http://localhost:5000`

---

### Step 3: Frontend Setup

#### Open New Terminal & Navigate to Project Root
```bash
cd DWM_Smart_Cart
```

#### Install Dependencies
```bash
npm install
```

#### Start Development Server (Port 3000)
```bash
npm start
```

---

### ✅ Verification

Your browser should automatically open to `http://localhost:3000`

**Check if everything works:**
1. ✓ Products load on homepage
2. ✓ Can add items to cart
3. ✓ AI Insights tab shows recommendations
4. ✓ Customer segment prediction works
5. ✓ Age group prediction displays
6. ✓ Graphs render correctly

---

## 🖥️ Running the Application

### Every Time You Start:

**Terminal 1 - Backend:**
```bash
cd DWM_Smart_Cart/backend
python server.py
```

**Terminal 2 - Frontend:**
```bash
cd DWM_Smart_Cart
npm start
```

**Access:** Open browser to `http://localhost:3000`

---

## ⚙️ Configuration

### Backend Port (Default: 5000)
Edit `backend/server.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Change port here
```

### Frontend API URL
Edit `src/App.js` if backend port changes:
```javascript
const API_BASE = 'http://localhost:5000/api';  // Update port here
```

---

## 🐛 Troubleshooting

### Database Issues

If you encounter database errors, delete and reinitialize:
```bash
Remove-Item smartcart.db -ErrorAction SilentlyContinue
python initialize.py
```

---

## 🎨 New Enhanced Features

### 📊 Interactive Graph Visualizations

Generate beautiful, interactive graphs with smooth animations:

```bash
cd backend
python generate_graphs.py
```

This creates:
- **Static PNG graphs** (exploratory analysis, accuracy metrics, association rules)
- **Interactive HTML graphs** (network graph, dashboard, recommendation flow)

Open the HTML files in your browser for:
- 🔍 Zoom and pan
- 🖱️ Hover tooltips with detailed metrics
- ✨ Smooth animations and transitions
- 🎯 Click-to-explore functionality

### 🎯 K-Means Customer Segmentation

**Elbow Method** automatically finds the optimal number of clusters:

```python
from ml_models import MLRecommendationEngine

ml_engine = MLRecommendationEngine()
optimal_k, k_range, inertias, silhouette_scores = ml_engine.find_optimal_clusters_elbow(max_k=10)
```

**Enhanced Clustering** with smart segment naming:

```python
result = ml_engine.train_kmeans_with_elbow(n_clusters=None)  # Auto-detect K
print(f"Silhouette Score: {result['silhouette_score']}")
```

### 💎 RFM Customer Analysis

Segment customers by behavior patterns:

```python
rfm_result = ml_engine.perform_rfm_segmentation()
# Returns: Champions 🏆, Loyal Customers 💎, Potential Loyalists 🌟, etc.
```

### 🔧 Data Warehouse Operations

**Data Quality Checks:**

```python
from data_warehouse import DataWarehouse

warehouse = DataWarehouse()
quality_report = warehouse.validate_data_quality()
# Checks: negative prices, null values, duplicates, orphaned records
```

**Incremental ETL Loads:**

```python
result = warehouse.incremental_load(batch_size=1000)
# Load only new records with metadata tracking
```

**ETL Statistics:**

```python
stats = warehouse.get_etl_stats()
# Returns: load history, records loaded/rejected, duration
```

---

## 🔌 Enhanced API Endpoints

### 🛒 Product & Recommendations
```
GET  /api/products
POST /api/recommendations
POST /api/ai-insights
POST /api/association-tree
POST /api/recommendation-graph
POST /api/predict-intent
```

### 📊 Analytics & Clustering (NEW)
```
GET  /api/cluster-analysis          # K-Means with elbow method
GET  /api/rfm-segmentation          # RFM customer segments
POST /api/analytics-dashboard       # Comprehensive analytics with charts
```

### 🔍 Data Quality & ETL (NEW)
```
GET  /api/data-quality              # Run data validation checks
GET  /api/etl-stats                 # ETL execution history
POST /api/refresh-aggregations      # Rebuild aggregation tables
```

### 📈 Interactive Graphs (NEW)
```
GET  /api/interactive-graph/network     # Network graph (HTML)
GET  /api/interactive-graph/dashboard   # Analytics dashboard (HTML)
GET  /api/interactive-graph/flow        # Recommendation flow (HTML)
POST /api/generate-graphs               # Generate all graphs
```

### ❤️ System
```
GET  /api/model-status
GET  /api/health
```

---

## 🧪 Running Tests

Run the comprehensive test suite:

```bash
cd backend
python test_enhanced_features.py
```

Tests include:
- ✅ K-Means clustering (elbow method, silhouette analysis)
- ✅ RFM segmentation
- ✅ Data quality validation
- ✅ ETL operations
- ✅ Graph generation
- ✅ Database schema

---

## 📈 Usage Examples

### Example 1: Get Customer Clusters

```bash
curl http://localhost:5000/api/cluster-analysis
```

Response:
```json
{
  "success": true,
  "optimal_k": 3,
  "elbow_data": {
    "k_values": [2, 3, 4, 5],
    "inertias": [245.2, 156.8, 98.3, 67.1],
    "silhouette_scores": [0.45, 0.62, 0.58, 0.51]
  },
  "clusters": {
    "customers": [...],
    "silhouette_score": 0.62,
    "n_clusters": 3
  }
}
```

### Example 2: Get RFM Segments

```bash
curl http://localhost:5000/api/rfm-segmentation
```

Response:
```json
{
  "success": true,
  "rfm_analysis": {
    "segments": [
      {
        "customer_id": 1,
        "customer_name": "Customer_1",
        "segment": "Champions 🏆",
        "RFM_score": 14,
        "R_score": 5,
        "F_score": 5,
        "M_score": 4
      }
    ],
    "summary": {
      "Champions 🏆": 2,
      "Loyal Customers 💎": 1,
      "At Risk ⚠️": 2
    }
  }
}
```

### Example 3: Analytics Dashboard with Charts

```bash
curl -X POST http://localhost:5000/api/analytics-dashboard \
  -H "Content-Type: application/json" \
  -d '{"cart_items": [1, 2, 3]}'
```

Response includes animated chart configurations for:
- 📊 Recommendation confidence bar chart
- 🍩 Category sales doughnut chart
- 📈 Support vs Lift scatter plot

---

## 🎨 Animation Features

All visualizations include light, minimalist animations:

- **Smooth transitions** - 300-800ms cubic-in-out easing
- **Fade-in effects** - Charts appear gracefully
- **Hover interactions** - Tooltips with detailed metrics
- **Zoom/Pan** - Interactive exploration (Plotly graphs)
- **Progressive loading** - Staggered element appearance

---

## 📚 Technical Details

### Graph Layout Algorithms

- **Kamada-Kawai**: Force-directed layout preventing node overlap
- **Spring Layout**: Natural clustering of related nodes
- **Collision Detection**: Automatic spacing adjustment

### Clustering Metrics

- **Inertia**: Within-cluster sum of squares (lower is better)
- **Silhouette Score**: Cluster separation quality (-1 to 1, higher is better)
- **Elbow Point**: Optimal K where improvement diminishes

### Data Quality Rules

1. **Negative Price Check**: Flags transactions with price < 0
2. **Null Check**: Identifies missing customer/product IDs
3. **Duplicate Detection**: Finds repeated transactions
4. **Referential Integrity**: Checks for orphaned records

---

## 👥 Contributors

1. **Mayuresh Sawant** - [LinkedIn](https://www.linkedin.com/in/contact-mayuresh-sawant)
2. **Vicky Pukale** - [LinkedIn](https://www.linkedin.com/in/vicky-pukale)
3. **Sunil Saini**

---

## 📄 License

This project is created for academic purposes as part of Data Warehouse & Mining (DWM) coursework.

---