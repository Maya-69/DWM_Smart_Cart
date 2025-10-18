# 🛒 SmartCart - AI-Powered Recommendation System

An intelligent e-commerce recommendation system that uses **Machine Learning** (Apriori Algorithm & Association Rule Mining) to suggest products based on real grocery shopping patterns. Built with **React** frontend and **Python Flask** backend.

---

## ✨ Features

- 🤖 **AI-Powered Recommendations** - Uses Apriori algorithm to find product associations
- 🌳 **Interactive Visualization** - Multi-level hierarchical recommendation graph
- 📊 **Real-Time Insights** - Confidence scores, lift metrics, and user purchase patterns
- 🎯 **Smart Cart Management** - Quantity tracking and inline recommendations
- 📈 **Decision Flow Trees** - Visual explanation of ML recommendation process
- 🔍 **Product Search** - Fast and responsive product filtering

---

## 🎯 Key Algorithms

### Apriori Algorithm
- **Support**: Frequency of itemset in transactions
- **Confidence**: Probability of rule being true
- **Lift**: How much more likely items are bought together vs independently

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

---

## 📂 Project Structure

<pre>
smartcart/
├── src/
│   ├── App.js                 # Main React component
│   ├── App.css                # Styling
│   └── index.js               # React entry point
├── backend/
│   ├── server.py              # Flask API server
│   ├── ml_models.py           # ML algorithms (Apriori)
│   ├── database.py            # SQLite operations
│   ├── data_warehouse.py      # Data processing
│   ├── load_dataset.py        # Dataset loader
│   └── initialize.py          # Setup script
├── Groceries_dataset.csv      # Training data (38,000+ transactions)
├── requirements.txt           # Python dependencies
└── README.md
</pre>

---

## 🚀 How to Run

### Prerequisites
- **Node.js** (v14+)
- **Python** 3.8+
- **npm** or yarn (npm recommended)

---

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/smartcart.git
cd smartcart
```

---

### Step 2: Backend Setup

```bash
cd \backend
```

#### Install Python Dependencies
```bash 
pip install -r requirements.txt
```

#### Initialize Database & Train ML Models

This will:
- Load the grocery dataset (38,000+ transactions)
- Train Apriori model for product associations
- Generate association rules
- Setup SQLite database

```bash
python initialize.py
```

#### Start Backend Server

```bash
python server.py
```

---

### Step 3: Frontend Setup

#### Install Dependencies
```bash
npm install
```

#### Start Development Server
```bash
npm start
```

---

## 🐛 Troubleshooting

### Database Issues

If you encounter database errors, delete and reinitialize:
```bash
Remove-Item smartcart.db -ErrorAction SilentlyContinue
python initialize.py
```
## Contributors :
1) Mayuresh Sawant  (Linkdin Link : https://www.linkedin.com/in/contact-mayuresh-sawant ) 
2) Vicky Pukale     (Linkdin Link : https://www.linkedin.com/in/vicky-pukale )
3) Sunil Saini      
