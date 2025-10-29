# 🚀 SmartCart - Quick Setup Guide

## Fresh PC Installation (Step-by-Step)

### 📋 Prerequisites Check

Before starting, ensure you have:
- [ ] **Node.js** v14+ ([Download](https://nodejs.org/))
- [ ] **Python** 3.8+ ([Download](https://www.python.org/downloads/))
- [ ] **Git** ([Download](https://git-scm.com/))

**Verify installations:**
```bash
node --version   # Should show v14 or higher
python --version # Should show 3.8 or higher
git --version    # Should show any version
```

---

## 🔧 Installation Steps

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Maya-69/DWM_Smart_Cart.git
cd DWM_Smart_Cart
```

---

### 2️⃣ Backend Setup (Python Flask)

#### Navigate to backend folder:
```bash
cd backend
```

#### Install Python dependencies:
```bash
pip install -r requirements.txt
```

**If using conda/virtual environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Then install
pip install -r requirements.txt
```

#### Initialize Database & ML Models:
```bash
python initialize.py
```

**This will take 1-2 minutes and will:**
- ✓ Load 38,765 grocery transactions
- ✓ Train Apriori algorithm (generates 4,588 rules)
- ✓ Create SQLite database with star schema
- ✓ Setup K-Means clustering
- ✓ Initialize data warehouse tables

**Expected output:**
```
Loading dataset...
Training Apriori model...
Creating database tables...
Inserting products...
Creating dimension tables...
✓ Initialization complete!
```

#### Start Flask Backend Server:
```bash
python server.py
```

**Expected output:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

**✅ Backend is now running on port 5000**

---

### 3️⃣ Frontend Setup (React)

#### Open a NEW terminal/command prompt

#### Navigate to project root:
```bash
cd DWM_Smart_Cart
```

#### Install Node dependencies:
```bash
npm install
```

**This will take 2-3 minutes** (downloads ~200 packages)

#### Start React Development Server:
```bash
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view smart-cart in the browser.

  Local:            http://localhost:3000
  On Your Network:  http://192.168.x.x:3000
```

**✅ Frontend is now running on port 3000**

Your browser should automatically open `http://localhost:3000`

---

## ✅ Verification Checklist

Once both servers are running, verify:

- [ ] **Homepage loads** with product grid
- [ ] **Can search** products using search bar
- [ ] **Can add items** to cart (click "+")
- [ ] **Cart updates** with quantity and total
- [ ] **AI Insights tab** appears after adding items
- [ ] **Recommendations show** with confidence scores
- [ ] **Customer segment** displays (Budget/Regular/Premium)
- [ ] **Age group prediction** shows with probabilities
- [ ] **Meal intent** predicts based on cart
- [ ] **Recommendation graph** renders (hierarchical tree)

---

## 🖥️ Daily Usage

Every time you want to run SmartCart:

### Terminal 1 - Backend:
```bash
cd DWM_Smart_Cart/backend
python server.py
```

### Terminal 2 - Frontend:
```bash
cd DWM_Smart_Cart
npm start
```

**Access:** `http://localhost:3000`

**To stop:** Press `Ctrl+C` in both terminals

---

## 🐛 Common Issues & Solutions

### Issue 1: Port 5000 already in use
**Error:** `Address already in use`

**Solution:**
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

---

### Issue 2: Port 3000 already in use
**Error:** `Something is already running on port 3000`

**Solution:** Press `Y` when prompted to use a different port (e.g., 3001)

---

### Issue 3: Module not found errors
**Error:** `ModuleNotFoundError: No module named 'flask'`

**Solution:**
```bash
cd backend
pip install -r requirements.txt
```

---

### Issue 4: Database errors
**Error:** `sqlite3.OperationalError` or `table not found`

**Solution:** Delete and reinitialize database
```bash
cd backend
rm smartcart.db  # Linux/Mac
del smartcart.db  # Windows
python initialize.py
```

---

### Issue 5: CORS errors in browser
**Error:** `Access to fetch at 'http://localhost:5000' blocked by CORS`

**Solution:** Ensure Flask server is running with `Flask-CORS` installed
```bash
pip install Flask-CORS
```

---

### Issue 6: React dependencies warning
**Warning:** `found X vulnerabilities`

**Solution:** (Optional) Update dependencies
```bash
npm audit fix
```

---

## 🔄 Update from GitHub

To get latest changes:

```bash
cd DWM_Smart_Cart
git pull origin main
pip install -r backend/requirements.txt  # Update Python packages
npm install  # Update Node packages
```

---

## 📊 Testing the Features

### Test Apriori Recommendations:
1. Add "Milk" to cart
2. Click "AI Insights"
3. Should recommend: Bread, Butter, Cheese (items often bought with milk)

### Test Customer Segmentation:
1. Add items totaling < ₹300 → "Budget Shopper 💰"
2. Add items totaling ₹300-800 → "Regular Shopper 🛒"
3. Add items totaling > ₹800 → "Premium Shopper 👑"

### Test Age Prediction:
1. Add snacks + beverages → Likely "18-25 🎓"
2. Add vegetables + fruits → Likely "35-45 👨‍👩‍👧" or "45-60 👴"
3. Add dairy + bakery → Likely "25-35 🏢"

### Test Meal Intent:
1. Add bread + butter + cheese → "Sandwich Time 🥪"
2. Add eggs + bacon + bread → "Breakfast Feast 🍳"
3. Add pasta + tomatoes + cheese → "Italian Night 🍝"

---

## 📁 Project Files Overview

```
DWM_Smart_Cart/
├── backend/
│   ├── server.py              # Flask API (7 endpoints)
│   ├── ml_models.py           # Apriori, K-Means, Naive Bayes
│   ├── database.py            # SQLite operations
│   ├── data_warehouse.py      # Star schema & ETL
│   ├── initialize.py          # Setup script
│   ├── requirements.txt       # Python dependencies
│   ├── smartcart.db          # SQLite database (created on init)
│   └── Groceries_dataset.csv # 38,765 transactions
├── src/
│   ├── App.js                # Main React component
│   └── App.css               # Styling
├── public/
│   └── index.html            # HTML template
├── package.json              # Node dependencies
├── README.md                 # Full documentation
└── PROJECT_EXPLANATION.txt   # Academic explanation (DWM)
```

---

## 🎓 For Academic Presentation

**Key Points to Mention:**

1. **Data Warehouse Architecture:** Star schema with fact and dimension tables
2. **ML Algorithms Used:** Apriori (4,588 rules), K-Means (3 clusters), Naive Bayes
3. **Dataset:** 38,765 real-world grocery transactions from Kaggle
4. **Features:** Real-time recommendations, customer segmentation, demographic prediction
5. **Tech Stack:** React + Flask + SQLite + Scikit-learn + mlxtend

**Demo Flow:**
1. Show empty cart → Add milk → Show recommendations
2. Add more items → Show segment changing (Budget → Regular → Premium)
3. Show age prediction updating based on categories
4. Show meal intent detection
5. Show recommendation graph visualization

---

## 📞 Support

For issues or questions:
- Check `PROJECT_EXPLANATION.txt` for detailed DWM concepts
- Check `README.md` for full documentation
- GitHub Issues: https://github.com/Maya-69/DWM_Smart_Cart/issues

---

**Happy Coding! 🚀**
