"""
SmartCart Flask API Server

Run this file to start the server:
    python server.py

Prerequisites:
    1. Run 'python initialize.py' first to train models
    2. Make sure smartcart.db exists
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import os
from data_warehouse import DataWarehouse
from ml_models import MLRecommendationEngine

app = Flask(__name__)
CORS(app)

# Initialize components
warehouse = DataWarehouse()
ml_engine = MLRecommendationEngine()

# Load trained models on startup
def load_trained_models():
    """Load pre-trained models from database"""
    print("\n" + "=" * 70)
    print("LOADING ML MODELS")
    print("=" * 70)
    
    if not os.path.exists('smartcart.db'):
        print("\n❌ Database not found!")
        print("   Please run: python initialize.py")
        return False
    
    # Train models (they load from database)
    print("\nLoading Apriori model...")
    result = ml_engine.train_apriori(min_support=0.01, min_confidence=0.3)
    
    if result and result['total_rules'] > 0:
        print(f"✓ Loaded {result['total_rules']} association rules")
        print("=" * 70)
        return True
    else:
        print("\n❌ No trained models found!")
        print("   Please run: python initialize.py")
        print("=" * 70)
        return False

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products from database"""
    conn = sqlite3.connect('smartcart.db')
    c = conn.cursor()
    c.execute('SELECT id, name, category, price, img FROM products LIMIT 50')
    products = [{'id': row[0], 'name': row[1], 'category': row[2], 
                 'price': row[3], 'img': row[4]} for row in c.fetchall()]
    conn.close()
    return jsonify(products)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get ML-based recommendations from trained model"""
    data = request.json
    cart_items = data.get('cart_items', [])
    
    if not cart_items:
        return jsonify([])
    
    # Check if models are loaded
    if not ml_engine.association_rules:
        return jsonify({
            'error': 'Models not loaded',
            'message': 'Please restart server or run initialize.py'
        }), 503
    
    # Get product names from IDs
    conn = sqlite3.connect('smartcart.db')
    c = conn.cursor()
    placeholders = ','.join('?' * len(cart_items))
    c.execute(f'SELECT name FROM products WHERE id IN ({placeholders})', cart_items)
    cart_names = [row[0] for row in c.fetchall()]
    conn.close()
    
    if not cart_names:
        return jsonify([])
    
    # Get recommendations
    recommendations = ml_engine.get_recommendations(cart_names, top_n=8)
    
    # Map to product details
    if recommendations:
        conn = sqlite3.connect('smartcart.db')
        c = conn.cursor()
        for rec in recommendations:
            c.execute('SELECT id, price, img, category FROM products WHERE name = ?', (rec['product'],))
            result = c.fetchone()
            if result:
                rec['id'] = result[0]
                rec['price'] = result[1]
                rec['img'] = result[2]
                rec['category'] = result[3]
                rec['name'] = rec['product']
        conn.close()
    
    return jsonify(recommendations)

@app.route('/api/ai-insights', methods=['POST'])
def get_ai_insights():
    """Get detailed decision tree explanation"""
    data = request.json
    cart_items = data.get('cart_items', [])
    
    if not cart_items:
        return jsonify({'error': 'No cart items'})
    
    # Check if models are loaded
    if not ml_engine.association_rules:
        return jsonify({
            'error': 'Models not loaded',
            'message': 'Please restart server or run initialize.py'
        }), 503
    
    # Get product names
    conn = sqlite3.connect('smartcart.db')
    c = conn.cursor()
    placeholders = ','.join('?' * len(cart_items))
    c.execute(f'SELECT name FROM products WHERE id IN ({placeholders})', cart_items)
    cart_names = [row[0] for row in c.fetchall()]
    conn.close()
    
    if not cart_names:
        return jsonify({'error': 'No valid products found'})
    
    # Generate decision tree
    tree = ml_engine.generate_detailed_decision_tree(cart_names)
    
    return jsonify(tree)

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Check if models are loaded"""
    return jsonify({
        'trained': len(ml_engine.association_rules) > 0,
        'total_rules': len(ml_engine.association_rules),
        'kmeans_ready': ml_engine.kmeans_model is not None,
        'naive_bayes_ready': ml_engine.naive_bayes_model is not None
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(ml_engine.association_rules) > 0
    })

if __name__ == '__main__':
    # Load models on startup
    models_loaded = load_trained_models()
    
    if not models_loaded:
        print("\n⚠️  WARNING: Server starting without trained models!")
        print("   Some endpoints will not work correctly.")
        print("   Please run 'python initialize.py' and restart.\n")
    
    # Start server
    print("\n🚀 Starting Flask server...")
    print("   URL: http://localhost:5000")
    print("   Endpoints:")
    print("   - GET  /api/products")
    print("   - POST /api/recommendations")
    print("   - POST /api/ai-insights")
    print("   - GET  /api/model-status")
    print("   - GET  /api/health\n")
    
    app.run(debug=True, port=5000, use_reloader=False)  # use_reloader=False prevents double loading
