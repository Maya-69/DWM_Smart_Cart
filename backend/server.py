from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
from data_warehouse import DataWarehouse
from ml_models import MLRecommendationEngine
import os

app = Flask(__name__)
CORS(app)

warehouse = DataWarehouse()
ml_engine = MLRecommendationEngine()

# Train models on startup if not already trained
def initialize_ml_models():
    """Load or train ML models on server startup"""
    print("=" * 60)
    print("INITIALIZING ML MODELS")
    print("=" * 60)
    
    # Check if we have trained models
    if not ml_engine.association_rules or len(ml_engine.association_rules) == 0:
        print("\n⚠️  No trained models found. Training now...")
        
        # Train Apriori (most important for recommendations)
        print("  - Training Apriori algorithm...")
        result = ml_engine.train_apriori(min_support=0.01, min_confidence=0.3)
        
        if result:
            print(f"  ✓ Trained successfully! Generated {result['total_rules']} association rules")
        else:
            print("  ✗ Training failed - no transaction data available")
            print("  → Run 'python initialize.py' to load data first")
            return False
        
        # Train K-Means
        print("  - Training K-Means clustering...")
        kmeans_result = ml_engine.train_kmeans_clustering(n_clusters=3)
        if kmeans_result:
            print(f"  ✓ Created {len(kmeans_result)} customer segments")
        
        # Train Naive Bayes
        print("  - Training Naive Bayes...")
        nb_result = ml_engine.train_naive_bayes()
        if nb_result:
            print(f"  ✓ Accuracy: {nb_result['accuracy']:.2%}")
    else:
        print(f"\n✓ Models already loaded: {len(ml_engine.association_rules)} association rules available")
    
    print("=" * 60)
    print("✓ ML ENGINE READY")
    print("=" * 60)
    return True

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
    """
    Get ML-based recommendations from trained model
    
    Request body:
    {
        "cart_items": [1, 2, 3]  // Product IDs
    }
    
    Response:
    [
        {
            "id": 4,
            "name": "Product Name",
            "price": 100,
            "img": "emoji",
            "category": "category",
            "confidence": 85.5,
            "support": 0.15,
            "lift": 2.3,
            "users_bought": 150,
            "reason": "Often bought with Product A"
        }
    ]
    """
    data = request.json
    cart_items = data.get('cart_items', [])
    
    if not cart_items:
        return jsonify([])
    
    # Check if models are trained
    if not ml_engine.association_rules:
        return jsonify({
            'error': 'Models not trained',
            'message': 'Please run initialize.py first or call /api/train-models'
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
    
    # Get recommendations from association rules (trained model)
    recommendations = ml_engine.get_recommendations(cart_names, top_n=8)
    
    # Map back to product IDs with full details
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
    """
    Get detailed decision tree explanation with purchase statistics
    
    Request body:
    {
        "cart_items": [1, 2, 3]  // Product IDs
    }
    
    Response:
    {
        "cart_items": ["product1", "product2"],
        "total_recommendations": 10,
        "decision_flow": {
            "step1": {...},
            "step2": {...},
            "step3": {...},
            "step4": {...}
        },
        "statistics": {...}
    }
    """
    data = request.json
    cart_items = data.get('cart_items', [])
    
    if not cart_items:
        return jsonify({'error': 'No cart items'})
    
    # Check if models are trained
    if not ml_engine.association_rules:
        return jsonify({
            'error': 'Models not trained',
            'message': 'Please run initialize.py first'
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
    
    # Generate detailed decision tree
    tree = ml_engine.generate_detailed_decision_tree(cart_names)
    
    return jsonify(tree)

@app.route('/api/train-models', methods=['POST'])
def train_models():
    """
    Manually trigger model training
    
    Response:
    {
        "kmeans": [...],
        "naive_bayes": {...},
        "association_rules": {
            "total_rules": 100,
            "sample_rules": [...]
        }
    }
    """
    results = {}
    
    # Train K-Means
    print("Training K-Means...")
    kmeans_result = ml_engine.train_kmeans_clustering(n_clusters=3)
    results['kmeans'] = kmeans_result
    
    # Train Naive Bayes
    print("Training Naive Bayes...")
    nb_result = ml_engine.train_naive_bayes()
    results['naive_bayes'] = nb_result
    
    # Train Association Rules (Apriori)
    print("Training Apriori...")
    rules_result = ml_engine.train_apriori(min_support=0.01, min_confidence=0.3)
    results['association_rules'] = rules_result
    
    return jsonify(results)

@app.route('/api/model-status', methods=['GET'])
def model_status():
    """
    Check if models are trained and ready
    
    Response:
    {
        "trained": true,
        "total_rules": 150,
        "kmeans_ready": true,
        "naive_bayes_ready": true
    }
    """
    return jsonify({
        'trained': len(ml_engine.association_rules) > 0,
        'total_rules': len(ml_engine.association_rules),
        'kmeans_ready': ml_engine.kmeans_model is not None,
        'naive_bayes_ready': ml_engine.naive_bayes_model is not None
    })

@app.route('/api/olap/rollup', methods=['GET'])
def olap_rollup():
    """OLAP Roll-up operation"""
    dimension = request.args.get('dimension', 'category')
    result = warehouse.olap_rollup(dimension)
    return jsonify(result)

@app.route('/api/olap/drilldown/<category>', methods=['GET'])
def olap_drilldown(category):
    """OLAP Drill-down operation"""
    result = warehouse.olap_drilldown(category)
    return jsonify(result)

@app.route('/api/init-data', methods=['POST'])
def initialize_data():
    """Initialize warehouse with sample data"""
    try:
        warehouse.load_from_csv('Groceries_dataset.csv')
        return jsonify({'status': 'success', 'message': 'Data warehouse initialized'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_trained': len(ml_engine.association_rules) > 0
    })

if __name__ == '__main__':
    # Initialize ML models on startup
    initialize_ml_models()
    
    # Start Flask server
    print("\n🚀 Starting Flask server on http://localhost:5000")
    print("   API endpoints:")
    print("   - GET  /api/products")
    print("   - POST /api/recommendations")
    print("   - POST /api/ai-insights")
    print("   - POST /api/train-models")
    print("   - GET  /api/model-status")
    print("\n")
    
    app.run(debug=True, port=5000)
