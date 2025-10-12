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

@app.route('/api/association-tree', methods=['POST'])
def get_association_tree():
    """NEW: Build hierarchical association tree for cart items"""
    data = request.json
    cart_items = data.get('cart_items', [])
    max_depth = data.get('max_depth', 5)

    if not cart_items:
        return jsonify({'tree': None, 'cart_names': []})

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
    cart_product_names = [row[0] for row in c.fetchall()]
    conn.close()

    if not cart_product_names:
        return jsonify({'tree': None, 'cart_names': []})

    # Build the tree recursively
    tree = build_association_tree_recursive(
        cart_product_names, 
        max_depth=max_depth,
        visited=set(cart_product_names)
    )

    return jsonify({
        'tree': tree if tree else [],
        'cart_names': cart_product_names
    })

def build_association_tree_recursive(items, max_depth, visited, current_depth=0):
    """Recursively build association tree"""
    if current_depth >= max_depth:
        return None

    # Get recommendations for current items
    recommendations = ml_engine.get_recommendations(items, top_n=10)

    if not recommendations:
        return None

    # Filter out already visited items to prevent cycles
    recommendations = [r for r in recommendations if r['product'] not in visited]

    if not recommendations:
        return None

    # Take top recommendations (limit to avoid explosion)
    top_recs = recommendations[:3]  # Limit to 3 children per node

    children = []
    for rec in top_recs:
        # Mark as visited
        new_visited = visited.copy()
        new_visited.add(rec['product'])

        # Recursively build children
        child_tree = build_association_tree_recursive(
            [rec['product']], 
            max_depth=max_depth,
            visited=new_visited,
            current_depth=current_depth + 1
        )

        children.append({
            'name': rec['product'],
            'confidence': round(rec['confidence'], 1),
            'support': rec['users_bought'],
            'lift': round(rec['lift'], 2),
            'children': child_tree if child_tree else []
        })

    return children

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
    
@app.route('/api/recommendation-graph', methods=['POST'])
def get_recommendation_graph():
    """Generate hierarchical recommendation graph"""
    data = request.json
    cart_items = data.get('cart_items', [])
    
    if not cart_items:
        return jsonify({'error': 'No items in cart'}), 400
    
    # Handle both string arrays and object arrays
    if isinstance(cart_items[0], dict):
        cart_names = [item['name'] for item in cart_items]
    else:
        cart_names = cart_items  # Already strings
    
    print(f"🛒 Generating graph for: {cart_names}")
    graph = ml_engine.generate_recommendation_graph(cart_names)
    print(f"📊 Graph generated with {len(graph.get('root', {}).get('children', []))} children")
    
    return jsonify(graph)

def predict_meal_intent(cart_items):
    """
    AI-powered meal intent predictor based on cart items
    """
    if not cart_items:
        return None
    
    # Get product names from cart
    product_names = [item['name'].lower() for item in cart_items]
    product_set = set(product_names)
    
    # Define meal patterns with emojis and confidence
    patterns = [
        {
            'name': 'Sandwich Time',
            'emoji': '🥪',
            'keywords': ['bread', 'butter', 'cheese', 'ham', 'turkey', 'lettuce', 'tomato'],
            'confidence': 0
        },
        {
            'name': 'Breakfast Feast',
            'emoji': '🍳',
            'keywords': ['eggs', 'bacon', 'bread', 'butter', 'milk', 'coffee', 'cereals'],
            'confidence': 0
        },
        {
            'name': 'Pasta Party',
            'emoji': '🍝',
            'keywords': ['pasta', 'spaghetti', 'cheese', 'tomato', 'sauce', 'meat'],
            'confidence': 0
        },
        {
            'name': 'Chicken Dish',
            'emoji': '🍗',
            'keywords': ['chicken', 'meat', 'rice', 'vegetables', 'spices', 'oil'],
            'confidence': 0
        },
        {
            'name': 'Salad Bowl',
            'emoji': '🥗',
            'keywords': ['lettuce', 'tomato', 'cucumber', 'vegetables', 'oil', 'vinegar'],
            'confidence': 0
        },
        {
            'name': 'Pizza Night',
            'emoji': '🍕',
            'keywords': ['cheese', 'tomato', 'meat', 'vegetables', 'dough', 'sauce'],
            'confidence': 0
        },
        {
            'name': 'Baking Session',
            'emoji': '🧁',
            'keywords': ['flour', 'sugar', 'eggs', 'butter', 'milk', 'chocolate'],
            'confidence': 0
        },
        {
            'name': 'Snack Attack',
            'emoji': '🍿',
            'keywords': ['chips', 'snack', 'candy', 'chocolate', 'cookies', 'popcorn'],
            'confidence': 0
        },
        {
            'name': 'Beverages Only',
            'emoji': '🥤',
            'keywords': ['juice', 'soda', 'water', 'coffee', 'tea', 'drinks'],
            'confidence': 0
        },
        {
            'name': 'Healthy Living',
            'emoji': '💪',
            'keywords': ['fruit', 'vegetables', 'yogurt', 'nuts', 'berries', 'organic'],
            'confidence': 0
        }
    ]
    
    # Calculate confidence for each pattern
    for pattern in patterns:
        matches = sum(1 for keyword in pattern['keywords'] if any(keyword in name for name in product_names))
        if matches > 0:
            pattern['confidence'] = (matches / len(pattern['keywords'])) * 100
    
    # Sort by confidence and get top 3
    patterns.sort(key=lambda x: x['confidence'], reverse=True)
    top_predictions = [p for p in patterns if p['confidence'] > 0][:3]
    
    if not top_predictions:
        return {
            'primary': {
                'name': 'Mixed Shopping',
                'emoji': '🛒',
                'confidence': 100,
                'description': 'Diverse selection of items'
            },
            'alternatives': []
        }
    
    primary = top_predictions[0]
    alternatives = top_predictions[1:] if len(top_predictions) > 1 else []
    
    return {
        'primary': {
            'name': primary['name'],
            'emoji': primary['emoji'],
            'confidence': round(primary['confidence']),
            'description': f"Based on {len([k for k in primary['keywords'] if any(k in n for n in product_names)])} matching ingredients"
        },
        'alternatives': [
            {
                'name': alt['name'],
                'emoji': alt['emoji'],
                'confidence': round(alt['confidence'])
            } for alt in alternatives
        ]
    }

@app.route('/api/predict-intent', methods=['POST'])
def predict_intent():
    data = request.json
    cart_items = data.get('cart_items', [])
    
    prediction = predict_meal_intent(cart_items)
    
    return jsonify({
        'success': True,
        'prediction': prediction
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
    print("     - GET  /api/products")
    print("     - POST /api/recommendations")
    print("     - POST /api/ai-insights")
    print("     - POST /api/association-tree    [NEW]")
    print("     - GET  /api/model-status")
    print("     - GET  /api/health\n")

    app.run(debug=True, port=5000, use_reloader=False)