"""
SmartCart Flask API Server
Enhanced with clustering analysis, interactive graphs, and data quality endpoints
Run this file to start the server:
    python server.py

Prerequisites:
    1. Run 'python initialize.py' first to train models
    2. Make sure smartcart.db exists
"""
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import sqlite3
import os
import json
from data_warehouse import DataWarehouse
from ml_models import MLRecommendationEngine
from generate_graphs import SmartCartVisualizer

app = Flask(__name__)
CORS(app)

# Initialize components
warehouse = DataWarehouse()
ml_engine = MLRecommendationEngine()
visualizer = None  # Initialize on demand

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

@app.route('/api/predict-customer-segment', methods=['POST'])
def predict_customer_segment():
    """Predict customer segment (Budget/Regular/Premium) using K-Means based on cart"""
    try:
        data = request.json
        cart_items = data.get('cart_items', [])
        
        if not cart_items:
            return jsonify({'success': False, 'error': 'No cart items'}), 400
        
        # Calculate metrics from cart
        total_spent = sum([item['price'] * item['quantity'] for item in cart_items])
        unique_products = len(set([item['id'] for item in cart_items]))
        avg_price = total_spent / len(cart_items) if cart_items else 0
        
        # Predict segment using K-Means
        segment_result = ml_engine.assign_customer_to_segment(cart_data=cart_items)
        
        # Calculate confidence based on distance to cluster center
        confidence = 75 + (segment_result.get('distance_score', 0) * 25)  # 75-100%
        
        return jsonify({
            'success': True,
            'segment': segment_result['segment_name'],
            'cluster_id': segment_result['cluster_id'],
            'confidence': min(confidence, 99),
            'metrics': {
                'total_spent': total_spent,
                'unique_products': unique_products,
                'avg_price': avg_price
            },
            'description': get_segment_description(segment_result['segment_name'])
        })
        
    except Exception as e:
        print(f"Error in predict_customer_segment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict-age-group', methods=['POST'])
def predict_age_group():
    """Predict customer age group using Naive Bayes based on cart items"""
    try:
        data = request.json
        cart_items = data.get('cart_items', [])
        
        if not cart_items:
            return jsonify({'success': False, 'error': 'No cart items'}), 400
        
        # Get product categories from cart
        conn = sqlite3.connect('smartcart.db')
        c = conn.cursor()
        
        placeholders = ','.join('?' * len(cart_items))
        cart_ids = [item['id'] for item in cart_items]
        c.execute(f'SELECT category FROM products WHERE id IN ({placeholders})', cart_ids)
        categories = [row[0] for row in c.fetchall()]
        conn.close()
        
        # Predict age group using Naive Bayes
        age_result = ml_engine.predict_age_from_cart(categories)
        
        return jsonify({
            'success': True,
            'age_group': age_result['age_group'],
            'confidence': age_result['confidence'],
            'probabilities': age_result['probabilities'],
            'top_categories': categories[:5],
            'reasoning': age_result.get('reasoning', '')
        })
        
    except Exception as e:
        print(f"Error in predict_age_group: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def get_segment_description(segment_name):
    """Get description for customer segment"""
    descriptions = {
        'Budget Shoppers 💰': 'Value-conscious shoppers who prioritize savings and essential items',
        'Regular Shoppers 🛒': 'Consistent shoppers with balanced spending patterns',
        'Premium Shoppers 👑': 'High-value customers who prefer quality and variety'
    }
    return descriptions.get(segment_name, 'Regular shopper with balanced preferences')

# ========== NEW ENHANCED ENDPOINTS ==========

@app.route('/api/cluster-analysis', methods=['GET'])
def cluster_analysis():
    """Get customer clustering analysis with elbow method"""
    try:
        # Find optimal clusters
        optimal_k, k_range, inertias, silhouette_scores = ml_engine.find_optimal_clusters_elbow(max_k=8)
        
        # Train with optimal K
        clustering_result = ml_engine.train_kmeans_with_elbow(n_clusters=optimal_k)
        
        if clustering_result:
            return jsonify({
                'success': True,
                'optimal_k': optimal_k,
                'elbow_data': {
                    'k_values': k_range,
                    'inertias': inertias,
                    'silhouette_scores': silhouette_scores
                },
                'clusters': clustering_result
            })
        else:
            return jsonify({'success': False, 'error': 'No data available'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rfm-segmentation', methods=['GET'])
def rfm_segmentation():
    """Get RFM (Recency, Frequency, Monetary) customer segmentation"""
    try:
        result = ml_engine.perform_rfm_segmentation()
        
        if result:
            return jsonify({
                'success': True,
                'rfm_analysis': result
            })
        else:
            return jsonify({'success': False, 'error': 'No data available'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/data-quality', methods=['GET'])
def data_quality_check():
    """Run data quality validation checks"""
    try:
        result = warehouse.validate_data_quality()
        return jsonify({
            'success': True,
            'quality_report': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/etl-stats', methods=['GET'])
def etl_statistics():
    """Get ETL execution statistics"""
    try:
        stats = warehouse.get_etl_stats()
        return jsonify({
            'success': True,
            'etl_history': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/refresh-aggregations', methods=['POST'])
def refresh_agg():
    """Refresh aggregation tables"""
    try:
        result = warehouse.refresh_aggregations()
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analytics-dashboard', methods=['POST'])
def analytics_dashboard():
    """Get comprehensive analytics with graph data for visualization"""
    try:
        data = request.json
        cart_items = data.get('cart_items', [])
        
        # Get product names
        if cart_items:
            conn = sqlite3.connect('smartcart.db')
            c = conn.cursor()
            placeholders = ','.join('?' * len(cart_items))
            c.execute(f'SELECT name FROM products WHERE id IN ({placeholders})', cart_items)
            cart_names = [row[0] for row in c.fetchall()]
            conn.close()
        else:
            cart_names = []
        
        # Get recommendations
        recommendations = ml_engine.get_recommendations(cart_names, top_n=10) if cart_names else []
        
        # Get clustering data
        try:
            clustering = ml_engine.train_kmeans_with_elbow(n_clusters=3)
        except:
            clustering = None
        
        # Get OLAP data
        olap_rollup = warehouse.olap_rollup('category')
        
        # Prepare chart configurations with animations
        response = {
            'success': True,
            'cart_items': cart_names,
            'recommendations': recommendations,
            'charts': {
                'recommendation_confidence': {
                    'type': 'bar',
                    'data': {
                        'labels': [r['product'][:20] for r in recommendations[:5]],
                        'datasets': [{
                            'label': 'Confidence %',
                            'data': [r['confidence'] for r in recommendations[:5]],
                            'backgroundColor': 'rgba(52, 152, 219, 0.7)',
                            'borderColor': 'rgba(52, 152, 219, 1)',
                            'borderWidth': 2
                        }]
                    },
                    'options': {
                        'animation': {
                            'duration': 800,
                            'easing': 'easeInOutQuart'
                        },
                        'plugins': {
                            'legend': {'display': True}
                        }
                    }
                },
                'category_sales': {
                    'type': 'doughnut',
                    'data': {
                        'labels': [item['category'] for item in olap_rollup[:6]],
                        'datasets': [{
                            'data': [item['total_sales'] for item in olap_rollup[:6]],
                            'backgroundColor': [
                                'rgba(255, 99, 132, 0.7)',
                                'rgba(54, 162, 235, 0.7)',
                                'rgba(255, 206, 86, 0.7)',
                                'rgba(75, 192, 192, 0.7)',
                                'rgba(153, 102, 255, 0.7)',
                                'rgba(255, 159, 64, 0.7)'
                            ]
                        }]
                    },
                    'options': {
                        'animation': {
                            'animateRotate': True,
                            'animateScale': True,
                            'duration': 1000
                        }
                    }
                },
                'lift_scatter': {
                    'type': 'scatter',
                    'data': {
                        'datasets': [{
                            'label': 'Support vs Lift',
                            'data': [
                                {'x': r['support'] * 100, 'y': r['lift'], 
                                 'r': r['confidence'] * 10}
                                for r in recommendations[:10]
                            ],
                            'backgroundColor': 'rgba(75, 192, 192, 0.5)'
                        }]
                    },
                    'options': {
                        'animation': {
                            'duration': 1200,
                            'easing': 'easeInOutBack'
                        }
                    }
                }
            },
            'clustering': clustering,
            'olap_summary': olap_rollup[:10]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/interactive-graph/<graph_type>', methods=['GET'])
def get_interactive_graph(graph_type):
    """Serve interactive HTML graphs"""
    try:
        graph_file_map = {
            'network': 'interactive_network_graph.html',
            'dashboard': 'interactive_dashboard.html',
            'flow': 'recommendation_flow.html'
        }
        
        filename = graph_file_map.get(graph_type)
        if not filename:
            return jsonify({'error': 'Invalid graph type'}), 400
        
        filepath = os.path.join('graphs', filename)
        
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='text/html')
        else:
            # Generate on-demand if not exists
            global visualizer
            if visualizer is None:
                visualizer = SmartCartVisualizer()
                visualizer.load_data()
                visualizer.calculate_apriori_metrics()
            
            if graph_type == 'network':
                visualizer.plot_interactive_network_graph()
            elif graph_type == 'dashboard':
                visualizer.plot_interactive_association_dashboard()
            elif graph_type == 'flow':
                visualizer.plot_animated_recommendation_flow()
            
            return send_file(filepath, mimetype='text/html')
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-graphs', methods=['POST'])
def generate_all_graphs():
    """Generate all static and interactive graphs"""
    try:
        global visualizer
        visualizer = SmartCartVisualizer()
        visualizer.generate_all_graphs()
        
        return jsonify({
            'success': True,
            'message': 'All graphs generated successfully',
            'files': [
                'exploratory_analysis.png',
                'apriori_accuracy_metrics.png',
                'association_rules_analysis.png',
                'recommendation_accuracy_metrics.png',
                'interactive_network_graph.html',
                'interactive_dashboard.html',
                'recommendation_flow.html'
            ]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500



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
    print("\n📡 ENDPOINTS:")
    print("\n   🛒 PRODUCT & RECOMMENDATIONS:")
    print("     - GET  /api/products")
    print("     - POST /api/recommendations")
    print("     - POST /api/ai-insights")
    print("     - POST /api/association-tree")
    print("     - POST /api/recommendation-graph")
    print("     - POST /api/predict-intent")
    print("\n   📊 ANALYTICS & CLUSTERING:")
    print("     - GET  /api/cluster-analysis          [NEW]")
    print("     - GET  /api/rfm-segmentation           [NEW]")
    print("     - POST /api/analytics-dashboard        [NEW]")
    print("\n   🔍 DATA QUALITY & ETL:")
    print("     - GET  /api/data-quality               [NEW]")
    print("     - GET  /api/etl-stats                  [NEW]")
    print("     - POST /api/refresh-aggregations       [NEW]")
    print("\n   📈 INTERACTIVE GRAPHS:")
    print("     - GET  /api/interactive-graph/network  [NEW]")
    print("     - GET  /api/interactive-graph/dashboard[NEW]")
    print("     - GET  /api/interactive-graph/flow     [NEW]")
    print("     - POST /api/generate-graphs            [NEW]")
    print("\n   ❤️  SYSTEM:")
    print("     - GET  /api/model-status")
    print("     - GET  /api/health\n")

    app.run(debug=True, port=5000, use_reloader=False)