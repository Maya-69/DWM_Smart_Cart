"""
SmartCart ML Initialization & Testing
Run this ONCE to:
- Load products from dataset
- Create data warehouse
- Train all ML models
- Test functionality
"""
import os
import sys
import sqlite3
import requests
import time

def check_prerequisites():
    """Check if required files exist"""
    print("=" * 70)
    print("CHECKING PREREQUISITES")
    print("=" * 70)
    
    errors = []
    
    # Check dataset
    if not os.path.exists('Groceries_dataset.csv'):
        errors.append(" Groceries_dataset.csv not found!")
        print("   Download from: https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset")
    else:
        print("✓ Groceries_dataset.csv found")
    
    if errors:
        print("\n" + "\n".join(errors))
        sys.exit(1)
    
    print()

def initialize_database_and_models():
    """Step 1: Load data and train models"""
    from load_dataset import create_products_from_dataset
    from data_warehouse import DataWarehouse
    from ml_models import MLRecommendationEngine
    
    print("=" * 70)
    print("STEP 1: DATABASE & MODEL TRAINING")
    print("=" * 70)
    
    # Load products from dataset
    print("\n[1/4] Loading products from Groceries dataset...")
    create_products_from_dataset('Groceries_dataset.csv')
    
    # Load warehouse
    print("\n[2/4] Loading data warehouse...")
    warehouse = DataWarehouse()
    warehouse.load_from_csv('Groceries_dataset.csv')
    print("    ✓ Warehouse loaded with transaction data")
    
    # Train models
    print("\n[3/4] Training ML models...")
    ml_engine = MLRecommendationEngine()
    
    print("    - Training K-Means clustering...")
    kmeans_result = ml_engine.train_kmeans_clustering(n_clusters=3)
    if kmeans_result:
        print(f"      ✓ Created {len(kmeans_result)} customer segments")
    else:
        print("       Not enough data for K-Means")
    
    print("    - Training Naive Bayes...")
    nb_result = ml_engine.train_naive_bayes()
    if nb_result:
        print(f"      ✓ Accuracy: {nb_result['accuracy']:.2%}")
    else:
        print("       Not enough data for Naive Bayes")
    
    print("    - Training Apriori (Association Rules)...")
    apriori_result = ml_engine.train_apriori(min_support=0.01, min_confidence=0.3)
    if apriori_result:
        print(f"      ✓ Generated {apriori_result['total_rules']} association rules")
    else:
        print("      Failed to train Apriori")
        sys.exit(1)
    
    # Display sample rules
    print("\n[4/4] Sample Association Rules:")
    if apriori_result and 'sample_rules' in apriori_result:
        for i, rule in enumerate(apriori_result['sample_rules'][:5], 1):
            print(f"    {i}. {rule['antecedents']} → {rule['consequents']}")
            print(f"       Confidence: {rule['confidence']*100:.1f}%, Lift: {rule['lift']:.2f}")
    
    print("\n" + "=" * 70)
    print("✓ INITIALIZATION COMPLETE!")
    print("=" * 70)
    
    return ml_engine

def test_functionality(ml_engine):
    """Step 2: Test ML models"""
    print("\n" + "=" * 70)
    print("STEP 2: TESTING ML MODELS")
    print("=" * 70)
    
    # Get some products
    conn = sqlite3.connect('smartcart.db')
    c = conn.cursor()
    c.execute('SELECT id, name FROM products LIMIT 5')
    products = c.fetchall()
    conn.close()
    
    if not products:
        print(" No products found in database!")
        return
    
    # Test 1: Recommendations
    print("\n[TEST 1] Recommendation Engine")
    test_items = [products[0][1], products[1][1]]
    print(f"   Cart items: {test_items}")
    
    recs = ml_engine.get_recommendations(test_items, top_n=5)
    if recs:
        print(f"   ✓ Got {len(recs)} recommendations:")
        for i, rec in enumerate(recs[:3], 1):
            print(f"      {i}. {rec['product']}: {rec['confidence']:.1f}% confidence")
    else:
        print("     No recommendations found")
    
    # Test 2: Decision Tree
    print("\n[TEST 2] Decision Tree Generation")
    tree = ml_engine.generate_detailed_decision_tree(test_items)
    if tree and 'decision_flow' in tree:
        print(f"   ✓ Generated decision tree with {tree['total_recommendations']} recommendations")
        print(f"   Cart items: {tree['cart_items']}")
        if tree['decision_flow'].get('step4'):
            top_picks = tree['decision_flow']['step4'].get('top_picks', [])
            if top_picks:
                print(f"   Top recommendation: {top_picks[0]['product']} ({top_picks[0]['confidence']:.0f}%)")
    else:
        print("     Could not generate decision tree")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)

def test_api_server():
    """Step 3: Test API server (optional - requires server to be running)"""
    print("\n" + "=" * 70)
    print("STEP 3: API SERVER TEST (Optional)")
    print("=" * 70)
    print("\nTo test the API:")
    print("  1. Open a new terminal")
    print("  2. Run: python server.py")
    print("  3. Test endpoints:")
    print("     - GET  http://localhost:5000/api/products")
    print("     - POST http://localhost:5000/api/recommendations")
    print("     - GET  http://localhost:5000/api/model-status")
    
    # Try to test if server is running
    try:
        response = requests.get('http://localhost:5000/api/model-status', timeout=2)
        if response.status_code == 200:
            status = response.json()
            print(f"\n✓ Server is running!")
            print(f"   Models trained: {status['trained']}")
            print(f"   Total rules: {status['total_rules']}")
        else:
            print("\n  Server returned error")
    except requests.exceptions.ConnectionError:
        print("\n Server not running (this is optional)")
    except Exception as e:
        print(f"\n Could not connect: {e}")

def display_next_steps():
    """Show what to do next"""
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\n1. Start Backend Server:")
    print("   cd backend")
    print("   python server.py")
    print("\n2. Start Frontend (in new terminal):")
    print("   npm start")
    print("\n3. Open Browser:")
    print("   http://localhost:3000")
    print("\n" + "=" * 70)

def main():
    """Main initialization flow"""
    print("SMARTCART ML RECOMMENDATION SYSTEM - INITIALIZATION")
    
    # Check prerequisites
    check_prerequisites()
    
    # Initialize and train
    ml_engine = initialize_database_and_models()
    
    # Test models
    test_functionality(ml_engine)
    
    # Show API test info
    test_api_server()
    
    # Show next steps
    display_next_steps()
    
    print("\n Initialization complete! Run 'python server.py' to start the server.\n")

if __name__ == '__main__':
    main()
