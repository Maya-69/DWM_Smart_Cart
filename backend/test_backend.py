"""
Test backend functionality
"""
import requests
import json

BASE_URL = 'http://localhost:5000/api'

def test_products():
    """Test getting products"""
    print("\n1. Testing GET /api/products...")
    response = requests.get(f'{BASE_URL}/products')
    if response.status_code == 200:
        products = response.json()
        print(f"   ✓ Got {len(products)} products")
        return products
    else:
        print(f"   ✗ Failed: {response.status_code}")
        return []

def test_recommendations(product_ids):
    """Test getting recommendations"""
    print("\n2. Testing POST /api/recommendations...")
    response = requests.post(
        f'{BASE_URL}/recommendations',
        json={'cart_items': product_ids}
    )
    if response.status_code == 200:
        recs = response.json()
        print(f"   ✓ Got {len(recs)} recommendations")
        for rec in recs[:3]:
            print(f"      - {rec['name']}: {rec['confidence']:.1f}% confidence")
        return recs
    else:
        print(f"   ✗ Failed: {response.status_code}")
        print(f"      Error: {response.json()}")
        return []

def test_ai_insights(product_ids):
    """Test AI insights"""
    print("\n3. Testing POST /api/ai-insights...")
    response = requests.post(
        f'{BASE_URL}/ai-insights',
        json={'cart_items': product_ids}
    )
    if response.status_code == 200:
        insights = response.json()
        print(f"   ✓ Got AI insights")
        print(f"      - Total recommendations: {insights.get('total_recommendations', 0)}")
        print(f"      - Cart items: {insights.get('cart_items', [])}")
        return insights
    else:
        print(f"   ✗ Failed: {response.status_code}")
        return None

def test_model_status():
    """Test model status"""
    print("\n4. Testing GET /api/model-status...")
    response = requests.get(f'{BASE_URL}/model-status')
    if response.status_code == 200:
        status = response.json()
        print(f"   ✓ Models trained: {status['trained']}")
        print(f"      - Total rules: {status['total_rules']}")
        return status
    else:
        print(f"   ✗ Failed: {response.status_code}")
        return None

if __name__ == '__main__':
    print("=" * 60)
    print("BACKEND API TESTS")
    print("=" * 60)
    
    try:
        # Test 1: Get products
        products = test_products()
        
        if products:
            # Test 2: Get recommendations for first 2 products
            test_product_ids = [products[0]['id'], products[1]['id']]
            print(f"\n   Using products: {[p['name'] for p in products[:2]]}")
            
            recs = test_recommendations(test_product_ids)
            
            # Test 3: Get AI insights
            if recs:
                test_ai_insights(test_product_ids)
            
            # Test 4: Model status
            test_model_status()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS COMPLETED")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to server!")
        print("   Make sure the server is running:")
        print("   python server.py")
