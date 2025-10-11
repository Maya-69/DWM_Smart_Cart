"""
Initialize complete system with real dataset
"""
from load_dataset import create_products_from_dataset
from data_warehouse import DataWarehouse
from ml_models import MLRecommendationEngine

def initialize_system():
    print("=" * 60)
    print("INITIALIZING SMARTCART RECOMMENDATION SYSTEM")
    print("=" * 60)
    
    # Step 1: Load products from dataset
    print("\n[1/4] Loading products from Groceries dataset...")
    create_products_from_dataset('Groceries_dataset.csv')
    
    # Step 2: Load warehouse
    print("\n[2/4] Loading data warehouse...")
    warehouse = DataWarehouse()
    warehouse.load_from_csv('Groceries_dataset.csv')
    
    # Step 3: Train models
    print("\n[3/4] Training ML models...")
    ml_engine = MLRecommendationEngine()
    
    print("  - Training K-Means clustering...")
    kmeans_result = ml_engine.train_kmeans_clustering(n_clusters=3)
    print(f"    ✓ Created {len(kmeans_result) if kmeans_result else 0} customer segments")
    
    print("  - Training Naive Bayes...")
    nb_result = ml_engine.train_naive_bayes()
    if nb_result:
        print(f"    ✓ Accuracy: {nb_result['accuracy']:.2%}")
    else:
        print("    ✗ Failed (need more data)")
    
    print("  - Training Apriori (Association Rules)...")
    # CORRECTED: Changed from train_fpgrowth to train_apriori
    fp_result = ml_engine.train_apriori(min_support=0.01, min_confidence=0.3)
    if fp_result:
        print(f"    ✓ Generated {fp_result['total_rules']} association rules")
    else:
        print("    ✗ Failed (need more data)")
    
    # Step 4: Display sample rules
    print("\n[4/4] Sample Association Rules:")
    if fp_result and 'sample_rules' in fp_result:
        for i, rule in enumerate(fp_result['sample_rules'][:5], 1):
            print(f"  {i}. {rule['antecedents']} → {rule['consequents']}")
            print(f"     Confidence: {rule['confidence']*100:.1f}%, Lift: {rule['lift']:.2f}")
    
    print("\n" + "=" * 60)
    print("✓ SYSTEM INITIALIZED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start backend: python server.py")
    print("2. Start frontend: npm start")
    print("3. Open http://localhost")

if __name__ == '__main__':
    initialize_system()
