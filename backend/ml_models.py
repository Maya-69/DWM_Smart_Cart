"""
ML Models: K-Means, Naive Bayes, Apriori (Custom Implementation)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
from itertools import combinations
import sqlite3
import json

class MLRecommendationEngine:
    def __init__(self, db_path='smartcart.db'):
        self.db_path = db_path
        self.kmeans_model = None
        self.naive_bayes_model = None
        self.association_rules = []
        self.scaler = StandardScaler()
    
    def train_kmeans_clustering(self, n_clusters=3):
        """K-MEANS: Customer segmentation"""
        conn = sqlite3.connect(self.db_path)
        query = """SELECT c.customer_id,
                          COUNT(DISTINCT f.product_id) as unique_products,
                          AVG(f.total_price) as avg_transaction,
                          SUM(f.total_price) as total_spent,
                          COUNT(f.transaction_id) as purchase_frequency
                   FROM dim_customer c
                   JOIN fact_transactions f ON c.customer_id = f.customer_id
                   GROUP BY c.customer_id"""
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return None
        
        X = df[['unique_products', 'avg_transaction', 'total_spent', 'purchase_frequency']].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        df['cluster'] = clusters
        df['cluster_name'] = df['cluster'].map({
            0: 'Budget Shoppers', 1: 'Regular Shoppers', 2: 'Premium Shoppers'
        })
        
        return df[['customer_id', 'cluster', 'cluster_name', 'total_spent']].to_dict('records')
    
    def train_naive_bayes(self):
        """NAIVE BAYES: Category prediction"""
        conn = sqlite3.connect(self.db_path)
        query = """SELECT c.age_group, c.location, c.customer_segment, 
                          f.category, COUNT(*) as purchase_count
                   FROM dim_customer c
                   JOIN fact_transactions f ON c.customer_id = f.customer_id
                   GROUP BY c.age_group, c.location, c.customer_segment, f.category"""
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return None
        
        le_age = LabelEncoder()
        le_loc = LabelEncoder()
        le_seg = LabelEncoder()
        le_cat = LabelEncoder()
        
        df['age_encoded'] = le_age.fit_transform(df['age_group'])
        df['loc_encoded'] = le_loc.fit_transform(df['location'])
        df['seg_encoded'] = le_seg.fit_transform(df['customer_segment'])
        df['cat_encoded'] = le_cat.fit_transform(df['category'])
        
        X = df[['age_encoded', 'loc_encoded', 'seg_encoded']].values
        y = df['cat_encoded'].values
        
        self.naive_bayes_model = MultinomialNB()
        self.naive_bayes_model.fit(X, y)
        
        self.encoders = {'age': le_age, 'location': le_loc, 'segment': le_seg, 'category': le_cat}
        
        return {"status": "trained", "accuracy": float(self.naive_bayes_model.score(X, y))}
    
    def train_apriori(self, min_support=0.01, min_confidence=0.3):
        """APRIORI ALGORITHM: Association rules mining"""
        conn = sqlite3.connect(self.db_path)
        query = """SELECT f.customer_id, f.time_id, p.name
                   FROM fact_transactions f
                   JOIN products p ON f.product_id = p.id
                   ORDER BY f.customer_id, f.time_id"""
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return None
        
        # Group by transaction
        df['transaction'] = df['customer_id'].astype(str) + '_' + df['time_id'].astype(str)
        transactions = df.groupby('transaction')['name'].apply(list).tolist()
        
        total_transactions = len(transactions)
        
        # Count item frequencies
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in set(transaction):
                item_counts[item] += 1
        
        # Filter by minimum support
        min_support_count = min_support * total_transactions
        frequent_items = {item: count for item, count in item_counts.items() 
                         if count >= min_support_count}
        
        # Generate association rules for pairs
        self.association_rules = []
        
        for transaction in transactions:
            unique_items = list(set(transaction))
            for item1, item2 in combinations(unique_items, 2):
                if item1 in frequent_items and item2 in frequent_items:
                    self.association_rules.append((item1, item2))
        
        # Calculate support, confidence, and lift
        rule_counts = defaultdict(int)
        for rule in self.association_rules:
            rule_counts[rule] += 1
        
        final_rules = []
        for (antecedent, consequent), count in rule_counts.items():
            support = count / total_transactions
            confidence = count / frequent_items[antecedent]
            
            # Calculate lift
            prob_consequent = frequent_items[consequent] / total_transactions
            lift = confidence / prob_consequent if prob_consequent > 0 else 0
            
            if confidence >= min_confidence:
                final_rules.append({
                    'antecedents': [antecedent],
                    'consequents': [consequent],
                    'support': support,
                    'confidence': confidence,
                    'lift': lift,
                    'users_bought': count
                })
        
        # Sort by confidence
        final_rules.sort(key=lambda x: x['confidence'], reverse=True)
        self.association_rules = final_rules
        
        return {
            'total_rules': len(final_rules),
            'sample_rules': final_rules[:10]
        }
    
    def get_recommendations(self, cart_items, top_n=5):
        """Get recommendations based on association rules"""
        if not self.association_rules:
            return []
        
        recommendations = []
        
        for rule in self.association_rules:
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            cart_set = set(cart_items)
            
            if antecedents.issubset(cart_set):
                for item in consequents:
                    if item not in cart_items:
                        recommendations.append({
                            'product': item,
                            'confidence': float(rule['confidence']) * 100,
                            'support': float(rule['support']),
                            'lift': float(rule['lift']),
                            'users_bought': int(rule.get('users_bought', 0)),
                            'reason': f"Often bought with {', '.join(antecedents)}",
                            'rule_strength': 'Strong' if rule['lift'] > 1.5 else 'Moderate'
                        })
        
        # Remove duplicates and sort
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['product'] not in seen:
                seen.add(rec['product'])
                unique_recs.append(rec)
        
        unique_recs.sort(key=lambda x: x['confidence'], reverse=True)
        return unique_recs[:top_n]
    
    def generate_detailed_decision_tree(self, cart_items):
        """
        Generate detailed decision tree with purchase statistics
        Shows: Cart → Frequently Bought Together → Top 4-5 recommendations with reasoning
        """
        if not self.association_rules:
            return {
                'error': 'No trained model available',
                'message': 'Please train the model first'
            }
        
        # Get all recommendations
        all_recs = self.get_recommendations(cart_items, top_n=20)
        
        # Categorize by confidence levels
        high_conf = [r for r in all_recs if r['confidence'] >= 60][:5]
        medium_conf = [r for r in all_recs if 30 <= r['confidence'] < 60][:4]
        low_conf = [r for r in all_recs if r['confidence'] < 30][:3]
        
        # Build tree structure
        tree = {
            'cart_items': cart_items,
            'total_recommendations': len(all_recs),
            'decision_flow': {
                'step1': {
                    'title': 'Cart Analysis',
                    'description': f'Analyzing {len(cart_items)} items in your cart',
                    'items': cart_items
                },
                'step2': {
                    'title': 'Finding Patterns',
                    'description': f'Searching through {len(self.association_rules)} association rules',
                    'rules_found': len([r for r in self.association_rules if any(item in str(r['antecedents']) for item in cart_items)])
                },
                'step3': {
                    'title': 'Confidence Filtering',
                    'description': 'Sorting recommendations by purchase probability',
                    'categories': {
                        'high': {
                            'range': '60-100%',
                            'count': len(high_conf),
                            'items': high_conf
                        },
                        'medium': {
                            'range': '30-60%',
                            'count': len(medium_conf),
                            'items': medium_conf
                        },
                        'low': {
                            'range': '0-30%',
                            'count': len(low_conf),
                            'items': low_conf
                        }
                    }
                },
                'step4': {
                    'title': 'Final Recommendations',
                    'description': 'Top picks based on highest confidence scores',
                    'top_picks': all_recs[:5]
                }
            },
            'statistics': {
                'total_customers_analyzed': sum(r.get('users_bought', 0) for r in all_recs[:5]),
                'avg_confidence': np.mean([r['confidence'] for r in all_recs]) if all_recs else 0,
                'avg_lift': np.mean([r['lift'] for r in all_recs]) if all_recs else 0
            }
        }
        
        return tree
    
    def generate_decision_tree(self, cart_items):
        """Legacy method - kept for backward compatibility"""
        return self.generate_detailed_decision_tree(cart_items)
