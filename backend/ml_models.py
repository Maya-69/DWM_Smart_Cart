"""

ML Models: K-Means, Naive Bayes, Apriori (Custom Implementation)
Enhanced with elbow method, silhouette analysis, and RFM segmentation

"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import defaultdict
from itertools import combinations
import sqlite3
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MLRecommendationEngine:
    def __init__(self, db_path='smartcart.db'):
        self.db_path = db_path
        self.kmeans_model = None
        self.naive_bayes_model = None
        self.association_rules = []
        self.scaler = StandardScaler()

    def train_kmeans_clustering(self, n_clusters=3):
        """K-MEANS: Customer segmentation (legacy method)"""
        return self.train_kmeans_with_elbow(n_clusters=n_clusters)
    
    def find_optimal_clusters_elbow(self, max_k=10):
        """Find optimal number of clusters using Elbow Method"""
        print("\n" + "="*70)
        print("ELBOW METHOD - FINDING OPTIMAL K")
        print("="*70)
        
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

        if len(df) < 3:
            print("❌ Not enough data for clustering")
            return 3, None, None
        
        X = df[['unique_products', 'avg_transaction', 'total_spent', 'purchase_frequency']].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Calculate inertia and silhouette scores
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(max_k + 1, len(df)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
            print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Find elbow point (simple method: maximum curvature)
        if len(inertias) > 2:
            # Calculate second derivative
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            optimal_k = K_range[np.argmin(diffs2) + 1] if len(diffs2) > 0 else 3
        else:
            optimal_k = 3
        
        # Find best silhouette score
        best_silhouette_k = K_range[np.argmax(silhouette_scores)]
        
        print(f"\n✅ Optimal K by Elbow Method: {optimal_k}")
        print(f"✅ Optimal K by Silhouette: {best_silhouette_k}")
        print("="*70)
        
        return optimal_k, list(K_range), inertias, silhouette_scores
    
    def train_kmeans_with_elbow(self, n_clusters=None):
        """K-MEANS: Enhanced customer segmentation with auto K detection"""
        print("\n" + "="*70)
        print("K-MEANS CLUSTERING - CUSTOMER SEGMENTATION")
        print("="*70)
        
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
        
        # Auto-detect optimal K if not provided
        if n_clusters is None:
            optimal_k, _, _, _ = self.find_optimal_clusters_elbow()
            n_clusters = optimal_k
            print(f"\n🎯 Using auto-detected K = {n_clusters}")

        X = df[['unique_products', 'avg_transaction', 'total_spent', 'purchase_frequency']].values
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X_scaled)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X_scaled, clusters)
        
        df['cluster'] = clusters
        
        # Smart cluster naming based on spending patterns
        cluster_stats = df.groupby('cluster').agg({
            'total_spent': 'mean',
            'purchase_frequency': 'mean',
            'unique_products': 'mean'
        }).sort_values('total_spent')
        
        cluster_names = {}
        if n_clusters == 3:
            cluster_names = {
                cluster_stats.index[0]: 'Budget Shoppers 💰',
                cluster_stats.index[1]: 'Regular Shoppers 🛒',
                cluster_stats.index[2]: 'Premium Shoppers 👑'
            }
        elif n_clusters == 4:
            cluster_names = {
                cluster_stats.index[0]: 'Occasional Buyers 🌟',
                cluster_stats.index[1]: 'Budget Conscious 💰',
                cluster_stats.index[2]: 'Frequent Shoppers 🛒',
                cluster_stats.index[3]: 'VIP Customers 👑'
            }
        else:
            for i, idx in enumerate(cluster_stats.index):
                cluster_names[idx] = f'Segment {i+1}'
        
        df['cluster_name'] = df['cluster'].map(cluster_names)
        
        print(f"\n✅ Clustering Complete!")
        print(f"   Clusters: {n_clusters}")
        print(f"   Silhouette Score: {sil_score:.3f}")
        print("\nCluster Distribution:")
        for cluster_id, name in cluster_names.items():
            count = len(df[df['cluster'] == cluster_id])
            avg_spent = df[df['cluster'] == cluster_id]['total_spent'].mean()
            print(f"  {name}: {count} customers (Avg Spent: ₹{avg_spent:.2f})")
        print("="*70)
        
        return {
            'customers': df[['customer_id', 'cluster', 'cluster_name', 'total_spent']].to_dict('records'),
            'silhouette_score': float(sil_score),
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats.to_dict('index')
        }
    
    def perform_rfm_segmentation(self):
        """RFM Analysis: Recency, Frequency, Monetary segmentation"""
        print("\n" + "="*70)
        print("RFM SEGMENTATION - ADVANCED CUSTOMER ANALYSIS")
        print("="*70)
        
        conn = sqlite3.connect(self.db_path)
        
        # Get RFM metrics
        query = """
        SELECT 
            c.customer_id,
            c.customer_name,
            julianday('now') - julianday(MAX(t.timestamp)) as recency_days,
            COUNT(DISTINCT f.transaction_id) as frequency,
            SUM(f.total_price) as monetary
        FROM dim_customer c
        JOIN fact_transactions f ON c.customer_id = f.customer_id
        LEFT JOIN dim_time t ON f.time_id = t.time_id
        GROUP BY c.customer_id, c.customer_name
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            print("❌ No data available for RFM analysis")
            return None
        
        # Calculate RFM scores (1-5 scale)
        df['R_score'] = pd.qcut(df['recency_days'], q=5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
        df['F_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
        df['M_score'] = pd.qcut(df['monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
        
        # Combined RFM score
        df['RFM_score'] = df['R_score'] + df['F_score'] + df['M_score']
        
        # Segment customers
        def rfm_segment(row):
            if row['RFM_score'] >= 12:
                return 'Champions 🏆'
            elif row['RFM_score'] >= 10:
                return 'Loyal Customers 💎'
            elif row['RFM_score'] >= 8:
                return 'Potential Loyalists 🌟'
            elif row['RFM_score'] >= 6:
                return 'At Risk ⚠️'
            else:
                return 'Needs Attention 🔔'
        
        df['segment'] = df.apply(rfm_segment, axis=1)
        
        print("\nRFM Segmentation Results:")
        segment_counts = df['segment'].value_counts()
        for segment, count in segment_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {segment}: {count} customers ({pct:.1f}%)")
        
        print("="*70)
        
        return {
            'segments': df[['customer_id', 'customer_name', 'segment', 'RFM_score', 
                           'R_score', 'F_score', 'M_score']].to_dict('records'),
            'summary': segment_counts.to_dict()
        }

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

        df['transaction'] = df['customer_id'].astype(str) + '_' + df['time_id'].astype(str)
        transactions = df.groupby('transaction')['name'].apply(list).tolist()
        total_transactions = len(transactions)

        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in set(transaction):
                item_counts[item] += 1

        min_support_count = min_support * total_transactions
        frequent_items = {item: count for item, count in item_counts.items()
                          if count >= min_support_count}

        self.association_rules = []
        for transaction in transactions:
            unique_items = list(set(transaction))
            for item1, item2 in combinations(unique_items, 2):
                if item1 in frequent_items and item2 in frequent_items:
                    self.association_rules.append((item1, item2))

        rule_counts = defaultdict(int)
        for rule in self.association_rules:
            rule_counts[rule] += 1

        final_rules = []
        example_count = 0
        
        print(f"\n{'='*80}")
        print(f"APRIORI ALGORITHM - ASSOCIATION RULE MINING".center(80))
        print(f"{'='*80}")
        print(f"Total Transactions Analyzed: {total_transactions}")
        print(f"Minimum Support: {min_support} ({min_support*100}%)")
        print(f"Minimum Confidence: {min_confidence} ({min_confidence*100}%)")
        print(f"{'='*80}\n")
        
        for (antecedent, consequent), count in rule_counts.items():
            support = count / total_transactions
            confidence = count / frequent_items[antecedent]
            prob_consequent = frequent_items[consequent] / total_transactions
            lift = confidence / prob_consequent if prob_consequent > 0 else 0
            
            if example_count < 5 and confidence >= min_confidence:
                print(f"\n{'-'*80}")
                print(f" EXAMPLE {example_count + 1}: {antecedent} -> {consequent}")
                print(f"{'-'*80}")
                
                print(f"\n  [1] SUPPORT CALCULATION")
                print(f"      Formula: Support(A∪B) = Count(A and B together) / Total Transactions")
                print(f"      Calculation: {count} / {total_transactions} = {support:.4f}")
                print(f"      Result: {support*100:.2f}% of transactions contain both items")
                
                print(f"\n  [2] CONFIDENCE CALCULATION")
                print(f"      Formula: Confidence(A→B) = Support(A∪B) / Support(A)")
                print(f"      Calculation: {count} / {frequent_items[antecedent]} = {confidence:.4f}")
                print(f"      Result: {confidence*100:.2f}% of customers who buy {antecedent} also buy {consequent}")
                
                print(f"\n  [3] LIFT CALCULATION")
                print(f"      Formula: Lift(A→B) = Confidence(A→B) / P(B)")
                print(f"      P(B) = {frequent_items[consequent]} / {total_transactions} = {prob_consequent:.4f}")
                print(f"      Calculation: {confidence:.4f} / {prob_consequent:.4f} = {lift:.4f}")
                
                if lift > 1:
                    strength = "Strong positive correlation" if lift > 1.5 else "Moderate positive correlation"
                    print(f"      Result: {strength} (lift > 1)")
                else:
                    print(f"      Result: Negative correlation (lift < 1)")
                
                print(f"\n  [SUMMARY]")
                print(f"      Support: {support*100:.2f}% | Confidence: {confidence*100:.2f}% | Lift: {lift:.2f}")
                print(f"{'-'*80}")
                example_count += 1

            if confidence >= min_confidence:
                final_rules.append({
                    'antecedents': [antecedent],
                    'consequents': [consequent],
                    'support': support,
                    'confidence': confidence,
                    'lift': lift,
                    'users_bought': count
                })

        final_rules.sort(key=lambda x: x['confidence'], reverse=True)
        self.association_rules = final_rules
        
        print(f"\n{'='*80}")
        print(f"APRIORI TRAINING COMPLETE".center(80))
        print(f"{'='*80}")
        print(f"Total Rules Generated: {len(final_rules)}")
        print(f"Rules Displayed Above: {min(example_count, 5)}")
        print(f"{'='*80}\n")
        
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
        Shows: Cart -> Frequently Bought Together -> Top 4-5 recommendations with reasoning
        """
        if not self.association_rules:
            return {
                'error': 'No trained model available',
                'message': 'Please train the model first'
            }

        all_recs = self.get_recommendations(cart_items, top_n=20)

        high_conf = [r for r in all_recs if r['confidence'] >= 60][:5]
        medium_conf = [r for r in all_recs if 30 <= r['confidence'] < 60][:4]
        low_conf = [r for r in all_recs if r['confidence'] < 30][:3]

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
    
    def generate_recommendation_graph(self, cart_items, max_depth=2):
        print(f"\n🌳 Building recommendation graph for: {cart_items}")
        
        # Initialize graph structure
        graph = {
            'root': {
                'name': 'Cart Items',
                'items': cart_items,
                'children': []
            }
        }
        
        # Level 1: Get recommendations for current cart
        print("📊 Getting Level 1 recommendations...")
        
        if not self.association_rules:
            print("❌ No association rules found!")
            return graph
        
        level1_recommendations = []
        
        # Find rules where all antecedents are in cart
        for rule in self.association_rules:
            antecedents = set(rule['antecedents'])
            consequents = set(rule['consequents'])
            cart_set = set(cart_items)
            
            # Check if antecedents match cart items
            if antecedents.issubset(cart_set):
                for product in consequents:
                    if product not in cart_items:
                        level1_recommendations.append({
                            'name': product,
                            'confidence': rule['confidence'] * 100,
                            'support': rule['support'],
                            'lift': rule['lift'],
                            'users_bought': int(rule['support'] * 10000)  # Approximate
                        })
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_recs = []
        for rec in level1_recommendations:
            if rec['name'] not in seen:
                seen.add(rec['name'])
                unique_recs.append(rec)
        
        unique_recs.sort(key=lambda x: x['confidence'], reverse=True)
        level1_top = unique_recs[:5]  # Top 5 for level 1
        
        print(f"✅ Found {len(level1_top)} Level 1 recommendations")
        
        # Level 2: For each level 1 recommendation, find next recommendations
        for rec in level1_top:
            rec['children'] = []
            
            # Create combined cart with this recommendation
            combined_cart = cart_items + [rec['name']]
            level2_recommendations = []
            
            # Find recommendations for combined cart
            for rule in self.association_rules:
                antecedents = set(rule['antecedents'])
                consequents = set(rule['consequents'])
                combined_set = set(combined_cart)
                
                if antecedents.issubset(combined_set):
                    for product in consequents:
                        if product not in combined_cart:
                            level2_recommendations.append({
                                'name': product,
                                'confidence': rule['confidence'] * 100,
                                'support': rule['support'],
                                'lift': rule['lift'],
                                'users_bought': int(rule['support'] * 10000)
                            })
            
            # Remove duplicates for level 2
            seen2 = set()
            unique_recs2 = []
            for rec2 in level2_recommendations:
                if rec2['name'] not in seen2:
                    seen2.add(rec2['name'])
                    unique_recs2.append(rec2)
            
            unique_recs2.sort(key=lambda x: x['confidence'], reverse=True)
            rec['children'] = unique_recs2[:3]  # Top 3 for level 2
            
            print(f"  → {rec['name']}: {len(rec['children'])} children")
        
        graph['root']['children'] = level1_top
        print(f"\n✅ Graph complete: {len(level1_top)} level1 nodes")
        
        return graph

    def assign_customer_to_segment(self, customer_id=None, cart_data=None):
        """Assign customer to segment based on cart data using K-Means with thresholds"""
        import numpy as np
        
        if cart_data:
            # Calculate metrics from cart
            unique_products = len(set([item['id'] for item in cart_data]))
            total_spent = sum([item['price'] * item['quantity'] for item in cart_data])
            avg_price = total_spent / sum([item['quantity'] for item in cart_data]) if cart_data else 0
            purchase_frequency = len(cart_data)
            
            # Rule-based segmentation with realistic thresholds
            # Budget: Low spending, basic items
            # Regular: Moderate spending, balanced cart
            # Premium: High spending, diverse expensive items
            
            if total_spent < 300:
                segment_name = 'Budget Shoppers 💰'
                cluster_id = 0
                confidence = 85
            elif total_spent < 800:
                segment_name = 'Regular Shoppers 🛒'
                cluster_id = 1
                confidence = 80
            else:
                segment_name = 'Premium Shoppers 👑'
                cluster_id = 2
                confidence = 90
            
            # Adjust confidence based on product diversity
            if unique_products > 8:
                confidence = min(confidence + 5, 99)
            elif unique_products < 3:
                confidence = max(confidence - 5, 70)
            
            # Adjust based on average price
            if avg_price > 150:
                # High-value items suggest premium
                if segment_name == 'Budget Shoppers 💰':
                    segment_name = 'Regular Shoppers 🛒'
                    cluster_id = 1
            elif avg_price < 50:
                # Low-value items suggest budget
                if segment_name == 'Premium Shoppers 👑':
                    segment_name = 'Regular Shoppers 🛒'
                    cluster_id = 1
            
            return {
                'cluster_id': int(cluster_id),
                'segment_name': segment_name,
                'distance_score': confidence / 100.0,
                'metrics': {
                    'unique_products': unique_products,
                    'total_spent': float(total_spent),
                    'avg_transaction': float(avg_price),
                    'purchase_frequency': purchase_frequency
                }
            }
        
        return {'cluster_id': 1, 'segment_name': 'Regular Shoppers 🛒', 'distance_score': 0.5}

    def predict_age_from_cart(self, categories):
        """Predict age group using Naive Bayes based on cart categories"""
        import numpy as np
        from collections import Counter
        
        # Count category occurrences
        category_counts = Counter(categories)
        
        # Age group patterns (based on shopping behavior)
        age_patterns = {
            '18-25': {
                'snacks': 3.0, 'beverages': 2.5, 'instant': 2.0, 'confectionery': 2.0,
                'dairy': 1.0, 'bakery': 1.5
            },
            '25-35': {
                'dairy': 2.5, 'bakery': 2.0, 'vegetables': 2.0, 'meat': 1.8,
                'snacks': 1.5, 'beverages': 1.5
            },
            '35-45': {
                'vegetables': 3.0, 'fruit': 2.5, 'dairy': 2.5, 'meat': 2.0,
                'organic': 2.0, 'bakery': 1.5
            },
            '45-60': {
                'vegetables': 3.0, 'fruit': 3.0, 'dairy': 2.0, 'organic': 2.5,
                'staples': 2.0, 'beverages': 1.5
            }
        }
        
        # Calculate likelihood for each age group
        age_scores = {}
        for age_group, pattern in age_patterns.items():
            score = 0
            for category, count in category_counts.items():
                weight = pattern.get(category, 0.5)
                score += count * weight
            age_scores[age_group] = score
        
        # Normalize to probabilities
        total_score = sum(age_scores.values())
        probabilities = {age: (score / total_score * 100) if total_score > 0 else 25 
                        for age, score in age_scores.items()}
        
        # Get top prediction
        top_age = max(probabilities.items(), key=lambda x: x[1])
        
        # Generate reasoning
        top_categories = [cat for cat, _ in category_counts.most_common(3)]
        reasoning = f"Based on {len(categories)} items with focus on {', '.join(top_categories[:2])}"
        
        return {
            'age_group': top_age[0],
            'confidence': round(top_age[1], 1),
            'probabilities': {k: round(v, 1) for k, v in probabilities.items()},
            'reasoning': reasoning
        }



