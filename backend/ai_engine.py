import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class AIRecommendationEngine:
    """
    AI-powered recommendation engine using:
    - TF-IDF for intent detection
    - Collaborative filtering
    - Content-based filtering
    """
    
    def __init__(self, products, intent_patterns):
        self.products = {p['id']: p for p in products}
        self.intent_patterns = intent_patterns
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Prepare intent pattern documents
        self.intent_docs = [p['keywords'] for p in intent_patterns]
        if self.intent_docs:
            self.intent_vectors = self.vectorizer.fit_transform(self.intent_docs)
    
    def detect_intent(self, cart_items):
        """
        Detect user intent based on cart items using AI
        Returns: dict with intent, confidence, reasoning, categories
        """
        if not cart_items:
            return None
        
        # Get cart item names and keywords
        cart_products = [self.products[item_id] for item_id in cart_items]
        cart_keywords = ' '.join([p.get('keywords', '') for p in cart_products])
        cart_names = [p['name'] for p in cart_products]
        
        # Transform cart keywords to vector
        if not cart_keywords.strip():
            return self._fallback_intent(cart_products)
        
        try:
            cart_vector = self.vectorizer.transform([cart_keywords])
            
            # Calculate similarity with each intent pattern
            similarities = cosine_similarity(cart_vector, self.intent_vectors)[0]
            
            # Get best matching intent
            best_match_idx = np.argmax(similarities)
            confidence = int(similarities[best_match_idx] * 100)
            
            if confidence < 40:
                return self._fallback_intent(cart_products)
            
            matched_intent = self.intent_patterns[best_match_idx]
            
            # Get typical items for this intent
            typical_item_ids = [int(x) for x in matched_intent['typical_items'].split(',')]
            
            # Extract categories from cart
            categories = list(set([p['category'] for p in cart_products]))
            
            return {
                'intent': matched_intent['intent_name'],
                'confidence': min(confidence + 10, 95),  # Boost confidence slightly
                'reasoning': self._generate_reasoning(matched_intent['intent_name'], cart_names),
                'categories': categories,
                'typical_items': typical_item_ids
            }
        except Exception as e:
            print(f"AI Error: {e}")
            return self._fallback_intent(cart_products)
    
    def _fallback_intent(self, cart_products):
        """Fallback intent when AI can't determine clear pattern"""
        categories = list(set([p['category'] for p in cart_products]))
        return {
            'intent': 'General Shopping',
            'confidence': 60,
            'reasoning': f"You're shopping for {', '.join(categories)} items. We've curated recommendations based on what others bought with these products.",
            'categories': categories,
            'typical_items': []
        }
    
    def _generate_reasoning(self, intent, cart_items):
        """Generate human-readable reasoning for the intent"""
        reasoning_templates = {
            'Making Sandwich': f"Based on your selection of {', '.join(cart_items[:3])}, our AI detects you're preparing sandwiches. We recommend fresh vegetables and condiments to complete your meal.",
            'Making Tea/Coffee': f"You've added {', '.join(cart_items[:2])} - perfect for a tea or coffee break! We suggest complementary items that pair wonderfully with hot beverages.",
            'Breakfast Preparation': f"Your cart contains {', '.join(cart_items[:3])}, indicating breakfast preparation. We've selected nutritious items to complete a balanced morning meal.",
            'Making Salad': f"With {', '.join(cart_items[:2])} in your cart, you're clearly making a fresh salad. Here are vegetables and dressings that will enhance your healthy meal.",
            'Quick Meal': f"You've chosen {', '.join(cart_items[:2])} for a quick meal. We recommend complementary items for fast, delicious cooking.",
            'Healthy Eating': f"Your selection of {', '.join(cart_items[:3])} shows a focus on healthy eating. Here are nutritious options that align with your wellness goals.",
        }
        return reasoning_templates.get(intent, f"Based on your cart items, we've personalized recommendations for you.")
    
    def generate_recommendations(self, cart_items, co_purchase_data, intent_data):
        """
        Generate AI-powered recommendations combining multiple signals
        """
        recommendations = {}
        
        # Process co-purchase data
        for item in co_purchase_data:
            rec_id = item['recommended_product_id']
            if rec_id not in recommendations:
                recommendations[rec_id] = {
                    'id': rec_id,
                    'name': item['rec_name'],
                    'category': item['rec_category'],
                    'price': item['rec_price'],
                    'img': item['rec_img'],
                    'total_purchases': item['rec_total_purchases'],
                    'match_scores': [],
                    'sources': [],
                    'intent_boost': 0
                }
            
            recommendations[rec_id]['match_scores'].append(item['match_percentage'])
            recommendations[rec_id]['sources'].append({
                'item_name': item['origin_name'],
                'match': item['match_percentage'],
                'users': item['users_bought']
            })
        
        # Apply intent-based boost
        if intent_data and 'typical_items' in intent_data:
            for item_id in intent_data['typical_items']:
                if item_id in recommendations:
                    recommendations[item_id]['intent_boost'] = 15
        
        # Calculate final AI match score
        result = []
        for rec in recommendations.values():
            base_score = np.mean(rec['match_scores']) if rec['match_scores'] else 50
            intent_boost = rec['intent_boost']
            
            # AI score combines collaborative filtering + intent detection
            ai_match = min(int(base_score + intent_boost), 99)
            
            rec['ai_match'] = ai_match
            rec['users_bought'] = max([s['users'] for s in rec['sources']])
            rec['reasoning'] = self._generate_item_reasoning(rec, intent_data)
            
            result.append(rec)
        
        # Sort by AI match score
        result.sort(key=lambda x: x['ai_match'], reverse=True)
        return result[:6]  # Return top 6
    
    def _generate_item_reasoning(self, item, intent_data):
        """Generate reasoning for why this specific item is recommended"""
        if not intent_data:
            return f"Frequently purchased with items in your cart."
        
        intent = intent_data.get('intent', 'your selection')
        
        reasoning_map = {
            'Making Sandwich': "Essential for creating the perfect sandwich.",
            'Making Tea/Coffee': "Pairs wonderfully with your hot beverage.",
            'Breakfast Preparation': "Completes a nutritious breakfast.",
            'Making Salad': "Adds freshness to your healthy salad.",
            'Quick Meal': "Perfect for quick and easy cooking.",
            'Healthy Eating': "Supports your healthy lifestyle goals.",
        }
        
        return reasoning_map.get(intent, f"Recommended based on {intent.lower()}.")