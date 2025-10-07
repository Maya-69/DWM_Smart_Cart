from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import json

app = Flask(__name__)
CORS(app)

# Database initialization
def init_db():
    conn = sqlite3.connect('smartcart.db')
    c = conn.cursor()
    
    # Products table
    c.execute('''CREATE TABLE IF NOT EXISTS products
                 (id INTEGER PRIMARY KEY,
                  name TEXT NOT NULL,
                  category TEXT NOT NULL,
                  price INTEGER NOT NULL,
                  img TEXT NOT NULL,
                  total_purchases INTEGER DEFAULT 0)''')
    
    # Co-purchase table (tracks items bought together)
    c.execute('''CREATE TABLE IF NOT EXISTS co_purchases
                 (product_id INTEGER,
                  recommended_product_id INTEGER,
                  match_percentage INTEGER,
                  users_bought INTEGER,
                  FOREIGN KEY (product_id) REFERENCES products(id),
                  FOREIGN KEY (recommended_product_id) REFERENCES products(id))''')
    
    # Insert sample data if empty
    c.execute('SELECT COUNT(*) FROM products')
    if c.fetchone()[0] == 0:
        products = [
            (1, 'Britannia Bread', 'bakery', 40, '🍞', 15000),
            (2, 'Amul Butter', 'dairy', 58, '🧈', 12000),
            (3, 'Amul Cheese', 'dairy', 125, '🧀', 8000),
            (4, 'Milk 1L', 'dairy', 62, '🥛', 20000),
            (5, 'Tea Powder', 'beverages', 180, '🍵', 10000),
            (6, 'Parle-G Biscuits', 'snacks', 25, '🍪', 18000),
            (7, 'Eggs (12 pcs)', 'dairy', 107, '🥚', 14000),
            (8, 'Jam Bottle', 'spreads', 145, '🫙', 6000),
            (9, 'Peanut Butter', 'spreads', 299, '🥜', 5000),
            (10, 'Coffee Powder', 'beverages', 220, '☕', 9000),
            (11, 'Sugar 1kg', 'grocery', 55, '🧂', 11000),
            (12, 'Maggi Noodles', 'instant', 48, '🍜', 16000),
        ]
        c.executemany('INSERT INTO products VALUES (?,?,?,?,?,?)', products)
        
        co_purchases = [
            # Bread recommendations
            (1, 2, 85, 12750), (1, 3, 72, 10800), (1, 8, 65, 9750), (1, 9, 58, 8700),
            # Butter recommendations
            (2, 1, 85, 12750), (2, 3, 68, 8160), (2, 8, 55, 6600),
            # Cheese recommendations
            (3, 1, 72, 10800), (3, 2, 68, 8160),
            # Milk recommendations
            (4, 5, 88, 17600), (4, 10, 78, 15600), (4, 6, 70, 14000), (4, 11, 62, 12400),
            # Tea recommendations
            (5, 4, 88, 17600), (5, 6, 82, 14760), (5, 11, 75, 13500),
            # Biscuits recommendations
            (6, 5, 82, 14760), (6, 4, 70, 14000),
            # Eggs recommendations
            (7, 1, 65, 9750), (7, 2, 60, 9000),
            # Jam recommendations
            (8, 1, 65, 9750), (8, 2, 55, 6600),
            # Peanut Butter recommendations
            (9, 1, 58, 8700),
            # Coffee recommendations
            (10, 4, 78, 15600), (10, 11, 68, 12240),
            # Sugar recommendations
            (11, 5, 75, 13500), (11, 4, 62, 12400), (11, 10, 68, 12240),
            # Maggi recommendations
            (12, 7, 45, 6300),
        ]
        c.executemany('INSERT INTO co_purchases VALUES (?,?,?,?)', co_purchases)
    
    conn.commit()
    conn.close()

# API Routes
@app.route('/api/products', methods=['GET'])
def get_products():
    conn = sqlite3.connect('smartcart.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM products')
    products = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(products)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    cart_items = request.json.get('cart_items', [])
    
    if not cart_items:
        return jsonify([])
    
    conn = sqlite3.connect('smartcart.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get co-purchase data for all cart items
    placeholders = ','.join('?' * len(cart_items))
    query = f'''
        SELECT 
            cp.recommended_product_id,
            cp.match_percentage,
            cp.users_bought,
            p.name,
            p.category,
            p.price,
            p.img,
            p.total_purchases
        FROM co_purchases cp
        JOIN products p ON cp.recommended_product_id = p.id
        WHERE cp.product_id IN ({placeholders})
        AND cp.recommended_product_id NOT IN ({placeholders})
    '''
    
    c.execute(query, cart_items + cart_items)
    recommendations = {}
    
    for row in c.fetchall():
        rec_id = row['recommended_product_id']
        if rec_id not in recommendations:
            recommendations[rec_id] = {
                'id': rec_id,
                'name': row['name'],
                'category': row['category'],
                'price': row['price'],
                'img': row['img'],
                'total_purchases': row['total_purchases'],
                'total_match': 0,
                'max_users': 0,
                'sources': []
            }
        
        recommendations[rec_id]['total_match'] += row['match_percentage']
        recommendations[rec_id]['max_users'] = max(recommendations[rec_id]['max_users'], row['users_bought'])
        recommendations[rec_id]['sources'].append({
            'match': row['match_percentage'],
            'users': row['users_bought']
        })
    
    # Calculate average match and sort
    result = []
    for rec in recommendations.values():
        rec['aiMatch'] = round(rec['total_match'] / len(cart_items))
        rec['usersBought'] = rec['max_users']
        result.append(rec)
    
    result.sort(key=lambda x: x['aiMatch'], reverse=True)
    conn.close()
    
    return jsonify(result[:4])

@app.route('/api/co-purchases/<int:product_id>', methods=['GET'])
def get_co_purchases(product_id):
    conn = sqlite3.connect('smartcart.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT 
            cp.*,
            p.name,
            p.img
        FROM co_purchases cp
        JOIN products p ON cp.recommended_product_id = p.id
        WHERE cp.product_id = ?
        ORDER BY cp.match_percentage DESC
    ''', (product_id,))
    
    co_purchases = [dict(row) for row in c.fetchall()]
    conn.close()
    return jsonify(co_purchases)

if __name__ == '__main__':
    init_db()
    print("✅ Database initialized!")
    print("🚀 Server running on http://localhost:5000")
    app.run(debug=True, port=5000)  