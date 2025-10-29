import sqlite3
import json

def init_database():
    """Initialize the SQLite database with tables and sample data"""
    conn = sqlite3.connect('smartcart.db')
    c = conn.cursor()
    
    # Products table
    c.execute('''CREATE TABLE IF NOT EXISTS products
                 (id INTEGER PRIMARY KEY,
                  name TEXT NOT NULL,
                  category TEXT NOT NULL,
                  price INTEGER NOT NULL,
                  img TEXT NOT NULL,
                  total_purchases INTEGER DEFAULT 0,
                  keywords TEXT)''')
    
    # Co-purchase patterns table
    c.execute('''CREATE TABLE IF NOT EXISTS co_purchases
                 (product_id INTEGER,
                  recommended_product_id INTEGER,
                  match_percentage INTEGER,
                  users_bought INTEGER,
                  FOREIGN KEY (product_id) REFERENCES products(id),
                  FOREIGN KEY (recommended_product_id) REFERENCES products(id),
                  PRIMARY KEY (product_id, recommended_product_id))''')
    
    # Intent patterns table (for AI learning)
    c.execute('''CREATE TABLE IF NOT EXISTS intent_patterns
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  intent_name TEXT NOT NULL,
                  keywords TEXT NOT NULL,
                  typical_items TEXT NOT NULL,
                  confidence_threshold INTEGER DEFAULT 70)''')
    
    # Check if data already exists
    c.execute('SELECT COUNT(*) FROM products')
    if c.fetchone()[0] == 0:
        # Insert products with keywords for AI matching
        products = [
            (1, 'Britannia Bread', 'bakery', 40, '🍞', 15000, 'bread,loaf,bakery,sandwich,toast'),
            (2, 'Amul Butter', 'dairy', 58, '🧈', 12000, 'butter,dairy,spread,toast'),
            (3, 'Amul Cheese', 'dairy', 125, '🧀', 8000, 'cheese,dairy,sandwich,pizza'),
            (4, 'Milk 1L', 'dairy', 62, '🥛', 20000, 'milk,dairy,beverage,tea,coffee,cereal'),
            (5, 'Tea Powder', 'beverages', 180, '🍵', 10000, 'tea,beverage,drink,chai'),
            (6, 'Parle-G Biscuits', 'snacks', 25, '🍪', 18000, 'biscuits,cookies,snacks,tea'),
            (7, 'Eggs (12 pcs)', 'dairy', 107, '🥚', 14000, 'eggs,protein,breakfast,omelette'),
            (8, 'Strawberry Jam', 'spreads', 145, '🫙', 6000, 'jam,spread,sweet,bread,toast'),
            (9, 'Peanut Butter', 'spreads', 299, '🥜', 5000, 'peanut,butter,spread,protein,sandwich'),
            (10, 'Coffee Powder', 'beverages', 220, '☕', 9000, 'coffee,beverage,drink,caffeine'),
            (11, 'Sugar 1kg', 'grocery', 55, '🧂', 11000, 'sugar,sweet,tea,coffee,baking'),
            (12, 'Maggi Noodles', 'instant', 48, '🍜', 16000, 'noodles,instant,maggi,quick,meal'),
            (13, 'Tomato Ketchup', 'condiments', 85, '🍅', 9500, 'ketchup,sauce,condiment,tomato'),
            (14, 'Mayonnaise', 'condiments', 165, '🥫', 7200, 'mayo,mayonnaise,sauce,sandwich,salad'),
            (15, 'Lettuce', 'vegetables', 35, '🥬', 5500, 'lettuce,vegetable,salad,sandwich,healthy'),
            (16, 'Tomatoes (500g)', 'vegetables', 45, '🍅', 12000, 'tomato,vegetable,salad,sandwich,fresh'),
            (17, 'Cucumber', 'vegetables', 30, '🥒', 8000, 'cucumber,vegetable,salad,fresh'),
            (18, 'Onions (1kg)', 'vegetables', 40, '🧅', 14000, 'onion,vegetable,cooking'),
            (19, 'Corn Flakes', 'breakfast', 180, '🌽', 8500, 'cereal,cornflakes,breakfast,milk'),
            (20, 'Oats', 'breakfast', 150, '🥣', 7000, 'oats,breakfast,healthy,porridge'),
            (21, 'Honey', 'organic', 250, '🍯', 6000, 'honey,sweet,natural,tea,healthy'),
            (22, 'Yogurt', 'dairy', 55, '🥛', 10000, 'yogurt,curd,dairy,breakfast,healthy'),
            (23, 'Orange Juice', 'beverages', 120, '🧃', 7500, 'juice,orange,beverage,breakfast,vitamin'),
            (24, 'Potato Chips', 'snacks', 40, '🥔', 13000, 'chips,snacks,potato,crispy'),
            (25, 'Chocolate Bar', 'confectionery', 60, '🍫', 11000, 'chocolate,sweet,candy,snack'),
        ]
        c.executemany('INSERT INTO products VALUES (?,?,?,?,?,?,?)', products)
        
        # Co-purchase patterns (expanded dataset)
        co_purchases = [
            # Bread-related
            (1, 2, 85, 12750), (1, 3, 72, 10800), (1, 8, 65, 9750), (1, 9, 58, 8700),
            (1, 14, 70, 9500), (1, 15, 55, 7200), (1, 16, 60, 8000),
            # Butter-related
            (2, 1, 85, 12750), (2, 3, 68, 8160), (2, 8, 55, 6600),
            # Cheese-related
            (3, 1, 72, 10800), (3, 2, 68, 8160), (3, 16, 50, 5500),
            # Milk-related
            (4, 5, 88, 17600), (4, 10, 78, 15600), (4, 6, 70, 14000), (4, 11, 62, 12400),
            (4, 19, 75, 13000), (4, 20, 65, 11000),
            # Tea-related
            (5, 4, 88, 17600), (5, 6, 82, 14760), (5, 11, 75, 13500), (5, 21, 60, 9000),
            # Biscuits-related
            (6, 5, 82, 14760), (6, 4, 70, 14000), (6, 10, 65, 10000),
            # Eggs-related
            (7, 1, 65, 9750), (7, 2, 60, 9000), (7, 16, 55, 7500),
            # Jam-related
            (8, 1, 65, 9750), (8, 2, 55, 6600),
            # Peanut butter-related
            (9, 1, 58, 8700), (9, 8, 45, 5000),
            # Coffee-related
            (10, 4, 78, 15600), (10, 11, 68, 12240), (10, 6, 60, 9500),
            # Sugar-related
            (11, 5, 75, 13500), (11, 4, 62, 12400), (11, 10, 68, 12240),
            # Noodles-related
            (12, 7, 45, 6300), (12, 13, 55, 7000),
            # Sandwich ingredients
            (14, 1, 70, 9500), (14, 3, 60, 7000), (14, 15, 65, 7500), (14, 16, 68, 8000),
            (15, 16, 80, 9000), (15, 14, 65, 7500), (15, 1, 55, 7200),
            (16, 15, 80, 9000), (16, 18, 75, 10000), (16, 14, 68, 8000),
            # Breakfast combos
            (19, 4, 90, 15000), (19, 23, 70, 10000), (19, 21, 55, 7000),
            (20, 4, 88, 14000), (20, 21, 75, 9500), (20, 22, 60, 7500),
            # Healthy combos
            (21, 20, 75, 9500), (21, 22, 65, 8000), (21, 5, 60, 9000),
            (22, 20, 60, 7500), (22, 19, 55, 6500), (22, 21, 65, 8000),
        ]
        c.executemany('INSERT INTO co_purchases VALUES (?,?,?,?)', co_purchases)
        
        # Intent patterns for AI
        intent_patterns = [
            (None, 'Making Sandwich', 'bread,butter,cheese,sandwich', '1,2,3,14,15,16', 80),
            (None, 'Making Tea/Coffee', 'tea,coffee,milk,sugar,biscuits', '4,5,6,10,11,21', 85),
            (None, 'Breakfast Preparation', 'cereal,milk,juice,eggs,bread', '1,4,7,19,20,22,23', 75),
            (None, 'Making Salad', 'lettuce,tomato,cucumber,onion', '15,16,17,18,14', 80),
            (None, 'Quick Meal', 'noodles,eggs,instant', '7,12,13', 70),
            (None, 'Healthy Eating', 'oats,honey,yogurt,juice', '20,21,22,23', 75),
        ]
        c.executemany('INSERT INTO intent_patterns VALUES (?,?,?,?,?)', intent_patterns)
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully!")

def get_all_products():
    """Fetch all products from database"""
    conn = sqlite3.connect('smartcart.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM products ORDER BY total_purchases DESC')
    products = [dict(row) for row in c.fetchall()]
    conn.close()
    return products

def get_categories():
    """Get unique categories"""
    conn = sqlite3.connect('smartcart.db')
    c = conn.cursor()
    c.execute('SELECT DISTINCT category FROM products ORDER BY category')
    categories = [row[0] for row in c.fetchall()]
    conn.close()
    return categories

def get_co_purchases(product_ids):
    """Get co-purchase data for given product IDs"""
    conn = sqlite3.connect('smartcart.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    placeholders = ','.join('?' * len(product_ids))
    query = f'''
        SELECT 
            cp.*,
            p.name as rec_name,
            p.category as rec_category,
            p.price as rec_price,
            p.img as rec_img,
            p.keywords as rec_keywords,
            p.total_purchases as rec_total_purchases,
            origin.name as origin_name
        FROM co_purchases cp
        JOIN products p ON cp.recommended_product_id = p.id
        JOIN products origin ON cp.product_id = origin.id
        WHERE cp.product_id IN ({placeholders})
        AND cp.recommended_product_id NOT IN ({placeholders})
        ORDER BY cp.match_percentage DESC
    '''
    
    c.execute(query, product_ids + product_ids)
    results = [dict(row) for row in c.fetchall()]
    conn.close()
    return results

def get_intent_patterns():
    """Get all intent patterns"""
    conn = sqlite3.connect('smartcart.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM intent_patterns')
    patterns = [dict(row) for row in c.fetchall()]
    conn.close()
    return patterns

def get_product_by_id(product_id):
    """Get a single product by ID"""
    conn = sqlite3.connect('smartcart.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM products WHERE id = ?', (product_id,))
    product = dict(c.fetchone())
    conn.close()
    return product