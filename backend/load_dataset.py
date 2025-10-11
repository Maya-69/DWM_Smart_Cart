"""
Load and preprocess Groceries Market Basket Dataset from Kaggle
"""
import pandas as pd
import sqlite3
import numpy as np

def load_groceries_dataset(csv_path='groceries_dataset.csv'):
    """
    Load the Groceries dataset and return products and transactions
    Dataset: https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Get unique products
    unique_items = df['itemDescription'].unique()
    print(f"Loaded {len(unique_items)} unique products from dataset")
    
    # Group transactions
    transactions = df.groupby('Member_number')['itemDescription'].apply(list).values
    print(f"Loaded {len(transactions)} transactions")
    
    return unique_items, transactions

def create_products_from_dataset(csv_path='groceries_dataset.csv', db_path='smartcart.db'):
    """
    Create products table from actual dataset items
    """
    unique_items, _ = load_groceries_dataset(csv_path)
    
    # Map to categories
    category_mapping = {
        'milk': 'dairy', 'yogurt': 'dairy', 'butter': 'dairy', 'cream': 'dairy',
        'cheese': 'dairy', 'curd': 'dairy', 'eggs': 'dairy',
        'bread': 'bakery', 'rolls': 'bakery', 'pastry': 'bakery',
        'vegetables': 'vegetables', 'fruit': 'fruit', 'citrus': 'fruit',
        'beef': 'meat', 'chicken': 'meat', 'pork': 'meat', 'sausage': 'meat',
        'ham': 'meat', 'frankfurter': 'meat',
        'soda': 'beverages', 'water': 'beverages', 'juice': 'beverages',
        'beer': 'beverages', 'wine': 'beverages', 'liquor': 'beverages',
        'coffee': 'beverages', 'whisky': 'beverages',
        'chocolate': 'snacks', 'snack': 'snacks', 'ice cream': 'snacks',
        'napkins': 'household', 'dishes': 'household', 'hygiene': 'household',
        'cat food': 'pet care', 'dog food': 'pet care', 'pet care': 'pet care',
        'frozen': 'frozen', 'canned': 'canned',
        'sugar': 'staples', 'margarine': 'staples', 'condensed milk': 'staples'
    }
    
    # Emojis for categories
    emoji_map = {
        'dairy': '🥛', 'bakery': '🍞', 'vegetables': '🥬', 'fruit': '🍎',
        'meat': '🥩', 'beverages': '🥤', 'snacks': '🍫', 'household': '🧻',
        'pet care': '🐾', 'frozen': '❄️', 'canned': '🥫', 'staples': '🌾'
    }
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Create products table
    c.execute('''CREATE TABLE IF NOT EXISTS products
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL UNIQUE,
                 category TEXT NOT NULL,
                 price INTEGER NOT NULL,
                 img TEXT NOT NULL,
                 total_purchases INTEGER DEFAULT 0,
                 keywords TEXT)''')
    
    # Insert products from dataset
    for idx, item in enumerate(unique_items, start=1):
        # Determine category
        category = 'other'
        for keyword, cat in category_mapping.items():
            if keyword in item.lower():
                category = cat
                break
        
        # Random price between 20-200
        price = np.random.randint(20, 200)
        
        # Get emoji
        emoji = emoji_map.get(category, '🛒')
        
        # Keywords for search
        keywords = item.replace('/', ' ').replace('-', ' ')
        
        c.execute('''INSERT OR IGNORE INTO products 
                    (name, category, price, img, total_purchases, keywords)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                 (item, category, price, emoji, 0, keywords))
    
    conn.commit()
    conn.close()
    print(f"Created {len(unique_items)} products in database")

if __name__ == '__main__':
    create_products_from_dataset()
