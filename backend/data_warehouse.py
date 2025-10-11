"""
Data Warehouse with OLAP Operations
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataWarehouse:
    def __init__(self, db_path='smartcart.db'):
        self.db_path = db_path
        self.init_warehouse_schema()
    
    def init_warehouse_schema(self):
        """Star Schema: Fact and Dimension tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # FACT TABLE
        c.execute("""CREATE TABLE IF NOT EXISTS fact_transactions
                    (transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     customer_id INTEGER,
                     product_id INTEGER,
                     time_id INTEGER,
                     quantity INTEGER,
                     total_price REAL,
                     discount REAL,
                     category TEXT)""")
        
        # DIMENSION: Customer
        c.execute("""CREATE TABLE IF NOT EXISTS dim_customer
                    (customer_id INTEGER PRIMARY KEY,
                     customer_name TEXT,
                     age_group TEXT,
                     location TEXT,
                     customer_segment TEXT,
                     registration_date TEXT)""")
        
        # DIMENSION: Time
        c.execute("""CREATE TABLE IF NOT EXISTS dim_time
                    (time_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     hour INTEGER,
                     day INTEGER,
                     month INTEGER,
                     year INTEGER,
                     day_of_week TEXT,
                     is_weekend INTEGER)""")
        
        conn.commit()
        conn.close()
    
    def load_from_csv(self, csv_path='groceries_dataset.csv'):
        """ETL: Load real dataset into warehouse"""
        df = pd.read_csv(csv_path)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Add sample customers
        customers = [
            (i, f'Customer_{i}', np.random.choice(['18-25', '25-35', '35-45', '45-60']),
             np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune']),
             np.random.choice(['Budget', 'Regular', 'Premium']),
             '2024-01-01')
            for i in range(1, 6)
        ]
        c.executemany("INSERT OR IGNORE INTO dim_customer VALUES (?, ?, ?, ?, ?, ?)", customers)
        
        # Add time dimension
        for i, date_str in enumerate(df['Date'].unique()[:100], start=1):
            date_obj = pd.to_datetime(date_str)
            c.execute("""INSERT OR IGNORE INTO dim_time VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                     (i, date_str, date_obj.hour, date_obj.day, date_obj.month,
                      date_obj.year, date_obj.strftime('%A'), 
                      1 if date_obj.weekday() >= 5 else 0))
        
        # Load transactions
        for _, row in df.iterrows():
            c.execute("SELECT id, category, price FROM products WHERE name = ?", (row['itemDescription'],))
            result = c.fetchone()
            if result:
                product_id, category, price = result
                customer_id = np.random.randint(1, 6)
                time_id = np.random.randint(1, 100)
                quantity = 1
                total_price = price * quantity
                discount = 0
                
                c.execute("""INSERT INTO fact_transactions 
                            (customer_id, product_id, time_id, quantity, total_price, discount, category)
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                         (customer_id, product_id, time_id, quantity, total_price, discount, category))
        
        conn.commit()
        conn.close()
        print("Warehouse loaded with real dataset!")
    
    def olap_rollup(self, dimension='category'):
        """OLAP Roll-up"""
        conn = sqlite3.connect(self.db_path)
        query = f"""SELECT {dimension}, SUM(total_price) as total_sales,
                          SUM(quantity) as total_quantity,
                          COUNT(DISTINCT customer_id) as unique_customers
                   FROM fact_transactions
                   GROUP BY {dimension}
                   ORDER BY total_sales DESC"""
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df.to_dict('records')
    
    def olap_drilldown(self, category):
        """OLAP Drill-down"""
        conn = sqlite3.connect(self.db_path)
        query = """SELECT p.name, SUM(f.total_price) as sales, SUM(f.quantity) as qty
                   FROM fact_transactions f
                   JOIN products p ON f.product_id = p.id
                   WHERE f.category = ?
                   GROUP BY p.name ORDER BY sales DESC LIMIT 10"""
        df = pd.read_sql_query(query, conn, params=(category,))
        conn.close()
        return df.to_dict('records')
