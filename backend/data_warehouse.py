"""
Data Warehouse with OLAP Operations
Enhanced with ETL metadata tracking, data validation, and incremental loads
"""
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class DataWarehouse:
    def __init__(self, db_path='smartcart.db'):
        self.db_path = db_path
        self.init_warehouse_schema()
    
    def init_warehouse_schema(self):
        """Star Schema: Fact and Dimension tables + ETL Metadata"""
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
                     category TEXT,
                     load_date TEXT DEFAULT CURRENT_TIMESTAMP)""")
        
        # DIMENSION: Customer
        c.execute("""CREATE TABLE IF NOT EXISTS dim_customer
                    (customer_id INTEGER PRIMARY KEY,
                     customer_name TEXT,
                     age_group TEXT,
                     location TEXT,
                     customer_segment TEXT,
                     registration_date TEXT,
                     last_updated TEXT DEFAULT CURRENT_TIMESTAMP)""")
        
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
        
        # ETL METADATA TABLE (New!)
        c.execute("""CREATE TABLE IF NOT EXISTS etl_metadata
                    (etl_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     load_date TEXT DEFAULT CURRENT_TIMESTAMP,
                     table_name TEXT,
                     records_loaded INTEGER,
                     records_rejected INTEGER,
                     status TEXT,
                     error_message TEXT,
                     duration_seconds REAL)""")
        
        # DATA QUALITY RULES TABLE (New!)
        c.execute("""CREATE TABLE IF NOT EXISTS data_quality_log
                    (log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     check_date TEXT DEFAULT CURRENT_TIMESTAMP,
                     rule_name TEXT,
                     table_name TEXT,
                     records_affected INTEGER,
                     severity TEXT,
                     details TEXT)""")
        
        # AGGREGATION TABLE for faster queries (New!)
        c.execute("""CREATE TABLE IF NOT EXISTS agg_daily_sales
                    (date TEXT,
                     category TEXT,
                     total_sales REAL,
                     total_quantity INTEGER,
                     unique_customers INTEGER,
                     avg_transaction REAL,
                     PRIMARY KEY (date, category))""")
        
        conn.commit()
        conn.close()
        print("✅ Enhanced warehouse schema initialized")
    
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
    
    def validate_data_quality(self):
        """Run data quality checks and log results"""
        print("\n" + "="*70)
        print("DATA QUALITY VALIDATION")
        print("="*70)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        issues_found = []
        
        # Check 1: Negative prices
        c.execute("""SELECT COUNT(*) FROM fact_transactions WHERE total_price < 0""")
        negative_prices = c.fetchone()[0]
        if negative_prices > 0:
            issues_found.append(('Negative Prices', 'fact_transactions', negative_prices, 'ERROR'))
            c.execute("""INSERT INTO data_quality_log (rule_name, table_name, records_affected, severity, details)
                        VALUES (?, ?, ?, ?, ?)""",
                     ('Negative Prices', 'fact_transactions', negative_prices, 'ERROR', 
                      'Found transactions with negative prices'))
        
        # Check 2: Null customer IDs
        c.execute("""SELECT COUNT(*) FROM fact_transactions WHERE customer_id IS NULL""")
        null_customers = c.fetchone()[0]
        if null_customers > 0:
            issues_found.append(('Null Customer IDs', 'fact_transactions', null_customers, 'WARNING'))
            c.execute("""INSERT INTO data_quality_log (rule_name, table_name, records_affected, severity, details)
                        VALUES (?, ?, ?, ?, ?)""",
                     ('Null Customer IDs', 'fact_transactions', null_customers, 'WARNING',
                      'Transactions without customer assignment'))
        
        # Check 3: Duplicate transactions
        c.execute("""SELECT customer_id, product_id, time_id, COUNT(*) as cnt
                    FROM fact_transactions
                    GROUP BY customer_id, product_id, time_id
                    HAVING cnt > 1""")
        duplicates = len(c.fetchall())
        if duplicates > 0:
            issues_found.append(('Duplicate Transactions', 'fact_transactions', duplicates, 'WARNING'))
            c.execute("""INSERT INTO data_quality_log (rule_name, table_name, records_affected, severity, details)
                        VALUES (?, ?, ?, ?, ?)""",
                     ('Duplicate Transactions', 'fact_transactions', duplicates, 'WARNING',
                      'Potential duplicate transaction records'))
        
        # Check 4: Orphaned records
        c.execute("""SELECT COUNT(*) FROM fact_transactions f
                    WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.id = f.product_id)""")
        orphaned = c.fetchone()[0]
        if orphaned > 0:
            issues_found.append(('Orphaned Records', 'fact_transactions', orphaned, 'ERROR'))
            c.execute("""INSERT INTO data_quality_log (rule_name, table_name, records_affected, severity, details)
                        VALUES (?, ?, ?, ?, ?)""",
                     ('Orphaned Records', 'fact_transactions', orphaned, 'ERROR',
                      'Transactions referencing non-existent products'))
        
        conn.commit()
        
        if issues_found:
            print("\n⚠️  Data Quality Issues Found:")
            for issue, table, count, severity in issues_found:
                print(f"  [{severity}] {issue}: {count} records in {table}")
        else:
            print("\n✅ All data quality checks passed!")
        
        print("="*70)
        conn.close()
        
        return {
            'total_issues': len(issues_found),
            'issues': [{'rule': i[0], 'table': i[1], 'count': i[2], 'severity': i[3]} 
                      for i in issues_found]
        }
    
    def incremental_load(self, csv_path='groceries_dataset.csv', batch_size=1000):
        """Incremental ETL: Load only new records"""
        start_time = datetime.now()
        print("\n" + "="*70)
        print("INCREMENTAL ETL LOAD")
        print("="*70)
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Get last load timestamp
        c.execute("""SELECT MAX(load_date) FROM etl_metadata 
                    WHERE table_name = 'fact_transactions' AND status = 'SUCCESS'""")
        last_load = c.fetchone()[0]
        
        try:
            df = pd.read_csv(csv_path)
            print(f"📊 Total records in CSV: {len(df)}")
            
            # Simulate incremental by taking records not already loaded
            c.execute("SELECT COUNT(*) FROM fact_transactions")
            existing_count = c.fetchone()[0]
            
            # Load in batches
            records_loaded = 0
            records_rejected = 0
            
            for i in range(existing_count, min(existing_count + batch_size, len(df))):
                row = df.iloc[i]
                
                # Data validation
                if pd.isna(row['itemDescription']):
                    records_rejected += 1
                    continue
                
                # Get product ID
                c.execute("SELECT id, category, price FROM products WHERE name = ?", 
                         (row['itemDescription'],))
                result = c.fetchone()
                
                if result:
                    product_id, category, price = result
                    customer_id = np.random.randint(1, 6)
                    time_id = np.random.randint(1, 100)
                    quantity = 1
                    total_price = price * quantity
                    
                    c.execute("""INSERT INTO fact_transactions 
                                (customer_id, product_id, time_id, quantity, total_price, discount, category)
                                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                             (customer_id, product_id, time_id, quantity, total_price, 0, category))
                    records_loaded += 1
                else:
                    records_rejected += 1
            
            # Log ETL metadata
            duration = (datetime.now() - start_time).total_seconds()
            c.execute("""INSERT INTO etl_metadata 
                        (table_name, records_loaded, records_rejected, status, duration_seconds)
                        VALUES (?, ?, ?, ?, ?)""",
                     ('fact_transactions', records_loaded, records_rejected, 'SUCCESS', duration))
            
            conn.commit()
            
            print(f"\n✅ ETL Complete!")
            print(f"   Records Loaded: {records_loaded}")
            print(f"   Records Rejected: {records_rejected}")
            print(f"   Duration: {duration:.2f}s")
            print("="*70)
            
            conn.close()
            
            return {
                'status': 'SUCCESS',
                'records_loaded': records_loaded,
                'records_rejected': records_rejected,
                'duration': duration
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            c.execute("""INSERT INTO etl_metadata 
                        (table_name, records_loaded, records_rejected, status, error_message, duration_seconds)
                        VALUES (?, ?, ?, ?, ?, ?)""",
                     ('fact_transactions', 0, 0, 'FAILED', str(e), duration))
            conn.commit()
            conn.close()
            
            print(f"\n❌ ETL Failed: {str(e)}")
            return {'status': 'FAILED', 'error': str(e)}
    
    def refresh_aggregations(self):
        """Refresh aggregation tables for faster queries"""
        print("\n♻️  Refreshing aggregation tables...")
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Clear existing aggregations
        c.execute("DELETE FROM agg_daily_sales")
        
        # Rebuild aggregations
        c.execute("""INSERT INTO agg_daily_sales
                    SELECT 
                        t.timestamp as date,
                        f.category,
                        SUM(f.total_price) as total_sales,
                        SUM(f.quantity) as total_quantity,
                        COUNT(DISTINCT f.customer_id) as unique_customers,
                        AVG(f.total_price) as avg_transaction
                    FROM fact_transactions f
                    JOIN dim_time t ON f.time_id = t.time_id
                    GROUP BY t.timestamp, f.category""")
        
        rows_inserted = c.rowcount
        conn.commit()
        conn.close()
        
        print(f"✅ Aggregations refreshed: {rows_inserted} rows")
        return {'rows_aggregated': rows_inserted}
    
    def get_etl_stats(self):
        """Get ETL execution statistics"""
        conn = sqlite3.connect(self.db_path)
        
        query = """SELECT 
                    load_date,
                    table_name,
                    records_loaded,
                    records_rejected,
                    status,
                    duration_seconds
                   FROM etl_metadata
                   ORDER BY load_date DESC
                   LIMIT 10"""
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df.to_dict('records') if len(df) > 0 else []
