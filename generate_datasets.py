import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define industries and their characteristics
INDUSTRIES = {
    'Electronics': ['CPUs', 'RAM', 'SSDs', 'Motherboards', 'GPUs', 'Power Supplies'],
    'Automotive': ['Bearings', 'Pistons', 'Brake Pads', 'Filters', 'Belts', 'Gaskets'],
    'Construction': ['Cement', 'Steel Rods', 'Pipes', 'Valves', 'Bolts', 'Concrete Blocks'],
    'Textiles': ['Cotton Fabric', 'Polyester', 'Buttons', 'Zippers', 'Thread', 'Dyes'],
    'Pharmaceuticals': ['APIs', 'Tablets', 'Capsules', 'Syringes', 'Vials', 'Packaging'],
    'Food & Beverage': ['Ingredients', 'Packaging', 'Bottles', 'Labels', 'Preservatives', 'Flavorings'],
    'Chemicals': ['Industrial Solvents', 'Acids', 'Polymers', 'Catalysts', 'Additives', 'Resins'],
    'Machinery': ['Gears', 'Motors', 'Pumps', 'Valves', 'Conveyors', 'Hydraulics'],
    'Plastics': ['PVC', 'HDPE', 'Polypropylene', 'Plastic Sheets', 'Molded Parts', 'Films'],
    'Metal Works': ['Aluminum Sheets', 'Steel Plates', 'Copper Wire', 'Brass Fittings', 'Metal Tubes', 'Castings']
}

# Countries and their risk profiles
COUNTRIES = {
    'India': {'political_risk': 'Low', 'trade_risk': 'Low', 'natural_disaster_risk': 'Medium'},
    'USA': {'political_risk': 'Low', 'trade_risk': 'Low', 'natural_disaster_risk': 'Low'},
    'China': {'political_risk': 'Medium', 'trade_risk': 'Medium', 'natural_disaster_risk': 'Medium'},
    'Germany': {'political_risk': 'Low', 'trade_risk': 'Low', 'natural_disaster_risk': 'Low'},
    'Japan': {'political_risk': 'Low', 'trade_risk': 'Low', 'natural_disaster_risk': 'High'},
    'Vietnam': {'political_risk': 'Medium', 'trade_risk': 'Low', 'natural_disaster_risk': 'High'},
    'Mexico': {'political_risk': 'Medium', 'trade_risk': 'Low', 'natural_disaster_risk': 'Medium'},
    'Brazil': {'political_risk': 'Medium', 'trade_risk': 'Medium', 'natural_disaster_risk': 'Medium'},
    'UK': {'political_risk': 'Low', 'trade_risk': 'Medium', 'natural_disaster_risk': 'Low'},
    'South Korea': {'political_risk': 'Low', 'trade_risk': 'Low', 'natural_disaster_risk': 'Medium'},
    'Thailand': {'political_risk': 'Medium', 'trade_risk': 'Low', 'natural_disaster_risk': 'High'},
    'Turkey': {'political_risk': 'High', 'trade_risk': 'Medium', 'natural_disaster_risk': 'High'},
}

CITIES = {
    'India': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Ahmedabad'],
    'USA': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'San Francisco'],
    'China': ['Shanghai', 'Beijing', 'Shenzhen', 'Guangzhou', 'Chengdu', 'Hangzhou'],
    'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne', 'Stuttgart'],
    'Japan': ['Tokyo', 'Osaka', 'Nagoya', 'Yokohama', 'Fukuoka', 'Sapporo'],
    'Vietnam': ['Hanoi', 'Ho Chi Minh City', 'Da Nang', 'Hai Phong', 'Can Tho'],
    'Mexico': ['Mexico City', 'Guadalajara', 'Monterrey', 'Puebla', 'Tijuana'],
    'Brazil': ['Sao Paulo', 'Rio de Janeiro', 'Brasilia', 'Salvador', 'Fortaleza'],
    'UK': ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow'],
    'South Korea': ['Seoul', 'Busan', 'Incheon', 'Daegu', 'Daejeon'],
    'Thailand': ['Bangkok', 'Chiang Mai', 'Phuket', 'Pattaya', 'Khon Kaen'],
    'Turkey': ['Istanbul', 'Ankara', 'Izmir', 'Bursa', 'Antalya'],
}

CREDIT_RATINGS = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB']
COMPANY_PREFIXES = ['Global', 'Premium', 'Elite', 'Alpha', 'Beta', 'Sigma', 'Apex', 'Prime', 'Superior', 'Advanced']
COMPANY_SUFFIXES = ['Inc.', 'Ltd.', 'Corp.', 'Solutions', 'Industries', 'Enterprises', 'Manufacturing', 'Supplies', 'Tech', 'Co.']

def generate_company_name(industry):
    """Generate a realistic company name"""
    prefix = random.choice(COMPANY_PREFIXES)
    industry_word = industry.split()[0] if ' ' in industry else industry
    suffix = random.choice(COMPANY_SUFFIXES)
    return f"{prefix} {industry_word} {suffix}"

def generate_suppliers(n=500):
    """Generate realistic supplier dataset"""
    suppliers = []
    
    for i in range(n):
        industry = random.choice(list(INDUSTRIES.keys()))
        country = random.choice(list(COUNTRIES.keys()))
        city = random.choice(CITIES[country])
        risks = COUNTRIES[country]
        
        # Generate financial metrics
        revenue_growth = round(np.random.normal(10, 8), 1)  # Average 10% with std 8%
        profit_margin = round(np.random.normal(12, 5), 1)   # Average 12% with std 5%
        
        # Credit rating distribution (more weighted towards higher ratings)
        credit_rating = random.choices(
            CREDIT_RATINGS,
            weights=[15, 12, 12, 10, 10, 8, 8, 7, 6, 5, 4, 3],
            k=1
        )[0]
        
        # Performance metrics
        on_time_delivery = round(np.random.beta(8, 2) * 100, 1)  # Skewed towards high values
        quality_score = round(np.random.beta(7, 2) * 5, 1)       # Out of 5, skewed high
        response_time = round(np.random.gamma(2, 2), 1)          # Hours, typically 2-8
        
        supplier = {
            'supplier_id': f'SUP{str(i+1).zfill(4)}',
            'name': generate_company_name(industry),
            'industry': industry,
            'country': country,
            'city': city,
            'address': f'{random.randint(1, 999)} {random.choice(["Industrial", "Business", "Corporate", "Tech"])} Street',
            'postal_code': f'{random.randint(10000, 99999)}',
            'political_risk': risks['political_risk'],
            'trade_risk': risks['trade_risk'],
            'natural_disaster_risk': risks['natural_disaster_risk'],
            'revenue_growth_yoy': revenue_growth,
            'net_profit_margin': profit_margin,
            'credit_rating': credit_rating,
            'on_time_delivery_rate': on_time_delivery,
            'quality_score': quality_score,
            'response_time_hours': response_time,
            'years_in_business': random.randint(2, 40),
            'employee_count': random.choice([50, 100, 200, 500, 1000, 2000, 5000]),
            'certification': random.choice(['ISO 9001', 'ISO 14001', 'ISO 45001', 'None']),
            'email': f'contact@{generate_company_name(industry).lower().replace(" ", "")}.com',
            'phone': f'+{random.randint(1, 99)}-{random.randint(100, 999)}-{random.randint(1000000, 9999999)}'
        }
        
        suppliers.append(supplier)
    
    return pd.DataFrame(suppliers)

def generate_purchase_orders(suppliers_df, n=4500):
    """Generate realistic purchase order dataset"""
    pos = []
    
    start_date = datetime.now() - timedelta(days=730)  # 2 years of history
    
    for i in range(n):
        supplier = suppliers_df.sample(1).iloc[0]
        industry = supplier['industry']
        items = INDUSTRIES[industry]
        
        # Order date within last 2 years
        order_date = start_date + timedelta(days=random.randint(0, 730))
        
        # Delivery typically 7-60 days after order
        expected_delivery = order_date + timedelta(days=random.randint(7, 60))
        
        # Actual delivery (some delayed, some early, most on time)
        delivery_variance = int(np.random.normal(0, 5))  # Mean 0, std 5 days
        actual_delivery = expected_delivery + timedelta(days=delivery_variance)
        
        # If order is in future, no actual delivery yet
        if actual_delivery > datetime.now():
            actual_delivery = None
            status = 'Pending'
            on_time = None
        else:
            # Determine if delivered on time
            on_time = actual_delivery <= expected_delivery
            status = 'Completed' if random.random() > 0.05 else 'Cancelled'  # 5% cancellation rate
        
        # Item details
        item_name = random.choice(items)
        quantity = random.choice([10, 20, 50, 100, 200, 500, 1000])
        unit_price = round(random.uniform(50, 5000), 2)
        subtotal = quantity * unit_price
        tax_rate = random.choice([0.05, 0.08, 0.10, 0.12, 0.18])
        tax = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax, 2)
        
        po = {
            'po_number': f'PO{str(i+1).zfill(5)}',
            'supplier_id': supplier['supplier_id'],
            'supplier_name': supplier['name'],
            'order_date': order_date.strftime('%Y-%m-%d'),
            'expected_delivery_date': expected_delivery.strftime('%Y-%m-%d'),
            'actual_delivery_date': actual_delivery.strftime('%Y-%m-%d') if actual_delivery else None,
            'status': status,
            'item_description': item_name,
            'quantity': quantity,
            'unit_price': unit_price,
            'subtotal': subtotal,
            'tax_rate': tax_rate,
            'tax_amount': tax,
            'total_amount': total,
            'currency': random.choice(['USD', 'EUR', 'INR', 'GBP', 'JPY']),
            'on_time_delivery': on_time,
            'payment_terms': random.choice(['Net 30', 'Net 45', 'Net 60', 'Immediate', 'Net 15']),
            'shipping_method': random.choice(['Air Freight', 'Sea Freight', 'Ground', 'Express']),
            'created_by': random.choice(['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Williams', 'David Brown']),
        }
        
        pos.append(po)
    
    return pd.DataFrame(pos)

def calculate_supplier_performance(suppliers_df, pos_df):
    """Calculate aggregated performance metrics for each supplier"""
    
    # Group by supplier and calculate metrics
    supplier_stats = pos_df.groupby('supplier_id').agg({
        'po_number': 'count',  # Total orders
        'on_time_delivery': lambda x: (x == True).sum() / len(x) * 100 if len(x) > 0 else 0,  # On-time %
        'total_amount': 'sum',  # Total business value
        'actual_delivery_date': 'count'  # Completed orders
    }).reset_index()
    
    supplier_stats.columns = ['supplier_id', 'total_orders', 'actual_on_time_rate', 'total_business_value', 'completed_orders']
    
    # Merge with original supplier data
    suppliers_enhanced = suppliers_df.merge(supplier_stats, on='supplier_id', how='left')
    
    # Fill NaN values for suppliers with no orders
    suppliers_enhanced['total_orders'] = suppliers_enhanced['total_orders'].fillna(0).astype(int)
    suppliers_enhanced['completed_orders'] = suppliers_enhanced['completed_orders'].fillna(0).astype(int)
    suppliers_enhanced['total_business_value'] = suppliers_enhanced['total_business_value'].fillna(0)
    suppliers_enhanced['actual_on_time_rate'] = suppliers_enhanced['actual_on_time_rate'].fillna(
        suppliers_enhanced['on_time_delivery_rate']
    )
    
    return suppliers_enhanced

if __name__ == '__main__':
    print("Generating supplier dataset...")
    suppliers_df = generate_suppliers(500)
    
    print("Generating purchase orders dataset...")
    pos_df = generate_purchase_orders(suppliers_df, 4500)
    
    print("Calculating supplier performance metrics...")
    suppliers_enhanced = calculate_supplier_performance(suppliers_df, pos_df)
    
    # Save to CSV
    print("Saving datasets...")
    suppliers_enhanced.to_csv('./data/suppliers.csv', index=False)
    pos_df.to_csv('./data/purchase_orders.csv', index=False)
    
    print(f"\n✅ Generated:")
    print(f"   - {len(suppliers_enhanced)} suppliers")
    print(f"   - {len(pos_df)} purchase orders")
    print(f"\nSample supplier data:")
    print(suppliers_enhanced.head())
    print(f"\nSample PO data:")
    print(pos_df.head())
    print(f"\nFiles saved to /home/claude/procurement_hub/data/")