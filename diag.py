#!/usr/bin/env python3
"""
Diagnostic script to identify why fixes aren't working
Run this from your project root directory: python test_fixes.py
"""

import pandas as pd
import numpy as np
import sys
import os

print("="*70)
print("SUPPLIER RECOMMENDATION SYSTEM - DIAGNOSTIC TOOL")
print("="*70)

# Check if files exist
print("\n1. CHECKING FILE LOCATIONS...")
files_ok = True

if os.path.exists('data/suppliers.csv'):
    print("  ✅ data/suppliers.csv found")
else:
    print("  ❌ data/suppliers.csv NOT found")
    files_ok = False

if os.path.exists('app.py'):
    print("  ✅ app.py found")
else:
    print("  ❌ app.py NOT found")
    files_ok = False

if os.path.exists('data/embedding_engine.py'):
    print("  ✅ data/embedding_engine.py found")
else:
    print("  ❌ data/embedding_engine.py NOT found") 
    files_ok = False

if not files_ok:
    print("\n❌ CRITICAL: Files missing! Make sure you're in the project root directory.")
    sys.exit(1)

# Load the data
print("\n2. LOADING DATA...")
try:
    df = pd.read_csv('data/suppliers.csv')
    print(f"  ✅ Loaded {len(df)} suppliers")
except Exception as e:
    print(f"  ❌ Error loading data: {e}")
    sys.exit(1)

# Check required columns
print("\n3. CHECKING DATA COLUMNS...")
required = ['supplier_id', 'name', 'industry', 'products', 'political_risk', 'trade_risk', 'natural_disaster_risk']
missing = [col for col in required if col not in df.columns]

if missing:
    print(f"  ❌ Missing columns: {missing}")
else:
    print("  ✅ All required columns present")

# Analyze risk data
print("\n4. ANALYZING RISK DATA...")
print("\nRaw Risk Values Distribution:")

for col in ['political_risk', 'trade_risk', 'natural_disaster_risk']:
    if col in df.columns:
        print(f"\n  {col}:")
        value_counts = df[col].value_counts()
        for val, count in value_counts.items():
            pct = count / len(df) * 100
            print(f"    {val}: {count} ({pct:.1f}%)")

# Import and test risk function
print("\n5. TESTING RISK CALCULATION...")
try:
    from app import compute_overall_risk_level, map_single_risk
    
    # Test with sample data
    test_row = {
        'political_risk': 'Low',
        'trade_risk': 'Low',
        'natural_disaster_risk': 'High'
    }
    result = compute_overall_risk_level(test_row)
    
    # Calculate what it should be
    score_map = {"Low": 1.0, "Mid": 2.0, "High": 3.0}
    pr = map_single_risk(test_row['political_risk'])
    tr = map_single_risk(test_row['trade_risk'])
    nr = map_single_risk(test_row['natural_disaster_risk'])
    avg = (score_map[pr] + score_map[tr] + score_map[nr]) / 3
    
    print(f"  Test case: Low, Low, High")
    print(f"  Average score: {avg:.2f}")
    print(f"  Result: {result}")
    print(f"  Expected: Mid (if avg {avg:.2f} is between 1.7 and 2.5)")
    
    if result == "Mid" and 1.7 <= avg < 2.5:
        print("  ✅ Risk calculation is using WEIGHTED AVERAGE (correct!)")
    elif result == "High":
        print("  ❌ Risk calculation is using HIGHEST WINS (old buggy version!)")
        print("     You need to replace app.py with the fixed version!")
    
except Exception as e:
    print(f"  ❌ Error testing risk calculation: {e}")
    print("     Make sure app.py is the updated version")

# Calculate actual risk distribution
print("\n6. ACTUAL RISK DISTRIBUTION IN YOUR DATA...")
try:
    df['overall_risk'] = df.apply(compute_overall_risk_level, axis=1)
    
    print("\nOverall Risk Levels:")
    for risk in ['Low', 'Mid', 'High']:
        count = (df['overall_risk'] == risk).sum()
        pct = count / len(df) * 100
        bar = '█' * int(pct / 2)
        print(f"  {risk:4s}: {bar} {count:4d} ({pct:5.1f}%)")
    
    if (df['overall_risk'] == 'High').sum() > len(df) * 0.9:
        print("\n  ⚠️  WARNING: >90% suppliers are High risk")
        print("     This suggests your data has genuinely high risk values")
        print("     The fix is working, but your data is actually high risk!")
        
except Exception as e:
    print(f"  ❌ Error calculating risk: {e}")

# Test embedding engine
print("\n7. TESTING EMBEDDING ENGINE...")
try:
    from data.embedding_engine import EmbeddingEngine
    
    # Check if products field is being weighted
    import inspect
    source = inspect.getsource(EmbeddingEngine.ensure)
    
    if 'products} {products}' in source or 'products field' in source.lower():
        print("  ✅ Embedding engine HAS product weighting code")
    else:
        print("  ❌ Embedding engine MISSING product weighting")
        print("     You need to replace data/embedding_engine.py!")
    
    # Test with real data
    engine = EmbeddingEngine()
    electronics = df[df['industry'] == 'Electronics'].copy()
    
    if len(electronics) > 0:
        # Create test data with one electronics and one non-electronics
        test_df = pd.concat([
            electronics.head(1),
            df[df['industry'] != 'Electronics'].head(1)
        ]).reset_index(drop=True)
        
        engine.ensure(test_df)
        results = engine.search('monitor', top_k=2)
        
        print(f"\n  Search test for 'monitor':")
        for sid, score in results:
            supplier = test_df[test_df['supplier_id'] == sid].iloc[0]
            print(f"    {supplier['industry']:15s} - Score: {score:.4f}")
        
        # Check if electronics supplier scored higher
        if len(results) >= 2:
            first_sid = results[0][0]
            first_industry = test_df[test_df['supplier_id'] == first_sid].iloc[0]['industry']
            
            if first_industry == 'Electronics':
                print("  ✅ Electronics supplier ranks first (correct!)")
            else:
                print("  ❌ Non-Electronics supplier ranks first")
                print("     Check if the fixed embedding_engine.py is deployed")
    else:
        print("  ⚠️  No Electronics suppliers in your data to test")
        
except Exception as e:
    print(f"  ❌ Error testing embedding: {e}")
    import traceback
    traceback.print_exc()

# Analyze Electronics suppliers with monitors
print("\n8. ELECTRONICS SUPPLIERS WITH MONITOR PRODUCTS...")
try:
    electronics = df[df['industry'] == 'Electronics']
    print(f"  Total Electronics suppliers: {len(electronics)}")
    
    if 'products' in electronics.columns:
        monitor_keywords = ['monitor', 'display', 'screen', 'lcd', 'led']
        has_monitors = electronics[
            electronics['products'].str.lower().str.contains('|'.join(monitor_keywords), na=False)
        ]
        
        print(f"  With monitor-related products: {len(has_monitors)}")
        
        if len(has_monitors) > 0:
            print("\n  Sample suppliers:")
            for idx, row in has_monitors.head(5).iterrows():
                risk = compute_overall_risk_level(row)
                print(f"    • {row['name']} ({row['supplier_id']}) - Risk: {risk}")
                if pd.notna(row['products']):
                    products_preview = str(row['products'])[:60]
                    print(f"      Products: {products_preview}...")
        else:
            print("\n  ⚠️  No Electronics suppliers have monitor-related products")
            print("     This could explain why monitors search isn't working!")
    
except Exception as e:
    print(f"  ❌ Error analyzing electronics: {e}")

# Final recommendations
print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

print("\n📋 DEPLOYMENT CHECKLIST:")
print("  [ ] Downloaded the LATEST fixed files from outputs folder")
print("  [ ] Replaced app.py in project root")
print("  [ ] Replaced data/embedding_engine.py")
print("  [ ] Restarted Flask app (python app.py)")
print("  [ ] Cleared browser cache (Ctrl+Shift+R)")

print("\n🔍 IF TESTS SHOW OLD CODE:")
print("  → The fixed files were not properly deployed")
print("  → Download again and make sure you replace the correct files")

print("\n📊 IF RISK DISTRIBUTION IS STILL 100% HIGH:")
print("  → Check test #5 above - if it shows 'HIGHEST WINS', files not updated")
print("  → If it shows 'WEIGHTED AVERAGE' but still all High, your data")
print("    genuinely has high risk values (e.g., most suppliers have 2-3 High risks)")

print("\n🔎 IF MONITOR SEARCH STILL SHOWS NON-ELECTRONICS:")
print("  → Check test #7 - should show 'product weighting code'")
print("  → Check test #8 - you need Electronics suppliers WITH monitor products")
print("  → If no suppliers have monitors in their products field, that's the issue")

print("\n💡 NEXT STEPS:")
print("  1. Review the test results above")
print("  2. If old code detected: re-download and deploy fixed files")
print("  3. If data issue: you may need to adjust your risk thresholds or product data")
print("  4. Run this script again after making changes")

print("\n" + "="*70)