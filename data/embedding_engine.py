import pandas as pd
sup = pd.read_csv("data/suppliers.csv")
print("unique industry values:", sorted(sup['industry'].dropna().unique()[:50]))
print("sample rows where industry contains 'elect':")
print(sup[sup['industry'].astype(str).str.lower().str.contains('elect', na=False)].head(10))
