# Smart Procurement Hub - AI-Powered Supplier Recommendation & Automated PO System

## 📋 Overview

An integrated procurement automation system that combines:
- **Smart Supplier Recommendation**: AI/ML-powered ranking based on multiple risk factors, financial health, and performance metrics
- **Automated PO Processing**: Streamlined purchase order creation with natural language support

## 🎯 Key Features

### 1. AI-Powered Supplier Recommendations
- Analyzes 500+ suppliers across 10 industries
- Multi-factor risk assessment:
  - Political instability
  - Trade disputes
  - Natural disaster exposure
- Financial health analysis:
  - Credit ratings
  - Revenue growth
  - Profit margins
- Performance metrics:
  - On-time delivery rates (from 4500+ historical POs)
  - Quality scores
  - Response times

### 2. Comprehensive Risk Profiling
- Regional risk assessment per supplier
- Overall risk scoring (Low/Medium/High)
- Visual risk distribution dashboards

### 3. Integrated PO Creation
- Search and select suppliers
- View detailed supplier profiles in modal
- Create POs directly from supplier details
- Natural language PO input support

### 4. Analytics & Insights
- Performance trend tracking
- Cost savings analysis
- Cycle time monitoring
- Risk distribution visualization

## 🗂️ Project Structure

```
procurement_hub/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── generate_datasets.py            # Dataset generator script
├── data/
│   ├── suppliers.csv              # 500 suppliers dataset
│   └── purchase_orders.csv        # 4500 POs dataset
├── models/
│   └── recommendation_engine.py   # ML recommendation algorithm
├── templates/
│   ├── base.html                  # Base template with navigation
│   ├── dashboard.html             # Dashboard with statistics
│   ├── suppliers.html             # Suppliers + PO creation page
│   ├── purchase_orders.html       # View all POs (TODO)
│   ├── po_details.html            # Individual PO details (TODO)
│   └── analytics.html             # Analytics & charts (TODO)
└── static/
    ├── css/
    └── js/
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Steps

1. **Install Dependencies**
```bash
cd /home/claude/procurement_hub
pip install -r requirements.txt --break-system-packages
```

2. **Generate Datasets** (Already done)
```bash
python generate_datasets.py
```

3. **Run the Application**
```bash
python app.py
```

4. **Access the Application**
Open browser and navigate to: `http://localhost:5000`

## 📊 Dataset Details

### Suppliers Dataset (500 entries)
- **Industries**: Electronics, Automotive, Construction, Textiles, Pharmaceuticals, Food & Beverage, Chemicals, Machinery, Plastics, Metal Works
- **Countries**: 12 countries with varying risk profiles
- **Attributes**: 
  - Basic info (name, location, contact)
  - Risk factors (political, trade, natural disasters)
  - Financial metrics (credit rating, revenue growth, profit margin)
  - Performance metrics (on-time delivery, quality score, response time)
  - Business history (years, employees, certifications)

### Purchase Orders Dataset (4500 entries)
- **Time Range**: 2 years of historical data
- **Attributes**:
  - Order details (PO number, dates, status)
  - Items (description, quantity, pricing)
  - Supplier information
  - Delivery tracking (expected vs actual)
  - Financial details (subtotal, tax, total)

## 🧮 Recommendation Algorithm

The system uses a weighted scoring model:

```python
Match Score = (
    Risk Score × 0.25 +
    Financial Score × 0.20 +
    Performance Score × 0.35 +
    History Score × 0.20
)
```

### Sub-scores:
1. **Risk Score (0-100)**:
   - Political risk: 40%
   - Trade risk: 35%
   - Natural disaster risk: 25%

2. **Financial Score (0-100)**:
   - Credit rating: 50%
   - Revenue growth: 25%
   - Profit margin: 25%

3. **Performance Score (0-100)**:
   - On-time delivery: 50%
   - Quality score: 30%
   - Response time: 20%

4. **History Score (0-100)**:
   - Total orders: 40%
   - Business value: 30%
   - Years in business: 30%

## 🎨 UI Features

- **Dark Mode Support**: Toggle between light and dark themes
- **Responsive Design**: Works on mobile, tablet, and desktop
- **Interactive Modals**: Detailed supplier information with inline PO creation
- **Search & Filter**: Find suppliers by name, industry, or location
- **Real-time Updates**: Dynamic content loading

## 📱 Pages

1. **Dashboard**: Overview statistics and risk distribution
2. **Suppliers & PO Creation**: Combined search, recommendation, and order creation
3. **View All POs**: List and filter all purchase orders (TODO)
4. **Analytics**: Charts and performance insights (TODO)

## 🔄 Current Status

### ✅ Completed
- [x] Dataset generation (500 suppliers, 4500 POs)
- [x] Recommendation engine with ML scoring
- [x] Flask application structure
- [x] Base template with navigation
- [x] Dashboard with statistics
- [x] Suppliers page with search & filter
- [x] Supplier details modal
- [x] Integrated PO creation form

### 🚧 In Progress
- [ ] Purchase orders list template
- [ ] Individual PO details page
- [ ] Analytics page with Chart.js
- [ ] NLP for natural language PO input
- [ ] Testing and bug fixes

## 🛠️ Technologies Used

- **Backend**: Flask (Python)
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, Tailwind CSS
- **Icons**: Material Symbols
- **Charts**: Chart.js (for analytics page)

## 📝 API Endpoints

- `GET /` - Dashboard
- `GET /suppliers` - Supplier recommendations (with search/filter)
- `GET /supplier/<supplier_id>` - Supplier details page
- `GET /api/supplier/<supplier_id>` - Supplier details JSON API
- `GET /purchase-orders` - List all POs
- `GET /po/<po_number>` - PO details
- `POST /create-po` - Create new PO
- `GET /analytics` - Analytics dashboard
- `POST /api/convert-to-po` - Convert natural language to PO structure

## 🎯 Next Steps

1. Complete remaining templates (PO list, PO details, Analytics)
2. Add Chart.js for analytics visualization
3. Implement NLP for natural language PO parsing
4. Add form validation
5. Implement database persistence (currently in-memory)
6. Add export functionality (PDF, Excel)
7. Deploy to production server

## 📄 License

License - Feel free to use and modify for your needs.

## 👥 Contributors

Created as part of the Smart Procurement Hub project.
