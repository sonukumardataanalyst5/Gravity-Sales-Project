import os
import json
import warnings
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy import stats

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ─── Data Loading & Cleaning ───────────────────────────────────────────────────

def load_and_clean_data():
    csv_path = os.path.join(os.path.dirname(__file__),
                            'Superstore Marketing Campaign Dataset.csv')
    df = pd.read_csv(csv_path)

    # 1. Handle missing Income → fill with median
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    df['Income'].fillna(df['Income'].median(), inplace=True)

    # 2. Date formatting
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='mixed', dayfirst=False)

    # 3. Derive Age
    current_year = 2026
    df['Age'] = current_year - df['Year_Birth']
    # Remove outlier ages (e.g., born 1893 → age 133)
    df = df[df['Age'].between(18, 100)].copy()

    # 4. Standardize Marital_Status
    status_map = {
        'Married': 'Married', 'Together': 'Together', 'Single': 'Single',
        'Divorced': 'Divorced', 'Widow': 'Widow', 'Alone': 'Single',
        'Absurd': 'Single', 'YOLO': 'Single'
    }
    df['Marital_Status'] = df['Marital_Status'].map(status_map).fillna('Single')

    # 4b. Merge '2n Cycle' into 'Basic' for Education
    df['Education'] = df['Education'].replace('2n Cycle', 'Basic')

    # 5. Derived columns
    spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                  'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)

    purchase_cols = ['NumDealsPurchases', 'NumWebPurchases',
                     'NumCatalogPurchases', 'NumStorePurchases']
    df['TotalPurchases'] = df[purchase_cols].sum(axis=1)

    df['HasChildren'] = ((df['Kidhome'] + df['Teenhome']) > 0).astype(int)

    # 6. Age groups
    bins = [17, 30, 40, 50, 60, 70, 100]
    labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '70+']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)

    # 7. Income groups
    ibins = [0, 25000, 50000, 75000, 100000, 200000]
    ilabels = ['<25K', '25K-50K', '50K-75K', '75K-100K', '100K+']
    df['Income_Group'] = pd.cut(df['Income'], bins=ibins, labels=ilabels)

    # 8. Top spending category per customer
    cat_map = {
        'MntWines': 'Wines', 'MntFruits': 'Fruits',
        'MntMeatProducts': 'Meat', 'MntFishProducts': 'Fish',
        'MntSweetProducts': 'Sweets', 'MntGoldProds': 'Gold'
    }
    df['TopCategory'] = df[spend_cols].idxmax(axis=1).map(cat_map)

    # Bulletproof catch-all for any straggling NaNs in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


DF = load_and_clean_data()

# ─── ML Model ──────────────────────────────────────────────────────────────────

def build_ml_model(df):
    feature_cols = ['Income', 'Age', 'TotalSpend', 'TotalPurchases',
                    'NumWebPurchases', 'NumCatalogPurchases',
                    'NumStorePurchases', 'NumDealsPurchases',
                    'NumWebVisitsMonth', 'Recency',
                    'MntWines', 'MntMeatProducts', 'MntFruits',
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                    'Kidhome', 'Teenhome', 'HasChildren']

    X = df[feature_cols].copy()
    y = df['Response'].copy()

    # ── Original model (before SMOTE) ──
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf_orig = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    clf_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = clf_orig.predict(X_test_orig)
    acc_orig = accuracy_score(y_test_orig, y_pred_orig)
    report_orig = classification_report(y_test_orig, y_pred_orig, output_dict=True)
    cm_orig = confusion_matrix(y_test_orig, y_pred_orig).tolist()
    fi_orig = dict(zip(feature_cols,
                       [round(float(v), 4) for v in clf_orig.feature_importances_]))
    fi_orig = dict(sorted(fi_orig.items(), key=lambda x: x[1], reverse=True))

    # Buyers found per 100 (original)
    total_test_orig = len(y_test_orig)
    buyers_found_orig = int(y_pred_orig.sum())
    buyers_per_100_orig = round(buyers_found_orig / total_test_orig * 100, 1)

    # ── SMOTE model ──
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.25, random_state=42, stratify=y_res
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    importances = dict(zip(feature_cols,
                           [round(float(v), 4) for v in clf.feature_importances_]))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    buyers_found_smote = int(y_pred.sum())
    total_test_smote = len(y_test)
    buyers_per_100_smote = round(buyers_found_smote / total_test_smote * 100, 1)

    def fmt_report(r):
        return {k: {kk: round(float(vv), 4) if isinstance(vv, (float, np.floating)) else vv
                    for kk, vv in v.items()} if isinstance(v, dict) else round(float(v), 4)
               for k, v in r.items()}

    return {
        'accuracy': round(float(acc), 4),
        'report': fmt_report(report),
        'confusion_matrix': cm,
        'feature_importance': importances,
        'smote_info': {
            'original_0': int((y == 0).sum()),
            'original_1': int((y == 1).sum()),
            'resampled_0': int((y_res == 0).sum()),
            'resampled_1': int((y_res == 1).sum()),
        },
        'original_model': {
            'accuracy': round(float(acc_orig), 4),
            'report': fmt_report(report_orig),
            'confusion_matrix': cm_orig,
            'feature_importance': fi_orig,
            'buyers_per_100': buyers_per_100_orig,
        },
        'smote_model': {
            'accuracy': round(float(acc), 4),
            'buyers_per_100': buyers_per_100_smote,
        },
        'model_comparison': {
            'original_accuracy': round(float(acc_orig) * 100, 1),
            'smote_accuracy': round(float(acc) * 100, 1),
            'original_buyers_per_100': buyers_per_100_orig,
            'smote_buyers_per_100': buyers_per_100_smote,
            'original_recall_1': round(float(report_orig.get('1', {}).get('recall', 0)) * 100, 1),
            'smote_recall_1': round(float(report.get('1', {}).get('recall', 0)) * 100, 1),
        }
    }


ML_RESULTS = build_ml_model(DF)

# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


def apply_filters(df):
    """Apply slicer filters from query params."""
    education = request.args.get('education')
    marital = request.args.get('marital')
    age_group = request.args.get('age_group')
    income_group = request.args.get('income_group')
    has_children = request.args.get('has_children')

    if education and education != 'All':
        df = df[df['Education'] == education]
    if marital and marital != 'All':
        df = df[df['Marital_Status'] == marital]
    if age_group and age_group != 'All':
        df = df[df['Age_Group'] == age_group]
    if income_group and income_group != 'All':
        df = df[df['Income_Group'] == income_group]
    if has_children and has_children != 'All':
        df = df[df['HasChildren'] == int(has_children)]
    return df


@app.route('/api/filters')
def get_filters():
    return jsonify({
        'education': ['All'] + sorted(DF['Education'].unique().tolist()),
        'marital': ['All'] + sorted(DF['Marital_Status'].unique().tolist()),
        'age_group': ['All'] + ['18-30', '31-40', '41-50', '51-60', '61-70', '70+'],
        'income_group': ['All'] + ['<25K', '25K-50K', '50K-75K', '75K-100K', '100K+'],
        'has_children': ['All', '0', '1'],
    })


@app.route('/api/kpis')
def get_kpis():
    df = apply_filters(DF.copy())
    if df.empty:
        return jsonify({'error': 'No data for selected filters'})

    total_customers = len(df)
    avg_income = round(float(df['Income'].mean()), 2)
    total_spend = round(float(df['TotalSpend'].sum()), 2)
    avg_spend = round(float(df['TotalSpend'].mean()), 2)
    response_rate = round(float(df['Response'].mean() * 100), 2)
    avg_recency = round(float(df['Recency'].mean()), 1)
    complain_rate = round(float(df['Complain'].mean() * 100), 2)
    avg_web_visits = round(float(df['NumWebVisitsMonth'].mean()), 1)

    return jsonify({
        'total_customers': total_customers,
        'avg_income': avg_income,
        'total_spend': total_spend,
        'avg_spend': avg_spend,
        'response_rate': response_rate,
        'avg_recency': avg_recency,
        'complain_rate': complain_rate,
        'avg_web_visits': avg_web_visits,
    })


@app.route('/api/spend_by_category')
def spend_by_category():
    df = apply_filters(DF.copy())
    cats = {
        'Wines': float(df['MntWines'].sum()),
        'Fruits': float(df['MntFruits'].sum()),
        'Meat': float(df['MntMeatProducts'].sum()),
        'Fish': float(df['MntFishProducts'].sum()),
        'Sweets': float(df['MntSweetProducts'].sum()),
        'Gold': float(df['MntGoldProds'].sum()),
    }
    return jsonify(cats)


@app.route('/api/purchase_channels')
def purchase_channels():
    df = apply_filters(DF.copy())
    channels = {
        'Web': int(df['NumWebPurchases'].sum()),
        'Catalog': int(df['NumCatalogPurchases'].sum()),
        'Store': int(df['NumStorePurchases'].sum()),
        'Deals': int(df['NumDealsPurchases'].sum()),
    }
    return jsonify(channels)


@app.route('/api/demographics')
def demographics():
    df = apply_filters(DF.copy())

    # Spend by education
    edu_spend = df.groupby('Education')['TotalSpend'].mean().round(2)

    # Spend by marital status
    mar_spend = df.groupby('Marital_Status')['TotalSpend'].mean().round(2)

    # Spend by age group
    age_spend = df.groupby('Age_Group', observed=False)['TotalSpend'].mean().fillna(0).round(2)

    # Response rate by education
    edu_response = (df.groupby('Education')['Response'].mean().fillna(0) * 100).round(2)

    # Response rate by marital status
    mar_response = (df.groupby('Marital_Status')['Response'].mean().fillna(0) * 100).round(2)

    # Response rate by age group
    age_response = (df.groupby('Age_Group', observed=False)['Response'].mean().fillna(0) * 100).round(2)

    # Customer count by education
    edu_count = df.groupby('Education').size()

    # Customer count by age group
    age_count = df.groupby('Age_Group', observed=False).size()

    return jsonify({
        'edu_spend': {str(k): float(v) for k, v in edu_spend.items()},
        'mar_spend': {str(k): float(v) for k, v in mar_spend.items()},
        'age_spend': {str(k): float(v) for k, v in age_spend.items()},
        'edu_response': {str(k): float(v) for k, v in edu_response.items()},
        'mar_response': {str(k): float(v) for k, v in mar_response.items()},
        'age_response': {str(k): float(v) for k, v in age_response.items()},
        'edu_count': {str(k): int(v) for k, v in edu_count.items()},
        'age_count': {str(k): int(v) for k, v in age_count.items()},
    })


@app.route('/api/top_customers')
def top_customers():
    df = apply_filters(DF.copy())
    top = df.nlargest(10, 'TotalSpend')[
        ['Id', 'Age', 'Education', 'Marital_Status', 'Income',
         'TotalSpend', 'TopCategory', 'Response']
    ]
    return jsonify(top.to_dict(orient='records'))


@app.route('/api/web_analysis')
def web_analysis():
    df = apply_filters(DF.copy())
    # Web visits vs purchases correlation
    web_data = df.groupby('NumWebVisitsMonth').agg(
        avg_web_purchases=('NumWebPurchases', 'mean'),
        avg_spend=('TotalSpend', 'mean'),
        count=('Id', 'count')
    ).round(2)
    return jsonify({
        str(k): {kk: float(vv) for kk, vv in v.items()}
        for k, v in web_data.to_dict(orient='index').items()
    })


@app.route('/api/numeric_stats')
def numeric_stats():
    df = apply_filters(DF.copy())
    num_cols = ['Income', 'Age', 'Recency', 'MntWines', 'MntFruits',
                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                'MntGoldProds', 'TotalSpend', 'NumWebPurchases',
                'NumCatalogPurchases', 'NumStorePurchases',
                'NumDealsPurchases', 'NumWebVisitsMonth', 'TotalPurchases']
    desc = df[num_cols].describe().round(2)
    # Add skewness and kurtosis
    skew = df[num_cols].skew().round(4)
    kurt = df[num_cols].kurtosis().round(4)
    desc.loc['skewness'] = skew
    desc.loc['kurtosis'] = kurt
    result = {}
    for col in desc.columns:
        result[col] = {str(k): float(v) for k, v in desc[col].items()}
    return jsonify(result)


@app.route('/api/correlation')
def correlation():
    df = apply_filters(DF.copy())
    num_cols = ['Income', 'Age', 'Recency', 'MntWines', 'MntFruits',
                'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                'MntGoldProds', 'TotalSpend', 'NumWebPurchases',
                'NumCatalogPurchases', 'NumStorePurchases',
                'NumDealsPurchases', 'NumWebVisitsMonth']
    corr = df[num_cols].corr().round(3)
    return jsonify({
        'columns': corr.columns.tolist(),
        'data': corr.values.tolist()
    })


@app.route('/api/ml_results')
def ml_results():
    return jsonify(ML_RESULTS)


@app.route('/api/campaign_by_segment')
def campaign_by_segment():
    df = apply_filters(DF.copy())
    # Acceptance rate by num web purchases bins
    web_bins = [0, 2, 5, 10, 30]
    web_labels = ['0-2', '3-5', '6-10', '10+']
    df['WebPurchaseBin'] = pd.cut(df['NumWebPurchases'], bins=web_bins, labels=web_labels)

    web_response = (df.groupby('WebPurchaseBin', observed=False)['Response'].mean().fillna(0) * 100).round(2)

    # Income vs response
    inc_response = (df.groupby('Income_Group', observed=False)['Response'].mean().fillna(0) * 100).round(2)

    # Recency bins
    rec_bins = [0, 20, 40, 60, 80, 100]
    rec_labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df['RecencyBin'] = pd.cut(df['Recency'], bins=rec_bins, labels=rec_labels)
    rec_response = (df.groupby('RecencyBin', observed=False)['Response'].mean().fillna(0) * 100).round(2)

    return jsonify({
        'web_purchase_response': {str(k): float(v) for k, v in web_response.items()},
        'income_response': {str(k): float(v) for k, v in inc_response.items()},
        'recency_response': {str(k): float(v) for k, v in rec_response.items()},
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
