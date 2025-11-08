import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Pipeline Studio",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# Professional CSS with shadows and hover effects
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0f172a;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .section-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 24px;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        border-radius: 2px;
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.08);
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
        transform: translateY(-2px);
    }
    
    .success-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
        border-left: 4px solid #10b981;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.08);
        transition: all 0.3s ease;
    }
    
    .success-box:hover {
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.15);
        transform: translateY(-2px);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border-left: 4px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.08);
        transition: all 0.3s ease;
    }
    
    .warning-box:hover {
        box-shadow: 0 8px 24px rgba(245, 158, 11, 0.15);
        transform: translateY(-2px);
    }
    
    .metric-card {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.12);
        border-color: #cbd5e1;
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.12);
        transform: translateY(-2px);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.875rem 1.75rem;
        border: none;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
        letter-spacing: 0.02em;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.35);
        background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.04);
    }
    
    .sidebar-section {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .sidebar-section:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    .tab-container {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
    }
    
    .feature-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #f8fafc;
        margin: 0.25rem;
        border-radius: 8px;
        font-size: 0.875rem;
        color: #475569;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    
    .feature-badge:hover {
        background: #f1f5f9;
        border-color: #cbd5e1;
        transform: translateX(2px);
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        transform: translateY(-3px);
        border-color: #cbd5e1;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1;
    }
    
    /* Professional icon replacements */
    .icon-chart::before { content: '📊'; }
    .icon-settings::before { content: '⚙️'; }
    .icon-target::before { content: '🎯'; }
    .icon-model::before { content: '🤖'; }
    .icon-deploy::before { content: '🚀'; }
    
    /* Hide default emojis in headers */
    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.01em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None

# Helper Functions
def identify_column_types(df):
    """Automatically identify column types"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    for col in categorical_cols[:]:
        try:
            pd.to_datetime(df[col], errors='raise')
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass
    
    return numeric_cols, categorical_cols, datetime_cols

def detect_outliers(df, col):
    """Detect outliers using IQR method"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return len(outliers), lower_bound, upper_bound

def analyze_skewness(df, numeric_cols):
    """Analyze skewness of numerical features"""
    skewness_data = []
    for col in numeric_cols:
        skew_val = skew(df[col].dropna())
        kurt_val = kurtosis(df[col].dropna())
        
        if abs(skew_val) < 0.5:
            interpretation = "Normal"
            color = "green"
        elif abs(skew_val) < 1:
            interpretation = "Moderate"
            color = "orange"
        else:
            interpretation = "High"
            color = "red"
            
        skewness_data.append({
            'Column': col,
            'Skewness': round(skew_val, 3),
            'Kurtosis': round(kurt_val, 3),
            'Status': interpretation
        })
    return pd.DataFrame(skewness_data)

def perform_eda(df):
    """Comprehensive EDA Analysis"""
    
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}", help="Number of samples in dataset")
    with col2:
        st.metric("Total Columns", df.shape[1], help="Number of features")
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", help="Dataset size in memory")
    with col4:
        dupes = df.duplicated().sum()
        st.metric("Duplicates", dupes, delta=f"-{dupes}" if dupes > 0 else "None", delta_color="inverse")
    
    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["First 10 Rows", "Last 10 Rows", "Random Sample"])
    with preview_tab1:
        st.dataframe(df.head(10), use_container_width=True, height=400)
    with preview_tab2:
        st.dataframe(df.tail(10), use_container_width=True, height=400)
    with preview_tab3:
        st.dataframe(df.sample(min(10, len(df))), use_container_width=True, height=400)
    
    numeric_cols, categorical_cols, datetime_cols = identify_column_types(df)
    
    st.markdown('<div class="section-title">Feature Types</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #2563eb; margin-bottom: 1rem; font-size: 1.1rem; font-weight: 600;">Numerical Features ({len(numeric_cols)})</h3>
            {''.join([f'<div class="feature-badge">{col}</div>' for col in numeric_cols[:10]])}
            {f'<div style="color: #64748b; margin-top: 1rem; font-size: 0.875rem;">...and {len(numeric_cols)-10} more</div>' if len(numeric_cols) > 10 else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #7c3aed; margin-bottom: 1rem; font-size: 1.1rem; font-weight: 600;">Categorical Features ({len(categorical_cols)})</h3>
            {''.join([f'<div class="feature-badge">{col}</div>' for col in categorical_cols[:10]])}
            {f'<div style="color: #64748b; margin-top: 1rem; font-size: 0.875rem;">...and {len(categorical_cols)-10} more</div>' if len(categorical_cols) > 10 else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #10b981; margin-bottom: 1rem; font-size: 1.1rem; font-weight: 600;">Datetime Features ({len(datetime_cols)})</h3>
            {''.join([f'<div class="feature-badge">{col}</div>' for col in datetime_cols[:10]])}
            {f'<div style="color: #64748b; margin-top: 1rem; font-size: 0.875rem;">...and {len(datetime_cols)-10} more</div>' if len(datetime_cols) > 10 else ''}
        </div>
        """, unsafe_allow_html=True)
    
    if numeric_cols:
        st.markdown('<div class="section-title">Statistical Summary</div>', unsafe_allow_html=True)
        st.dataframe(df[numeric_cols].describe().T.style.background_gradient(cmap='Blues', subset=['mean', 'std']), 
                     use_container_width=True, height=400)
        
        st.markdown('<div class="section-title">Skewness Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><strong>Analysis Insight:</strong> Highly skewed features should be transformed using PowerTransformer for better model performance. Normal distribution (|skewness| < 0.5) works best for most algorithms.</div>', unsafe_allow_html=True)
        
        skewness_df = analyze_skewness(df, numeric_cols)
        st.dataframe(skewness_df, use_container_width=True, height=400)
        
        highly_skewed = skewness_df[skewness_df['Status'].str.contains('High')]
        if len(highly_skewed) > 0:
            st.markdown(f'<div class="warning-box"><strong>Alert:</strong> {len(highly_skewed)} feature(s) are highly skewed. Enable PowerTransformer in the pipeline to fix this automatically.</div>', unsafe_allow_html=True)
    
    if categorical_cols:
        st.markdown('<div class="section-title">Categorical Features Analysis</div>', unsafe_allow_html=True)
        cat_stats = pd.DataFrame({
            'Column': categorical_cols,
            'Unique Values': [df[col].nunique() for col in categorical_cols],
            'Most Frequent': [df[col].mode()[0] if len(df[col].mode()) > 0 else None for col in categorical_cols],
            'Frequency': [df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0 for col in categorical_cols],
            'Frequency %': [(df[col].value_counts().iloc[0] / len(df) * 100) if len(df[col]) > 0 else 0 for col in categorical_cols]
        })
        cat_stats['Frequency %'] = cat_stats['Frequency %'].round(2)
        st.dataframe(cat_stats, use_container_width=True, height=400)
    
    st.markdown('<div class="section-title">Data Quality Report</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Values Analysis")
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum().values,
            'Percentage': (df.isnull().sum() / len(df) * 100).round(2).values
        }).sort_values('Percentage', ascending=False)
        
        missing_df = missing_df[missing_df['Missing'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#ef4444' if x > 50 else '#f59e0b' if x > 20 else '#10b981' for x in missing_df['Percentage']]
            ax.barh(missing_df['Column'], missing_df['Percentage'], color=colors)
            ax.set_xlabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
            ax.set_title('Missing Values Distribution', fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.markdown('<div class="success-box"><strong>Excellent!</strong> No missing values detected in any column.</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Outlier Detection (IQR Method)")
        if numeric_cols:
            outlier_data = []
            for col in numeric_cols:
                n_outliers, lower, upper = detect_outliers(df, col)
                pct = (n_outliers/len(df)*100)
                status = "High" if pct > 10 else "Moderate" if pct > 5 else "Low"
                outlier_data.append({
                    'Column': col,
                    'Status': status,
                    'Count': n_outliers,
                    'Percentage': f"{pct:.2f}%"
                })
            outlier_df = pd.DataFrame(outlier_data)
            st.dataframe(outlier_df, use_container_width=True)
            
            st.markdown('<div class="info-box"><strong>Note:</strong> Outliers are NOT removed in pipelines (cannot be applied to new data). Use RobustScaler if outliers are present.</div>', unsafe_allow_html=True)
    
    if numeric_cols and len(numeric_cols) > 1:
        st.markdown('<div class="section-title">Correlation Analysis</div>', unsafe_allow_html=True)
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                   vmin=-1, vmax=1)
        ax.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': round(corr_matrix.iloc[i, j], 3),
                        'Strength': 'Very Strong' if abs(corr_matrix.iloc[i, j]) > 0.9 else 'Strong'
                    })
        
        if high_corr:
            st.markdown("#### Highly Correlated Features (|r| > 0.7)")
            st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
            st.markdown('<div class="warning-box"><strong>Multicollinearity Alert:</strong> Consider removing one feature from each highly correlated pair to improve model performance.</div>', unsafe_allow_html=True)
    
    if numeric_cols:
        st.markdown('<div class="section-title">Distribution Analysis</div>', unsafe_allow_html=True)
        
        selected_cols = st.multiselect(
            "Select features to visualize:",
            numeric_cols,
            default=numeric_cols[:min(4, len(numeric_cols))],
            key="dist_viz"
        )
        
        if selected_cols:
            n_cols = 2
            n_rows = (len(selected_cols) + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
            
            for idx, col in enumerate(selected_cols):
                axes[idx].hist(df[col].dropna(), bins=40, edgecolor='white', alpha=0.8, 
                             color='#2563eb', linewidth=0.5)
                axes[idx].set_title(f'{col}', fontsize=14, fontweight='bold', pad=10)
                axes[idx].set_xlabel(col, fontsize=11)
                axes[idx].set_ylabel('Frequency', fontsize=11)
                axes[idx].grid(alpha=0.3, linestyle='--')
                axes[idx].spines['top'].set_visible(False)
                axes[idx].spines['right'].set_visible(False)
            
            for idx in range(len(selected_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)

def build_pipeline(config, numeric_cols, categorical_cols, task_type):
    """Build production-ready sklearn pipeline with ColumnTransformer"""
    
    numeric_steps = []
    
    if config['handle_missing']:
        strategy = config['numeric_imputation']
        numeric_steps.append(('imputer', SimpleImputer(strategy=strategy)))
    
    if config['fix_skewness']:
        numeric_steps.append(('power', PowerTransformer(method='yeo-johnson', standardize=True)))
    
    elif config['scale_features']:
        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        numeric_steps.append(('scaler', scaler_map[config['scaling_method']]))
    
    numeric_transformer = Pipeline(steps=numeric_steps) if numeric_steps else 'passthrough'
    
    categorical_steps = []
    
    if config['handle_missing']:
        strategy = 'most_frequent' if config['categorical_imputation'] == 'mode' else 'constant'
        fill_value = 'Unknown' if config['categorical_imputation'] == 'unknown' else None
        categorical_steps.append(('imputer', SimpleImputer(strategy=strategy, fill_value=fill_value)))
    
    if config['encode_categorical']:
        from sklearn.preprocessing import OneHotEncoder
        categorical_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)))
    
    categorical_transformer = Pipeline(steps=categorical_steps) if categorical_steps else 'passthrough'
    
    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    return preprocessor

def train_model_pipeline(df, config, target_col, task_type):
    """Train model with full pipeline"""
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    label_encoder = None
    if task_type == 'classification':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        st.session_state.label_encoder = label_encoder
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['test_size'], random_state=42, 
            stratify=y if task_type == 'classification' else None
        )
    except ValueError:
        st.warning("Stratification failed (rare classes detected). Splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config['test_size'], random_state=42
        )
    
    numeric_cols, categorical_cols, _ = identify_column_types(X_train)
    
    preprocessor = build_pipeline(config, numeric_cols, categorical_cols, task_type)
    
    if task_type == 'classification':
        if config['model_type'] == 'Random Forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        elif config['model_type'] == 'Logistic Regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'model__C': [0.01, 0.1, 1.0, 10.0],
                'model__penalty': ['l2']
            }
        else:
            model = SVC(random_state=42)
            param_grid = {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto']
            }
    else:
        if config['model_type'] == 'Random Forest':
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            param_grid = {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            }
        elif config['model_type'] == 'Ridge':
            model = Ridge(random_state=42)
            param_grid = {
                'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            }
        else:
            model = SVR()
            param_grid = {
                'model__C': [0.1, 1.0, 10.0],
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto']
            }
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    if config['use_grid_search']:
        scoring = 'f1_weighted' if task_type == 'classification' else 'r2'
        
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, 
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_scores = grid_search.cv_results_['mean_test_score']
        best_cv_score = grid_search.best_score_
    else:
        pipeline.fit(X_train, y_train)
        best_pipeline = pipeline
        best_params = None
        scoring = 'f1_weighted' if task_type == 'classification' else 'r2'
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring)
        best_cv_score = cv_scores.mean()
    
    y_pred = best_pipeline.predict(X_test)
    
    results = {
        'pipeline': best_pipeline,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'task_type': task_type,
        'label_encoder': label_encoder,
        'best_params': best_params,
        'cv_scores': cv_scores,
        'best_cv_score': best_cv_score
    }
    
    if task_type == 'classification':
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        results['classification_report'] = classification_report(y_test, y_pred, 
                                                                 target_names=label_encoder.classes_ if label_encoder else None)
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    else:
        results['r2'] = r2_score(y_test, y_pred)
        results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        results['mae'] = np.mean(np.abs(y_test - y_pred))
    
    return results

# Main App Header
st.markdown('<div class="main-title">ML Pipeline Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Production-ready machine learning pipelines with best practices built-in</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## Configuration")
    
    uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.markdown(f'<div class="success-box"><strong>Dataset Loaded:</strong> {uploaded_file.name}<br>Dimensions: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    st.divider()
    
    if st.session_state.df is not None:
        st.markdown("### Pipeline Settings")
        
        with st.expander("Missing Values", expanded=True):
            handle_missing = st.checkbox("Handle Missing Values", value=True, 
                                        help="Automatically impute missing values")
            numeric_imputation = st.selectbox("Numerical Strategy:", 
                                             ['mean', 'median'],
                                             help="Mean: symmetric distributions\nMedian: skewed or with outliers")
            categorical_imputation = st.selectbox("Categorical Strategy:", 
                                                 ['mode', 'unknown'],
                                                 help="Mode: most frequent value\nUnknown: new category")
        
        with st.expander("Transformations"):
            fix_skewness = st.checkbox("Fix Skewness (PowerTransformer)", value=True,
                                      help="Automatically normalizes skewed distributions")
            st.caption("Recommended for features with |skewness| > 1")
            
            if not fix_skewness:
                scale_features = st.checkbox("Scale Features", value=True)
                if scale_features:
                    scaling_method = st.selectbox("Scaling Method:", 
                                                 ['standard', 'robust', 'minmax'],
                                                 help="Standard: SVM/Logistic\nRobust: with outliers\nMinMax: Neural nets")
            else:
                scale_features = False
                scaling_method = 'standard'
                st.info("PowerTransformer includes standardization")
        
        with st.expander("Encoding"):
            encode_categorical = st.checkbox("One-Hot Encode Categorical", value=True,
                                           help="Convert categories to binary columns")
        
        config = {
            'handle_missing': handle_missing,
            'numeric_imputation': numeric_imputation,
            'categorical_imputation': categorical_imputation,
            'fix_skewness': fix_skewness,
            'scale_features': scale_features,
            'scaling_method': scaling_method,
            'encode_categorical': encode_categorical
        }
        
        st.divider()
        st.markdown("### Quick Stats")
        st.metric("Dataset Size", f"{st.session_state.df.shape[0]:,} × {st.session_state.df.shape[1]}")
        st.metric("Missing Values", f"{st.session_state.df.isnull().sum().sum():,}")
        st.metric("Duplicates", f"{st.session_state.df.duplicated().sum():,}")

# Main Content
if st.session_state.df is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["Exploratory Analysis", "Build Pipeline", "Model Performance", "Deploy"])
    
    with tab1:
        perform_eda(st.session_state.df)
    
    with tab2:
        st.markdown('<div class="section-title">Build ML Pipeline</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h4 style="margin-top: 0;">What makes this pipeline production-ready?</h4>
        <ul style="margin-bottom: 0;">
            <li><strong>No Data Leakage:</strong> All transformers fit only on training data</li>
            <li><strong>ColumnTransformer:</strong> Separate handling for numeric/categorical features</li>
            <li><strong>PowerTransformer:</strong> Fixes skewness automatically (better than log)</li>
            <li><strong>GridSearchCV:</strong> Automatic hyperparameter tuning with cross-validation</li>
            <li><strong>Reusable:</strong> Save once, use everywhere with joblib</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Step 1: Define Your Task")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox("Select Target Variable:", 
                                     st.session_state.df.columns,
                                     help="The column you want to predict")
            
            if st.session_state.df[target_col].dtype == 'object' or st.session_state.df[target_col].nunique() < 20:
                default_task = 0
            else:
                default_task = 1
            
            task_type = st.radio("Task Type:", 
                                ['classification', 'regression'],
                                index=default_task,
                                help="Classification: predict categories\nRegression: predict numbers")
        
        with col2:
            if task_type == 'classification':
                st.markdown("#### Target Distribution")
                target_counts = st.session_state.df[target_col].value_counts()
                st.bar_chart(target_counts)
                
                if len(target_counts) > 10:
                    st.warning(f"{len(target_counts)} classes detected. Consider grouping rare classes.")
                
                imbalance_ratio = target_counts.max() / target_counts.min()
                if imbalance_ratio > 3:
                    st.warning(f"Class imbalance detected (ratio: {imbalance_ratio:.1f}:1). F1-score will be used instead of accuracy.")
            else:
                st.markdown("#### Target Distribution")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(st.session_state.df[target_col].dropna(), bins=30, color='#2563eb', edgecolor='white')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)
                st.pyplot(fig)
        
        st.markdown("### Step 2: Choose Model & Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if task_type == 'classification':
                model_type = st.selectbox("Model:", 
                                         ['Random Forest', 'Logistic Regression', 'SVM'],
                                         help="Random Forest: robust, handles non-linear\nLogistic: interpretable, fast\nSVM: powerful for complex patterns")
            else:
                model_type = st.selectbox("Model:", 
                                         ['Random Forest', 'Ridge', 'SVR'],
                                         help="Random Forest: robust, handles non-linear\nRidge: regularized linear\nSVR: powerful for complex patterns")
        
        with col2:
            test_size = st.slider("Test Split:", 0.1, 0.4, 0.2, 0.05,
                                 help="Proportion of data for testing")
            st.caption(f"Train: {int((1-test_size)*100)}% | Test: {int(test_size*100)}%")
        
        with col3:
            use_grid_search = st.checkbox("GridSearchCV", value=True,
                                         help="Automatically find best hyperparameters")
            if use_grid_search:
                st.info("5-fold cross-validation enabled")
        
        train_config = {**config, 'model_type': model_type, 'test_size': test_size, 'use_grid_search': use_grid_search}
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Train Pipeline", type="primary", use_container_width=True):
                with st.spinner("Training pipeline... This may take a minute."):
                    try:
                        results = train_model_pipeline(st.session_state.df, train_config, target_col, task_type)
                        st.session_state.model_results = results
                        st.session_state.pipeline = results['pipeline']
                        
                        st.markdown(f'<div class="success-box"><strong>Pipeline trained successfully!</strong><br>CV Score: {results["best_cv_score"]:.4f}</div>', unsafe_allow_html=True)
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
                        st.exception(e)
    
    with tab3:
        if st.session_state.model_results:
            results = st.session_state.model_results
            
            st.markdown('<div class="section-title">Model Performance Report</div>', unsafe_allow_html=True)
            
            if results['best_params']:
                st.markdown("### Optimal Hyperparameters")
                
                params_display = {k.replace('model__', ''): v for k, v in results['best_params'].items()}
                
                cols = st.columns(len(params_display))
                for idx, (param, value) in enumerate(params_display.items()):
                    with cols[idx]:
                        st.metric(param, value)
            
            st.markdown("### Cross-Validation Results")
            st.markdown(f'<div class="info-box"><strong>CV Score:</strong> {results["best_cv_score"]:.4f} (average across 5 folds)<br>This score is more reliable than a single train/test split.</div>', unsafe_allow_html=True)
            
            if results['task_type'] == 'classification':
                st.markdown("### Classification Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{results['accuracy']:.4f}")
                with col2:
                    st.metric("F1-Score", f"{results['f1_score']:.4f}", 
                             help="Better metric for imbalanced datasets")
                with col3:
                    st.metric("CV Score", f"{results['best_cv_score']:.4f}")
                with col4:
                    st.metric("Test Samples", len(results['y_test']))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Detailed Report")
                    st.text(results['classification_report'])
                
                with col2:
                    st.markdown("#### Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    cm = results['confusion_matrix']
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               cbar_kws={'label': 'Count'},
                               xticklabels=st.session_state.label_encoder.classes_ if st.session_state.label_encoder else 'auto',
                               yticklabels=st.session_state.label_encoder.classes_ if st.session_state.label_encoder else 'auto')
                    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
                    ax.set_xlabel('Predicted Label', fontsize=11)
                    ax.set_ylabel('True Label', fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                if 'Random Forest' in train_config['model_type']:
                    st.markdown("### Feature Importance")
                    
                    model = results['pipeline'].named_steps['model']
                    preprocessor = results['pipeline'].named_steps['preprocessor']
                    feature_names = []
                    
                    if 'num' in [name for name, _, _ in preprocessor.transformers_]:
                        num_features = preprocessor.transformers_[0][2]
                        feature_names.extend(num_features)
                    
                    if 'cat' in [name for name, _, _ in preprocessor.transformers_]:
                        cat_transformer = preprocessor.named_transformers_['cat']
                        if 'onehot' in cat_transformer.named_steps:
                            cat_features = cat_transformer.named_steps['onehot'].get_feature_names_out()
                            feature_names.extend(cat_features)
                    
                    if len(feature_names) == len(model.feature_importances_):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#2563eb')
                        ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
                        ax.set_title('Top 15 Important Features', fontsize=14, fontweight='bold', pad=15)
                        ax.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
            
            else:
                st.markdown("### Regression Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{results['r2']:.4f}",
                             help="1.0 = perfect predictions, 0.0 = baseline")
                with col2:
                    st.metric("RMSE", f"{results['rmse']:.4f}",
                             help="Root Mean Squared Error")
                with col3:
                    st.metric("MAE", f"{results['mae']:.4f}",
                             help="Mean Absolute Error")
                with col4:
                    st.metric("Test Samples", len(results['y_test']))
                
                st.markdown("### Predictions vs Actual Values")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(results['y_test'], results['y_pred'], alpha=0.6, color='#2563eb', s=50)
                    
                    min_val = min(results['y_test'].min(), results['y_pred'].min())
                    max_val = max(results['y_test'].max(), results['y_pred'].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                    
                    ax.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
                    ax.set_title('Predictions vs Actual', fontsize=14, fontweight='bold', pad=15)
                    ax.legend()
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    residuals = results['y_test'] - results['y_pred']
                    ax.scatter(results['y_pred'], residuals, alpha=0.6, color='#7c3aed', s=50)
                    ax.axhline(y=0, color='r', linestyle='--', lw=2)
                    ax.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
                    ax.set_ylabel('Residuals', fontsize=11, fontweight='bold')
                    ax.set_title('Residual Plot', fontsize=14, fontweight='bold', pad=15)
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                if 'Random Forest' in train_config['model_type']:
                    st.markdown("### Feature Importance")
                    
                    model = results['pipeline'].named_steps['model']
                    preprocessor = results['pipeline'].named_steps['preprocessor']
                    feature_names = []
                    
                    if 'num' in [name for name, _, _ in preprocessor.transformers_]:
                        num_features = preprocessor.transformers_[0][2]
                        feature_names.extend(num_features)
                    
                    if 'cat' in [name for name, _, _ in preprocessor.transformers_]:
                        cat_transformer = preprocessor.named_transformers_['cat']
                        if 'onehot' in cat_transformer.named_steps:
                            cat_features = cat_transformer.named_steps['onehot'].get_feature_names_out()
                            feature_names.extend(cat_features)
                    
                    if len(feature_names) == len(model.feature_importances_):
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['Feature'], importance_df['Importance'], color='#7c3aed')
                        ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
                        ax.set_title('Top 15 Important Features', fontsize=14, fontweight='bold', pad=15)
                        ax.grid(axis='x', alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
        
        else:
            st.markdown("""
            <div class="info-box" style="text-align: center; padding: 3rem;">
                <h2>No Model Trained Yet</h2>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    Go to the <strong>"Build Pipeline"</strong> tab to train your first model!
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="section-title">Deploy Your Pipeline</div>', unsafe_allow_html=True)
        
        if st.session_state.pipeline:
            st.markdown('<div class="success-box"><strong>Pipeline Ready!</strong> Your model is trained and ready for deployment.</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Export Pipeline")
                
                pipeline_bytes = joblib.dump(st.session_state.pipeline, 'model_pipeline.pkl')
                
                with open('model_pipeline.pkl', 'rb') as f:
                    st.download_button(
                        label="Download Pipeline (.pkl)",
                        data=f,
                        file_name="ml_pipeline.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                
                if st.session_state.label_encoder:
                    joblib.dump(st.session_state.label_encoder, 'label_encoder.pkl')
                    with open('label_encoder.pkl', 'rb') as f:
                        st.download_button(
                            label="Download Label Encoder (.pkl)",
                            data=f,
                            file_name="label_encoder.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
            
            with col2:
                st.markdown("### Pipeline Summary")
                
                results = st.session_state.model_results
                st.metric("Task Type", results['task_type'].title())
                st.metric("Model", train_config['model_type'])
                
                if results['task_type'] == 'classification':
                    st.metric("F1-Score", f"{results['f1_score']:.4f}")
                else:
                    st.metric("R² Score", f"{results['r2']:.4f}")
            
            st.markdown("---")
            
            st.markdown("### Deployment Code")
            
            st.markdown("#### Python Script")
            
            code = f"""
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('ml_pipeline.pkl')
"""
            
            if st.session_state.label_encoder:
                code += """
label_encoder = joblib.load('label_encoder.pkl')
"""
            
            code += """
# Prepare new data (must have same columns as training data, excluding target)
new_data = pd.DataFrame({{
    # Your new data here
}})

# Make predictions
predictions = pipeline.predict(new_data)
"""
            
            if st.session_state.label_encoder:
                code += """
# Decode predictions to original labels
predictions_decoded = label_encoder.inverse_transform(predictions)
print(predictions_decoded)
"""
            else:
                code += """
print(predictions)
"""
            
            st.code(code, language='python')
            
            st.markdown("#### FastAPI Deployment Example")
            
            fastapi_code = """
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
pipeline = joblib.load('ml_pipeline.pkl')

class PredictionInput(BaseModel):
    # Define your input fields here
    pass

@app.post("/predict")
def predict(input_data: PredictionInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = pipeline.predict(df)
    return {{"prediction": prediction.tolist()}}
"""
            st.code(fastapi_code, language='python')
            
            st.markdown("""
            <div class="info-box">
            <h4>Deployment Checklist</h4>
            <ul>
                <li>Pipeline includes all preprocessing steps</li>
                <li>No data leakage (transformers fit on training data only)</li>
                <li>New data must have same features (excluding target)</li>
                <li>Missing values handled automatically by pipeline</li>
                <li>Categorical encoding handled automatically</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown("""
            <div class="warning-box" style="text-align: center; padding: 3rem;">
                <h2>No Pipeline to Deploy</h2>
                <p style="font-size: 1.1rem; margin-top: 1rem;">
                    Train a model first in the <strong>"Build Pipeline"</strong> tab!
                </p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">Welcome to ML Pipeline Studio</h1>
        <p style="font-size: 1.2rem; color: #64748b; margin-bottom: 2rem;">
            Build production-ready machine learning pipelines in minutes
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #2563eb;">Comprehensive EDA</h3>
            <ul style="margin-top: 1rem; color: #64748b;">
                <li>Automatic data quality checks</li>
                <li>Skewness analysis with recommendations</li>
                <li>Interactive visualizations</li>
                <li>Correlation analysis</li>
                <li>Missing value detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #7c3aed;">Smart Pipelines</h3>
            <ul style="margin-top: 1rem; color: #64748b;">
                <li>PowerTransformer for skewness</li>
                <li>ColumnTransformer for mixed types</li>
                <li>Prevents data leakage</li>
                <li>GridSearchCV for hyperparameters</li>
                <li>F1-score for imbalanced data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #10b981;">Production Ready</h3>
            <ul style="margin-top: 1rem; color: #64748b;">
                <li>One-file deployment with joblib</li>
                <li>FastAPI integration examples</li>
                <li>Cross-validation included</li>
                <li>Label encoder for inverse transform</li>
                <li>Comprehensive documentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box" style="text-align: center;">
        <h3>Getting Started</h3>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            Upload your dataset using the sidebar to begin building your ML pipeline!
        </p>
    </div>
    """, unsafe_allow_html=True)