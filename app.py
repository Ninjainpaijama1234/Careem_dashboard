import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# 1. Page Configuration
# ---------------------
st.set_page_config(
    page_title="Advanced RFM Intelligence Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for nicer UI
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# 2. Helper Functions
# -------------------

@st.cache_data
def generate_dummy_data():
    """Generates realistic dummy data if no file is uploaded."""
    np.random.seed(42)
    n_users = 1000
    
    dates = pd.date_range(end=dt.datetime.today(), periods=365)
    signup_dates = np.random.choice(dates, n_users)
    
    data = []
    for i in range(n_users):
        user_id = f"CUST_{1000+i}"
        signup_date = signup_dates[i]
        
        # Simulate Recency (days since last order)
        days_since = np.random.randint(1, 300)
        last_order = dt.datetime.today() - dt.timedelta(days=days_since)
        
        # Simulate Frequency & Monetary based on some correlation
        orders = np.random.randint(1, 50)
        # Higher orders usually means slightly higher spend, but varied
        spend = orders * np.random.uniform(10, 100) 
        
        # Additional metadata
        cuisine = np.random.choice(['Italian', 'Asian', 'Burgers', 'Healthy', 'Indian'], p=[0.3, 0.2, 0.2, 0.2, 0.1])
        platform = np.random.choice(['iOS', 'Android', 'Web'], p=[0.6, 0.3, 0.1])
        
        data.append([user_id, signup_date, last_order, days_since, orders, spend, cuisine, platform])
        
    df = pd.DataFrame(data, columns=['user_id', 'signup_date', 'last_order_date', 'days_since_last_order', 'total_orders', 'total_spend', 'favorite_cuisine', 'last_platform_used'])
    
    # Calculate a mock 'Customer Lifetime Value' for display
    df['customer_lifetime_value'] = df['total_spend'] * 1.2 # Simplified logic
    return df

@st.cache_data
def load_data(uploaded_file):
    """Loads data from CSV or generates dummy data."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        df = generate_dummy_data()

    # Standardization: Ensure columns exist or rename similar ones
    # (In a real app, you'd add column mapping logic here)
    
    # Date Conversion
    for col in ['signup_date', 'last_order_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    return df

def calculate_rfm(df, r_weight, f_weight, m_weight):
    """Calculates RFM scores, Weighted Scores, and Segments."""
    
    # 1. Calculate Quintiles (1-5)
    # Recency: Lower days = Higher Score (5)
    df['R_Rank'] = pd.qcut(df['days_since_last_order'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    # Frequency: Higher count = Higher Score (5)
    df['F_Rank'] = pd.qcut(df['total_orders'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    # Monetary: Higher spend = Higher Score (5)
    df['M_Rank'] = pd.qcut(df['total_spend'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # 2. Weighted Score
    df['RFM_Weighted_Score'] = (r_weight * df['R_Rank'] + f_weight * df['F_Rank'] + m_weight * df['M_Rank'])

    # 3. Text Segment Definition (More granular rule-based mapping)
    def define_segment(row):
        r, f, m = row['R_Rank'], row['F_Rank'], row['M_Rank']
        avg = (r + f + m) / 3
        
        if r >= 5 and f >= 5 and m >= 5:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r <= 2 and f >= 4:
            return 'At-Risk'
        elif r <= 2 and f <= 2:
            return 'Lost'
        elif row['RFM_Weighted_Score'] >= 4:
            return 'Potential Loyalists'
        elif 2 <= row['RFM_Weighted_Score'] < 4:
            return 'Needs Attention'
        else:
            return 'Hibernating'

    df['Segment'] = df.apply(define_segment, axis=1)
    
    return df

@st.cache_resource
def train_prediction_models(df):
    """Trains simple ML models for the Prediction tab."""
    # Prepare Features
    X = df[['R_Rank', 'F_Rank', 'M_Rank', 'days_since_last_order', 'total_orders', 'total_spend']]
    
    # 1. Churn Prediction (Classification)
    # Proxy target: If Recency Rank is 1 (Lowest), consider them 'Churned' for training purposes
    y_churn = (df['R_Rank'] == 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, max_depth=5)
    clf.fit(X_train, y_train)
    churn_acc = accuracy_score(y_test, clf.predict(X_test))

    # 2. Next Month Spend Prediction (Regression)
    # Proxy target: Total Spend * random noise to simulate future variation
    # In real life, you would use historical 'next month' data.
    y_spend = df['total_spend'] * np.random.uniform(0.8, 1.2, len(df))
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_spend, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=50, max_depth=5)
    reg.fit(X_train_r, y_train_r)
    
    return clf, reg, churn_acc

# 3. Sidebar Configuration
# ------------------------
with st.sidebar:
    st.image("https://placehold.co/300x100/000000/FFFFFF?text=Insight+Engine", use_container_width=True)
    st.header("📂 Data Import")
    uploaded_file = st.file_uploader("Upload CSV (or use auto-generated data)", type=["csv"])
    
    if uploaded_file is None:
        st.info("ℹ️ Using auto-generated demo data.")
    
    st.markdown("---")
    st.header("⚙️ RFM Weights")
    st.caption("Adjust importance for segmentation logic.")
    r_weight = st.slider("Recency (R)", 0.0, 2.0, 1.0, 0.1)
    f_weight = st.slider("Frequency (F)", 0.0, 2.0, 1.0, 0.1)
    m_weight = st.slider("Monetary (M)", 0.0, 2.0, 1.0, 0.1)
    
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")

# 4. Main Application
# -------------------
st.title("🧠 Advanced Customer Intelligence Hub")

# Load & Process
df_raw = load_data(uploaded_file)

if df_raw is not None:
    df_rfm = calculate_rfm(df_raw.copy(), r_weight, f_weight, m_weight)
    
    # Create Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Overview", 
        "🧩 Segmentation & Actions", 
        "👤 Customer Deep Dive",
        "🔮 Predictive Analytics"
    ])

    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.subheader("Business Pulse")
        
        # Top Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Active Customers", f"{df_rfm['user_id'].nunique():,}", delta="+5% vs last mo")
        m2.metric("Total Revenue", f"${df_rfm['total_spend'].sum():,.0f}")
        m3.metric("Avg Order Value", f"${df_rfm['total_spend'].mean():.2f}")
        m4.metric("Avg Frequency", f"{df_rfm['total_orders'].mean():.1f} orders")

        st.markdown("### 📈 Revenue & Activity Trends")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Income Distribution**")
            # Histogram of spend
            fig_hist = px.histogram(df_rfm, x="total_spend", nbins=30, 
                                    color_discrete_sequence=['#4CAF50'],
                                    title="Distribution of Customer Spend")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with c2:
            st.markdown("**Recency Heatmap**")
            # Altair heatmap of Recency vs Frequency
            heatmap = alt.Chart(df_rfm).mark_rect().encode(
                x=alt.X('R_Rank:O', title='Recency Score (1-5)'),
                y=alt.Y('F_Rank:O', title='Frequency Score (1-5)'),
                color=alt.Color('mean(total_spend):Q', scale=alt.Scale(scheme='greens'), title='Avg Spend'),
                tooltip=['count()', 'mean(total_spend)']
            ).properties(title="Where is the money coming from? (R vs F)").interactive()
            st.altair_chart(heatmap, use_container_width=True)

    # --- TAB 2: SEGMENTATION ---
    with tab2:
        st.subheader("Segment Analysis & Extraction")
        
        # 3D Scatter Plot for RFM
        st.markdown("**3D Visualization of RFM Clusters**")
        fig_3d = px.scatter_3d(
            df_rfm, x='days_since_last_order', y='total_orders', z='total_spend',
            color='Segment', opacity=0.7,
            hover_data=['user_id'],
            title="Recency vs Frequency vs Monetary (3D View)"
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=500)
        st.plotly_chart(fig_3d, use_container_width=True)

        # Segment Breakdown
        col_seg1, col_seg2 = st.columns([2, 1])
        
        with col_seg1:
            st.markdown("**Segment Size & Monetary Value**")
            segment_stats = df_rfm.groupby('Segment').agg({
                'user_id': 'count',
                'total_spend': 'sum'
            }).reset_index()
            
            # Dual Axis Chart
            base = alt.Chart(segment_stats).encode(x='Segment')
            bar = base.mark_bar(color='#a1d99b').encode(y='total_spend', tooltip=['Segment', 'total_spend'])
            line = base.mark_line(color='#31a354').encode(y='user_id')
            
            st.altair_chart((bar + line).interactive(), use_container_width=True)
            st.caption("Bars = Total Spend | Line = Number of Customers")

        with col_seg2:
            st.markdown("**Download Segments**")
            st.write("Export specific lists for email marketing.")
            
            for seg in df_rfm['Segment'].unique():
                seg_df = df_rfm[df_rfm['Segment'] == seg]
                csv = seg_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label=f"📥 {seg} ({len(seg_df)})",
                    data=csv,
                    file_name=f"segment_{seg.lower().replace(' ', '_')}.csv",
                    mime='text/csv',
                    key=f"dl_{seg}"
                )

    # --- TAB 3: DEEP DIVE ---
    with tab3:
        st.subheader("Single Customer 360° View")
        
        search_col, _ = st.columns([1, 2])
        with search_col:
            selected_user = st.selectbox("Search Customer ID", df_rfm['user_id'].unique())
        
        if selected_user:
            user_data = df_rfm[df_rfm['user_id'] == selected_user].iloc[0]
            
            # KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            k1.info(f"Segment: **{user_data['Segment']}**")
            k2.metric("RFM Weighted Score", f"{user_data['RFM_Weighted_Score']:.2f}")
            k3.metric("Lifetime Spend", f"${user_data['total_spend']:.2f}")
            k4.metric("Days Inactive", f"{user_data['days_since_last_order']} days")
            
            # Radar Chart Comparison
            st.markdown("#### vs. Average Customer")
            
            avg_metrics = df_rfm[['R_Rank', 'F_Rank', 'M_Rank']].mean()
            
            categories = ['Recency Rank', 'Frequency Rank', 'Monetary Rank']
            
            fig_radar = go.Figure()
            
            # User Data
            fig_radar.add_trace(go.Scatterpolar(
                r=[user_data['R_Rank'], user_data['F_Rank'], user_data['M_Rank']],
                theta=categories,
                fill='toself',
                name=f'Customer {selected_user}'
            ))
            
            # Average Data
            fig_radar.add_trace(go.Scatterpolar(
                r=[avg_metrics['R_Rank'], avg_metrics['F_Rank'], avg_metrics['M_Rank']],
                theta=categories,
                fill='toself',
                name='Average Customer',
                line_color='gray',
                opacity=0.5
            ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.markdown("#### Raw Data Record")
            st.dataframe(pd.DataFrame([user_data]))

    # --- TAB 4: PREDICTION (NEW) ---
    with tab4:
        st.subheader("🔮 Predictive Analytics (Beta)")
        st.markdown("""
        We use Random Forest models to estimate future behavior based on current RFM metrics.
        *Note: This is trained on the fly using the current dataset.*
        """)
        
        # Train Model Button (to save resources)
        if st.button("Train Prediction Models"):
            with st.spinner("Training AI Models..."):
                clf, reg, acc = train_prediction_models(df_rfm)
                
                # Make Predictions for whole dataset
                X_pred = df_rfm[['R_Rank', 'F_Rank', 'M_Rank', 'days_since_last_order', 'total_orders', 'total_spend']]
                df_rfm['Predicted_Churn_Prob'] = clf.predict_proba(X_pred)[:, 1]
                df_rfm['Predicted_Next_Month_Spend'] = reg.predict(X_pred)
                
                st.success(f"Models Trained! Churn Model Accuracy: {acc:.1%}")
                
                # Visualization of Predictions
                p1, p2 = st.columns(2)
                
                with p1:
                    st.markdown("**Customers Most Likely to Churn (>80% Prob)**")
                    high_risk = df_rfm[df_rfm['Predicted_Churn_Prob'] > 0.8].sort_values('total_spend', ascending=False).head(10)
                    st.dataframe(high_risk[['user_id', 'Segment', 'Predicted_Churn_Prob', 'total_spend']].style.format({
                        'Predicted_Churn_Prob': '{:.1%}',
                        'total_spend': '${:,.2f}'
                    }))
                    
                with p2:
                    st.markdown("**Predicted Highest Spenders Next Month**")
                    high_potential = df_rfm.sort_values('Predicted_Next_Month_Spend', ascending=False).head(10)
                    st.dataframe(high_potential[['user_id', 'Segment', 'Predicted_Next_Month_Spend']].style.format({
                        'Predicted_Next_Month_Spend': '${:,.2f}'
                    }))
                
                # Download Predictions
                st.markdown("---")
                csv_pred = df_rfm.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Dataset with Predictions",
                    data=csv_pred,
                    file_name="rfm_predictions.csv",
                    mime="text/csv"
                )
        else:
            st.info("Click the button above to train the machine learning models on your data.")

else:
    st.error("Could not load data. Please refresh the page.")
