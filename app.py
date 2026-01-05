import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
# ---------------------
st.set_page_config(
    page_title="InsightX: Ultimate RFM Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px;
        color: #555;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50;
        color: white;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Data & Helper Functions
# --------------------------

@st.cache_data
def generate_dummy_data():
    """Generates a rich, realistic dataset for demonstration."""
    np.random.seed(42)
    n_users = 2000  # Increased sample size
    
    dates = pd.date_range(end=dt.datetime.today(), periods=730)
    signup_dates = np.random.choice(dates, n_users)
    
    data = []
    for i in range(n_users):
        user_id = f"CUST_{10000+i}"
        signup_date = signup_dates[i]
        
        # Simulate specialized behavior patterns
        persona = np.random.choice(['Whale', 'Regular', 'Newbie', 'Lapsed'], p=[0.05, 0.4, 0.3, 0.25])
        
        if persona == 'Whale':
            orders = np.random.randint(20, 100)
            spend = orders * np.random.uniform(50, 200)
            days_since = np.random.randint(1, 20)
        elif persona == 'Regular':
            orders = np.random.randint(5, 30)
            spend = orders * np.random.uniform(20, 80)
            days_since = np.random.randint(5, 60)
        elif persona == 'Newbie':
            orders = np.random.randint(1, 5)
            spend = orders * np.random.uniform(15, 60)
            days_since = np.random.randint(1, 30)
        else: # Lapsed
            orders = np.random.randint(1, 15)
            spend = orders * np.random.uniform(10, 50)
            days_since = np.random.randint(90, 365)
            
        last_order = dt.datetime.today() - dt.timedelta(days=days_since)
        
        # Dimensions
        cuisine = np.random.choice(['Italian', 'Asian', 'Fast Food', 'Healthy', 'Mexican', 'Vegan'], 
                                 p=[0.25, 0.2, 0.2, 0.15, 0.15, 0.05])
        platform = np.random.choice(['iOS App', 'Android App', 'Website', 'Call Center'], 
                                  p=[0.5, 0.3, 0.15, 0.05])
        age_group = np.random.choice(['18-24', '25-34', '35-44', '45+'], p=[0.2, 0.4, 0.3, 0.1])
        
        data.append([user_id, signup_date, last_order, days_since, orders, spend, cuisine, platform, age_group])
        
    df = pd.DataFrame(data, columns=[
        'user_id', 'signup_date', 'last_order_date', 'days_since_last_order', 
        'total_orders', 'total_spend', 'favorite_cuisine', 'platform', 'age_group'
    ])
    
    # Customer Lifetime Value (Proxy)
    df['clv'] = df['total_spend'] * np.random.uniform(1.0, 1.5, n_users)
    return df

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file:
        try:
            return pd.read_csv(uploaded_file)
        except:
            return None
    return generate_dummy_data()

def calculate_rfm(df, r_w, f_w, m_w):
    # Quantiles
    df['R_Rank'] = pd.qcut(df['days_since_last_order'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df['F_Rank'] = pd.qcut(df['total_orders'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_Rank'] = pd.qcut(df['total_spend'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # Weighted Score
    df['RFM_Score'] = (r_w * df['R_Rank'] + f_w * df['F_Rank'] + m_w * df['M_Rank'])
    
    # Advanced Segmentation Logic
    def segment_customer(row):
        score = row['RFM_Score']
        r, f = row['R_Rank'], row['F_Rank']
        
        if r == 5 and f == 5: return 'Champions'
        if r >= 3 and f >= 4: return 'Loyal Customers'
        if r >= 4 and f <= 2: return 'Promising New'
        if r >= 3 and f <= 3: return 'Potential Loyalists'
        if r <= 2 and f >= 4: return 'At Risk'
        if r <= 2 and f <= 2: return 'Lost'
        if score > 3.5: return 'Need Attention'
        return 'Hibernating'

    df['Segment'] = df.apply(segment_customer, axis=1)
    return df

@st.cache_resource
def train_models(df):
    """Trains predictive models and returns feature importance."""
    X = df[['R_Rank', 'F_Rank', 'M_Rank', 'days_since_last_order', 'total_orders', 'total_spend']]
    
    # 1. Churn Prediction
    y_churn = (df['R_Rank'] <= 2).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_churn, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    # 2. Spend Prediction
    y_spend = df['total_spend'] * np.random.uniform(0.9, 1.3, len(df))
    reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    reg.fit(X, y_spend)
    
    return clf, reg, acc, X.columns

@st.cache_resource
def find_similar_customers(df, current_user_id, n_neighbors=5):
    """Finds similar customers using NearestNeighbors."""
    features = ['days_since_last_order', 'total_orders', 'total_spend', 'R_Rank', 'F_Rank', 'M_Rank']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='ball_tree').fit(X_scaled)
    
    # Find index of current user
    try:
        user_idx = df[df['user_id'] == current_user_id].index[0]
        distances, indices = nbrs.kneighbors([X_scaled[user_idx]])
        
        # Exclude self
        similar_indices = indices[0][1:]
        return df.iloc[similar_indices]
    except:
        return pd.DataFrame()

# 3. Sidebar
# ----------
with st.sidebar:
    st.image("https://placehold.co/300x120/2c3e50/FFFFFF?text=InsightX+AI", use_container_width=True)
    st.markdown("### 🎛️ Control Panel")
    
    uploaded_file = st.file_uploader("Data Source", type=["csv"])
    if not uploaded_file:
        st.caption("🚀 Running on Synthetic Engine")

    st.divider()
    
    st.markdown("### ⚖️ RFM Weights")
    r_w = st.slider("Recency Impact", 0.0, 2.0, 1.2, help="Higher weight prioritizes recent activity")
    f_w = st.slider("Frequency Impact", 0.0, 2.0, 0.8)
    m_w = st.slider("Monetary Impact", 0.0, 2.0, 1.0)
    
    st.divider()
    st.info("💡 **Pro Tip:** Check the 'Predictive Analytics' tab to forecast next month's revenue.")

# 4. Main App Logic
# -----------------
df_raw = load_data(uploaded_file)
if df_raw is not None:
    df = calculate_rfm(df_raw.copy(), r_w, f_w, m_w)
    
    st.title("InsightX: Customer Intelligence Platform")
    
    tabs = st.tabs([
        "📊 Executive Overview", 
        "🧩 Advanced Segmentation", 
        "👤 360° Customer View", 
        "🔮 AI & Future Analytics"
    ])
    
    # --- TAB 1: EXECUTIVE OVERVIEW ---
    with tabs[0]:
        # Top Level KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_rev = df['total_spend'].sum()
        kpi1.metric("Total Revenue", f"${total_rev:,.0f}", "+12.5%")
        kpi2.metric("Active Users", f"{df['user_id'].nunique():,}", "+34")
        kpi3.metric("Avg Order Value", f"${df['total_spend'].mean():.2f}", "-2.1%")
        kpi4.metric("Churn Rate (Est)", f"{(len(df[df['Segment']=='Lost'])/len(df))*100:.1f}%", "-0.5%")
        
        st.markdown("---")
        
        # Row 2: Visualizations
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.subheader("📈 User Growth Trajectory")
            # Aggregate signups by month
            df_growth = df.set_index('signup_date').resample('M')['user_id'].count().reset_index()
            df_growth['cumulative'] = df_growth['user_id'].cumsum()
            
            fig_area = px.area(df_growth, x='signup_date', y='cumulative', 
                               labels={'cumulative': 'Total Users', 'signup_date': 'Date'},
                               color_discrete_sequence=['#2ecc71'])
            fig_area.update_layout(hovermode="x unified", height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_area, use_container_width=True)
            
        with c2:
            st.subheader("🍕 Market Composition")
            # Sunburst Chart: Platform -> Cuisine
            fig_sun = px.sunburst(df, path=['platform', 'favorite_cuisine'], values='total_spend',
                                  color='total_spend', color_continuous_scale='RdBu')
            fig_sun.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_sun, use_container_width=True)

        # Row 3: Cohort/Heatmap Analysis style
        st.subheader("🔥 RFM Correlation Heatmap")
        st.caption("How do Frequency and Monetary Value correlate? (Color = Recency Rank)")
        
        fig_bubble = px.scatter(df, x='total_orders', y='total_spend', 
                                color='R_Rank', size='total_spend',
                                hover_data=['user_id', 'Segment'],
                                color_continuous_scale='Viridis')
        fig_bubble.update_layout(height=400)
        st.plotly_chart(fig_bubble, use_container_width=True)

    # --- TAB 2: SEGMENTATION ---
    with tabs[1]:
        st.subheader("🧩 Strategic Segmentation Architecture")
        
        c_seg1, c_seg2 = st.columns([2, 1])
        
        with c_seg1:
            st.markdown("**Segment Distribution (Treemap)**")
            # Treemap is excellent for hierarchical data or categorical size comparison
            seg_tree_data = df.groupby('Segment').agg({'user_id':'count', 'total_spend':'sum'}).reset_index()
            fig_tree = px.treemap(seg_tree_data, path=['Segment'], values='total_spend',
                                  color='user_id', color_continuous_scale='Blues',
                                  hover_data=['user_id'],
                                  title="Size by Revenue (Color = User Count)")
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with c_seg2:
            st.markdown("**Segment Performance**")
            st.dataframe(
                df.groupby('Segment')[['total_spend', 'total_orders']].mean().sort_values('total_spend', ascending=False).style.format("${:,.2f}"),
                height=350
            )

        st.markdown("---")
        st.subheader("🔄 Customer Journey Flow (Parallel Categories)")
        st.caption("Trace the path: Platform -> Age Group -> Segment")
        
        fig_parcat = px.parallel_categories(df, dimensions=['platform', 'age_group', 'Segment'],
                                            color='total_spend', color_continuous_scale=px.colors.sequential.Inferno)
        fig_parcat.update_layout(height=500)
        st.plotly_chart(fig_parcat, use_container_width=True)
        
        # Interactive ROI Simulator
        st.markdown("---")
        with st.expander("💰 Marketing Campaign ROI Simulator", expanded=True):
            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                target_seg = st.selectbox("Target Segment", df['Segment'].unique())
            with sim_c2:
                budget = st.number_input("Campaign Budget ($)", 1000, 50000, 5000)
            with sim_c3:
                conv_rate = st.slider("Est. Conversion Rate (%)", 1, 20, 5) / 100
                
            target_users = df[df['Segment'] == target_seg]
            avg_ticket = target_users['total_spend'].mean() / target_users['total_orders'].mean() if target_users['total_orders'].mean() > 0 else 0
            
            est_revenue = len(target_users) * conv_rate * avg_ticket
            roi = (est_revenue - budget) / budget * 100
            
            st.metric("Estimated ROI", f"{roi:.1f}%", delta_color="normal" if roi > 0 else "inverse")
            st.write(f"Targeting **{len(target_users)}** users in '{target_seg}'. Expected Revenue: **${est_revenue:,.2f}**")

    # --- TAB 3: DEEP DIVE ---
    with tabs[2]:
        st.subheader("👤 Individual Customer Intelligence")
        
        col_search, col_profile = st.columns([1, 3])
        with col_search:
            uid = st.selectbox("Select Customer", df['user_id'].unique())
            user = df[df['user_id'] == uid].iloc[0]
            
            st.markdown("### Next Best Action")
            if user['Segment'] == 'Champions':
                st.success("🌟 **Upsell**: Offer Premium Membership")
            elif user['Segment'] == 'At Risk':
                st.error("🛑 **Retain**: Send 20% Off Coupon")
            elif user['Segment'] == 'Loyal Customers':
                st.info("💬 **Engage**: Request Product Review")
            else:
                st.warning("📢 **Nurture**: Send Educational Content")
                
        with col_profile:
            # Profile Header
            prof_c1, prof_c2, prof_c3 = st.columns(3)
            prof_c1.metric("Lifetime Value", f"${user['total_spend']:,.2f}")
            prof_c2.metric("Orders", user['total_orders'])
            prof_c3.metric("Last Seen", f"{user['days_since_last_order']} days ago")
            
            # Radar Chart
            st.markdown("**Trait Comparison vs Average**")
            avgs = df.mean(numeric_only=True)
            
            categories = ['Recency Score', 'Frequency Score', 'Monetary Score']
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[user['R_Rank'], user['F_Rank'], user['M_Rank']],
                theta=categories, fill='toself', name='This User'
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=[avgs['R_Rank'], avgs['F_Rank'], avgs['M_Rank']],
                theta=categories, fill='toself', name='Average', line=dict(dash='dash')
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=300)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        st.subheader("👯 Similar Customers (Look-alikes)")
        st.caption("Users with similar spending habits, frequency, and recency.")
        
        similar_df = find_similar_customers(df, uid)
        if not similar_df.empty:
            st.dataframe(
                similar_df[['user_id', 'Segment', 'total_spend', 'total_orders', 'favorite_cuisine']],
                use_container_width=True
            )
        else:
            st.warning("Not enough data to find neighbors.")

    # --- TAB 4: PREDICTIVE ANALYTICS ---
    with tabs[3]:
        st.subheader("🔮 Predictive AI Engine")
        
        if st.button("🚀 Train & Run Models", type="primary"):
            with st.spinner("Training Random Forest Models..."):
                clf, reg, acc, feature_names = train_models(df)
                
                # Predictions
                X_pred = df[['R_Rank', 'F_Rank', 'M_Rank', 'days_since_last_order', 'total_orders', 'total_spend']]
                df['Prob_Churn'] = clf.predict_proba(X_pred)[:, 1]
                df['Pred_Spend_Next_Mo'] = reg.predict(X_pred) / 12  # Rough monthly estimate
                
            st.success(f"Training Complete. Model Accuracy: {acc:.1%}")
            
            # Feature Importance
            st.markdown("### 🧠 What drives Customer Churn?")
            feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': clf.feature_importances_})
            fig_imp = px.bar(feat_imp.sort_values('Importance', ascending=True), 
                             x='Importance', y='Feature', orientation='h',
                             title="Feature Importance (Churn Model)")
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Predictions Table
            c_pred1, c_pred2 = st.columns(2)
            
            with c_pred1:
                st.markdown("#### 🚨 High Flight Risk (>80%)")
                risk_df = df[df['Prob_Churn'] > 0.8].sort_values('total_spend', ascending=False).head(10)
                st.dataframe(risk_df[['user_id', 'Segment', 'Prob_Churn', 'total_spend']].style.background_gradient(subset=['Prob_Churn'], cmap='Reds'))
                
            with c_pred2:
                st.markdown("#### 💎 Predicted Top Spenders (Next Month)")
                spend_df = df.sort_values('Pred_Spend_Next_Mo', ascending=False).head(10)
                st.dataframe(spend_df[['user_id', 'Segment', 'Pred_Spend_Next_Mo']].style.format({'Pred_Spend_Next_Mo': '${:,.2f}'}))
                
            # Distribution of Churn Probability
            fig_dist = px.histogram(df, x='Prob_Churn', nbins=50, title="Distribution of Churn Probability Across Base",
                                    color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig_dist, use_container_width=True)
            
        else:
            st.info("Click the button above to initialize the Machine Learning pipeline.")
            
            st.markdown("### Methodology")
            st.markdown("""
            1. **Churn Model**: Random Forest Classifier trained on Recency thresholds.
            2. **Revenue Model**: Random Forest Regressor trained on historical spending patterns.
            3. **Look-alikes**: K-Nearest Neighbors (KNN) algorithm to find vectors in 6D space.
            """)
