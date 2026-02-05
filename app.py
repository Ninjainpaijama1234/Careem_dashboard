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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Page Configuration
# ---------------------
st.set_page_config(
    page_title="InsightX: Careem AI Challenge",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI (Careem-inspired accents)
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
        background-color: #00A550; /* Careem Green */
        color: white;
    }
    /* Buttons */
    div.stButton > button:first-child {
        background-color: #00A550;
        color: white;
        border-radius: 8px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# 2. Data & Helper Functions
# --------------------------

@st.cache_data
def generate_dummy_data():
    """Generates a rich, realistic dataset for demonstration."""
    np.random.seed(42)
    n_users = 3000
    
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
    
    # 1. Churn Prediction (Proxy: Recency Rank low)
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
    st.markdown("### 🚀 InsightX")
    st.caption("AI Challenge Demo by [Your Name]")
    
    uploaded_file = st.file_uploader("Data Source", type=["csv"])
    if not uploaded_file:
        st.info("Running on Synthetic Data")

    st.divider()
    
    st.markdown("### ⚖️ RFM Weights")
    r_w = st.slider("Recency Impact", 0.0, 2.0, 1.2)
    f_w = st.slider("Frequency Impact", 0.0, 2.0, 0.8)
    m_w = st.slider("Monetary Impact", 0.0, 2.0, 1.0)
    
    st.divider()
    st.success("Note: Tab 5 covers Challenge #1 (AI Campaign Architect). Tab 4 covers Challenge #2 (Predictive Engine).")

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
        "🔮 AI & Future Analytics",
        "📣 AI Campaign Architect" # NEW TAB
    ])
    
    # --- TAB 1: EXECUTIVE OVERVIEW ---
    with tabs[0]:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        total_rev = df['total_spend'].sum()
        kpi1.metric("Total Revenue", f"${total_rev:,.0f}", "+12.5%")
        kpi2.metric("Active Users", f"{df['user_id'].nunique():,}", "+34")
        kpi3.metric("Avg Order Value", f"${df['total_spend'].mean():.2f}", "-2.1%")
        kpi4.metric("Churn Rate (Est)", f"{(len(df[df['Segment']=='Lost'])/len(df))*100:.1f}%", "-0.5%")
        
        st.markdown("---")
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.subheader("📈 User Growth Trajectory")
            df_growth = df.set_index('signup_date').resample('M')['user_id'].count().reset_index()
            df_growth['cumulative'] = df_growth['user_id'].cumsum()
            fig_area = px.area(df_growth, x='signup_date', y='cumulative', color_discrete_sequence=['#00A550'])
            fig_area.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_area, use_container_width=True)
            
        with c2:
            st.subheader("🍕 Market Composition")
            fig_sun = px.sunburst(df, path=['platform', 'favorite_cuisine'], values='total_spend', color='total_spend')
            fig_sun.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_sun, use_container_width=True)

    # --- TAB 2: SEGMENTATION ---
    with tabs[1]:
        st.subheader("🧩 Strategic Segmentation Architecture")
        
        with st.expander("📥 **Bulk Segment Export (Download Center)**", expanded=True):
            st.info("Select a segment below to download the specific customer list.")
            segments = sorted(df['Segment'].unique())
            cols = st.columns(4) 
            for i, seg in enumerate(segments):
                seg_df = df[df['Segment'] == seg]
                csv = seg_df.to_csv(index=False).encode('utf-8')
                with cols[i % 4]:
                    st.download_button(
                        label=f"📥 {seg} ({len(seg_df)})",
                        data=csv,
                        file_name=f"segment_{seg.lower().replace(' ', '_')}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )

        st.markdown("---")
        c_seg1, c_seg2 = st.columns([2, 1])
        with c_seg1:
            fig_tree = px.treemap(df.groupby('Segment').agg({'user_id':'count', 'total_spend':'sum'}).reset_index(), 
                                  path=['Segment'], values='total_spend', color='user_id', color_continuous_scale='Mint')
            st.plotly_chart(fig_tree, use_container_width=True)
            
        with c_seg2:
            st.dataframe(df.groupby('Segment')[['total_spend', 'total_orders']].mean().sort_values('total_spend', ascending=False).style.format("${:,.2f}"), height=350)

    # --- TAB 3: DEEP DIVE ---
    with tabs[2]:
        col_search, col_profile = st.columns([1, 3])
        with col_search:
            uid = st.selectbox("Select Customer", df['user_id'].unique())
            user = df[df['user_id'] == uid].iloc[0]
            st.markdown("### Next Best Action")
            if user['Segment'] == 'Champions':
                st.success("🌟 Upsell: Premium Membership")
            elif user['Segment'] == 'At Risk':
                st.error("🛑 Retain: 20% Off Coupon")
            else:
                st.warning("📢 Nurture: Educate")
                
        with col_profile:
            prof_c1, prof_c2, prof_c3 = st.columns(3)
            prof_c1.metric("Lifetime Value", f"${user['total_spend']:,.2f}")
            prof_c2.metric("Orders", user['total_orders'])
            prof_c3.metric("Last Seen", f"{user['days_since_last_order']} days ago")
            
            avgs = df.mean(numeric_only=True)
            categories = ['Recency Score', 'Frequency Score', 'Monetary Score']
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=[user['R_Rank'], user['F_Rank'], user['M_Rank']], theta=categories, fill='toself', name='User'))
            fig_radar.add_trace(go.Scatterpolar(r=[avgs['R_Rank'], avgs['F_Rank'], avgs['M_Rank']], theta=categories, fill='toself', name='Avg'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=300)
            st.plotly_chart(fig_radar, use_container_width=True)

    # --- TAB 4: PREDICTIVE ANALYTICS (Enhanced for Challenge #2) ---
    with tabs[3]:
        st.subheader("🔮 Predictive Growth Engine")
        st.caption("Challenge #2: Predict which segment is most likely to churn or convert.")
        
        if st.button("🚀 Train & Run Models", type="primary"):
            with st.spinner("Training Random Forest Models..."):
                clf, reg, acc, feature_names = train_models(df)
                
                # Predictions
                X_pred = df[['R_Rank', 'F_Rank', 'M_Rank', 'days_since_last_order', 'total_orders', 'total_spend']]
                df['Prob_Churn'] = clf.predict_proba(X_pred)[:, 1]
                df['Pred_Spend_Next_Mo'] = reg.predict(X_pred) / 12
                
            st.success(f"Training Complete. Model Accuracy: {acc:.1%}")
            
            # --- NEW: SEGMENT LEVEL CHURN ANALYSIS ---
            st.markdown("---")
            st.markdown("### 📂 Segment-Level Intelligence")
            st.caption("Aggregated risk and value profiles to identify strategic focus areas.")
            
            # Calculate aggregate metrics
            seg_metrics = df.groupby('Segment').agg(
                avg_churn_prob=('Prob_Churn', 'mean'),
                avg_pred_spend=('Pred_Spend_Next_Mo', 'mean'),
                users=('user_id', 'nunique')
            ).reset_index()

            # Identify High Priority (High Value + High Risk)
            high_value_threshold = df['Pred_Spend_Next_Mo'].median()
            high_risk_threshold = 0.5
            priority_seg = seg_metrics[
                (seg_metrics['avg_pred_spend'] >= high_value_threshold) & 
                (seg_metrics['avg_churn_prob'] >= high_risk_threshold)
            ]
            
            col_seg_metrics, col_priority = st.columns(2)
            
            with col_seg_metrics:
                st.markdown("**Churn Risk by Segment**")
                st.dataframe(
                    seg_metrics.sort_values('avg_churn_prob', ascending=False),
                    column_config={
                        "avg_churn_prob": st.column_config.ProgressColumn(
                            "Avg Churn Prob", format="%.2f", min_value=0, max_value=1
                        ),
                        "avg_pred_spend": st.column_config.NumberColumn(
                            "Pred. Monthly Spend", format="$%.2f"
                        )
                    },
                    use_container_width=True
                )
                
            with col_priority:
                st.markdown("**🎯 Priority Focus: High Risk / High Value**")
                if not priority_seg.empty:
                    st.error(f"Attention Needed: {priority_seg['Segment'].tolist()}")
                    st.dataframe(priority_seg, use_container_width=True)
                else:
                    st.success("No segments currently meet the High Risk + High Value threshold.")

            # --- EXISTING VISUALIZATIONS ---
            st.markdown("---")
            st.markdown("### 🚦 Risk vs Reward Matrix")
            fig_risk = px.scatter(df, x='Prob_Churn', y='Pred_Spend_Next_Mo',
                                  color='Segment', size='total_spend',
                                  title="Predicted Churn Probability vs Predicted Future Spend",
                                  labels={'Prob_Churn': 'Probability of Churn (0-1)', 'Pred_Spend_Next_Mo': 'Predicted Next Month Spend'},
                                  hover_data=['user_id'])
            fig_risk.add_hline(y=df['Pred_Spend_Next_Mo'].mean(), line_dash="dash", line_color="grey")
            fig_risk.add_vline(x=0.5, line_dash="dash", line_color="red")
            st.plotly_chart(fig_risk, use_container_width=True)
            
        else:
            st.info("Click the button above to initialize the Machine Learning pipeline.")

    # --- TAB 5: AI CAMPAIGN ARCHITECT (New for Challenge #1) ---
    with tabs[4]:
        st.subheader("📣 AI Campaign Architect")
        st.caption("Challenge #1: Generate a multi-channel marketing plan with consistent brand voice.")
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            product = st.text_area(
                "Product / Offer Description",
                "Careem Food: 30% off for Champions segment this weekend on orders above 60 AED in Dubai.",
                height=100
            )
            
            c_input1, c_input2 = st.columns(2)
            with c_input1:
                target_segment_ai = st.selectbox(
                    "Target Customer Segment",
                    sorted(df['Segment'].unique())
                )
                brand_voice = st.selectbox(
                    "Brand Voice",
                    ["Careem – friendly & witty", "Professional & trustworthy", "Youthful & bold"]
                )
            with c_input2:
                primary_goal = st.selectbox(
                    "Primary Campaign Goal",
                    ["Drive first order", "Increase frequency", "Win back lapsed users", "Upsell high value users"]
                )
                
        with col_right:
            st.info("💡 **How it works:** This tool constructs a sophisticated prompt chain ensuring your brand voice remains consistent across Email, Push, and Social channels.")
        
        if st.button("Generate Campaign Blueprint", type="primary"):
            st.divider()
            st.markdown("### 🧠 Copy-Ready Prompt for ChatGPT / Claude")
            st.markdown("Copy the block below and paste it into your LLM of choice:")
            
            prompt_text = f"""
You are a senior lifecycle marketing manager at a Super App like Careem in MENA.

Your task: design a multi-channel campaign (email, push, in-app banner, and social ad) for the following offer:

PRODUCT/OFFER:
\"\"\"{product}\"\"\"

CONTEXT:
- Target RFM segment: {target_segment_ai}
- Primary goal: {primary_goal}
- Audience: mobile-first users in GCC, price sensitive but convenience-driven
- Brand voice: {brand_voice} (keep it consistent across all assets)

OUTPUT:
1. 1-paragraph campaign strategy (who, why now, what message).
2. Channel plan table: channel, objective, core message, CTA, timing.
3. Copy:
   - Email: subject line + preview + body outline (not full long email).
   - Push: 3 options, max 40 characters each.
   - In-app banner: headline + subcopy.
   - Social ad (Instagram story): hook line + 2 supporting points.
4. Add one simple A/B test idea for subject line or hook.
            """
            st.code(prompt_text, language="markdown")
            
            st.markdown("### ✍️ Example Hook Suggestions (Generated based on strategy)")
            st.write("- **Emotional:** \"Your favourite biryani is just one tap away—this weekend, we’ve got your cravings covered.\"")
            st.write("- **Data-driven:** \"Champions like you save an average of 42 AED this weekend with our 30% off Careem Food offer.\"")
            st.write("- **Humorous:** \"Don’t cook. It’s 40 degrees outside and your oven doesn’t need encouragement.\"")
