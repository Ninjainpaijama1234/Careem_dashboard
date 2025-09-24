import streamlit as st
import pandas as pd
import altair as alt
import datetime as dt

# 1. Page Configuration
# ---------------------
st.set_page_config(
    page_title="Advanced RFM Segmentation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Helper Functions
# -------------------
@st.cache_data
def load_data(uploaded_file):
    """
    Loads data from a CSV file, handles date conversion, and returns a DataFrame.
    Caches the data to improve performance.
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            # Load default dataset if no file is uploaded
            df = pd.read_csv('careem_food_enhanced_user_data.csv')

        # Convert date columns to datetime objects, handling potential errors
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
        df['last_order_date'] = pd.to_datetime(df['last_order_date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error("The default 'careem_food_enhanced_user_data.csv' was not found. Please upload a file.")
        return None

def calculate_rfm(df, r_weight, f_weight, m_weight):
    """
    Calculates Recency, Frequency, and Monetary scores and a final weighted RFM score.
    Also segments customers based on the final score.
    """
    # Create copies of scores to avoid modifying original dataframe
    df['R_Score'] = pd.qcut(df['days_since_last_order'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(int)
    df['F_Score'] = pd.qcut(df['total_orders'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_Score'] = pd.qcut(df['total_spend'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Calculate weighted RFM score
    df['RFM_Score'] = (r_weight * df['R_Score'] +
                       f_weight * df['F_Score'] +
                       m_weight * df['M_Score'])

    # Define segment labels based on RFM score quantiles
    score_bins = pd.qcut(df['RFM_Score'], 5, labels=False, duplicates='drop')
    segment_map = {
        0: 'Lost',
        1: 'At-Risk',
        2: 'Potential Loyalists',
        3: 'Loyal Customers',
        4: 'Champions'
    }
    df['Segment'] = score_bins.map(segment_map)
    # Handle cases where scores might not fall into 5 distinct bins
    if df['Segment'].isnull().any():
        df['Segment'] = df['Segment'].cat.add_categories('Undefined').fillna('Undefined')


    return df

def get_segment_info(segment):
    """
    Returns a description and marketing suggestions for a given customer segment.
    """
    info = {
        "Champions": {
            "description": "Your most valuable customers. They order frequently, spend the most, and have ordered very recently. They are the backbone of your business.",
            "suggestions": "Reward them with loyalty programs, exclusive offers, and early access to new features. Engage them for feedback and encourage them to become brand ambassadors."
        },
        "Loyal Customers": {
            "description": "High-frequency and high-value customers who are consistently engaged. They are reliable and form a core part of your revenue.",
            "suggestions": "Upsell higher-value products, ask for reviews, and engage them with personalized content to maintain their loyalty. Offer subscription models if applicable."
        },
        "Potential Loyalists": {
            "description": "These are recent customers with average frequency or spend. They have the potential to become Loyal Customers or even Champions.",
            "suggestions": "Offer membership or loyalty programs, provide personalized recommendations, and run targeted promotions to increase their order frequency and value."
        },
        "At-Risk": {
            "description": "Customers who used to order frequently or spend a lot, but haven't ordered in a while. They are on the verge of churning.",
            "suggestions": "Launch personalized win-back campaigns with special discounts or offers. Send reminders and ask for feedback to understand their reasons for inactivity."
        },
        "Lost": {
            "description": "These are your churned customers. They have the lowest recency, frequency, and monetary values.",
            "suggestions": "Conduct surveys to understand why they left. A small, carefully targeted campaign might reactivate a few, but the focus should be on preventing other segments from becoming 'Lost'."
        }
    }
    return info.get(segment, {"description": "No description available.", "suggestions": ""})


# 3. Sidebar Configuration
# ------------------------
with st.sidebar:
    st.image("https://placehold.co/300x100/000000/FFFFFF?text=RFM+Dashboard", use_column_width=True)
    st.title("📊 Dashboard Controls")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your customer data (CSV)", type=["csv"])
    st.markdown("---")

    st.header("⚙️ RFM Weight Configuration")
    st.markdown("Adjust the relative importance of Recency, Frequency, and Monetary factors.")
    r_weight = st.slider("Recency Weight", 0.0, 1.0, 0.4, 0.05)
    f_weight = st.slider("Frequency Weight", 0.0, 1.0, 0.4, 0.05)
    m_weight = st.slider("Monetary Weight", 0.0, 1.0, 0.2, 0.05)

    st.info(f"""
    **Current Weights:**
    - Recency: **{r_weight:.2f}**
    - Frequency: **{f_weight:.2f}**
    - Monetary: **{m_weight:.2f}**
    - Total: **{r_weight+f_weight+m_weight:.2f}**
    """)
    if abs(r_weight + f_weight + m_weight - 1.0) > 0.01:
        st.warning("For best results, the sum of weights should ideally be 1.0.")


# 4. Main Application
# -------------------
st.title("🚀 Advanced Customer RFM Segmentation Dashboard")
st.markdown("An interactive tool to understand and segment your customer base using the RFM model.")

df_raw = load_data(uploaded_file)

if df_raw is not None:
    df_rfm = calculate_rfm(df_raw.copy(), r_weight, f_weight, m_weight)

    tab1, tab2, tab3 = st.tabs(["📊  **Dashboard Overview**", "🧩  **RFM Segmentation Analysis**", "👤  **Customer Deep Dive**"])

    # -- Tab 1: Dashboard Overview --
    with tab1:
        st.header("High-Level Business Metrics")
        st.markdown("A snapshot of key performance indicators from your customer data.")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Total Customers", f"{df_rfm['user_id'].nunique():,}")
        kpi_cols[1].metric("Total Revenue", f"${df_rfm['total_spend'].sum():,.2f}")
        kpi_cols[2].metric("Average Orders per Customer", f"{df_rfm['total_orders'].mean():.2f}")
        kpi_cols[3].metric("Average Spend per Order", f"${(df_rfm['total_spend']/df_rfm['total_orders']).mean():,.2f}")

        st.markdown("---")
        st.header("Customer Behavior Analysis")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.subheader("Customer Acquisition Over Time")
            acq_data = df_rfm.set_index('signup_date').resample('M')['user_id'].count().reset_index()
            acq_chart = alt.Chart(acq_data).mark_area(
                line={'color':'darkgreen'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color='darkgreen', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x='signup_date:T',
                y='user_id:Q'
            ).properties(height=300)
            st.altair_chart(acq_chart, use_container_width=True)

        with chart_cols[1]:
            st.subheader("Order Recency Distribution")
            recency_hist = alt.Chart(df_rfm).mark_bar().encode(
                alt.X("days_since_last_order:Q", bin=alt.Bin(maxbins=30), title="Days Since Last Order"),
                alt.Y("count()", title="Number of Customers")
            ).properties(height=300)
            st.altair_chart(recency_hist, use_container_width=True)

    # -- Tab 2: RFM Segmentation Analysis --
    with tab2:
        st.header("Dynamic RFM Segmentation")
        st.markdown("Segments are calculated based on the weights you set in the sidebar. Explore the characteristics of each group.")

        seg_dist_cols = st.columns([0.6, 0.4])
        with seg_dist_cols[0]:
            st.subheader("Customer Segment Distribution")
            segment_counts = df_rfm['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'count']
            
            pie_chart = alt.Chart(segment_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="count", type="quantitative"),
                color=alt.Color(field="Segment", type="nominal",
                    scale=alt.Scale(domain=['Champions', 'Loyal Customers', 'Potential Loyalists', 'At-Risk', 'Lost'],
                                    range=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']))
            ).properties(height=350)
            st.altair_chart(pie_chart, use_container_width=True)
        
        with seg_dist_cols[1]:
            st.subheader("Segment Value Breakdown")
            segment_monetary = df_rfm.groupby('Segment')['total_spend'].sum().sort_values(ascending=False).reset_index()
            monetary_bar = alt.Chart(segment_monetary).mark_bar().encode(
                x=alt.X('total_spend:Q', title='Total Revenue'),
                y=alt.Y('Segment:N', sort='-x')
            ).properties(height=350)
            st.altair_chart(monetary_bar, use_container_width=True)

        st.markdown("---")
        st.header("Deep Dive into Segments")
        st.markdown("Click on a segment name to understand its characteristics and get actionable marketing suggestions.")
        
        segments_in_data = df_rfm['Segment'].unique()
        for segment in ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At-Risk', 'Lost']:
            if segment in segments_in_data:
                with st.expander(f"**{segment}** ({df_rfm[df_rfm['Segment'] == segment].shape[0]} customers)"):
                    info = get_segment_info(segment)
                    seg_info_cols = st.columns(2)
                    with seg_info_cols[0]:
                        st.markdown(f"**Description:** {info['description']}")
                        st.markdown(f"**Marketing Suggestions:** {info['suggestions']}")
                    with seg_info_cols[1]:
                        st.markdown("**Segment Averages:**")
                        segment_data = df_rfm[df_rfm['Segment'] == segment]
                        avg_recency = int(segment_data['days_since_last_order'].mean())
                        avg_frequency = round(segment_data['total_orders'].mean(), 2)
                        avg_monetary = round(segment_data['total_spend'].mean(), 2)
                        st.metric("Avg. Recency", f"{avg_recency} days")
                        st.metric("Avg. Frequency", f"{avg_frequency} orders")
                        st.metric("Avg. Monetary Value", f"${avg_monetary:,.2f}")

    # -- Tab 3: Customer Deep Dive --
    with tab3:
        st.header("Search and Analyze Individual Customers")
        st.markdown("Select a customer to view their profile, order history summary, and RFM classification.")
        
        customer_id = st.selectbox(
            "Select Customer ID:",
            options=df_rfm['user_id'].unique()
        )
        
        if customer_id:
            customer_data = df_rfm[df_rfm['user_id'] == customer_id].iloc[0]
            
            st.subheader(f"👤 Customer Profile: {customer_id}")
            
            card_cols = st.columns(3)
            with card_cols[0]:
                with st.container(border=True):
                    st.markdown(f"**Segment:** `{customer_data['Segment']}`")
                    st.markdown(f"**RFM Score:** `{customer_data['RFM_Score']:.2f}`")
            with card_cols[1]:
                with st.container(border=True):
                    st.markdown(f"**Signup Date:** `{customer_data['signup_date'].strftime('%Y-%m-%d')}`")
                    st.markdown(f"**Last Order Date:** `{customer_data['last_order_date'].strftime('%Y-%m-%d')}`")
            with card_cols[2]:
                with st.container(border=True):
                    st.markdown(f"**Last Used Platform:** `{customer_data['last_platform_used']}`")
                    st.markdown(f"**Favorite Cuisine:** `{customer_data['favorite_cuisine']}`")

            st.markdown("---")
            st.subheader("Lifetime Value and Order Summary")
            summary_cols = st.columns(4)
            summary_cols[0].metric("Total Spend", f"${customer_data['total_spend']:,.2f}")
            summary_cols[1].metric("Total Orders", f"{customer_data['total_orders']:,}")
            summary_cols[2].metric("Days Since Last Order", f"{customer_data['days_since_last_order']:,}")
            summary_cols[3].metric("Customer Lifetime Value", f"${customer_data['customer_lifetime_value']:,.2f}")

            st.subheader("Raw Data")
            st.json(customer_data.to_json())


else:
    st.info("Awaiting for a CSV file to be uploaded. The dashboard will activate once data is available.")
