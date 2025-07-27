import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px # Import Plotly Express for interactive charts
from datetime import datetime, timedelta # Import datetime and timedelta for date calculations

st.set_page_config(layout="wide", page_title="HealthKart Influencer Dashboard")

# --- Helper function to capitalize DataFrame headers for display ---
def get_capitalized_column_config(df):
    """
    Generates a column_config dictionary for st.dataframe to capitalize column headers.
    Converts 'column_name' to 'Column Name'.
    """
    config = {}
    # Handle regular columns
    for col in df.columns:
        config[col] = st.column_config.Column(label=col.replace('_', ' ').title())
    # Handle index name if it exists and is not the default (e.g., for set_index results)
    if df.index.name is not None:
        config[df.index.name] = st.column_config.Column(label=df.index.name.replace('_', ' ').title())
    return config

# --- 1. Data Ingestion & Preprocessing ---
@st.cache_data
def load_data():
    try:
        df_influencers = pd.read_csv('influencers.csv')
        df_posts = pd.read_csv('posts.csv')
        df_tracking = pd.read_csv('tracking_data.csv')
        df_payouts = pd.read_csv('payouts.csv')

        # Convert date columns to datetime objects
        df_posts['date'] = pd.to_datetime(df_posts['date'])
        df_tracking['date'] = pd.to_datetime(df_tracking['date'])

        # Prepare influencers_df for merging with posts_df to get influencer category and platform
        # Explicitly select and rename columns to avoid suffix issues and ensure desired names
        influencers_for_posts_merge = df_influencers[['ID', 'category', 'platform']].copy()
        influencers_for_posts_merge.rename(columns={
            'category': 'influencer_category_for_post',
            'platform': 'influencer_platform_for_post'
        }, inplace=True)

        # Merge posts with prepared influencer data
        posts_merged = pd.merge(df_posts, influencers_for_posts_merge,
                                left_on='influencer_id', right_on='ID', how='left')
        # Drop duplicate 'ID' column from the merge if it exists
        posts_merged.drop(columns=['ID'], inplace=True, errors='ignore')

        # Merge tracking data with influencers and payouts
        # This part seems correct as it explicitly renames columns
        df_tracking_with_influencer = pd.merge(df_tracking, df_influencers[['ID', 'name', 'category', 'platform']],
                                               left_on='influencer_id', right_on='ID', how='left')
        df_tracking_with_influencer.rename(columns={'name': 'influencer_name', 'category': 'influencer_category', 'platform': 'influencer_platform'}, inplace=True)
        df_tracking_with_influencer.drop(columns=['ID'], inplace=True, errors='ignore') # Drop duplicate ID column

        # For ROAS calculation, we need to associate payouts with tracking data.
        # Aggregate payout per influencer first, then merge.
        # This prevents issues with 'basis' and 'rate' not aligning with individual tracking rows.
        payouts_agg = df_payouts.groupby('influencer_id')['total_payout'].sum().reset_index()
        tracking_full = pd.merge(df_tracking_with_influencer, payouts_agg, on='influencer_id', how='left')
        tracking_full['total_payout'].fillna(0, inplace=True) # Fill NaN payouts with 0

        # Calculate campaign cost for ROAS - this is just a simplistic sum of payouts
        # If a campaign involves multiple influencers, their payouts sum up to the campaign cost.
        campaign_payouts = df_tracking_with_influencer.groupby('campaign')['influencer_id'].nunique().reset_index()
        campaign_payouts = pd.merge(campaign_payouts, payouts_agg, on='influencer_id', how='left')
        campaign_payouts = campaign_payouts.groupby('campaign')['total_payout'].sum().reset_index()
        campaign_payouts.rename(columns={'total_payout': 'campaign_total_payout'}, inplace=True)

        return df_influencers, posts_merged, tracking_full, df_payouts, campaign_payouts
    except FileNotFoundError:
        st.error("Data CSV files not found. Please run `data_generator.py` first to generate the data.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading or processing data: {e}")
        st.stop()

# Add a loading spinner for data ingestion
with st.spinner("Loading and preprocessing data..."):
    influencers_df, posts_df, tracking_df, payouts_df, campaign_payouts_df = load_data()

# --- Helper Functions for Calculations ---
def calculate_roas(revenue, ad_spend):
    if ad_spend == 0 or pd.isna(ad_spend):
        return np.inf if revenue > 0 else 0 # Infinite if revenue > 0 and no spend, 0 if no revenue and no spend
    return revenue / ad_spend

def calculate_incremental_roas_simple(total_revenue, ad_spend, baseline_percentage=0.2):
    """
    Simulates incremental ROAS by assuming a baseline organic revenue.
    For this simulation, we'll assume a fixed percentage of total revenue
    would have occurred organically without the influencer campaign.
    """
    baseline_revenue = total_revenue * baseline_percentage
    incremental_revenue = total_revenue - baseline_revenue
    if ad_spend == 0 or pd.isna(ad_spend):
        return np.inf if incremental_revenue > 0 else 0
    return incremental_revenue / ad_spend

# Function to reset filters
def reset_filters():
    st.session_state.date_range_option = "All Time"
    st.session_state.date_range_custom = (
        tracking_df['date'].min().date() if not tracking_df.empty else (datetime.now() - timedelta(days=365)).date(),
        tracking_df['date'].max().date() if not tracking_df.empty else datetime.now().date()
    )
    # Reset multiselect filters to empty lists
    st.session_state.selected_products = []
    st.session_state.selected_categories = []
    st.session_state.selected_platforms = []
    st.session_state.min_tracking_events = 1 # Reset to default minimum of 1

# Initialize session state for filters if not already set
if 'date_range_option' not in st.session_state:
    st.session_state.date_range_option = "All Time"
if 'date_range_custom' not in st.session_state:
    min_date_data = tracking_df['date'].min().date() if not tracking_df.empty else (datetime.now() - timedelta(days=365)).date()
    max_date_data = tracking_df['date'].max().date() if not tracking_df.empty else datetime.now().date()
    st.session_state.date_range_custom = (min_date_data, max_date_data)
if 'selected_products' not in st.session_state:
    st.session_state.selected_products = [] # Default to empty list
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = [] # Default to empty list
if 'selected_platforms' not in st.session_state:
    st.session_state.selected_platforms = [] # Default to empty list
if 'min_tracking_events' not in st.session_state:
    st.session_state.min_tracking_events = 1 # Default value

# --- Streamlit Dashboard Layout ---
st.title("📈 HealthKart Influencer Campaign Dashboard")

# --- Sidebar Filters ---
st.sidebar.header("Filter Data")

# Reset Filters Button
if st.sidebar.button("Reset All Filters"):
    reset_filters()
    st.rerun() # Rerun the app to apply filter reset

# Date range filter options
date_options = ["All Time", "Last 30 Days", "Last 90 Days", "Last 180 Days", "Last Year", "Custom Range"]
selected_date_option = st.sidebar.radio(
    "Quick Date Range",
    options=date_options,
    key="date_range_option" # Use session state for persistence
)

min_date_data = tracking_df['date'].min().date() if not tracking_df.empty else (datetime.now() - timedelta(days=365)).date()
max_date_data = tracking_df['date'].max().date() if not tracking_df.empty else datetime.now().date()

start_date, end_date = min_date_data, max_date_data # Default to all time

if selected_date_option == "Last 30 Days":
    start_date = max_date_data - timedelta(days=30)
elif selected_date_option == "Last 90 Days":
    start_date = max_date_data - timedelta(days=90)
elif selected_date_option == "Last 180 Days":
    start_date = max_date_data - timedelta(days=180)
elif selected_date_option == "Last Year":
    start_date = max_date_data - timedelta(days=365)
elif selected_date_option == "Custom Range":
    custom_date_range = st.sidebar.date_input(
        "Select Custom Date Range",
        st.session_state.date_range_custom,
        min_value=min_date_data,
        max_value=max_date_data,
        key="custom_date_input" # Use a unique key for the widget
    )
    if len(custom_date_range) == 2:
        start_date, end_date = custom_date_range[0], custom_date_range[1]
        st.session_state.date_range_custom = custom_date_range # Update session state
else: # All Time
    start_date, end_date = min_date_data, max_date_data

# Filter DataFrames based on the determined date range
filtered_tracking_df = tracking_df[(tracking_df['date'].dt.date >= start_date) & (tracking_df['date'].dt.date <= end_date)].copy()
filtered_posts_df = posts_df[(posts_df['date'].dt.date >= start_date) & (posts_df['date'].dt.date <= end_date)].copy()


all_brands_products = sorted(tracking_df['product'].unique())
selected_products = st.sidebar.multiselect(
    "Select Product(s) (Brand)",
    all_brands_products,
    default=st.session_state.selected_products, # Default to empty list from session state
    key="selected_products"
)

all_influencer_categories = sorted(influencers_df['category'].unique())
selected_categories = st.sidebar.multiselect(
    "Select Influencer Category",
    all_influencer_categories,
    default=st.session_state.selected_categories, # Default to empty list from session state
    key="selected_categories"
)

all_platforms = sorted(influencers_df['platform'].unique())
selected_platforms = st.sidebar.multiselect(
    "Select Platform",
    all_platforms,
    default=st.session_state.selected_platforms, # Default to empty list from session state
    key="selected_platforms"
)

# New filter: Minimum Tracking Events per Influencer
min_tracking_events = st.sidebar.number_input(
    "Minimum Tracking Events per Influencer",
    min_value=1,
    value=st.session_state.min_tracking_events, # Use session state for value
    step=1,
    key="min_tracking_events_input"
)
st.session_state.min_tracking_events = min_tracking_events # Update session state


# Apply filters - Modified to handle empty multiselects (meaning "select all" for that filter)
if not filtered_tracking_df.empty:
    if selected_products: # Only filter if products are actually selected
        filtered_tracking_df = filtered_tracking_df[filtered_tracking_df['product'].isin(selected_products)]
    if selected_categories: # Only filter if categories are actually selected
        filtered_tracking_df = filtered_tracking_df[filtered_tracking_df['influencer_category'].isin(selected_categories)]
    if selected_platforms: # Only filter if platforms are actually selected
        filtered_tracking_df = filtered_tracking_df[filtered_tracking_df['influencer_platform'].isin(selected_platforms)]

if not filtered_posts_df.empty:
    # --- DEBUGGING LINES START ---
    st.subheader("DEBUG: Filtered Posts DataFrame Info") # Changed header for clarity
    st.write("Columns of filtered_posts_df:", filtered_posts_df.columns)
    # Apply capitalized column config and increase rows for better visibility
    st.dataframe(filtered_posts_df.head(10), column_config=get_capitalized_column_config(filtered_posts_df) | {
        "URL": st.column_config.Column(label="URL", width="large"), # Explicitly keep URL label as is if preferred
        "influencer_category_for_post": st.column_config.Column(label="Influencer Category For Post", width="large"),
        "influencer_platform_for_post": st.column_config.Column(label="Influencer Platform For Post", width="large"),
        "date": st.column_config.DateColumn(label="Date", format="YYYY-MM-DD") # Format date to show only YYYY-MM-DD
    })
    # --- DEBUGGING LINES END ---

    if selected_categories:
        filtered_posts_df = filtered_posts_df[filtered_posts_df['influencer_category_for_post'].isin(selected_categories)]
    if selected_platforms:
        filtered_posts_df = filtered_posts_df[filtered_posts_df['influencer_platform_for_post'].isin(selected_platforms)]


# --- Dashboard Sections ---
tab1, tab2, tab3, tab4 = st.tabs(["Campaign Performance", "Influencer Insights", "Payout Tracking", "Raw Data"])

with tab1:
    st.header("📊 Campaign Performance")

    if filtered_tracking_df.empty:
        st.info("No data available for the selected filters in Campaign Performance.")
    else:
        # Aggregate data for campaign performance
        campaign_summary = filtered_tracking_df.groupby('campaign').agg(
            total_revenue=('revenue', 'sum'),
            total_orders=('orders', 'sum')
        ).reset_index()

        # Merge with campaign_payouts_df to get campaign total payout
        campaign_summary = pd.merge(campaign_summary, campaign_payouts_df, on='campaign', how='left')
        campaign_summary['campaign_total_payout'].fillna(0, inplace=True) # Fill NaN with 0

        campaign_summary['ROAS'] = campaign_summary.apply(
            lambda row: calculate_roas(row['total_revenue'], row['campaign_total_payout']), axis=1
        )
        campaign_summary['Incremental_ROAS'] = campaign_summary.apply(
            lambda row: calculate_incremental_roas_simple(row['total_revenue'], row['campaign_total_payout']), axis=1
        )

        st.subheader("Overall Campaign Summary")

        # KPI Cards for overall performance
        total_revenue_overall = campaign_summary['total_revenue'].sum()
        total_payout_overall = campaign_summary['campaign_total_payout'].sum()
        overall_roas = calculate_roas(total_revenue_overall, total_payout_overall)

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.metric(label="Total Revenue", value=f"₹{total_revenue_overall:,.2f}")
        with col_kpi2:
            st.metric(label="Total Payout", value=f"₹{total_payout_overall:,.2f}")
        with col_kpi3:
            st.metric(label="Overall ROAS", value=f"{overall_roas:.2f}x")

        st.dataframe(campaign_summary.set_index('campaign').style.format({
            'total_revenue': '₹{:,.2f}',
            'campaign_total_payout': '₹{:,.2f}',
            'ROAS': '{:.2f}x',
            'Incremental_ROAS': '{:.2f}x'
        }), column_config=get_capitalized_column_config(campaign_summary.set_index('campaign')))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue vs Payout by Campaign")
            # Plotly Bar Chart: Revenue vs Payout
            fig_rev_payout = px.bar(
                campaign_summary,
                x='campaign',
                y=['total_revenue', 'campaign_total_payout'],
                barmode='group',
                title='Total Revenue and Payout per Campaign',
                labels={'value': 'Amount (₹)', 'campaign': 'Campaign'},
                color_discrete_map={'total_revenue': '#4CAF50', 'campaign_total_payout': '#FFC107'}
            )
            fig_rev_payout.update_layout(xaxis_title="Campaign", yaxis_title="Amount (₹)")
            fig_rev_payout.update_xaxes(tickangle=45)
            st.plotly_chart(fig_rev_payout, use_container_width=True)

        with col2:
            st.subheader("ROAS by Campaign")
            # Plotly Bar Chart: ROAS
            fig_roas = px.bar(
                campaign_summary,
                x='campaign',
                y='ROAS',
                title='Return on Ad Spend per Campaign',
                labels={'ROAS': 'ROAS (x)', 'campaign': 'Campaign'},
                color_discrete_sequence=['#2196F3']
            )
            fig_roas.update_layout(xaxis_title="Campaign", yaxis_title="ROAS (x)")
            fig_roas.update_xaxes(tickangle=45)
            st.plotly_chart(fig_roas, use_container_width=True)

        st.subheader("Campaign Performance Over Time")
        # Group by week for time series
        campaign_time_series = filtered_tracking_df.groupby([pd.Grouper(key='date', freq='W'), 'campaign']).agg(
            total_revenue=('revenue', 'sum')
        ).reset_index()

        # Merge with campaign_payouts (need to rethink how payout changes over time in simulated data)
        # For simplicity, we'll assume total campaign payout is spread evenly or assigned at start.
        # Here, we'll just show revenue over time.
        campaign_time_series = pd.merge(campaign_time_series, campaign_payouts_df, on='campaign', how='left')
        campaign_time_series['campaign_total_payout'].fillna(0, inplace=True) # Fill NaN with 0

        # Calculate weekly ROAS - this is an oversimplification for simulated data
        campaign_time_series['weekly_roas'] = campaign_time_series.apply(
            lambda row: calculate_roas(row['total_revenue'], row['campaign_total_payout'] / ( (end_date - start_date).days / 7) if (end_date - start_date).days > 0 else row['campaign_total_payout'] ), axis=1
        ) # Distribute payout over the weeks

        # --- Added check for empty DataFrame before plotting ---
        if not campaign_time_series.empty:
            # Plotly Line Chart: Revenue Over Time
            fig_time = px.line(
                campaign_time_series,
                x='date',
                y='total_revenue',
                color='campaign',
                markers=True,
                title='Total Revenue by Campaign Over Time (Weekly)',
                labels={'total_revenue': 'Revenue (₹)', 'date': 'Date', 'campaign': 'Campaign'}
            )
            fig_time.update_layout(xaxis_title="Date", yaxis_title="Revenue (₹)")
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No campaign performance over time data available for the selected filters.")


with tab2:
    st.header("👤 Influencer Insights")

    if filtered_tracking_df.empty:
        st.info("No data available for the selected filters in Influencer Insights.")
    else:
        # Aggregate data for influencer performance
        influencer_summary = filtered_tracking_df.groupby(['influencer_id', 'influencer_name', 'influencer_category', 'influencer_platform']).agg(
            total_revenue=('revenue', 'sum'),
            total_orders=('orders', 'sum'),
            # Count the number of tracking events for each influencer
            num_tracking_events=('user_id', 'count')
        ).reset_index()

        # --- Apply the new filter for minimum tracking events ---
        influencer_summary = influencer_summary[influencer_summary['num_tracking_events'] >= min_tracking_events]

        # Merge with aggregated payouts
        influencer_summary = pd.merge(influencer_summary, payouts_df.groupby('influencer_id')['total_payout'].sum().reset_index(),
                                       on='influencer_id', how='left')
        influencer_summary['total_payout'].fillna(0, inplace=True)

        influencer_summary['ROAS'] = influencer_summary.apply(
            lambda row: calculate_roas(row['total_revenue'], row['total_payout']), axis=1
        )
        influencer_summary['Incremental_ROAS'] = influencer_summary.apply(
            lambda row: calculate_incremental_roas_simple(row['total_revenue'], row['total_payout']), axis=1
        )

        st.subheader("Top Influencers by ROAS")
        # Check if influencer_summary is empty after applying min_tracking_events filter
        if not influencer_summary.empty:
            top_roas_influencers = influencer_summary.sort_values(by='ROAS', ascending=False).head(10)
            # Apply capitalized column config
            st.dataframe(top_roas_influencers.style.format({
                'total_revenue': '₹{:,.2f}',
                'total_payout': '₹{:,.2f}',
                'ROAS': '{:.2f}x',
                'Incremental_ROAS': '{:.2f}x'
            }), column_config=get_capitalized_column_config(top_roas_influencers))
        else:
            st.info("No influencers meet the selected filters and minimum tracking events criteria for ROAS.")


        st.subheader("Top Influencers by Total Revenue")
        if not influencer_summary.empty:
            top_revenue_influencers = influencer_summary.sort_values(by='total_revenue', ascending=False).head(10)
            # Apply capitalized column config
            st.dataframe(top_revenue_influencers.style.format({
                'total_revenue': '₹{:,.2f}',
                'total_payout': '₹{:,.2f}',
                'ROAS': '{:.2f}x',
                'Incremental_ROAS': '{:.2f}x'
            }), column_config=get_capitalized_column_config(top_revenue_influencers))
        else:
            st.info("No influencers meet the selected filters and minimum tracking events criteria for Total Revenue.")


        st.subheader("Best Performing Influencer Categories")
        if not influencer_summary.empty:
            category_summary = influencer_summary.groupby('influencer_category').agg(
                avg_roas=('ROAS', 'mean'),
                total_revenue=('total_revenue', 'sum'),
                total_payout=('total_payout', 'sum')
            ).reset_index().sort_values(by='avg_roas', ascending=False)
            # Apply capitalized column config
            st.dataframe(category_summary.style.format({
                'total_revenue': '₹{:,.2f}',
                'total_payout': '₹{:,.2f}',
                'avg_roas': '{:.2f}x'
            }), column_config=get_capitalized_column_config(category_summary))
        else:
            st.info("No influencer categories meet the selected filters and minimum tracking events criteria.")


        # Post performance (using filtered_posts_df)
        if not filtered_posts_df.empty:
            st.subheader("Post Performance Metrics")
            post_metrics = filtered_posts_df.groupby('influencer_platform_for_post').agg(
                avg_reach=('reach', 'mean'),
                avg_likes=('likes', 'mean'),
                avg_comments=('comments', 'mean'),
                total_posts=('URL', 'count')
            ).reset_index().sort_values(by='total_posts', ascending=False)
            # Apply capitalized column config
            st.dataframe(post_metrics.style.format({
                'avg_reach': '{:,.0f}',
                'avg_likes': '{:,.0f}',
                'avg_comments': '{:,.0f}'
            }), column_config=get_capitalized_column_config(post_metrics))
        else:
            st.info("No post data available for the selected filters.")


with tab3:
    st.header("💸 Payout Tracking")

    st.subheader("All Payout Records")
    if payouts_df.empty:
        st.info("No payout data available.")
    else:
        # Merge payout details with influencer names for clarity
        payouts_display = pd.merge(payouts_df, influencers_df[['ID', 'name']], left_on='influencer_id', right_on='ID', how='left')
        payouts_display.drop(columns=['ID'], inplace=True, errors='ignore') # Use errors='ignore' for robustness
        payouts_display.rename(columns={'name': 'influencer_name'}, inplace=True)

        # Apply capitalized column config
        st.dataframe(payouts_display.style.format({
            'rate': '₹{:,.2f}',
            'total_payout': '₹{:,.2f}'
        }), column_config=get_capitalized_column_config(payouts_display))

        st.subheader("Payouts by Basis (Post vs Order)")
        # Plotly Bar Chart: Payouts by Basis
        fig_payout_basis = px.bar(
            payouts_df.groupby('basis')['total_payout'].sum().reset_index(),
            x='basis',
            y='total_payout',
            title='Total Payout by Basis',
            labels={'total_payout': 'Total Payout (₹)', 'basis': 'Payout Basis'},
            color_discrete_sequence=['#FF9800', '#03A9F4']
        )
        fig_payout_basis.update_layout(xaxis_title="Payout Basis", yaxis_title="Total Payout (₹)")
        st.plotly_chart(fig_payout_basis, use_container_width=True)


with tab4:
    st.header("🗄️ Raw Data Previews")
    st.markdown("Here you can see the raw (or slightly merged) datasets.")

    st.subheader("Influencers Data")
    # Apply capitalized column config
    st.dataframe(influencers_df, column_config=get_capitalized_column_config(influencers_df))

    st.subheader("Posts Data (Merged with Influencers)")
    # Apply column_config to posts_df for better visibility of long names and capitalized headers
    st.dataframe(posts_df, column_config=get_capitalized_column_config(posts_df) | {
        "URL": st.column_config.Column(label="URL", width="large"), # Explicitly keep URL label as is if preferred
        "influencer_category_for_post": st.column_config.Column(label="Influencer Category For Post", width="large"),
        "influencer_platform_for_post": st.column_config.Column(label="Influencer Platform For Post", width="large"),
        "date": st.column_config.DateColumn(label="Date", format="YYYY-MM-DD") # Format date to show only YYYY-MM-DD
    })

    st.subheader("Tracking Data (Merged with Influencers and Payouts)")
    # Apply capitalized column config
    st.dataframe(tracking_df, column_config=get_capitalized_column_config(tracking_df))

    st.subheader("Payouts Data")
    # Apply capitalized column config
    st.dataframe(payouts_df, column_config=get_capitalized_column_config(payouts_df))

    # Optional: Export to CSV
    st.subheader("Export Data")
    st.download_button(
        label="Download Filtered Tracking Data as CSV",
        data=filtered_tracking_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_tracking_data.csv',
        mime='text/csv',
    )
    # Ensure influencer_summary is defined before trying to export it
    if 'influencer_summary' in locals() and not influencer_summary.empty:
        st.download_button(
            label="Download Influencer Summary as CSV",
            data=influencer_summary.to_csv(index=False).encode('utf-8'),
            file_name='influencer_summary.csv',
            mime='text/csv',
        )
    else:
        st.info("Influencer summary not available for download (check filters).")