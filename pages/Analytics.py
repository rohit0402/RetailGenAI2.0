import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Analytics Dashboard",
    layout="wide"
)

st.title("📈 Analytics Dashboard")

# Check if data exists
if "df" in st.session_state and st.session_state.df is not None:

    df = st.session_state.df.copy()

    # -----------------------------------
    # Data Type Safety
    # -----------------------------------

    numeric_columns = [
        'units_sold',
        'ROI',
        'marketing_spend',
        'Revenue',
        'Profit',
        'Cost'
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors='coerce'
            )

    # -----------------------------------
    # Sidebar Filters
    # -----------------------------------

    st.sidebar.markdown("## 🎯 Filters")

    # campaign_name Filter
    if 'campaign_name' in df.columns:

        selected_campaign_name = st.sidebar.selectbox(
            "Select campaign_name",
            ["All"] + sorted(
                df['campaign_name']
                .dropna()
                .unique()
                .tolist()
            )
        )

        if selected_campaign_name != "All":

            df = df[
                df['campaign_name'] == selected_campaign_name
            ]

    # -----------------------------------
    # Raw Data Preview
    # -----------------------------------

    st.markdown("### Raw Data Preview")

    st.dataframe(df.head(10))

    # -----------------------------------
    # Data Quality Checks
    # -----------------------------------

    st.markdown("## 🛡️ Data Quality Checks")

    missing_values = df.isnull().sum().sum()

    duplicate_rows = df.duplicated().sum()

    quality1, quality2 = st.columns(2)

    with quality1:

        st.metric(
            "Missing Values",
            int(missing_values)
        )

    with quality2:

        st.metric(
            "Duplicate Rows",
            int(duplicate_rows)
        )

    if missing_values > 0:

        st.warning(
            "Dataset contains missing values."
        )

    if duplicate_rows > 0:

        st.warning(
            "Dataset contains duplicate rows."
        )

    # -----------------------------------
    # KPI SECTION
    # -----------------------------------

    st.markdown("## 📊 Business KPIs")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # Total Sales
    with kpi1:

        if 'units_sold' in df.columns:

            st.metric(
                "Total Sales",
                f"{df['units_sold'].sum():,.0f}"
            )

    # Average ROI
    with kpi2:

        if 'ROI' in df.columns:

            st.metric(
                "Average ROI",
                f"{df['ROI'].mean():.2f}%"
            )

    # Total marketing_spend
    with kpi3:

        if 'marketing_spend' in df.columns:

            st.metric(
                "Total marketing_spend",
                f"{df['marketing_spend'].sum():,.0f}"
            )

    # Best campaign_name
    with kpi4:

        if (
            'campaign_name' in df.columns and
            'units_sold' in df.columns and
            not df.empty
        ):

            campaign_name_sales = (
                df.groupby('campaign_name')['units_sold']
                .sum()
            )

            if not campaign_name_sales.empty:

                best_campaign_name = (
                    campaign_name_sales.idxmax()
                )

                st.metric(
                    "Top campaign_name",
                    best_campaign_name
                )

    # -----------------------------------
    # Revenue / Profit KPIs
    # -----------------------------------

    if (
        'Revenue' in df.columns or
        'Profit' in df.columns
    ):

        st.markdown("## 💰 Revenue Insights")

        rev1, rev2 = st.columns(2)

        with rev1:

            if 'Revenue' in df.columns:

                st.metric(
                    "Total Revenue",
                    f"${df['Revenue'].sum():,.0f}"
                )

        with rev2:

            if 'Profit' in df.columns:

                st.metric(
                    "Total Profit",
                    f"${df['Profit'].sum():,.0f}"
                )

    # -----------------------------------
    # Visualizations
    # -----------------------------------

    st.markdown("## 📈 Quick Visualizations")

    col1, col2 = st.columns(2)

    # -----------------------------------
    # Sales by campaign_name
    # -----------------------------------

    with col1:

        if (
            'units_sold' in df.columns and
            'campaign_name' in df.columns and
            not df.empty
        ):

            st.subheader(
                "Sales Volume by campaign_name"
            )

            chart_data = (
                df.groupby('campaign_name')
                ['units_sold']
                .sum()
            )

            st.bar_chart(chart_data)

        else:

            st.info(
                "Upload retail campaign_name data to view charts."
            )

    # -----------------------------------
    # marketing_spend Over Time
    # -----------------------------------

    with col2:

        if (
            'start_date' in df.columns and
            'marketing_spend' in df.columns
        ):

            st.subheader(
                "marketing_spend Over Time"
            )

            time_df = df.copy()

            time_df['start_date'] = pd.to_datetime(
                time_df['start_date'],
                errors='coerce'
            )

            time_df = time_df.dropna(
                subset=['start_date']
            )

            if not time_df.empty:

                time_df = time_df.sort_values(
                    'start_date'
                )

                time_df.set_index(
                    'start_date',
                    inplace=True
                )

                st.line_chart(
                    time_df['marketing_spend']
                )

            else:

                st.warning(
                    "No valid start_date data available."
                )

        else:

            st.info(
                "Upload retail campaign_name data to see marketing_spend trends."
            )

    # -----------------------------------
    # Revenue by campaign_name
    # -----------------------------------

    if (
        'Revenue' in df.columns and
        'campaign_name' in df.columns
    ):

        st.markdown("## 💵 Revenue Analysis")

        revenue_chart = (
            df.groupby('campaign_name')
            ['Revenue']
            .sum()
        )

        st.bar_chart(revenue_chart)

    # -----------------------------------
    # Profit by campaign_name
    # -----------------------------------

    if (
        'Profit' in df.columns and
        'campaign_name' in df.columns
    ):

        st.markdown("## 🏆 Profit Analysis")

        profit_chart = (
            df.groupby('campaign_name')
            ['Profit']
            .sum()
        )

        st.bar_chart(profit_chart)

else:

    st.warning(
        "No structured data (CSV/JSON) available. "
        "Please upload a file on the main page first."
    )

    if st.button("Go to Main Page"):

        st.switch_page("app.py")