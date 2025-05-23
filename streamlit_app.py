import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime

# Config
st.set_page_config(page_title="Product Sales Dashboard", layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("fake_web_server_logs.csv")

df = load_data()

# Define buyer statuses and their probabilities
statuses = ["No Buy", "Interested", "Buy"]
probabilities = [0.5, 0.3, 0.2]  # Adjust this to tweak dominance

# Assign random buyer status
df["Buyer_Status"] = np.random.choice(statuses, size=len(df), p=probabilities)

# Optional binary columns for analysis
df["Buy"] = (df["Buyer_Status"] == "Buy").astype(int)
df["Interest"] = (df["Buyer_Status"] == "Interested").astype(int)

import numpy as np

# Assign revenue only to buyers
df["Revenue"] = np.where(df["Buy"] == 1, np.random.randint(50, 250, size=len(df)), 0)

print(df.head())




# Add Buy column at source level, not filtered level
#df["Interest"] = df["Event_Type"].apply(lambda x: 1 if x in ["Webinar", "Conference"] else 0)
#df["Buy"] = df["Event_Type"].apply(lambda x: 1 if x in ["Workshop", "Demo Day", "Hackathon"] else 0)



# Sidebar - Filters
with st.sidebar:
    st.title("📊 Product Sales Dashboard")
    st.subheader("🔎 Filters")

    selected_event = st.multiselect("📌 Event Types", options=sorted(df["Event_Type"].unique()), default=df["Event_Type"].unique())
    selected_country = st.multiselect("🌍 Countries", options=sorted(df["Country"].unique()), default=df["Country"].unique())
    selected_va = st.radio("🤖 Used Virtual Assistant?", ["All", "Yes", "No"], horizontal=True)

# Apply filters
filtered_df = df[df["Event_Type"].isin(selected_event) & df["Country"].isin(selected_country)]
if selected_va != "All":
    filtered_df = filtered_df[filtered_df["Virtual_Assistant"] == selected_va]

# Main Navigation (TOP)
section = st.radio(
    label="📂 Select Section",
    options=["Overview", "Event Performance","Customer Behaviour", "Sales Insights", "Forecast & Prediction"], 
    horizontal=True
)

# ---------------------- SECTION: OVERVIEW ---------------------- #
if section == "Overview":
    st.subheader("📊 Overview Dashboard")

    # Format date
    filtered_df["Date"] = pd.to_datetime(filtered_df["Timestamp"], format="%d/%m/%Y %H:%M").dt.date
    daily_counts = filtered_df.groupby("Date").size()

    # --- REAL KPIs ---
    total_events = len(filtered_df)
    unique_visitors = filtered_df["IP_Address"].nunique()
    unique_countries = filtered_df["Country"].nunique()
    avg_daily = daily_counts.mean()

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📥 Total Interactions", f"{total_events:,}")
    col2.metric("👤 Unique Visitors", f"{unique_visitors:,}")
    col3.metric("🌍 Countries", f"{unique_countries}")
    col4.metric("📅 Avg Events/Day", f"{int(avg_daily):,}")  #changed here


    #st.markdown("---")

    # 📊 Top 5 Countries Bar Chart
    #st.markdown("🌍 Top Countries by Engagement")
    top_countries = filtered_df["Country"].value_counts().nlargest(6).reset_index()
    top_countries.columns = ["Country", "Count"]

    fig_top_countries = px.bar(
        top_countries,
        x="Country",
        y="Count",
        color="Count",
        color_continuous_scale="Blues",
        title="📶 Top 6 Countries",
        hover_data={"Country": True, "Count": ":,"}
    )
    fig_top_countries.update_layout(
        height=300,
        yaxis_title="Total Count per Annum",
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_x=0.5
    )

    # 🤖 Virtual Assistant Usage Donut
    #st.markdown("🤖 Virtual Assistant Engagement")
    va_counts = filtered_df["Virtual_Assistant"].value_counts().reset_index()
    va_counts.columns = ["Virtual_Assistant", "Count"]

    # Define a matching blue color palette for consistency
    blues_palette = ["#08306B", "#4292C6"]  # dark to light blue
    
    fig_va_use = px.pie(
        va_counts,
        names="Virtual_Assistant",
        values="Count",
        title="VA vs Non-VA Engagement",
        hole=0.5,
        color_discrete_sequence=blues_palette  # Matching blue tones
)
    fig_va_use.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}: %{value}",
        pull=[0.1 if i == 0 else 0 for i in range(len(va_counts))]
)
    fig_va_use.update_layout(
        height=300,
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        title_x=0.5
)


    # Layout side-by-side
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_top_countries, use_container_width=True)
    col2.plotly_chart(fig_va_use, use_container_width=True)



# ---------------------- SECTION: DEEP DIVE ANALYTICS ---------------------- #
elif section == "Event Performance":
    import plotly.express as px
    from datetime import datetime

    df_time = filtered_df.copy()
    df_time["Date"] = pd.to_datetime(df_time["Timestamp"], format="%d/%m/%Y %H:%M").dt.date
    df_time["Hour"] = pd.to_datetime(df_time["Timestamp"], format="%d/%m/%Y %H:%M").dt.hour

    # --- TOP ROW ---
    top_col1, top_col2 = st.columns(2)

    with top_col1:
        st.markdown("🌍Treemap: Country & Event Type")
        treemap_df = filtered_df.groupby(["Country", "Event_Type"]).size().reset_index(name="Count")
        fig_treemap = px.treemap(
            treemap_df, path=["Country", "Event_Type"], values="Count",
            color="Count", color_continuous_scale="Viridis"
        )
        fig_treemap.update_layout(
            height=200,
            margin=dict(t=20, l=5, r=5, b=5)
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    with top_col2:
        st.markdown("📈 Daily Events Types")
        daily_events = df_time.groupby("Date").size().reset_index(name="Daily Event Type Count") #Changed here
        fig_area = px.area(
            daily_events, x="Date", y="Daily Event Type Count",
            color_discrete_sequence=["#00cc96"]
        )
        fig_area.update_layout(
            height=200,
            margin=dict(t=20, l=5, r=5, b=5)
        )
        st.plotly_chart(fig_area, use_container_width=True)

    # --- BOTTOM ROW ---
    bot_col1, bot_col2 = st.columns(2)

    with bot_col1:
        st.markdown("📊 Event Types per Hour for each Weekday") #Changed here

        # Extract weekday and hour
        df_time["Weekday"] = pd.to_datetime(df_time["Timestamp"], format="%d/%m/%Y %H:%M").dt.day_name()
        df_time["Hour"] = pd.to_datetime(df_time["Timestamp"], format="%d/%m/%Y %H:%M").dt.hour

        # Group and count
        weekday_df = df_time.groupby(["Weekday", "Hour"]).size().reset_index(name="Event Count")

        # Define weekday order for dropdown and display
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_df["Weekday"] = pd.Categorical(weekday_df["Weekday"], categories=weekday_order, ordered=True)

        # Dropdown for weekday selection
        selected_day = st.selectbox("Select a weekday to view hourly event trends:", weekday_order)

        # Filter based on selected weekday
        selected_df = weekday_df[weekday_df["Weekday"] == selected_day]

        # Plot interactive line chart
        fig_line = px.line(
            selected_df,
            x="Hour",
            y="Event Count",
            title=f"Hourly Event Trend on {selected_day}",
            markers=True,
            labels={"Hour": "Hours of The Day", "Event Count": "Total of Event Type"}  # Changed here
    )

        fig_line.update_layout(
            height=180,
            margin=dict(t=30, l=5, r=5, b=5),
            xaxis=dict(dtick=1)
    )

    st.plotly_chart(fig_line, use_container_width=True)



# ---------------------- SECTION: FORECAST ---------------------- #
elif section == "Forecast & Prediction":
    #st.markdown("## 🔮 Forecast & Prediction (Next 12 Months)")

    # Prepare data
    forecast_df = filtered_df.copy()
    forecast_df["Timestamp"] = pd.to_datetime(forecast_df["Timestamp"], format="%d/%m/%Y %H:%M")
    forecast_df["YearMonth"] = forecast_df["Timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly_events = forecast_df.groupby("YearMonth").size().reset_index(name="Event Count")

    if monthly_events.empty or len(monthly_events) < 3:
        st.warning("⚠️ Not enough monthly data points available for forecasting.")
    else:
        # Convert dates to ordinal values for regression
        monthly_events["ds_ordinal"] = monthly_events["YearMonth"].map(datetime.toordinal)
        X = monthly_events["ds_ordinal"].values.reshape(-1, 1)
        y = monthly_events["Event Count"]

        # Polynomial Regression
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)

        # Forecast next 12 months
        last_month = monthly_events["YearMonth"].max()
        future_months = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=12, freq='MS')
        X_future = future_months.map(datetime.toordinal).values.reshape(-1, 1)
        X_future_poly = poly.transform(X_future)
        y_pred = model.predict(X_future_poly)

        forecast_result = pd.DataFrame({
            "Month": future_months,
            "Predicted Events": y_pred.astype(int)
        })

        # Two-column layout
        col1, col2 = st.columns(2)

        # Historical vs Forecasted Line Chart
        with col1:
            st.markdown("📈 Historical + Forecasted Events (Monthly)")
            #st.markdown("This projection shows product activity trends and predictions across the next 12 months.")
            fig1, ax1 = plt.subplots()
            ax1.plot(monthly_events["YearMonth"], y, label="Historical", marker='o')
            ax1.plot(forecast_result["Month"], forecast_result["Predicted Events"], label="Forecast", linestyle="--", marker='x')
            ax1.set_title("Monthly Event Forecast")
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Event Count")
            ax1.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig1)

        # Forecast Summary
        with col2:
            st.markdown("📋 Forecast Summary (Next 12 Months)")
            forecast_result["Month"] = forecast_result["Month"].dt.strftime("%B %Y")
            st.dataframe(forecast_result.head(12), use_container_width=True)

            peak_month = forecast_result.loc[forecast_result["Predicted Events"].idxmax()]
            avg_forecast = int(forecast_result["Predicted Events"].mean())

            #st.metric(label="📅 Predicted Peak Month", value=peak_month["Month"], delta=f'{peak_month["Predicted Events"]} Events')
            #st.metric(label="📊 Avg Forecasted Events/Month", value=avg_forecast)

        # Trend interpretation
        #delta = forecast_result["Predicted Events"].iloc[-1] - y.iloc[-1]
        #if delta > 0:
        #    trend = f"📈 An upward trend is expected, with a projected increase of {delta} events."
        #elif delta < 0:
        #    trend = f"📉 A decrease of {abs(delta)} events is expected over the next 12 months."
        #else:
        #    trend = "➡️ The event trend appears stable in the forecast."

        #st.success(f"**Trend Insight:** {trend}")


# ---------------------- SECTION: SALES INSIGHTS ---------------------- #
# ---------------------- SECTION: SALES INSIGHTS ---------------------- #
elif section == "Sales Insights":
    sales_df = filtered_df.copy()

    # --- Basic Counts --- #
    total_users = len(sales_df)
    buy_count = sales_df["Buy"].sum()
    interest_count = sales_df["Interest"].sum()
    no_buy_count = total_users - buy_count

    # --- Revenue --- #
    total_revenue = sales_df.loc[sales_df["Buy"] == 1, "Revenue"].sum()
    avg_revenue_per_user = total_revenue / total_users if total_users > 0 else 0
    revenue_per_buyer = total_revenue / buy_count if buy_count > 0 else 0

    # --- KPI Calculations --- #
    lead_conversion_rate = (buy_count / interest_count) * 100 if interest_count > 0 else 0
    overall_conversion_rate = (buy_count / total_users) * 100 if total_users > 0 else 0
    drop_off_rate = (no_buy_count / total_users) * 100 if total_users > 0 else 0

    # --- Color Helper --- #
    def get_color_style(value, good, warn, reverse=False):
        if reverse:
            return "limegreen" if value <= good else "orange" if value <= warn else "red"
        else:
            return "limegreen" if value >= good else "orange" if value >= warn else "red"

    lead_color = get_color_style(lead_conversion_rate, 50, 30)
    drop_color = get_color_style(drop_off_rate, 20, 40, reverse=True)
    overall_color = get_color_style(overall_conversion_rate, 40, 25)
    rev_color = get_color_style(avg_revenue_per_user, 100, 50)
    buyer_rev_color = get_color_style(revenue_per_buyer, 150, 80)

    # --- KPI Card --- #
    def kpi_card(title, value, color, tooltip=None):
        tooltip_attr = f'title="{tooltip}"' if tooltip else ""
        st.markdown(f"""
        <div style="padding: 0.4rem; border-radius: 8px; background-color: #1e1e1e;
                    border: 1px solid #333; box-shadow: 0 0 5px rgba(0, 255, 0, 0.15);
                    text-align: center; margin-bottom: 0.6rem;" {tooltip_attr}>
            <h5 style="margin: 0 0 5px 0; color: #d4d4d4; font-size:0.9rem; font-family: 'Segoe UI';">{title}</h5>
            <p style="color: {color}; font-size: 1rem; font-weight: 600; margin: 0;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- KPI Summary --- #
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: kpi_card("Total Buyers", f"{buy_count:,}", "#f5f5f5")
    with col2: kpi_card("Interested Users", f"{interest_count:,}", "#f5f5f5")
    with col3: kpi_card("Non-Buyers", f"{no_buy_count:,}", "#f5f5f5")
    with col4: kpi_card("Conversion Rate", f"{lead_conversion_rate:.2f}%", lead_color)
    with col5: kpi_card("Drop-Off Rate", f"{drop_off_rate:.2f}%", drop_color)

    col6, col7, col8 = st.columns(3)
    with col6: kpi_card("Overall Conversion Rate", f"{overall_conversion_rate:.2f}%", overall_color)
    with col7: kpi_card("Total Revenue", f"${total_revenue:,.2f}", "#f5f5f5")
    with col8: kpi_card("Revenue/Buyer", f"${revenue_per_buyer:,.2f}", buyer_rev_color)

    # --- Charts in One Row --- #
    colA, colB, colC = st.columns(3)

    # --- Chart 1: Revenue Gauge --- #
    with colA:
        st.markdown("📈 Revenue Progress")
        current_revenue = sales_df["Revenue"].sum()
        revenue_target = 25000000
        percentage = round((current_revenue / revenue_target) * 100, 2)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_revenue,
            delta={'reference': revenue_target},
            number={'prefix': "$", 'valueformat': ',.0f'},
            title={'text': "Revenue vs Target", 'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [0, revenue_target], 'tickcolor': 'white'},
                'bar': {'color': "limegreen"},
                'steps': [
                    {'range': [0, revenue_target * 0.5], 'color': "#2a2a2a"},
                    {'range': [revenue_target * 0.5, revenue_target * 0.8], 'color': "#005f5f"},
                    {'range': [revenue_target * 0.8, revenue_target], 'color': "#007f7f"},
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': revenue_target}
            }
        ))
        fig.update_layout(height=260, paper_bgcolor='#0e1117', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    # --- Chart 2: Donut Chart --- #
    with colB:
        st.markdown("🌍 Buyers by Country")
        buyers_by_country = sales_df.groupby("Country")["Buy"].sum().reset_index()
        buyers_by_country.columns = ["Country", "Buyers"]
        
        fig_pie_country = px.pie(
            buyers_by_country,
            names="Country",
            values="Buyers",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
    )
        
        fig_pie_country.update_traces(
            textinfo="percent+label",
            textposition="inside",  # ✅ This moves the text outside slices
            textfont_size=14,
            textfont_color="white",  # ✅ Ensure label text is readable
            insidetextorientation='radial',
            marker=dict(line=dict(color='#0e1117', width=2))  # slight outline for better visibility
    )
        
        fig_pie_country.update_layout(
            height=200,
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font_color='white',
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(
                font=dict(color="white"),
                orientation="v",
                bgcolor="#1c1c1c",
                bordercolor="#333",
                borderwidth=1
        )
    )
        st.plotly_chart(fig_pie_country, use_container_width=True)


    # --- Chart 3: Monthly Revenue --- #
    with colC:
        st.markdown("📅 Monthly Revenue")
        sales_df["Timestamp"] = pd.to_datetime(sales_df["Timestamp"], dayfirst=True)
        sales_df["Month"] = pd.to_datetime(sales_df["Timestamp"].dt.to_period("M").astype(str))

        monthly_revenue = sales_df.groupby("Month")["Revenue"].sum().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_revenue["Month"],
            y=monthly_revenue["Revenue"],
            mode="lines+markers",
            line=dict(color="deepskyblue"),
            name="Revenue"
        ))

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue ($)",
            height=260,
            paper_bgcolor='#0e1117',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            margin=dict(t=40, b=40, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)













#-------------------------------CUSTOMER BEHAVIOUR------------------------------------------------
elif section == "Customer Behaviour":
    import plotly.express as px
    from datetime import datetime

    # Prepare data
    df_time = filtered_df.copy()
    df_time["Hour"] = pd.to_datetime(df_time["Timestamp"], format="%d/%m/%Y %H:%M").dt.hour

    # --- TOP ROW ---
    col1, col2 = st.columns(2, gap="small")

    # 🔹 1. Event Distribution by Job Type
    with col1:
        st.markdown("📊 Events Categorized by Profession Type")
        job_event_df = filtered_df.groupby(["Job_Type", "Event_Type"]).size().reset_index(name="Count")
        fig_job_event = px.bar(
            job_event_df, x="Job_Type", y="Count", color="Event_Type", barmode="group",
            height=200, color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_job_event.update_layout(
            margin=dict(t=10, l=0, r=0, b=10),
            xaxis_title="Profession",
            yaxis_title="Total Events Counts") #**
        st.plotly_chart(fig_job_event, use_container_width=True)

    # 🔹 2. Hourly Activity by Job Type
    with col2:
        st.markdown("⏰ Hourly Breakdown by Profession In a Day")
        hourly_job = df_time.groupby(["Hour", "Job_Type"]).size().reset_index(name="Count")
        fig_hourly = px.line(
            hourly_job, x="Hour", y="Count", color="Job_Type",
            height=200
        )
        fig_hourly.update_layout(
            margin=dict(t=10, l=0, r=0, b=10),
            yaxis_title="Total No. of Users")
        st.plotly_chart(fig_hourly, use_container_width=True)

    #st.markdown("---", unsafe_allow_html=True)

    # --- BOTTOM ROW ---
    col3, col4 = st.columns(2, gap="small")

    # 🔹 3. Frequent Event Types by Country
    with col3:
        st.markdown("🌍 Frequency of Event Types by Country")
        event_country_df = filtered_df.groupby(["Country", "Event_Type"]).size().reset_index(name="Count")
        fig_event_country = px.bar(
            event_country_df, x="Country", y="Count", color="Event_Type", barmode="stack",
            height=200, color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_event_country.update_layout(
            margin=dict(t=10, l=0, r=0, b=10),
            yaxis_title="Total No. of Events")
        st.plotly_chart(fig_event_country, use_container_width=True)

    # 🔹 4. Conversion Rate by Job Type (Donut)
    with col4:
        st.markdown("🧾 Annual Product Usage by Profession")
        conversion_df = filtered_df.groupby("Job_Type")["Buy"].sum().reset_index(name="Conversions")
        fig_donut = px.pie(
            conversion_df, names="Job_Type", values="Conversions", hole=0.5,
            height=200, color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig_donut.update_layout(margin=dict(t=10, l=0, r=0, b=10))
        st.plotly_chart(fig_donut, use_container_width=True)





