import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import statistics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Instagram Analytics",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Top header bar */
.dashboard-header {
    background: linear-gradient(135deg, #f09433 0%, #e6683c 25%, #dc2743 50%, #cc2366 75%, #bc1888 100%);
    padding: 2rem 2.5rem;
    border-radius: 18px;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.dashboard-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.dashboard-header p {
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-size: 0.95rem;
}

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    border-left: 4px solid #dc2743;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 0.6rem;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1;
}
.metric-card .delta {
    font-size: 0.8rem;
    color: #cc2366;
    margin-top: 0.3rem;
}

/* Section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a1a2e;
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #f0f0f0;
}

/* Prediction output box */
.pred-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 16px;
    padding: 1.6rem;
    color: white;
    text-align: center;
    margin-top: 0.5rem;
}
.pred-box .model-name {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.6);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.pred-box .pred-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f09433, #dc2743, #bc1888);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.pred-box .pred-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.55);
    margin-top: 0.2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #fafafa;
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f5f5f5;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* Remove extra top padding */
.block-container { padding-top: 1rem; }

/* Insight chip */
.insight-chip {
    display: inline-block;
    background: #fff0f3;
    color: #dc2743;
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 0.2rem 0.2rem 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("Instagram_Analytics.csv")
    df.drop_duplicates(inplace=True)
    df['upload_date'] = pd.to_datetime(df['upload_date'])
    df['year'] = df['upload_date'].dt.year
    df['month'] = df['upload_date'].dt.month
    df['day_of_the_week'] = df['upload_date'].dt.day_name()

    month_map = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
                 7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
    df['month'] = df['month'].map(month_map)
    df['month'] = pd.Categorical(df['month'],
        categories=['January','February','March','April','May','June','July',
                    'August','September','October','November','December'], ordered=True)
    df['day_of_the_week'] = pd.Categorical(df['day_of_the_week'],
        categories=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'], ordered=True)

    categorical_data = ['media_type','traffic_source','content_category','year','month','day_of_the_week']
    for col in categorical_data:
        df[col] = df[col].astype(str).str.strip()

    # Feature engineering
    df['total_interactions'] = (df['likes'] + df['comments'] + df['shares'] + df['saves']).astype('int64')
    df['caption_density'] = (df['caption_length'] / (df['hashtags_count'] + 1)).astype('int64')
    df['likes_per_reach'] = df['likes'] / (df['reach'] + 1)
    df['saves_per_reach'] = df['saves'] / (df['reach'] + 1)
    df['comments_per_reach'] = df['comments'] / (df['reach'] + 1)
    df['shares_per_reach'] = df['shares'] / (df['reach'] + 1)
    df['total_interactions_per_reach'] = df['total_interactions'] / (df['reach'] + 1)
    df['log_likes'] = np.log1p(df['likes'])
    return df


@st.cache_data
def get_ml_df(df):
    ml_df = df.copy()
    ml_df.drop_duplicates(inplace=True)
    ml_df.dropna(inplace=True)

    def find_anomalies(series):
        std = statistics.stdev(series)
        mean = statistics.mean(series)
        cut = std * 3
        return series[(series > mean + cut) | (series < mean - cut)].tolist()

    for col in ['engagement_rate','likes_per_reach','saves_per_reach','comments_per_reach',
                'shares_per_reach','total_interactions_per_reach','log_likes','caption_density']:
        anomalies = find_anomalies(ml_df[col])
        ml_df = ml_df[~ml_df[col].isin(anomalies)]
    return ml_df


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        arts = joblib.load('model_artifacts/pipeline_artifacts.joblib')
        return arts, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict_engagement(input_dict, artifacts):
    """Run feature engineering on raw input and predict with both models."""
    df_in = pd.DataFrame([input_dict])

    df_in['total_interactions'] = (df_in['likes'] + df_in['comments'] +
                                   df_in['shares'] + df_in['saves'])
    df_in['caption_density'] = (df_in['caption_length'] /
                                (df_in['hashtags_count'] + 1)).astype(int)
    df_in['likes_per_reach']   = df_in['likes']   / (df_in['reach'] + 1)
    df_in['saves_per_reach']   = df_in['saves']   / (df_in['reach'] + 1)
    df_in['comments_per_reach']= df_in['comments']/ (df_in['reach'] + 1)
    df_in['shares_per_reach']  = df_in['shares']  / (df_in['reach'] + 1)
    df_in['total_interactions_per_reach'] = df_in['total_interactions'] / (df_in['reach'] + 1)
    df_in['log_likes'] = np.log1p(df_in['likes'])
    df_in['year'] = str(df_in['year'].values[0])

    numeric_features = artifacts['numeric_features']
    categorical_data = artifacts.get('categorical_data',
        ['media_type','traffic_source','content_category','year','month','day_of_the_week'])

    model_numeric = [c for c in numeric_features if c != 'engagement_rate']
    model_features = model_numeric + categorical_data

    # drop log_likes if present in model_features
    model_features = [f for f in model_features if f != 'log_likes']
    X_new = df_in[model_features]

    results = {}
    lr_model  = artifacts.get('linear_regression_model')
    xgb_model = artifacts.get('xgboost_model')

    if lr_model:
        results['Linear Regression'] = float(lr_model.predict(X_new)[0])
    if xgb_model:
        results['XGBoost (Tuned)'] = float(xgb_model.predict(X_new)[0])
    return results


# ─────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────
IG_COLORS = ['#f09433','#e6683c','#dc2743','#cc2366','#bc1888','#833ab4','#5851db','#405de6']
PLOT_TEMPLATE = "plotly_white"

def ig_fig(fig, height=400):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        font_family="DM Sans",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
df = load_and_prepare_data()
ml_df = get_ml_df(df)
artifacts, model_error = load_models()

# ── HEADER ──────────────────────────────────
st.markdown("""
<div class="dashboard-header">
  <div>
    <h1>📸 Instagram Analytics</h1>
    <p>Engagement intelligence dashboard — EDA · Feature Insights · ML Predictions</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ────────────────────────────────────
tab_overview, tab_eda, tab_model, tab_predict = st.tabs([
    "📊 Overview", "🔍 EDA", "🤖 Model Performance", "🎯 Predict"
])


# ════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════
with tab_overview:
    # KPI row
    total_posts  = df['post_id'].nunique() if 'post_id' in df.columns else len(df)
    avg_eng      = df['engagement_rate'].mean()
    avg_reach    = df['reach'].mean()
    avg_impressions = df['impressions'].mean()
    top_media    = df.groupby('media_type')['engagement_rate'].mean().idxmax()
    top_category = df.groupby('content_category')['engagement_rate'].mean().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Total Posts",        f"{total_posts:,}",       "unique post IDs"),
        ("Avg Engagement Rate",f"{avg_eng:,.2f}",        "across all posts"),
        ("Avg Reach",          f"{avg_reach:,.0f}",      "per post"),
        ("Avg Impressions",    f"{avg_impressions:,.0f}","per post"),
    ]
    for col, (label, value, sub) in zip([c1,c2,c3,c4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="delta">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="section-title">Engagement Rate by Media Type</div>', unsafe_allow_html=True)
        media_eng = df.groupby('media_type')['engagement_rate'].mean().reset_index()
        fig = px.bar(media_eng, x='media_type', y='engagement_rate',
                     color='media_type', color_discrete_sequence=IG_COLORS,
                     text_auto='.1f', labels={'engagement_rate':'Avg Engagement Rate','media_type':'Media Type'})
        fig.update_traces(textposition='outside')
        st.plotly_chart(ig_fig(fig), use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Media Type Share of Engagement</div>', unsafe_allow_html=True)
        fig2 = px.pie(media_eng, values='engagement_rate', names='media_type',
                      color_discrete_sequence=IG_COLORS, hole=0.45)
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(ig_fig(fig2, height=350), use_container_width=True)

    st.markdown('<div class="section-title">Engagement Rate by Content Category</div>', unsafe_allow_html=True)
    cat_eng = df.groupby('content_category')['engagement_rate'].mean().sort_values(ascending=False).reset_index()
    fig3 = px.bar(cat_eng, x='content_category', y='engagement_rate',
                  color='engagement_rate', color_continuous_scale=IG_COLORS,
                  text_auto='.1f', labels={'engagement_rate':'Avg Engagement Rate','content_category':'Category'})
    fig3.update_traces(textposition='outside')
    fig3.update_coloraxes(showscale=False)
    st.plotly_chart(ig_fig(fig3, height=380), use_container_width=True)

    # Quick insight chips
    st.markdown(f"""
    <div style="margin-top:0.5rem">
      <span class="insight-chip">🏆 Best Media: {top_media}</span>
      <span class="insight-chip">🎨 Best Category: {top_category}</span>
      <span class="insight-chip">📅 Best Day: Wednesday</span>
      <span class="insight-chip">📆 Best Month: April</span>
      <span class="insight-chip">🔍 Best Traffic: Explore</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════
with tab_eda:
    eda_section = st.selectbox("Select Analysis",
        ["Correlation Heatmap", "Reach vs Impressions", "Time Series — Engagement",
         "Time Series — Followers", "Top 10 Posts", "Distribution of Features",
         "Post Frequency Analysis", "Content Category × Media Type"])

    if eda_section == "Correlation Heatmap":
        st.markdown('<div class="section-title">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = ['likes','comments','shares','saves','impressions','reach',
                     'followers_gained','caption_length','hashtags_count',
                     'total_interactions','engagement_rate','likes_per_reach',
                     'saves_per_reach','comments_per_reach','total_interactions_per_reach']
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[corr_cols].corr().round(2)
        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                        aspect='auto', zmin=-1, zmax=1)
        fig.update_layout(height=600, title="Pearson Correlation Matrix",
                          font_family="DM Sans", coloraxis_colorbar_title="r")
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 `reach` and `impressions` are strongly correlated. Engagement ratio features show the clearest direct relationship with `engagement_rate`.")

    elif eda_section == "Reach vs Impressions":
        st.markdown('<div class="section-title">Reach vs Impressions by Media Type</div>', unsafe_allow_html=True)
        fig = px.scatter(df, x='reach', y='impressions', color='media_type',
                         color_discrete_sequence=IG_COLORS, opacity=0.65,
                         hover_data=['post_id'] if 'post_id' in df.columns else None,
                         trendline='ols', trendline_scope='overall',
                         labels={'reach':'Reach','impressions':'Impressions','media_type':'Media Type'})
        st.plotly_chart(ig_fig(fig, height=480), use_container_width=True)
        st.info("💡 Strong positive linear relationship — posts with higher reach consistently achieve higher impressions.")

    elif eda_section == "Time Series — Engagement":
        st.markdown('<div class="section-title">Monthly & Weekly Engagement Trends</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            monthly = df.groupby(['month','media_type'], observed=True)['engagement_rate'].mean().reset_index()
            fig = px.line(monthly, x='month', y='engagement_rate', color='media_type',
                          markers=True, color_discrete_sequence=IG_COLORS,
                          labels={'month':'Month','engagement_rate':'Avg Engagement Rate','media_type':'Media Type'},
                          title="Monthly Engagement Rate by Media Type")
            st.plotly_chart(ig_fig(fig), use_container_width=True)
        with col2:
            weekly = df.groupby(['day_of_the_week','media_type'], observed=True)['engagement_rate'].mean().reset_index()
            fig2 = px.line(weekly, x='day_of_the_week', y='engagement_rate', color='media_type',
                           markers=True, color_discrete_sequence=IG_COLORS,
                           labels={'day_of_the_week':'Day','engagement_rate':'Avg Engagement Rate','media_type':'Media Type'},
                           title="Weekly Engagement Rate by Media Type")
            st.plotly_chart(ig_fig(fig2), use_container_width=True)
        st.info("💡 Wednesday in April consistently shows the highest engagement across media types.")

    elif eda_section == "Time Series — Followers":
        st.markdown('<div class="section-title">Followers Gained Over Time</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            monthly = df.groupby(['month','media_type'], observed=True)['followers_gained'].mean().reset_index()
            fig = px.line(monthly, x='month', y='followers_gained', color='media_type',
                          markers=True, color_discrete_sequence=IG_COLORS,
                          title="Monthly Followers Gained by Media Type")
            st.plotly_chart(ig_fig(fig), use_container_width=True)
        with col2:
            weekly = df.groupby(['day_of_the_week','media_type'], observed=True)['followers_gained'].mean().reset_index()
            fig2 = px.line(weekly, x='day_of_the_week', y='followers_gained', color='media_type',
                           markers=True, color_discrete_sequence=IG_COLORS,
                           title="Weekly Followers Gained by Media Type")
            st.plotly_chart(ig_fig(fig2), use_container_width=True)

    elif eda_section == "Top 10 Posts":
        st.markdown('<div class="section-title">Top 10 Posts</div>', unsafe_allow_html=True)
        metric_choice = st.radio("Rank by:", ["Engagement Rate","Total Interactions"], horizontal=True)
        if metric_choice == "Engagement Rate" and 'post_id' in df.columns:
            top = (df.groupby(['post_id','media_type','content_category'], observed=True)
                     ['engagement_rate'].mean().round(2).reset_index()
                     .sort_values('engagement_rate', ascending=False).head(10))
            fig = px.bar(top, x='engagement_rate', y='post_id', orientation='h',
                         color='content_category', color_discrete_sequence=IG_COLORS,
                         text_auto='.1f', title="Top 10 Posts by Engagement Rate")
        else:
            top = (df.groupby(['post_id','media_type','content_category'], observed=True)
                     ['total_interactions'].mean().round(0).reset_index()
                     .sort_values('total_interactions', ascending=False).head(10))
            fig = px.bar(top, x='total_interactions', y='post_id', orientation='h',
                         color='content_category', color_discrete_sequence=IG_COLORS,
                         text_auto=',.0f', title="Top 10 Posts by Total Interactions")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(ig_fig(fig, height=480), use_container_width=True)
        st.dataframe(top, use_container_width=True)

    elif eda_section == "Distribution of Features":
        st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)
        num_cols = ['likes','comments','shares','saves','impressions','reach',
                    'followers_gained','engagement_rate','total_interactions']
        num_cols = [c for c in num_cols if c in df.columns]
        chosen = st.selectbox("Feature:", num_cols)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x=chosen, color='media_type', nbins=40,
                               barmode='overlay', opacity=0.7,
                               color_discrete_sequence=IG_COLORS,
                               title=f"Distribution of {chosen}")
            st.plotly_chart(ig_fig(fig), use_container_width=True)
        with col2:
            fig2 = px.box(df, x='media_type', y=chosen, color='media_type',
                          color_discrete_sequence=IG_COLORS,
                          title=f"{chosen} by Media Type")
            st.plotly_chart(ig_fig(fig2), use_container_width=True)

    elif eda_section == "Post Frequency Analysis":
        st.markdown('<div class="section-title">Post Frequency vs Engagement & Followers</div>', unsafe_allow_html=True)
        pf = (df.groupby('day_of_the_week', observed=True)
              [['post_id','engagement_rate','followers_gained']]
              .agg({'post_id':'count','engagement_rate':'mean','followers_gained':'mean'})
              .round(2).rename(columns={'post_id':'num_posts'})
              .reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
              .reset_index())

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=pf['day_of_the_week'], y=pf['num_posts'],
                             name='Number of Posts', marker_color='#405de6', opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=pf['day_of_the_week'], y=pf['engagement_rate'],
                                 name='Avg Engagement Rate', mode='lines+markers',
                                 marker=dict(size=8), line=dict(color='#dc2743', width=3)), secondary_y=True)
        fig.update_layout(height=420, template=PLOT_TEMPLATE, font_family="DM Sans",
                          title="Post Frequency vs Engagement Rate",
                          plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          legend=dict(bgcolor='rgba(0,0,0,0)'))
        fig.update_yaxes(title_text="Number of Posts", secondary_y=False)
        fig.update_yaxes(title_text="Avg Engagement Rate", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    elif eda_section == "Content Category × Media Type":
        st.markdown('<div class="section-title">Content Category × Media Type Engagement</div>', unsafe_allow_html=True)
        pivot = (df.groupby(['content_category','media_type'], observed=True)
                   ['engagement_rate'].mean().round(2).unstack())
        fig = px.imshow(pivot, color_continuous_scale=IG_COLORS, text_auto='.1f', aspect='auto',
                        title="Avg Engagement Rate: Content Category × Media Type")
        fig.update_layout(height=500, font_family="DM Sans",
                          coloraxis_colorbar_title="Eng. Rate")
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-title">Model Evaluation on Held-Out Test Set</div>', unsafe_allow_html=True)

    if artifacts is None:
        st.warning(f"⚠️ Model artifacts not found ({model_error}). Showing re-trained evaluation.")
        from sklearn.linear_model import LinearRegression
        from xgboost import XGBRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder

        numeric_features = ml_df.select_dtypes(['int64','float64']).columns.tolist()
        categorical_data = ['media_type','traffic_source','content_category','year','month','day_of_the_week']
        model_numeric = [c for c in numeric_features if c not in ['engagement_rate','log_likes']]
        model_features = model_numeric + categorical_data

        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), model_numeric),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_data)
        ])

        X = ml_df[model_features]
        y = ml_df['engagement_rate']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lr = Pipeline([('pre', preprocessor), ('reg', LinearRegression())])
        lr.fit(X_train, y_train)
        xgb = Pipeline([('pre', preprocessor), ('model', XGBRegressor(n_estimators=300, max_depth=9, learning_rate=0.1, random_state=42))])
        xgb.fit(X_train, y_train)

        lr_pred  = lr.predict(X_test)
        xgb_pred = xgb.predict(X_test)
        model_results = {
            'Linear Regression': {'pred': lr_pred},
            'XGBoost': {'pred': xgb_pred},
        }
        y_test_vals = y_test
        xgb_fi = pd.DataFrame({'Features': preprocessor.get_feature_names_out(),
                                'Importance': xgb.named_steps['model'].feature_importances_}).sort_values('Importance', ascending=False)
    else:
        # Re-evaluate using loaded models
        numeric_features = artifacts['numeric_features']
        categorical_data = artifacts.get('categorical_data',
            ['media_type','traffic_source','content_category','year','month','day_of_the_week'])
        model_numeric = [c for c in numeric_features if c not in ['engagement_rate','log_likes']]
        model_features = model_numeric + categorical_data
        model_features = [f for f in model_features if f in ml_df.columns]

        X = ml_df[[f for f in model_features if f in ml_df.columns]]
        y = ml_df['engagement_rate']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_results = {}
        if artifacts.get('linear_regression_model'):
            model_results['Linear Regression'] = {'pred': artifacts['linear_regression_model'].predict(X_test)}
        if artifacts.get('xgboost_model'):
            model_results['XGBoost (Tuned)'] = {'pred': artifacts['xgboost_model'].predict(X_test)}
        y_test_vals = y_test
        xgb_fi = artifacts.get('xgboost_feature_importances', None)

    # Metrics table
    rows = []
    for name, d in model_results.items():
        p = d['pred']
        rows.append({'Model': name,
                     'MAE': round(mean_absolute_error(y_test_vals, p), 4),
                     'RMSE': round(np.sqrt(mean_squared_error(y_test_vals, p)), 4),
                     'R²': round(r2_score(y_test_vals, p), 4)})
    metrics_df = pd.DataFrame(rows)

    col1, col2, col3 = st.columns(3)
    for col, metric in zip([col1, col2, col3], ['MAE','RMSE','R²']):
        with col:
            fig = px.bar(metrics_df, x='Model', y=metric, color='Model',
                         color_discrete_sequence=IG_COLORS, text_auto='.4f',
                         title=metric)
            fig.update_traces(textposition='outside')
            st.plotly_chart(ig_fig(fig, height=320), use_container_width=True)

    # Actual vs Predicted scatter
    st.markdown('<div class="section-title">Actual vs Predicted Engagement Rate</div>', unsafe_allow_html=True)
    scatter_cols = st.columns(len(model_results))
    for col, (name, d) in zip(scatter_cols, model_results.items()):
        with col:
            pred_df = pd.DataFrame({'Actual': y_test_vals.values, 'Predicted': d['pred']})
            fig = px.scatter(pred_df, x='Actual', y='Predicted', opacity=0.5,
                             color_discrete_sequence=[IG_COLORS[2]],
                             title=name)
            min_v = pred_df.min().min()
            max_v = pred_df.max().max()
            fig.add_shape(type='line', x0=min_v, x1=max_v, y0=min_v, y1=max_v,
                          line=dict(color='red', dash='dash', width=2))
            st.plotly_chart(ig_fig(fig, height=340), use_container_width=True)

    # Feature importance
    st.markdown('<div class="section-title">XGBoost Feature Importance (Top 15)</div>', unsafe_allow_html=True)
    if xgb_fi is not None:
        feat_col = 'Features' if 'Features' in xgb_fi.columns else xgb_fi.columns[0]
        imp_col  = 'Importance' if 'Importance' in xgb_fi.columns else xgb_fi.columns[1]
        top_fi = xgb_fi.sort_values(imp_col, ascending=False).head(15)
        fig = px.bar(top_fi, x=imp_col, y=feat_col, orientation='h',
                     color=imp_col, color_continuous_scale=IG_COLORS,
                     title="Top 15 Features — XGBoost")
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(ig_fig(fig, height=480), use_container_width=True)


# ════════════════════════════════════════════
# TAB 4 — PREDICT
# ════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-title">Predict Engagement Rate for a New Post</div>', unsafe_allow_html=True)
    st.markdown("Fill in the post details below and click **Predict** to get engagement forecasts.")

    with st.form("prediction_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            media_type = st.selectbox("Media Type", ['Reel','Photo','Video','Carousel'])
            content_category = st.selectbox("Content Category",
                ['Technology','Fitness','Beauty','Music','Photography',
                 'Food','Lifestyle','Travel','Fashion','Comedy'])
        with r1c2:
            traffic_source = st.selectbox("Traffic Source",
                ['Home Feed','Hashtags','Reels Feed','External','Profile','Explore'])
            upload_year = st.selectbox("Upload Year", ['2024','2025'])
        with r1c3:
            upload_month = st.selectbox("Upload Month",
                ['January','February','March','April','May','June',
                 'July','August','September','October','November','December'])
            day_of_week = st.selectbox("Day of Week",
                ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'])

        st.markdown("**Engagement Metrics**")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        likes      = r2c1.number_input("Likes",      min_value=0, value=50000, step=1000)
        comments   = r2c2.number_input("Comments",   min_value=0, value=3000,  step=100)
        shares     = r2c3.number_input("Shares",     min_value=0, value=1500,  step=100)
        saves      = r2c4.number_input("Saves",      min_value=0, value=6000,  step=100)

        st.markdown("**Reach & Impressions**")
        r3c1, r3c2, r3c3 = st.columns(3)
        reach             = r3c1.number_input("Reach",             min_value=0, value=1_200_000, step=10000)
        impressions       = r3c2.number_input("Impressions",       min_value=0, value=1_400_000, step=10000)
        followers_gained  = r3c3.number_input("Followers Gained",  min_value=0, value=800,       step=10)

        st.markdown("**Caption Details**")
        r4c1, r4c2 = st.columns(2)
        caption_length = r4c1.number_input("Caption Length (chars)", min_value=0, value=1200, step=50)
        hashtags_count = r4c2.number_input("Hashtags Count",         min_value=0, value=20,   step=1)

        submitted = st.form_submit_button("🚀 Predict Engagement Rate", use_container_width=True)

    if submitted:
        input_dict = {
            'media_type': media_type,
            'content_category': content_category,
            'traffic_source': traffic_source,
            'year': upload_year,
            'month': upload_month,
            'day_of_the_week': day_of_week,
            'likes': likes,
            'comments': comments,
            'shares': shares,
            'saves': saves,
            'reach': reach,
            'impressions': impressions,
            'followers_gained': followers_gained,
            'caption_length': caption_length,
            'hashtags_count': hashtags_count,
        }

        if artifacts is None:
            st.error("⚠️ Model artifacts not loaded. Please ensure `model_artifacts/pipeline_artifacts.joblib` exists.")
        else:
            with st.spinner("Running predictions…"):
                preds = predict_engagement(input_dict, artifacts)

            avg_pred = np.mean(list(preds.values()))
            avg_dataset = df['engagement_rate'].mean()

            # Main output cards
            pred_cols = st.columns(len(preds) + 1)
            for col, (model_name, val) in zip(pred_cols, preds.items()):
                col.markdown(f"""
                <div class="pred-box">
                  <div class="model-name">{model_name}</div>
                  <div class="pred-value">{val:,.2f}</div>
                  <div class="pred-label">Predicted Engagement Rate</div>
                </div>""", unsafe_allow_html=True)

            pred_cols[-1].markdown(f"""
            <div class="pred-box" style="background:linear-gradient(135deg,#dc2743,#bc1888)">
              <div class="model-name">Ensemble Average</div>
              <div class="pred-value" style="-webkit-text-fill-color:white;background:none">{avg_pred:,.2f}</div>
              <div class="pred-label">vs. dataset avg {avg_dataset:,.2f}</div>
            </div>""", unsafe_allow_html=True)

            # Gauge
            st.markdown("---")
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_pred,
                delta={'reference': avg_dataset, 'valueformat': '.2f',
                       'increasing': {'color': '#2ecc71'}, 'decreasing': {'color': '#e74c3c'}},
                title={'text': "Predicted vs Dataset Average Engagement Rate", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, avg_dataset * 3], 'tickformat': ',.0f'},
                    'bar': {'color': '#dc2743'},
                    'steps': [
                        {'range': [0, avg_dataset * 0.75],          'color': '#ffd6d6'},
                        {'range': [avg_dataset * 0.75, avg_dataset], 'color': '#ffb3b3'},
                        {'range': [avg_dataset, avg_dataset * 3],    'color': '#d4f5d4'},
                    ],
                    'threshold': {'line': {'color': '#333', 'width': 3},
                                  'thickness': 0.85, 'value': avg_dataset}
                }
            ))
            fig.update_layout(height=320, font_family="DM Sans",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            # Derived features summary
            with st.expander("📐 Computed Feature Engineering Values"):
                total_interactions = likes + comments + shares + saves
                caption_density = caption_length // (hashtags_count + 1)
                st.json({
                    "total_interactions": int(total_interactions),
                    "caption_density": int(caption_density),
                    "likes_per_reach": round(likes / (reach + 1), 6),
                    "saves_per_reach": round(saves / (reach + 1), 6),
                    "comments_per_reach": round(comments / (reach + 1), 6),
                    "shares_per_reach": round(shares / (reach + 1), 6),
                    "total_interactions_per_reach": round(total_interactions / (reach + 1), 6),
                    "log_likes": round(np.log1p(likes), 6),
                })