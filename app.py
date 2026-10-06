import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# DARK THEME
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       GLOBAL PAGE
       ======================================================== */

    .stApp {
        background-color: #0b0f14 !important;
        color: #e5e7eb !important;
    }

    .main {
        background-color: #0b0f14 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Remove default header background */
    header[data-testid="stHeader"] {
        background-color: #0b0f14 !important;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #0a0e14 !important;
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #27303d !important;
    }

    /* ========================================================
       TEXT
       ======================================================== */

    h1, h2, h3, h4, h5, h6 {
        color: #f3f4f6 !important;
    }

    p {
        color: #aeb7c4 !important;
    }

    label {
        color: #cbd5e1 !important;
    }

    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #f3f4f6 !important;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }

    .subtitle {
        font-size: 15px;
        color: #8f9baa !important;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 650;
        color: #f3f4f6 !important;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: #151a22 !important;
        border-color: #303846 !important;
        color: #f3f4f6 !important;
    }

    div[data-baseweb="select"] span {
        color: #f3f4f6 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #aeb7c4 !important;
    }

    /* Dropdown menu */
    div[role="listbox"] {
        background-color: #151a22 !important;
        border: 1px solid #303846 !important;
    }

    div[role="option"] {
        background-color: #151a22 !important;
        color: #e5e7eb !important;
    }

    div[role="option"]:hover {
        background-color: #222936 !important;
    }

    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    div[data-testid="stNumberInput"] input {
        background-color: #151a22 !important;
        color: #f3f4f6 !important;
        border-color: #303846 !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #151a22 !important;
        color: #cbd5e1 !important;
        border-color: #303846 !important;
    }

    /* ========================================================
       BUTTON
       ======================================================== */

    .stButton > button {
        background-color: #334155 !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
        height: 44px !important;
        font-weight: 600 !important;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #475569 !important;
        border-color: #64748b !important;
    }

    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background-color: #121821 !important;
        border: 1px solid #252e3a !important;
        border-radius: 12px !important;
        padding: 18px !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #8f9baa !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f3f4f6 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #94a3b8 !important;
    }

    /* ========================================================
       MULTISELECT
       ======================================================== */

    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        background-color: #151a22 !important;
    }

    div[data-testid="stMultiSelect"] span {
        color: #f3f4f6 !important;
    }

    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    div[data-testid="stProgress"] > div {
        background-color: #252c36 !important;
    }

    div[data-testid="stProgress"] > div > div {
        background-color: #64748b !important;
    }

    /* ========================================================
       ALERT / INFO
       ======================================================== */

    div[data-testid="stAlert"] {
        background-color: #151a22 !important;
        border: 1px solid #303846 !important;
        color: #cbd5e1 !important;
    }

    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #252e3a !important;
    }

    /* ========================================================
       HIDE STREAMLIT EXTRAS
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("customer_churn_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    return model, encoders


@st.cache_data
def load_roc_data():

    try:

        with open("roc_data.pkl", "rb") as f:
            return pickle.load(f)

    except:

        return None


@st.cache_data
def load_dataset():

    possible_files = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "Telco-Customer-Churn.csv"
    ]

    for file in possible_files:

        try:
            return pd.read_csv(file)

        except FileNotFoundError:
            continue

    return None


model, encoders = load_model()
roc_data = load_roc_data()
df = load_dataset()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size:24px;
        font-weight:700;
        color:#f3f4f6;
        margin-bottom:4px;
    ">
        Customer Churn
    </div>

    <div style="
        font-size:14px;
        color:#8f9baa;
    ">
        Machine Learning Analytics
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Performance",
        "Customer Analytics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <div style="
        font-size:13px;
        color:#7f8a99;
        margin-bottom:6px;
    ">
        FINAL MODEL
    </div>

    <div style="
        font-size:17px;
        font-weight:600;
        color:#e5e7eb;
        margin-bottom:20px;
    ">
        Random Forest
    </div>

    <div style="
        font-size:13px;
        color:#7f8a99;
        margin-bottom:5px;
    ">
        TEST ACCURACY
    </div>

    <div style="
        font-size:16px;
        font-weight:600;
        color:#e5e7eb;
        margin-bottom:20px;
    ">
        77.50%
    </div>

    <div style="
        font-size:13px;
        color:#7f8a99;
        margin-bottom:5px;
    ">
        ROC-AUC
    </div>

    <div style="
        font-size:16px;
        font-weight:600;
        color:#e5e7eb;
    ">
        0.8330
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Customer Churn Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning dashboard for predicting and analysing telecommunications customer churn.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "Prediction":

    st.markdown(
        '<div class="section-title">Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the customer's demographic, service and billing information "
        "to generate a churn prediction."
    )

    st.markdown(
        '<div class="section-title">Customer Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )


    # --------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )


    # --------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------

    with col3:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=1.0
        )

        st.write("")

        predict_button = st.button(
            "Generate Churn Prediction",
            use_container_width=True
        )




# ============================================================
# PREDICTION RESULT
# ============================================================

    if predict_button:

        input_data = pd.DataFrame({

            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })


        # --------------------------------------------------------
        # APPLY SAVED ENCODERS
        # --------------------------------------------------------

        for column, encoder in encoders.items():

            if column in input_data.columns:

                try:

                    input_data[column] = encoder.transform(
                        input_data[column]
                    )

                except ValueError:

                    st.error(
                        f"Invalid value provided for {column}."
                    )

                    st.stop()


        # --------------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------------

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        probability_percent = probability * 100


        st.markdown("---")

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )


        # --------------------------------------------------------
        # RESULT COLUMNS
        # --------------------------------------------------------

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            if prediction == 1:

                st.metric(
                    label="Predicted Customer Status",
                    value="Likely to Churn"
                )

            else:

                st.metric(
                    label="Predicted Customer Status",
                    value="Likely to Stay"
                )


        with result_col2:

            st.metric(
                label="Churn Probability",
                value=f"{probability_percent:.2f}%"
            )


        # --------------------------------------------------------
        # PROBABILITY BAR
        # --------------------------------------------------------

        st.markdown(
            '<div class="section-title">Churn Probability</div>',
            unsafe_allow_html=True
        )

        st.progress(
            float(probability)
        )


        # --------------------------------------------------------
        # RISK INTERPRETATION
        # --------------------------------------------------------

        if probability >= 0.70:

            risk = "High churn risk"

            explanation = (
                "The customer has a relatively high predicted probability "
                "of churn and may require proactive retention attention."
            )

        elif probability >= 0.40:

            risk = "Moderate churn risk"

            explanation = (
                "The customer shows some characteristics associated with "
                "churn. Monitoring and targeted engagement may be useful."
            )

        else:

            risk = "Low churn risk"

            explanation = (
                "The customer has a relatively low predicted probability "
                "of churn based on the provided information."
            )


        st.markdown(
            '<div class="section-title">Prediction Interpretation</div>',
            unsafe_allow_html=True
        )

        st.info(
            f"**{risk}.** {explanation}"
        )


# ============================================================
# PAGE 2 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Evaluation of the final tuned Random Forest model on unseen test data."
    )


    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.metric(
            "Accuracy",
            "77.50%"
        )

    with metric2:
        st.metric(
            "Churn Recall",
            "62%"
        )

    with metric3:
        st.metric(
            "Churn F1",
            "60%"
        )

    with metric4:
        st.metric(
            "ROC-AUC",
            "0.8330"
        )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    comparison_df = pd.DataFrame({

        "Model": [
            "Decision Tree",
            "Random Forest",
            "XGBoost"
        ],

        "Mean CV Accuracy": [
            71.37,
            77.51,
            76.64
        ]
    })


    fig_comparison = px.bar(

        comparison_df,

        x="Model",

        y="Mean CV Accuracy",

        text="Mean CV Accuracy",

        labels={
            "Mean CV Accuracy": "Mean CV Accuracy (%)"
        },

        color_discrete_sequence=[
            "#64748b"
        ]
    )


    fig_comparison.update_traces(

        texttemplate="%{text:.2f}%",

        textposition="outside",

        marker_line_width=0
    )


    fig_comparison.update_layout(

        height=430,

        plot_bgcolor="#121821",

        paper_bgcolor="#121821",

        font={
            "color": "#d1d5db",
            "size": 13
        },

        title_font={
            "color": "#f3f4f6"
        },

        yaxis={

            "range": [0, 85],

            "gridcolor": "#28313d",

            "zerolinecolor": "#374151",

            "title_font": {
                "color": "#9ca3af"
            },

            "tickfont": {
                "color": "#9ca3af"
            }
        },

        xaxis={

            "gridcolor": "rgba(0,0,0,0)",

            "title_font": {
                "color": "#9ca3af"
            },

            "tickfont": {
                "color": "#d1d5db"
            }
        },

        showlegend=False,

        margin=dict(
            l=50,
            r=30,
            t=50,
            b=50
        )
    )


    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )


    cm = [
        [859, 177],
        [140, 233]
    ]


    fig_cm = go.Figure(

        data=go.Heatmap(

            z=cm,

            x=[
                "Predicted No Churn",
                "Predicted Churn"
            ],

            y=[
                "Actual No Churn",
                "Actual Churn"
            ],

            text=cm,

            texttemplate="%{text}",

            textfont={
                "size": 18,
                "color": "#f3f4f6"
            },

            hovertemplate=
            "Actual: %{y}<br>"
            "Predicted: %{x}<br>"
            "Count: %{z}"
            "<extra></extra>",

            colorscale=[
                [0, "#202631"],
                [0.5, "#475569"],
                [1, "#94a3b8"]
            ],

            showscale=False
        )
    )


    fig_cm.update_layout(

        height=410,

        plot_bgcolor="#121821",

        paper_bgcolor="#121821",

        font={
            "color": "#d1d5db"
        },

        xaxis=dict(
            title="Prediction",
            tickfont={
                "color": "#cbd5e1"
            }
        ),

        yaxis=dict(
            title="Actual",
            tickfont={
                "color": "#cbd5e1"
            }
        ),

        margin=dict(
            l=80,
            r=30,
            t=30,
            b=70
        )
    )


    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


    # ========================================================
    # ROC CURVE
    # ========================================================

    st.markdown(
        '<div class="section-title">ROC Curve</div>',
        unsafe_allow_html=True
    )


    if roc_data is not None:

        fpr = roc_data["fpr"]

        tpr = roc_data["tpr"]

        auc = roc_data["roc_auc"]


        fig_roc = go.Figure()


        # Actual ROC curve

        fig_roc.add_trace(

            go.Scatter(

                x=fpr,

                y=tpr,

                mode="lines",

                name=f"Random Forest (AUC = {auc:.4f})",

                line=dict(

                    color="#94a3b8",

                    width=3
                ),

                hovertemplate=
                "False Positive Rate: %{x:.3f}"
                "<br>True Positive Rate: %{y:.3f}"
                "<extra></extra>"
            )
        )


        # Random classifier

        fig_roc.add_trace(

            go.Scatter(

                x=[0, 1],

                y=[0, 1],

                mode="lines",

                name="Random classifier",

                line=dict(

                    color="#475569",

                    width=2,

                    dash="dash"
                )
            )
        )


        fig_roc.update_layout(

            height=460,

            plot_bgcolor="#121821",

            paper_bgcolor="#121821",

            font={
                "color": "#d1d5db"
            },

            xaxis=dict(

                title="False Positive Rate",

                range=[0, 1],

                gridcolor="#28313d",

                zerolinecolor="#374151",

                tickfont={
                    "color": "#9ca3af"
                },

                title_font={
                    "color": "#cbd5e1"
                }
            ),

            yaxis=dict(

                title="True Positive Rate",

                range=[0, 1],

                gridcolor="#28313d",

                zerolinecolor="#374151",

                tickfont={
                    "color": "#9ca3af"
                },

                title_font={
                    "color": "#cbd5e1"
                }
            ),

            legend=dict(

                font={
                    "color": "#cbd5e1"
                },

                bgcolor="#121821"
            ),

            margin=dict(
                l=60,
                r=40,
                t=30,
                b=60
            )
        )


        st.plotly_chart(
            fig_roc,
            use_container_width=True
        )


    else:

        st.warning(
            "ROC data file not found."
        )


# ============================================================
# PAGE 3 — CUSTOMER ANALYTICS
# ============================================================

elif page == "Customer Analytics":

    st.markdown(
        '<div class="section-title">Customer Analytics</div>',
        unsafe_allow_html=True
    )

    if df is None:

        st.error(
            "Customer dataset not found. Place the original "
            "Telco Customer Churn CSV in the same folder as app.py."
        )

        st.stop()


    analytics_df = df.copy()


    # ========================================================
    # DATA PREPARATION
    # ========================================================

    analytics_df["TotalCharges"] = pd.to_numeric(
        analytics_df["TotalCharges"],
        errors="coerce"
    )

    analytics_df["TotalCharges"] = (
        analytics_df["TotalCharges"]
        .fillna(0)
    )


    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    total_customers = len(analytics_df)

    churned_customers = (
        analytics_df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100

    avg_monthly = analytics_df[
        "MonthlyCharges"
    ].mean()


    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(
            "Customers",
            f"{total_customers:,}"
        )


    with metric2:

        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )


    with metric3:

        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%"
        )


    with metric4:

        st.metric(
            "Avg Monthly Charges",
            f"${avg_monthly:.2f}"
        )


    # ========================================================
    # FILTER
    # ========================================================

    st.markdown(
        '<div class="section-title">Explore Customer Segments</div>',
        unsafe_allow_html=True
    )


    contract_options = sorted(
        analytics_df["Contract"]
        .dropna()
        .unique()
        .tolist()
    )


    contract_filter = st.multiselect(

        "Contract Type",

        options=contract_options,

        default=contract_options
    )


    filtered_df = analytics_df[
        analytics_df["Contract"].isin(
            contract_filter
        )
    ]


    # ========================================================
    # CHURN DISTRIBUTION
    # ========================================================

    chart1, chart2 = st.columns(2)


    with chart1:

        churn_counts = (
            filtered_df["Churn"]
            .value_counts()
            .reset_index()
        )

        churn_counts.columns = [
            "Churn Status",
            "Customers"
        ]


        fig_churn = px.bar(

            churn_counts,

            x="Churn Status",

            y="Customers",

            text="Customers",

            labels={
                "Churn Status": "",
                "Customers": "Customers"
            },

            color_discrete_sequence=[
                "#64748b"
            ]
        )


        fig_churn.update_traces(

            textposition="outside",

            marker_line_width=0
        )


        fig_churn.update_layout(

            title={
                "text": "Customer Churn Distribution",
                "font": {
                    "color": "#f3f4f6",
                    "size": 18
                }
            },

            height=400,

            plot_bgcolor="#121821",

            paper_bgcolor="#121821",

            font={
                "color": "#d1d5db"
            },

            yaxis=dict(
                gridcolor="#28313d",
                tickfont={
                    "color": "#9ca3af"
                }
            ),

            xaxis=dict(
                tickfont={
                    "color": "#d1d5db"
                }
            ),

            showlegend=False
        )


        st.plotly_chart(
            fig_churn,
            use_container_width=True
        )


    # ========================================================
    # CONTRACT CHURN
    # ========================================================

    with chart2:

        contract_churn = pd.crosstab(

            filtered_df["Contract"],

            filtered_df["Churn"],

            normalize="index"

        ) * 100


        contract_churn = (

            contract_churn

            .reset_index()

            .melt(

                id_vars="Contract",

                var_name="Churn",

                value_name="Percentage"
            )
        )


        fig_contract = px.bar(

            contract_churn,

            x="Contract",

            y="Percentage",

            color="Churn",

            barmode="group",

            text_auto=".1f",

            color_discrete_map={

                "No": "#475569",

                "Yes": "#94a3b8"
            },

            labels={
                "Percentage": "Customers (%)"
            }
        )


        fig_contract.update_layout(

            title={
                "text": "Churn Rate by Contract Type",
                "font": {
                    "color": "#f3f4f6",
                    "size": 18
                }
            },

            height=400,

            plot_bgcolor="#121821",

            paper_bgcolor="#121821",

            font={
                "color": "#d1d5db"
            },

            yaxis=dict(
                gridcolor="#28313d",
                tickfont={
                    "color": "#9ca3af"
                }
            ),

            xaxis=dict(
                tickfont={
                    "color": "#d1d5db"
                }
            ),

            legend=dict(
                font={
                    "color": "#cbd5e1"
                }
            )
        )


        st.plotly_chart(
            fig_contract,
            use_container_width=True
        )


    # ========================================================
    # CHURN RATE BY TENURE GROUP
    # ========================================================

    st.markdown(
        '<div class="section-title">Churn Rate by Customer Tenure</div>',
        unsafe_allow_html=True
    )

    # Create meaningful tenure groups
    filtered_df["TenureGroup"] = pd.cut(
        filtered_df["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=[
            "0–12 months",
            "13–24 months",
            "25–36 months",
            "37–48 months",
            "49–60 months",
            "61–72 months"
        ]
    )

    # Calculate churn rate for each tenure group
    tenure_churn = (
        filtered_df
        .groupby("TenureGroup", observed=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )

    # Interactive bar chart
    fig_tenure = px.bar(
        tenure_churn,
        x="TenureGroup",
        y="Churn Rate",
        text="Churn Rate",
        labels={
            "TenureGroup": "Customer Tenure",
            "Churn Rate": "Churn Rate (%)"
        },
        color_discrete_sequence=["#64748b"]
    )

    fig_tenure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_width=0,
        hovertemplate=(
            "<b>%{x}</b>"
            "<br>Churn Rate: %{y:.1f}%"
            "<extra></extra>"
        )
    )

    fig_tenure.update_layout(
        title={
            "text": "Churn Rate by Customer Tenure",
            "font": {
                "color": "#f3f4f6",
                "size": 18
            }
        },

        height=430,

        plot_bgcolor="#121821",
        paper_bgcolor="#121821",

        font={
            "color": "#d1d5db"
        },

        xaxis=dict(
            title="Customer Tenure",
            gridcolor="rgba(0,0,0,0)",
            tickfont={
                "color": "#d1d5db"
            },
            title_font={
                "color": "#cbd5e1"
            }
        ),

        yaxis=dict(
            title="Churn Rate (%)",
            gridcolor="#28313d",
            zerolinecolor="#374151",
            tickfont={
                "color": "#9ca3af"
            },
            title_font={
                "color": "#cbd5e1"
            }
        ),

        showlegend=False,

        margin=dict(
            l=60,
            r=30,
            t=60,
            b=60
        )
    )

    st.plotly_chart(
        fig_tenure,
        use_container_width=True
    )
    # ========================================================
    # INTERNET SERVICE
    # ========================================================

    service_churn = pd.crosstab(

        filtered_df["InternetService"],

        filtered_df["Churn"],

        normalize="index"

    ) * 100


    service_churn = (

        service_churn

        .reset_index()

        .melt(

            id_vars="InternetService",

            var_name="Churn",

            value_name="Percentage"
        )
    )


    fig_service = px.bar(

        service_churn,

        x="InternetService",

        y="Percentage",

        color="Churn",

        barmode="group",

        text_auto=".1f",

        color_discrete_map={

            "No": "#475569",

            "Yes": "#94a3b8"
        },

        labels={
            "Percentage": "Customers (%)"
        }
    )


    fig_service.update_layout(

        title={
            "text": "Churn Rate by Internet Service",
            "font": {
                "color": "#f3f4f6",
                "size": 18
            }
        },

        height=420,

        plot_bgcolor="#121821",

        paper_bgcolor="#121821",

        font={
            "color": "#d1d5db"
        },

        yaxis=dict(
            gridcolor="#28313d",
            tickfont={
                "color": "#9ca3af"
            }
        ),

        xaxis=dict(
            tickfont={
                "color": "#d1d5db"
            }
        ),

        legend=dict(
            font={
                "color": "#cbd5e1"
            }
        )
    )


    st.plotly_chart(
        fig_service,
        use_container_width=True
    )


    # ========================================================
    # PAYMENT METHOD
    # ========================================================

    payment_churn = pd.crosstab(

        filtered_df["PaymentMethod"],

        filtered_df["Churn"],

        normalize="index"

    ) * 100


    payment_churn = (

        payment_churn

        .reset_index()

        .melt(

            id_vars="PaymentMethod",

            var_name="Churn",

            value_name="Percentage"
        )
    )


    fig_payment = px.bar(

        payment_churn,

        x="PaymentMethod",

        y="Percentage",

        color="Churn",

        barmode="group",

        text_auto=".1f",

        color_discrete_map={

            "No": "#475569",

            "Yes": "#94a3b8"
        },

        labels={
            "Percentage": "Customers (%)"
        }
    )


    fig_payment.update_layout(

        title={
            "text": "Churn Rate by Payment Method",
            "font": {
                "color": "#f3f4f6",
                "size": 18
            }
        },

        height=450,

        plot_bgcolor="#121821",

        paper_bgcolor="#121821",

        font={
            "color": "#d1d5db"
        },

        yaxis=dict(
            gridcolor="#28313d",
            tickfont={
                "color": "#9ca3af"
            }
        ),

        xaxis=dict(
            tickfont={
                "color": "#d1d5db"
            }
        ),

        legend=dict(
            font={
                "color": "#cbd5e1"
            }
        )
    )


    st.plotly_chart(
        fig_payment,
        use_container_width=True
    )


    # ========================================================
    # KEY OBSERVATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Key Observations</div>',
        unsafe_allow_html=True
    )


    st.info(
        "Use the interactive charts to explore customer behaviour "
        "across contract type, tenure, monthly charges, internet "
        "service and payment method. Select specific contract types "
        "from the filter to analyse individual customer segments."
    )