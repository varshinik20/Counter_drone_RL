from pathlib import Path

import pandas as pd
import streamlit as st
from training_curves import extract_scalar_dataframe


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Counter-UAV Defense Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- PATHS ----------------
RESULTS_DIR = Path("results")
SUMMARY_CSV = RESULTS_DIR / "summary_metrics.csv"
SIM_CSV = RESULTS_DIR / "simulation_metrics.csv"
DAMAGE_IMG = RESULTS_DIR / "damage_distribution.png"
TRACK_UTIL_IMG = RESULTS_DIR / "tracking_utilization.png"
CORR_IMG = RESULTS_DIR / "correlation_analysis.png"
MODEL_PATH = RESULTS_DIR / "models" / "commander_policy.zip"
VIDEO_PATH = RESULTS_DIR / "demo.mp4"


# ---------------- STYLING ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #111827;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .sub-title {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.4rem;
    }

    .card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        padding: 1rem 1.2rem;
        border-radius: 18px;
        box-shadow: 0 6px 22px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .metric-title {
        font-size: 0.9rem;
        color: #64748b;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.2rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }

    .small-note {
        color: #64748b;
        font-size: 0.92rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        overflow: hidden;
    }

    .status-good {
        color: #16a34a;
        font-weight: 700;
    }

    .status-warn {
        color: #dc2626;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- HELPERS ----------------
def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def read_summary():
    if file_exists(SUMMARY_CSV):
        return pd.read_csv(SUMMARY_CSV)
    return None


def read_simulation_metrics():
    if file_exists(SIM_CSV):
        return pd.read_csv(SIM_CSV)
    return None


def get_policy_value(df, policy_name, column_name):
    try:
        row = df[df["policy"] == policy_name]
        if row.empty:
            return None
        return float(row.iloc[0][column_name])
    except Exception:
        return None


def fmt(v):
    if v is None:
        return "--"
    return f"{v:.2f}"


def compute_threat_details(row):
    zone_value = row.get("zone_value", 5)
    damage = row.get("damage_value", 5)
    dist = row.get("distance_to_zone", 30)
    time = row.get("time_to_zone", 10)

    urgency_dist = max(0, 60 - dist) * 0.25
    urgency_time = max(0, 20 - time) * 1.2

    score = zone_value * 5 + damage * 2.5 + urgency_dist + urgency_time

    return {
        "Total Threat Score": score,
        "Zone Importance": zone_value * 5,
        "Damage Potential": damage * 2.5,
        "Distance Urgency": urgency_dist,
        "Time Urgency": urgency_time,
    }


# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Performance Metrics",
        "Training Curves",
        "Explainability",
        "Graphs",
        "Simulation Data",
        "Demo Video",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Status")

model_status = "Available" if file_exists(MODEL_PATH) else "Missing"
results_status = "Available" if file_exists(SUMMARY_CSV) else "Missing"

st.sidebar.write(f"**Model File:** {model_status}")
st.sidebar.write(f"**Result Summary:** {results_status}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Expected Files")
st.sidebar.caption("Place generated outputs in the `results/` folder.")


# ---------------- LOAD DATA ----------------
summary_df = read_summary()
sim_df = read_simulation_metrics()


# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Counter-UAV Defense Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">White-theme analytics dashboard for Deep RL based drone swarm defense, including damage, tracking, weapon utilization, plots, training curves, explainability, and demo visualization.</div>',
    unsafe_allow_html=True,
)


# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.markdown('<div class="section-title">System Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Trained Model</div>
                <div class="metric-value">{'Yes' if file_exists(MODEL_PATH) else 'No'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Summary Metrics</div>
                <div class="metric-value">{'Ready' if summary_df is not None else 'Not Found'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">Simulation Records</div>
                <div class="metric-value">{len(sim_df) if sim_df is not None else 0}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Project Summary</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
        This dashboard presents the performance of a reinforcement learning based counter-drone defense system.
        It compares DeepRL against classical and random baselines using key metrics such as cumulative damage,
        in-tracking time, and weapon utilization. It also displays generated graphs, training curves, explainability,
        and simulation video output.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if summary_df is not None:
        st.markdown('<div class="section-title">Quick Comparison</div>', unsafe_allow_html=True)

        rl_damage = get_policy_value(summary_df, "DeepRL", "damage_pct")
        cl_damage = get_policy_value(summary_df, "Classic", "damage_pct")
        rl_track = get_policy_value(summary_df, "DeepRL", "tracking_pct")
        rl_util = get_policy_value(summary_df, "DeepRL", "weapon_utilization_pct")

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-title">DeepRL Damage</div>
                    <div class="metric-value">{fmt(rl_damage)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k2:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-title">DeepRL Tracking</div>
                    <div class="metric-value">{fmt(rl_track)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with k3:
            st.markdown(
                f"""
                <div class="card">
                    <div class="metric-title">DeepRL Weapon Utilization</div>
                    <div class="metric-value">{fmt(rl_util)}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if rl_damage is not None and cl_damage is not None:
            if rl_damage < cl_damage:
                st.success(
                    f"DeepRL is outperforming Classic control in damage reduction ({rl_damage:.2f}% vs {cl_damage:.2f}%)."
                )
            else:
                st.warning(
                    f"Classic control is currently outperforming DeepRL in damage reduction ({cl_damage:.2f}% vs {rl_damage:.2f}%). Retraining/tuning is recommended."
                )
    else:
        st.info("Run the compare command first to generate summary metrics.")


# ---------------- PERFORMANCE ----------------
elif page == "Performance Metrics":
    st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)

    if summary_df is not None:
        st.dataframe(summary_df, use_container_width=True)

        st.markdown(
            '<div class="small-note">Recommended interpretation: lower damage is better, while higher tracking and weapon utilization are better.</div>',
            unsafe_allow_html=True,
        )

        if "policy" in summary_df.columns:
            chart_df = summary_df.set_index("policy")
            st.bar_chart(chart_df[["damage_pct", "tracking_pct", "weapon_utilization_pct"]])
    else:
        st.warning("Summary metrics not found. Run compare first.")


# ---------------- TRAINING CURVES ----------------
elif page == "Training Curves":
    st.markdown('<div class="section-title">Training Curves</div>', unsafe_allow_html=True)

    tb_df = extract_scalar_dataframe("results/tensorboard_logs")

    if tb_df is None or tb_df.empty:
        st.warning("No TensorBoard scalar logs found. Train the model first with tensorboard logging enabled.")
    else:
        reward_df = tb_df[tb_df["tag"] == "rollout/ep_rew_mean"].copy()

        if not reward_df.empty:
            st.markdown('<div class="card">Episode Reward Mean vs Training Steps</div>', unsafe_allow_html=True)
            reward_chart = reward_df[["step", "value"]].rename(columns={"value": "ep_rew_mean"})
            reward_chart = reward_chart.set_index("step")
            st.line_chart(reward_chart)

            st.markdown(
                '<div class="small-note">A rising reward curve generally indicates the policy is learning better interception decisions over time.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Reward curve tag not found in TensorBoard logs.")
            st.write("Available tags:", sorted(tb_df["tag"].unique()))

        other_tags = [t for t in tb_df["tag"].unique() if t != "rollout/ep_rew_mean"]

        if other_tags:
            st.markdown('<div class="section-title">Other Training Signals</div>', unsafe_allow_html=True)

            selected_tag = st.selectbox("Select a training metric", other_tags)

            metric_df = tb_df[tb_df["tag"] == selected_tag][["step", "value"]].copy()
            metric_df = metric_df.set_index("step")

            st.line_chart(metric_df)
            st.markdown(
                f'<div class="small-note">Currently showing: <b>{selected_tag}</b></div>',
                unsafe_allow_html=True,
            )


# ---------------- EXPLAINABILITY ----------------
elif page == "Explainability":
    st.markdown('<div class="section-title">AI Decision Explainability</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
        This section explains how the AI prioritizes drone interception based on threat scoring.
        The agent selects targets that maximize risk reduction using a combination of:
        <ul>
            <li>Zone importance</li>
            <li>Damage potential</li>
            <li>Distance to sensitive zones</li>
            <li>Time to impact</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Simulated Threat Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        zone_value = st.slider("Zone Importance", 1, 10, 5)
        damage_value = st.slider("Drone Damage Potential", 1, 20, 6)

    with col2:
        distance = st.slider("Distance to Zone", 0, 60, 25)
        time_to_zone = st.slider("Time to Impact", 0, 20, 8)

    sample = {
        "zone_value": zone_value,
        "damage_value": damage_value,
        "distance_to_zone": distance,
        "time_to_zone": time_to_zone,
    }

    result = compute_threat_details(sample)

    st.markdown('<div class="section-title">Threat Score Breakdown</div>', unsafe_allow_html=True)

    df = pd.DataFrame(result.items(), columns=["Component", "Contribution"])
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("Component"))

    st.markdown('<div class="section-title">AI Decision Explanation</div>', unsafe_allow_html=True)

    explanation = f"""
    The AI prioritizes this drone because:

    • Zone Importance contributes significantly ({result["Zone Importance"]:.2f})  
    • Damage Potential is high ({result["Damage Potential"]:.2f})  
    • Distance urgency adds ({result["Distance Urgency"]:.2f})  
    • Time urgency adds ({result["Time Urgency"]:.2f})  

    👉 Final Threat Score: {result["Total Threat Score"]:.2f}

    Higher scores mean higher interception priority.
    """

    st.markdown(f"<div class='card'>{explanation}</div>", unsafe_allow_html=True)


# ---------------- GRAPHS ----------------
elif page == "Graphs":
    st.markdown('<div class="section-title">Generated Result Graphs</div>', unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown('<div class="card">Damage Distribution</div>', unsafe_allow_html=True)
        if file_exists(DAMAGE_IMG):
            st.image(str(DAMAGE_IMG), use_container_width=True)
        else:
            st.info("damage_distribution.png not found")

    with row1_col2:
        st.markdown('<div class="card">Tracking & Weapon Utilization</div>', unsafe_allow_html=True)
        if file_exists(TRACK_UTIL_IMG):
            st.image(str(TRACK_UTIL_IMG), use_container_width=True)
        else:
            st.info("tracking_utilization.png not found")

    st.markdown('<div class="card">Correlation Analysis</div>', unsafe_allow_html=True)
    if file_exists(CORR_IMG):
        st.image(str(CORR_IMG), use_container_width=True)
    else:
        st.info("correlation_analysis.png not found")


# ---------------- SIMULATION DATA ----------------
elif page == "Simulation Data":
    st.markdown('<div class="section-title">Simulation Metrics CSV</div>', unsafe_allow_html=True)

    if sim_df is not None:
        st.dataframe(sim_df, use_container_width=True)

        if "policy" in sim_df.columns:
            st.markdown('<div class="section-title">Episode Counts by Policy</div>', unsafe_allow_html=True)
            count_df = sim_df["policy"].value_counts().rename_axis("policy").reset_index(name="episodes")
            st.dataframe(count_df, use_container_width=True)
    else:
        st.warning("simulation_metrics.csv not found. Run compare first.")


# ---------------- VIDEO ----------------
elif page == "Demo Video":
    st.markdown('<div class="section-title">Demo Video</div>', unsafe_allow_html=True)

    if file_exists(VIDEO_PATH):
        with open(VIDEO_PATH, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
    else:
        st.info("No demo video found. Save your simulation video as results/demo.mp4")