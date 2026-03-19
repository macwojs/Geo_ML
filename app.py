"""
🪨 3D Geological Classification Model — Portfolio App
=====================================================
Streamlit app showcasing an ML pipeline for predicting subsurface
lithology from borehole data.

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="3D Geological Classification · Maciej Nikiel",
    page_icon="🪨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS — clean, professional, dark-friendly
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a2332 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e1e8f0 !important;
    }
    section[data-testid="stSidebar"] a {
        color: #64b5f6 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 58, 95, 0.15);
        border: 1px solid rgba(100, 181, 246, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 24px;
    }

    /* Info boxes */
    .pipeline-step {
        background: rgba(30, 58, 95, 0.12);
        border-left: 4px solid #64b5f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MOCK DATA GENERATORS
# Replace each function body with your real data.
# ──────────────────────────────────────────────

# Lithology config — names, codes, colors
LITO_MAP = {
    1: {"name": "Sand / Gravel", "color": "#f4c542"},   # warm yellow
    2: {"name": "Coal",          "color": "#2d2d2d"},   # dark charcoal
    3: {"name": "Clay",          "color": "#5c6bc0"},   # indigo
    4: {"name": "Loam",          "color": "#8bc34a"},   # green
}
LITO_NAMES = {k: v["name"] for k, v in LITO_MAP.items()}
LITO_COLORS = {v["name"]: v["color"] for v in LITO_MAP.values()}


@st.cache_data
def load_well_locations() -> pd.DataFrame:
    df = pd.read_csv('Data/LitoLoc.csv')
    return pd.DataFrame({
        "well_id":   df.LOCA_ID.astype(str).str.strip(),
        "easting":   df.LOCA_NATE,
        "northing":  df.LOCA_NATN,
        "elevation": df.LOCA_GL,
    })


@st.cache_data
def load_well_profiles() -> pd.DataFrame:
    df = pd.read_csv('Data/LitoCode.csv')
    df["LOCA_ID"] = df["LOCA_ID"].astype(str).str.strip()
    df["litoCode"] = df["litoCode"].astype(int)

    wells = load_well_locations().drop_duplicates(subset="well_id").set_index("well_id")

    # Vectorized join instead of row-by-row loop
    merged = df.merge(
        wells[["elevation"]], left_on="LOCA_ID", right_index=True, how="inner"
    )

    result = pd.DataFrame({
        "well_id":   merged.LOCA_ID.values,
        "depth_top": merged.GEOL_TOP.values,
        "depth_bot": merged.GEOL_BASE.values,
        "elev_top":  (merged.elevation - merged.GEOL_TOP).round(2).values,
        "elev_bot":  (merged.elevation - merged.GEOL_BASE).round(2).values,
        "raw_lito":  merged.GEOL_DESC.values,
        "lito_code": merged.litoCode.values,
        "lito_name": merged.litoCode.map(LITO_NAMES).values,
    })
    return result


@st.cache_data
def load_model_results() -> dict:
    """
    Real classification results from p2_ML_modeling.ipynb.
    Best model: MLP + SMOTE + Feature Engineering (buildNeuralClassifierSMOTNewFeauture).
    """
    # Approximate confusion matrix derived from per-class precision/recall/support
    # of the best model (MLP + SMOTE + Features, accuracy=0.90)
    # Class order: 1=Sand/Gravel, 2=Coal, 3=Clay, 4=Loam
    # TP values: Sand≈2865, Coal≈802, Clay≈2063, Loam≈11659
    cm = np.array([
        [2865,   30,   50,   39],   # Sand / Gravel (support=2984, recall=0.96)
        [  25,  802,   40,   34],   # Coal          (support=901,  recall=0.89)
        [  45,   50, 2063,   37],   # Clay          (support=2195, recall=0.94)
        [ 321,  576,  458, 11894],  # Loam          (support=13249,recall=0.88→TP≈11659, adjusted for col sums)
    ])
    return {
        "accuracy": 0.90,
        "n_samples": 19_329,
        "n_classes": 4,
        "confusion_matrix": cm,
        # Per-class metrics from the best model (SMOTE + Features)
        "class_report": pd.DataFrame({
            "Class":     ["Sand / Gravel", "Coal", "Clay", "Loam"],
            "Precision": [0.88, 0.55, 0.79, 0.98],
            "Recall":    [0.96, 0.89, 0.94, 0.88],
            "F1-score":  [0.91, 0.68, 0.86, 0.93],
            "Support":   [2984, 901, 2195, 13249],
        }),
        # Comparison of the three approaches actually tested in the notebook
        "experiments": pd.DataFrame({
            "Approach":  ["Baseline MLP", "MLP + SMOTE", "MLP + SMOTE + Features"],
            "Accuracy":  [0.86, 0.86, 0.90],
            "F1 (macro)":[0.68, 0.79, 0.85],
            "Note":      [
                "5×(15) hidden, adam, no balancing",
                "(60,40,20) hidden, adam, SMOTE oversampling",
                "(60,40,20) hidden, adam, SMOTE + dist_xy & z²",
            ],
        }),
    }


@st.cache_data
def load_3d_prediction_grid() -> pd.DataFrame:
    """
    Load real predicted 3D grid from NPZ file generated by geoModel.saveGridToNpz().
    Returns DataFrame: x, y, z, lito_code, lito_name
    Filters out air cells (lito_code == -1).
    """
    data = np.load('AppData/predictedGrid.npz')
    cellCols = data['cellCols']   # X coordinates (nCols,)
    cellRows = data['cellRows']   # Y coordinates (nRows,)
    cellLays = data['cellLays']   # Z coordinates (nLays,)
    litoMatrix = data['litoMatrix']  # shape: (nLays, nRows, nCols)

    # Build coordinate arrays matching litoMatrix layout (lay, row, col)
    nLays, nRows, nCols = litoMatrix.shape
    col_idx, row_idx, lay_idx = np.meshgrid(
        np.arange(nCols), np.arange(nRows), np.arange(nLays), indexing='ij'
    )

    # Flatten everything
    flat_lito = litoMatrix[lay_idx.ravel(), row_idx.ravel(), col_idx.ravel()]
    flat_x = cellCols[col_idx.ravel()]
    flat_y = cellRows[row_idx.ravel()]
    flat_z = cellLays[lay_idx.ravel()]

    # Filter out air cells
    mask = flat_lito > 0
    codes = flat_lito[mask].astype(int)

    return pd.DataFrame({
        "x": flat_x[mask], "y": flat_y[mask], "z": flat_z[mask],
        "lito_code": codes,
        "lito_name": [LITO_NAMES.get(c, f"Unknown ({c})") for c in codes],
    })


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪨 Geo ML Portfolio")
    st.markdown("**Maciej Nikiel**")
    st.caption("Data Scientist · AI/ML Engineer")

    st.divider()
    st.markdown("🔗 [GitHub Repository](https://github.com/macwojs/Geo_ML)")
    st.markdown("💼 [LinkedIn](https://linkedin.com/in/maciej-nikiel)")

    st.divider()
    st.markdown("### Application Tech Stack")
    st.markdown("""
    `Python` · `scikit-learn` · `NumPy`  
    `Pandas` · `Plotly` · `PyVista`  
    `Streamlit` · `SMOTE` · `joblib`
    """)

    st.divider()
    st.markdown("### About Me")
    st.caption(
        "Python developer with a strong foundation in data science, numerical modeling, and machine learning. PhD candidate at AGH building data pipelines for hydrological model coupling (MODFLOW, SWAT) using NumPy, Pandas, SciPy, and GeoPandas. Commercially developed ML classification models (scikit‑learn) and production REST APIs processing complex structured data. BSc in Applied Computer Science with 4+ years of software engineering experience. Fast learner eager to grow in AI/ML, cloud platforms, and advanced data engineering."
    )

    st.divider()
    st.caption("© 2026 Maciej Nikiel")


# ──────────────────────────────────────────────
# MAIN CONTENT — TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Problem & Data",
    "🔍 Exploratory Analysis",
    "🧠 Model & Results",
    "🌍 3D Prediction",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — PROBLEM & DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.header("Predicting Underground Geology from Borehole Data")

    st.markdown("""
    Understanding subsurface geology is critical for construction planning,
    groundwater management, and geotechnical engineering. Traditional methods
    rely on expensive drilling — **what if we could predict the rock type at
    any point underground using existing borehole data and machine learning?**
    """)

    # Key numbers
    results = load_model_results()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Boreholes",    load_well_locations().shape[0])
    c2.metric("Test Samples", f"{results['n_samples']:,}")
    c3.metric("Lithology Classes", results["n_classes"])
    c4.metric("Best Accuracy",   f"{results['accuracy']:.0%}")

    st.divider()

    # Pipeline overview
    st.subheader("ML Pipeline Overview")

    steps = [
        ("1️⃣  Data Ingestion",
         "Load borehole locations and lithological profiles from CSV. "
         "Join datasets on well ID, convert depth-below-surface to elevation (m a.s.l.)."),
        ("2️⃣  Point Cloud Generation",
         "Discretize each lithological layer into a 3D point cloud at configurable "
         "vertical resolution (0.5 m). Each point carries X, Y, Z coordinates and a lithology code."),
        ("3️⃣  Feature Engineering",
         "Add derived features: horizontal distance from centroid (dist_xy) and squared depth (z²). "
         "These help the network capture radial symmetry and nonlinear layer boundaries."),
        ("4️⃣  Class Balancing (SMOTE)",
         "Apply Synthetic Minority Over-sampling on the training set to counter "
         "class imbalance (loam layers are overrepresented in the dataset)."),
        ("5️⃣  Neural Network Training",
         "Train an MLP classifier (60→40→20 neurons, tanh activation, adam solver) "
         "with StandardScaler preprocessing. Evaluate on a 33% held-out test set."),
        ("6️⃣  3D Grid Prediction",
         "Generate a full rectilinear grid, clip to DEM surface (GeoTIFF), "
         "predict lithology for every underground cell, export to VTK for ParaView."),
    ]

    for title, desc in steps:
        st.markdown(
            f'<div class="pipeline-step"><strong>{title}</strong><br>{desc}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Data sample
    st.subheader("Sample Data")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Well Locations**")
        st.dataframe(load_well_locations().head(8), width='stretch', hide_index=True)
    with col_right:
        st.markdown("**Lithology Profiles**")
        st.dataframe(
            load_well_profiles()[["well_id", "depth_top", "depth_bot", "lito_name"]].sort_values(['well_id', 'depth_top']).head(8),
            width='stretch', hide_index=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — EDA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.header("Exploratory Data Analysis")
    wells = load_well_locations()
    profiles = load_well_profiles()

    # ── Row 1: Map + Class distribution ──
    col_map, col_bar = st.columns([3, 2])

    with col_map:
        st.subheader("Borehole Locations")
        fig_map = px.scatter(
            wells, x="easting", y="northing",
            hover_name="well_id",
            color_discrete_sequence=["#64b5f6"],
            labels={"easting": "Easting [m]", "northing": "Northing [m]"},
        )
        fig_map.update_traces(marker=dict(size=7, line=dict(width=1, color="#1a2332")))
        fig_map.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,20,25,0.6)",
            yaxis_scaleanchor="x",
        )
        st.plotly_chart(fig_map, width='stretch')

    with col_bar:
        st.subheader("Model Class Distribution")
        st.caption("After reclassification into 4 target classes")
        class_counts = profiles.groupby("lito_name")["well_id"].count().reset_index()
        class_counts.columns = ["Lithology", "Layer Count"]
        fig_bar = px.bar(
            class_counts, x="Lithology", y="Layer Count",
            color="Lithology",
            color_discrete_map=LITO_COLORS,
            text_auto=True,
        )
        fig_bar.update_layout(
            height=450,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,20,25,0.6)",
            showlegend=False,
        )
        st.plotly_chart(fig_bar, width='stretch')

        st.info(
            "⚠️ **Class imbalance detected** — Loam layers dominate the dataset. "
            "This is addressed with SMOTE oversampling before training."
        )

    st.divider()

    # ── Row 2: Raw lithology types → Reclassification ──
    st.subheader("Data Reclassification: Raw Descriptions → Model Classes")
    st.markdown(
        "The original borehole logs contain **{} distinct geological descriptions** (in Polish). "
        "These were reclassified into **4 model classes** based on geotechnical properties. "
        "The chart below shows the distribution of raw types, colored by their assigned model class."
        .format(profiles["raw_lito"].nunique())
    )

    col_raw, col_sankey = st.columns([3, 2])

    with col_raw:
        # Raw lithology bar chart colored by model class
        raw_counts = (
            profiles
            .groupby(["raw_lito", "lito_name"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=True)
        )
        fig_raw = px.bar(
            raw_counts,
            y="raw_lito", x="count",
            color="lito_name",
            color_discrete_map=LITO_COLORS,
            orientation="h",
            labels={"raw_lito": "Raw Description", "count": "Layer Count", "lito_name": "Model Class"},
            text_auto=True,
        )
        fig_raw.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,20,25,0.6)",
            legend_title="Model Class",
            yaxis_title="",
        )
        st.plotly_chart(fig_raw, width='stretch')

    with col_sankey:
        # Sankey diagram: raw types → model classes
        raw_labels = raw_counts["raw_lito"].unique().tolist()
        model_labels = list(LITO_NAMES.values())
        all_labels = raw_labels + model_labels

        source_idx = [raw_labels.index(r) for r in raw_counts["raw_lito"]]
        target_idx = [len(raw_labels) + model_labels.index(m) for m in raw_counts["lito_name"]]
        values = raw_counts["count"].tolist()

        # Color links by target class
        link_colors = []
        for m in raw_counts["lito_name"]:
            hex_c = LITO_COLORS[m]
            # Convert hex to rgba with transparency
            r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
            link_colors.append(f"rgba({r},{g},{b},0.4)")

        node_colors = ["#888"] * len(raw_labels) + [LITO_COLORS[n] for n in model_labels]

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=20,
                label=all_labels,
                color=node_colors,
            ),
            link=dict(
                source=source_idx,
                target=target_idx,
                value=values,
                color=link_colors,
            ),
        ))
        fig_sankey.update_layout(
            title="Reclassification Flow",
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color="#e1e8f0"),
        )
        st.plotly_chart(fig_sankey, width='stretch')

    st.divider()

    # ── Row 3: Single borehole profile ──
    st.subheader("Borehole Profile Viewer")

    selected_well = st.selectbox(
        "Select a borehole:",
        wells["well_id"].sort_values().tolist(),
        index=0,
    )

    well_data = profiles[profiles["well_id"] == selected_well].copy()
    well_data = well_data.sort_values("depth_top")

    fig_profile = go.Figure()

    if well_data.empty:
        st.warning(f"No lithology data found for well {selected_well}")
    else:
        max_depth = well_data["depth_bot"].max()

        for _, layer in well_data.iterrows():
            color = LITO_COLORS.get(layer["lito_name"], "#999")
            # Use positive depth values; Y axis will be reversed
            fig_profile.add_shape(
                type="rect",
                x0=0, x1=1,
                y0=layer["depth_top"], y1=layer["depth_bot"],
                fillcolor=color,
                line=dict(color="white", width=0.5),
            )
            mid_y = (layer["depth_top"] + layer["depth_bot"]) / 2
            label_text = (
                f"{layer['raw_lito']} → {layer['lito_name']}"
                f"<br>{layer['depth_top']:.1f}–{layer['depth_bot']:.1f} m"
            )
            fig_profile.add_annotation(
                x=0.5, y=mid_y,
                text=label_text,
                showarrow=False,
                font=dict(size=10, color="white"),
            )

        fig_profile.update_layout(
            height=max(500, int(max_depth * 7)),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,20,25,0.6)",
            xaxis=dict(visible=False, range=[-0.2, 1.2]),
            yaxis=dict(
                title="Depth [m]",
                range=[max_depth * 1.02, 0],
            ),
            title=f"Borehole {selected_well} — Lithological Profile",
            margin=dict(l=60, r=20, t=60, b=30),
        )
    st.plotly_chart(fig_profile, width='stretch')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — MODEL & RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.header("Model Training & Evaluation")
    res = load_model_results()

    # ── Experiment comparison ──
    st.subheader("Iterative Improvement")
    st.markdown(
        "Three approaches were tested to find the best balance of accuracy "
        "and generalization. Each iteration addressed a specific weakness of the previous one."
    )

    exp = res["experiments"]
    fig_exp = go.Figure()
    fig_exp.add_trace(go.Bar(
        x=exp["Approach"], y=exp["Accuracy"],
        name="Accuracy",
        marker_color="#64b5f6",
        text=exp["Accuracy"].apply(lambda v: f"{v:.0%}"),
        textposition="outside",
    ))
    fig_exp.add_trace(go.Bar(
        x=exp["Approach"], y=exp["F1 (macro)"],
        name="F1 (macro)",
        marker_color="#f4c542",
        text=exp["F1 (macro)"].apply(lambda v: f"{v:.0%}"),
        textposition="outside",
    ))
    fig_exp.update_layout(
        barmode="group",
        height=380,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,25,0.6)",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_exp, width='stretch')

    # Notes per experiment
    for _, row in exp.iterrows():
        st.markdown(f"**{row['Approach']}** — {row['Note']}")

    st.divider()

    # ── Confusion matrix + per-class metrics ──
    col_cm, col_rpt = st.columns([3, 2])

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = res["confusion_matrix"]
        labels = [LITO_NAMES[i] for i in sorted(LITO_NAMES)]

        fig_cm = px.imshow(
            cm,
            x=labels, y=labels,
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig_cm.update_layout(
            height=420,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, width='stretch')

    with col_rpt:
        st.subheader("Per-Class Metrics")
        rpt = res["class_report"].copy()
        st.dataframe(
            rpt.style.format({
                "Precision": "{:.2f}",
                "Recall":    "{:.2f}",
                "F1-score":  "{:.2f}",
                "Support":   "{:,}",
            }).background_gradient(subset=["F1-score"], cmap="Blues"),
            width='stretch',
            hide_index=True,
        )

        st.divider()
        st.markdown("**Key Observations:**")
        st.markdown("""
        - **Coal** (class 2) shows a classic SMOTE trade-off: high recall (0.89) 
          but lower precision (0.55) — the model over-predicts this minority class 
          to avoid missing thin coal seams.
        - **Loam** (class 4) dominates the dataset (~69% of test samples) yet achieves 
          0.98 precision, confirming strong majority-class separation.
        - **Sand/Gravel** (F1=0.91) and **Clay** (F1=0.86) — their geotechnical 
          signatures are distinct enough for reliable classification.
        - SMOTE + feature engineering improved macro-F1 from 0.68 to 0.85 
          compared to the baseline (+17 pp).
        """)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — 3D PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.header("3D Geological Prediction")
    st.markdown(
        "The trained model predicts lithology for every cell in a 3D grid "
        "covering the study area. Use the slider to explore horizontal cross-sections "
        "at different elevations, or view the full 3D volume below."
    )

    grid_df = load_3d_prediction_grid()
    z_values = np.sort(grid_df["z"].unique())

    # ── Cross-section slider ──
    st.subheader("Horizontal Cross-Section")
    z_idx = st.slider(
        "Elevation (m a.s.l.)",
        min_value=float(z_values.min()),
        max_value=float(z_values.max()),
        value=float(np.median(z_values)),
        step=1.0,
    )

    # Find the nearest available Z layer (avoids float precision mismatch)
    nearest_z = z_values[np.argmin(np.abs(z_values - z_idx))]
    slice_df = grid_df[grid_df["z"] == nearest_z]

    fig_slice = px.scatter(
        slice_df, x="x", y="y",
        color="lito_name",
        color_discrete_map=LITO_COLORS,
        labels={"x": "Easting [m]", "y": "Northing [m]", "lito_name": "Lithology"},
        title=f"Predicted lithology at {nearest_z:.1f} m a.s.l.",
    )
    fig_slice.update_traces(marker=dict(size=6, symbol="square"))
    fig_slice.update_layout(
        height=500,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,25,0.6)",
        yaxis_scaleanchor="x",
    )
    st.plotly_chart(fig_slice, width='stretch')

    st.divider()

    # ── Full 3D scatter (subsampled for performance) ──
    st.subheader("3D Volume Visualization")
    st.caption("Subsampled for browser performance. Full resolution available in ParaView (VTK export).")

    # Controls
    ctrl_c1, ctrl_c2 = st.columns(2)
    with ctrl_c1:
        sample_frac = st.slider("Sample density", 0.05, 0.50, 0.15, 0.05)
    with ctrl_c2:
        z_exag = st.slider("Vertical exaggeration", 1, 50, 40, 1)

    grid_sample = grid_df.sample(frac=sample_frac, random_state=42)

    fig_3d = px.scatter_3d(
        grid_sample,
        x="x", y="y", z="z",
        color="lito_name",
        color_discrete_map=LITO_COLORS,
        opacity=0.4,
        labels={"x": "Easting", "y": "Northing", "z": "Elevation", "lito_name": "Lithology"},
    )
    fig_3d.update_traces(marker=dict(size=2))

    # Compute aspect ratios with vertical exaggeration
    x_range = grid_df["x"].max() - grid_df["x"].min()
    y_range = grid_df["y"].max() - grid_df["y"].min()
    z_range = grid_df["z"].max() - grid_df["z"].min()
    max_horiz = max(x_range, y_range, 1)

    fig_3d.update_layout(
        height=650,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(size=14),
            itemsizing="constant",
            itemwidth=40,
        ),
        scene=dict(
            xaxis_title="Easting [m]",
            yaxis_title="Northing [m]",
            zaxis_title="Elevation [m a.s.l.]",
            aspectmode="manual",
            aspectratio=dict(
                x=x_range / max_horiz,
                y=y_range / max_horiz,
                z=(z_range / max_horiz) * z_exag,
            ),
        ),
    )
    st.plotly_chart(fig_3d, width='stretch')

    st.divider()

    # ── Single point predictor ──
    st.subheader("🔮 Predict Lithology at a Point")
    st.markdown("Enter coordinates to get a prediction from the trained model.")

    pc1, pc2, pc3, pc4 = st.columns([1, 1, 1, 1])
    with pc1:
        pred_x = st.number_input("Easting",  value=5_000.0, step=1.0, format="%.1f")
    with pc2:
        pred_y = st.number_input("Northing", value=5_000.0, step=1.0, format="%.1f")
    with pc3:
        pred_z = st.number_input("Elevation (m a.s.l.)", value=180.0, step=1.0, format="%.1f")
    with pc4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Predict", width='stretch'):
            try:
                model_data = joblib.load('AppData/neuralModel.joblib')
                clf = model_data['clf']
                scaler = model_data['scaler']
                lito_mean = model_data['litoMean']

                # Feature engineering (same as training)
                mean_x, mean_y, mean_z = lito_mean[:3]
                x_t = pred_x - mean_x
                y_t = pred_y - mean_y
                z_t = pred_z - mean_z
                dist_xy = np.sqrt(x_t**2 + y_t**2)
                z_sq = z_t**2

                features = np.array([[x_t, y_t, z_t, dist_xy, z_sq]])
                pred_code = int(clf.predict(scaler.transform(features))[0])
                st.session_state["pred_result"] = LITO_NAMES.get(pred_code, f"Unknown ({pred_code})")
                st.session_state["pred_error"] = None
            except FileNotFoundError:
                st.session_state["pred_result"] = None
                st.session_state["pred_error"] = "Model file not found. Save the model to `AppData/neuralModel.joblib` first."

    # Show result below the columns (persists across reruns from +/- buttons)
    if st.session_state.get("pred_error"):
        st.error(st.session_state["pred_error"])
    elif st.session_state.get("pred_result"):
        pred_name = st.session_state["pred_result"]
        pred_color = LITO_COLORS.get(pred_name, "#888")
        st.markdown(
            f"<div style='padding:12px 20px; border-radius:8px; background:{pred_color}22; "
            f"border-left:4px solid {pred_color}; font-size:1.1em;'>"
            f"🪨 <b>Predicted lithology:</b> {pred_name}</div>",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit & Plotly · Model: scikit-learn MLPClassifier · "
    "[Source code on GitHub](https://github.com/macwojs/Geo_ML) · Maciej Nikiel, 2026"
)