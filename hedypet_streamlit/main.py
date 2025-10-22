import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import extra_streamlit_components as stx
import json 
import os 

# Set page config
st.set_page_config(page_title="FDG 100 Healthy Humans", layout="wide")

# Add custom CSS to reduce top padding/margin
st.markdown("""
<style>
    /* Reduce top padding but leave room for header */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 0rem;
    }
    /* Keep header compact and fixed */
    header[data-testid="stHeader"] {
        background-color: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        height: 2.5rem;
        position: fixed;
        top: 0;
        z-index: 999;
    }
    /* Dark mode header background */
    [data-testid="stHeader"] {
        background-color: var(--background-color);
    }
    /* Reduce sidebar top padding */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem !important;
    }
    [data-testid="stSidebarContent"] {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Legacy data loader - no longer used
# @st.cache_data
# def generate_mu_data():
#     return pd.read_pickle(os.path.join(os.path.dirname(__file__),"v1.pkl.gz"))

@st.cache_data
def generate_norm_data():
    return pd.read_pickle(os.path.join(os.path.dirname(__file__),"norm2.pkl")).set_index("Subject")

@st.cache_data
def load_means_data():
    """Load the static organ means data"""
    return pd.read_pickle(os.path.join(os.path.dirname(__file__), "means.pkl.gz"))

@st.cache_data
def load_tacs_data():
    """Load the time activity curves data"""
    return pd.read_pickle(os.path.join(os.path.dirname(__file__), "tacs.pkl.gz"))

@st.cache_data
def load_patlak_data():
    """Load the Patlak analysis data"""
    return pd.read_pickle(os.path.join(os.path.dirname(__file__), "patlak.pkl.gz"))

def calculate_summary_stats(df, group_cols, organs, erosion=1, norm="SUL Decazes Denominator [Bq/mL]",uncertainty="std"):
    """Calculate mean and std for grouped data"""
    # This function now simply USES the group_cols list it's given.
    # It no longer modifies it.
    assert erosion in [0, 1, 2, 3], erosion
    df = df[df["Erosion Iterations"] == erosion].copy()

    # Filter by selected organs
    df = df[df.region.isin(organs)]

    norm_df = generate_norm_data()[norm]
    df["mult"] = df["Subject"].map(norm_df)
    df["mu"] /= df["mult"]

    # Group by the provided columns, which now correctly includes 'region'
    summary = df.groupby(['t'] + group_cols)['mu'].agg(['mean', 'std', 'count']).reset_index()
    if uncertainty == "95CI":
        summary['std'] = 1.96*summary['std'] / summary["count"].apply(np.sqrt)
    elif uncertainty == "95PI":
        summary['std'] = 1.96*summary['std']

    summary['std'] = summary['std'].fillna(0)

    # Sort by the x-axis value to ensure lines are drawn correctly.
    summary = summary.sort_values(['t'] + group_cols)

    return summary

# Load data
means_data = load_means_data()  # Load static organ means data
tacs_data = load_tacs_data()  # Load time activity curves data
patlak_data = load_patlak_data()  # Load Patlak analysis data

_norm = generate_norm_data()
norm_options = {
    "SUV": "SUV Denominator [Bq/mL]",
    "SUL (Janma)": "SUL Janma Denominator [Bq/mL]",
    "SUL (James)": "SUL James Denominator [Bq/mL]",
    "SUL (Decazes)": "SUL Decazes Denominator [Bq/mL]"
}

list(_norm.columns)

with open(os.path.join(os.path.dirname(__file__),"totalseg_ct_classes.json"), "r") as handle:
    ts_classes_orig = json.load(handle)

ts_classes = {v:int(k) for k,v in ts_classes_orig.items()}

# Create tab bar with extra-streamlit-components
chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="Time Activity Curves", description=""),
    stx.TabBarItemData(id="tab2", title="Organ Means", description=""),
    stx.TabBarItemData(id="tab4", title="Patlak", description=""),
    stx.TabBarItemData(id="tab5", title="About", description=""),
], default="tab1")

# Show appropriate controls in sidebar based on active tab
with st.sidebar:
    # Use custom HTML with negative margin to pull content up (leave room for collapse button)
    st.markdown('''
        <h1 style="margin-top: -2rem; margin-bottom: 0.25rem;">hedyPET</h1>
        <p style="margin-top: 0; margin-bottom: 1rem; color: gray; font-size: 0.9rem;">Data Explorer</p>
    ''', unsafe_allow_html=True)

    if chosen_id == "tab1":
        st.subheader("Controls")

        # Get available regions from tacs data
        available_regions_tacs = sorted(tacs_data['region'].unique())

        # Color grouping for time activity curves
        grouping_variables = ["None", "Age", "Sex", "Erosion", "Normalization", "Organ"]
        color_group_tacs = st.selectbox("Color grouping:", grouping_variables, index=5, help="Group curves by color", key="tacs_color")  # index=5 is "Organ"

        # Organ selector (conditional based on grouping)
        if color_group_tacs != "Organ":
            # When not grouping by organ, select single organ
            default_organ_tacs = "Liver" if "Liver" in available_regions_tacs else available_regions_tacs[0]
            organ_select_tacs = st.selectbox("Organ", available_regions_tacs, index=available_regions_tacs.index(default_organ_tacs), help="Select organ to analyze")
            organs_select_tacs = [organ_select_tacs]
        else:
            # When grouping by organ, allow multiple organ selection
            default_organs_multi_tacs = ["Liver", "Gray matter", "Skeletal muscle"] if all(organ in available_regions_tacs for organ in ["Liver", "Gray matter", "Skeletal muscle"]) else available_regions_tacs[:3]
            organs_select_tacs = st.multiselect("Organs to compare", available_regions_tacs, default=default_organs_multi_tacs, help="Select organs to include in the comparison")

        # Normalization selector (conditional based on grouping)
        if color_group_tacs != "Normalization":
            # Default to SUL (Decazes)
            norm_keys = list(norm_options.keys())
            default_index = norm_keys.index("SUL (Decazes)") if "SUL (Decazes)" in norm_keys else 0
            norm_selector_tacs = st.selectbox("Normalization",options=norm_keys,index=default_index,help="Select normalization method for the PET signal", key="tacs_norm")
            norm_select_tacs = norm_options[norm_selector_tacs]
            selected_norms_tacs = [norm_select_tacs]
            selected_norm_names_tacs = [norm_selector_tacs]
        else:
            # Allow multiple normalization selection when grouping by normalization
            default_norm_names_tacs = ["SUV", "SUL (Janma)", "SUL (James)", "SUL (Decazes)"]
            available_defaults_tacs = [norm for norm in default_norm_names_tacs if norm in norm_options.keys()]
            selected_norm_names_tacs = st.multiselect("Normalizations", options=list(norm_options.keys()), default=available_defaults_tacs, help="Select normalization methods to compare", key="tacs_norms")
            selected_norms_tacs = [norm_options[name] for name in selected_norm_names_tacs]
            norm_selector_tacs = "Multiple"

        # Erosion options (conditional based on grouping)
        erosion_options = ["None", "1 iteration"]
        if color_group_tacs != "Erosion":
            erosion_select_tacs = st.selectbox("Erosion", options=erosion_options, index=1, help="Number of erosion iterations applied to organ mask", key="tacs_erosion")
        else:
            erosion_select_tacs = None  # Will show all erosion levels

        # Confidence interval options
        uncertainty_options = ["None", "1 Standard Deviation", "95% Confidence Interval", "95% Prediction Interval"]
        uncertainty_tacs = st.selectbox("Uncertainty bands:", uncertainty_options, index=2, help="Type of uncertainty bands to display around the mean curve. Confidence Interval shows uncertainty of the mean, Prediction Interval shows expected range for individual observations.", key="tacs_uncertainty")  # index=2 is "95% Confidence Interval"

        # Time axis selector
        time_axis_options = ["Time (seconds)", "Frame Index"]
        time_axis_select = st.selectbox("X-axis", options=time_axis_options, index=0, help="Select x-axis type", key="time_axis")
        use_actual_time = (time_axis_select == "Time (seconds)")

        # Time range filter - adjust based on time type
        if use_actual_time:
            # Use actual time in minutes for slider
            max_time_minutes = 70.0
            time_range_minutes = st.slider(
                "Time (minutes):",
                min_value=0.0,
                max_value=max_time_minutes,
                value=(0.0, max_time_minutes),
                step=0.5,
                help="Filter data to a specific time range in minutes"
            )
            # Convert back to seconds for filtering
            time_range = (time_range_minutes[0] * 60, time_range_minutes[1] * 60)
            time_column = 't'
        else:
            # Use frame numbers
            time_range = st.slider(
                "Frame range:",
                min_value=0,
                max_value=int(tacs_data['Frame Index'].max()),
                value=(0, int(tacs_data['Frame Index'].max())),
                help="Filter data to a specific frame range"
            )
            time_column = 'Frame Index'
    
    elif chosen_id == "tab2":
        st.subheader("Controls")
        
        # Grouping options - two dimensions
        grouping_variables = ["None", "Age", "Sex", "Erosion", "Normalization", "Organ"]
        
        x_axis_group = st.selectbox("X-axis grouping:", grouping_variables, index=5, help="Primary grouping for x-axis")
        color_group = st.selectbox("Color grouping:", grouping_variables, index=4, help="Secondary grouping for color hues")
        
        # Get available regions from means data
        available_regions = sorted(means_data['region'].unique())
        
        # Organ selector (conditional based on grouping)
        if x_axis_group != "Organ" and color_group != "Organ":
            # When not grouping by organ, select single organ
            default_organ = "Liver" if "Liver" in available_regions else available_regions[0]
            organ_select_static = st.selectbox("Organ", available_regions, index=available_regions.index(default_organ), help="Select organ to analyze")
            organs_select_static = [organ_select_static]  # Convert to list for consistency
        else:
            # When grouping by organ, allow multiple organ selection
            default_organs_multi = ["Liver", "Aorta"] if all(organ in available_regions for organ in ["Liver", "Aorta"]) else available_regions[:2]
            organs_select_static = st.multiselect("Organs to compare", available_regions, default=default_organs_multi, help="Select organs to include in the comparison")

        # Normalization selector (conditional based on grouping)
        if x_axis_group != "Normalization" and color_group != "Normalization":
            norm_selector_static = st.selectbox("Normalization",options=list(norm_options.keys()),index=1,help="Select normalization method for the PET signal")
            norm_select_static = norm_options[norm_selector_static]
            selected_norms = [norm_select_static]  # Single normalization
            selected_norm_names = [norm_selector_static]
        else:
            # Allow multiple normalization selection when grouping by normalization
            default_norm_names = ["SUV", "SUL (Janma)", "SUL (James)", "SUL (Decazes)"]
            # Filter to only include available options
            available_defaults = [norm for norm in default_norm_names if norm in norm_options.keys()]
            selected_norm_names = st.multiselect("Normalizations", options=list(norm_options.keys()), default=available_defaults, help="Select normalization methods to compare")
            selected_norms = [norm_options[name] for name in selected_norm_names]
            norm_selector_static = "Multiple"  # For display purposes

        # Erosion options (for when erosion is NOT selected as grouping variable)
        erosion_options = ["None", "1 iteration"]
        if x_axis_group != "Erosion" and color_group != "Erosion":
            erosion_select_static = st.selectbox("Erosion", options=erosion_options, index=1, help="Number of erosion iterations applied to organ mask")
        else:
            erosion_select_static = None  # Will show all erosion levels
    
    elif chosen_id == "tab4":
        st.subheader("Controls")

        # Get available regions from Patlak data
        available_regions_patlak = sorted(patlak_data['region'].unique())

        # Grouping options for two dimensions (similar to static tab)
        grouping_variables_patlak = ["None", "Age", "Sex", "Erosion", "Input Function", "Organ", "Frames"]

        x_axis_group_patlak = st.selectbox("X-axis grouping:", grouping_variables_patlak, index=5, help="Primary grouping for x-axis", key="patlak_x_axis")  # index=5 is "Organ"
        color_group_patlak = st.selectbox("Color grouping:", grouping_variables_patlak, index=6, help="Secondary grouping for color hues", key="patlak_color")  # index=6 is "Frames"

        # Organ selector (conditional based on grouping)
        if x_axis_group_patlak != "Organ" and color_group_patlak != "Organ":
            # When not grouping by organ, select single organ
            default_organ_patlak = "Brain" if "Brain" in available_regions_patlak else available_regions_patlak[0]
            organ_select_patlak = st.selectbox("Organ", available_regions_patlak, index=available_regions_patlak.index(default_organ_patlak), help="Select organ to analyze", key="patlak_organ")
            organs_select_patlak = [organ_select_patlak]
        else:
            # When grouping by organ, allow multiple organ selection
            default_organs_multi_patlak = ["White matter", "Gray matter"] if all(organ in available_regions_patlak for organ in ["White matter", "Gray matter"]) else available_regions_patlak[:2]
            organs_select_patlak = st.multiselect("Organs to compare", available_regions_patlak, default=default_organs_multi_patlak, help="Select organs to include in the comparison", key="patlak_organs")

        # Input function selector (combination of voi_volume and aorta_voi)
        # Get unique combinations of input function parameters and sort by aorta_voi
        input_functions = patlak_data[['voi_volume', 'aorta_voi']].drop_duplicates()
        # Sort by aorta_voi: Ascending, Top (or Ascending top), Descending top, Descending bottom
        sort_order = {'Ascending': 0, 'Top': 1, 'Ascending top': 1, 'Descending top': 2, 'Descending bottom': 3}
        input_functions['sort_key'] = input_functions['aorta_voi'].map(lambda x: sort_order.get(x, 999))
        input_functions = input_functions.sort_values('sort_key').drop('sort_key', axis=1)

        # Abbreviate aorta_voi names
        def abbreviate_aorta_voi(name):
            name = name.replace('Ascending', 'Asc.')
            name = name.replace('Descending', 'Des.')
            name = name.replace('top', 'top')
            name = name.replace('bottom', 'bot.')
            return name

        input_function_labels = [f"{abbreviate_aorta_voi(row['aorta_voi'])} ({row['voi_volume']})" for _, row in input_functions.iterrows()]

        if x_axis_group_patlak != "Input Function" and color_group_patlak != "Input Function":
            # Single input function selection
            # Try to find "Des. bot. (1mL, 3px width)" as default
            default_label = "Des. bot. (1mL, 3px width)"
            default_input_idx = input_function_labels.index(default_label) if default_label in input_function_labels else 0
            input_function_select = st.selectbox(
                "Input Function",
                options=input_function_labels,
                index=default_input_idx,
                help="Select input function defined by aorta VOI and volume",
                key="patlak_input"
            )
            selected_input_functions = [input_function_select]
        else:
            # Multiple input function selection
            # Default to "Des. bot. (1mL, 3px width)" if available
            default_label = "Des. bot. (1mL, 3px width)"
            default_inputs = [default_label] if default_label in input_function_labels else input_function_labels[:1]
            selected_input_functions = st.multiselect(
                "Input Functions",
                options=input_function_labels,
                default=default_inputs,
                help="Select input functions to compare",
                key="patlak_inputs"
            )

        # Erosion options (conditional based on grouping)
        erosion_options = ["None", "1 iteration"]
        if x_axis_group_patlak != "Erosion" and color_group_patlak != "Erosion":
            erosion_select_patlak = st.selectbox("Erosion", options=erosion_options, index=1, help="Number of erosion iterations applied to organ mask", key="patlak_erosion")
            selected_erosions_patlak = None  # Not used when not grouping by erosion
        else:
            # When grouping by erosion, allow multiple erosion selection
            default_erosions = ["None", "1 iteration"]
            selected_erosions_patlak = st.multiselect("Erosions to compare", options=erosion_options, default=default_erosions, help="Select erosion iterations to include in the comparison", key="patlak_erosions")
            erosion_select_patlak = None  # Not used when grouping by erosion

        # Frame selector - single value or range depending on grouping
        min_frames = int(patlak_data['Regression Frames'].min())
        max_frames = int(patlak_data['Regression Frames'].max())

        # Time vector for dynamic PET frames (in seconds)
        time_vector = [1,3,5,7,9,11.0,13.0,15.0,17.0,19.0,21.0,23.0,25.0,27.0,29.0,31.0,33.0,35.0,37.0,39.0,42.5,47.5,52.5,57.5,62.5,67.5,72.5,77.5,82.5,87.5,95.0,105,115,125,135,145,155,165,175,185,195,205,215,225,235,270,330,390,450,510,570,660,780,900,1020,1140,1260,1380,1500,1620,1740,1950,2250,2550,2850,3150,3450,3750,4050]

        def get_patlak_time_range(n_frames):
            """Get the time range for Patlak analysis using the last N frames"""
            # Patlak uses the last n_frames from the time vector
            if n_frames <= 0 or n_frames > len(time_vector):
                return None, None

            start_time_sec = time_vector[-n_frames]
            end_time_sec = time_vector[-1]

            # Convert to minutes
            start_time_min = start_time_sec / 60.0
            end_time_min = end_time_sec / 60.0

            return start_time_min, end_time_min

        if x_axis_group_patlak == "Frames" or color_group_patlak == "Frames":
            # Range slider when grouping by frames
            # Default range is 4-6 if available, otherwise use min-max
            default_range_start = 4 if 4 >= min_frames and 4 <= max_frames else min_frames
            default_range_end = 6 if 6 >= min_frames and 6 <= max_frames else max_frames
            frames_range = st.slider(
                "Number of frames range:",
                min_value=min_frames,
                max_value=max_frames,
                value=(default_range_start, default_range_end),
                help="Select range of frames to include in Patlak analysis"
            )
            # Display corresponding time ranges for the selected frame range
            start_min_min, start_max_min = get_patlak_time_range(frames_range[0])
            end_min_min, end_max_min = get_patlak_time_range(frames_range[1])
            if start_min_min is not None and end_max_min is not None:
                st.caption(f"Time range: {frames_range[0]} frames ({start_min_min:.1f}-{start_max_min:.1f} min) to {frames_range[1]} frames ({end_min_min:.1f}-{end_max_min:.1f} min)")
            selected_frames = None  # Not used in this mode
        else:
            # Single value slider when not grouping by frames
            # Default to 5 frames if available, otherwise use max
            default_frame = 5 if 5 >= min_frames and 5 <= max_frames else max_frames
            selected_frames = st.slider(
                "Number of frames:",
                min_value=min_frames,
                max_value=max_frames,
                value=default_frame,
                help="Select number of frames used in Patlak analysis"
            )
            # Display corresponding time range
            start_min, end_min = get_patlak_time_range(selected_frames)
            if start_min is not None and end_min is not None:
                st.caption(f"Time range: {start_min:.1f}-{end_min:.1f} min (last {selected_frames} frames)")
            frames_range = None  # Not used in this mode

# Ensure variables have default values for the non-active tab
if chosen_id != "tab1":
    # Default values for time series controls
    color_group_tacs = "Organ"
    organ_select_tacs = "Liver"
    organs_select_tacs = ["Liver", "Gray matter", "Skeletal muscle"]
    norm_selector_tacs = "SUL (Decazes)"
    selected_norms_tacs = ["SUL Decazes Denominator [Bq/mL]"]
    selected_norm_names_tacs = ["SUL (Decazes)"]
    erosion_select_tacs = "1 iteration"
    time_range = (0, 4200)  # Full time range in seconds
    uncertainty_tacs = "95% Confidence Interval"
    use_actual_time = True
    time_column = 't'

if chosen_id != "tab2":
    # Default values for static analysis controls
    x_axis_group = "Organ"
    color_group = "Normalization"
    organ_select_static = "Liver"
    organs_select_static = ["Liver", "Aorta"]
    norm_selector_static = "SUV"
    selected_norms = ["SUV Denominator [Bq/mL]", "SUL Janma Denominator [Bq/mL]", "SUL James Denominator [Bq/mL]", "SUL Decazes Denominator [Bq/mL]"]
    selected_norm_names = ["SUV", "SUL (Janma)", "SUL (James)", "SUL (Decazes)"]
    erosion_select_static = "1 iteration"

if chosen_id != "tab4":
    # Default values for Patlak controls
    x_axis_group_patlak = "Organ"
    color_group_patlak = "Frames"
    organ_select_patlak = "Brain"
    organs_select_patlak = ["White matter", "Gray matter"]
    selected_input_functions = ["Des. bot. (1mL, 3px width)"]
    erosion_select_patlak = "1 iteration"
    selected_erosions_patlak = None
    selected_frames = 5
    frames_range = (4, 6)

if chosen_id == "tab1":
    # --- Time Activity Curves Tab using TACS data ---

    # Use the tacs data
    tacs_df = tacs_data.copy()
    
    # Merge with demographic data from norm file
    demo_data = generate_norm_data().reset_index()
    tacs_df = tacs_df.merge(demo_data[['Subject', 'sex', 'demographic-group']], on='Subject', how='left')
    
    # Filter by time range using the selected time column
    tacs_df = tacs_df[(tacs_df[time_column] >= time_range[0]) & (tacs_df[time_column] <= time_range[1])]
    
    # Filter by selected organs (using region names directly)
    tacs_df = tacs_df[tacs_df.region.isin(organs_select_tacs)]
    
    # Apply erosion filter if erosion is not used for grouping
    if erosion_select_tacs is not None:
        # Map erosion option to actual value in data
        erosion_mapping = {
            "None": 0,
            "1 iteration": 1
        }
        erosion_value = erosion_mapping[erosion_select_tacs]
        tacs_df = tacs_df[tacs_df["Erosion Iterations"] == erosion_value]
    
    # Apply normalization(s) - create separate rows for each normalization when grouping by normalization
    if color_group_tacs == "Normalization":
        # Create multiple rows for different normalizations
        normalized_dfs = []
        for norm_name, norm_col in zip(selected_norm_names_tacs, selected_norms_tacs):
            df_norm = tacs_df.copy()
            norm_df = generate_norm_data()[norm_col]
            df_norm["mult"] = df_norm["Subject"].map(norm_df)
            df_norm["mu"] /= df_norm["mult"]
            df_norm["normalization"] = norm_name  # Add normalization as a grouping column
            normalized_dfs.append(df_norm)
        tacs_df = pd.concat(normalized_dfs, ignore_index=True)
    else:
        # Single normalization
        norm_df = generate_norm_data()[selected_norms_tacs[0]]
        tacs_df["mult"] = tacs_df["Subject"].map(norm_df)
        tacs_df["mu"] /= tacs_df["mult"]
    
    # Use region as organ name directly
    tacs_df["organ_name"] = tacs_df["region"]
    
    # Check if we have any data left after filtering
    if tacs_df.empty:
        st.warning("No data available for the selected organs and filters.")
        st.stop()
    
    # Map grouping variables to column names
    def get_column_name_tacs(group_var):
        if group_var == "Age":
            return 'demographic-group'
        elif group_var == "Sex":
            return 'sex'
        elif group_var == "Erosion":
            return 'Erosion Iterations'
        elif group_var == "Normalization":
            return 'normalization'
        elif group_var == "Organ":
            return 'organ_name'
        else:
            return None
    
    # Determine color mapping
    color_col_tacs = get_column_name_tacs(color_group_tacs)
    
    # Create title components
    title_parts = []
    if color_group_tacs != "None":
        title_parts.append(f"colored by {color_group_tacs}")
    
    title_suffix = "Curves" if norm_selector_tacs == "Multiple" else f"{norm_selector_tacs} Curves"
    title = f"Time Activity {title_suffix}"
    if title_parts:
        title += " " + ", ".join(title_parts)
    
    # Create the line plot
    colors = [
        "#EF553B",
        "#636EFA", 
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    
    fig_tacs = go.Figure()
    
    if color_col_tacs:
        # Group by color variable and create separate traces
        color_groups = sorted(tacs_df[color_col_tacs].unique())
        
        for i, color_group_val in enumerate(color_groups):
            group_data = tacs_df[tacs_df[color_col_tacs] == color_group_val]
            
            # Calculate mean and std across subjects for each time point
            summary_data = group_data.groupby(time_column)['mu'].agg(['mean', 'std', 'count']).reset_index()
            summary_data = summary_data.sort_values(time_column)
            
            # Apply uncertainty calculation based on selected option
            if uncertainty_tacs == "95% Confidence Interval":
                summary_data['uncertainty'] = 1.96 * summary_data['std'] / summary_data["count"].apply(np.sqrt)
            elif uncertainty_tacs == "95% Prediction Interval":
                summary_data['uncertainty'] = 1.96 * summary_data['std']
            elif uncertainty_tacs == "1 Standard Deviation":
                summary_data['uncertainty'] = summary_data['std']
            else:  # "None"
                summary_data['uncertainty'] = 0
            
            summary_data['uncertainty'] = summary_data['uncertainty'].fillna(0)
            
            trace_color = colors[i % len(colors)]
            
            # Create hover template based on time column
            if use_actual_time:
                hover_template = (
                    f'<b>{color_group_val}</b><br>'
                    'Time: %{x:.0f}s<br>'
                    'Value: %{y:.3f}<br>'
                    '<extra></extra>'
                )
            else:
                hover_template = (
                    f'<b>{color_group_val}</b><br>'
                    'Frame: %{x:.0f}<br>'
                    'Value: %{y:.3f}<br>'
                    '<extra></extra>'
                )
            
            fig_tacs.add_trace(go.Scatter(
                x=summary_data[time_column],
                y=summary_data['mean'],
                name=str(color_group_val),
                mode='lines+markers',
                line=dict(color=trace_color, width=2),
                marker=dict(size=4),
                hovertemplate=hover_template
            ))
            
            # Add error bands if uncertainty is not "None"
            if uncertainty_tacs != "None" and len(summary_data) > 0:
                color_rgb = px.colors.hex_to_rgb(trace_color)
                fillcolor = f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 0.2)'
                
                fig_tacs.add_trace(go.Scatter(
                    x=summary_data[time_column],
                    y=summary_data['mean'] + summary_data['uncertainty'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_tacs.add_trace(go.Scatter(
                    x=summary_data[time_column],
                    y=summary_data['mean'] - summary_data['uncertainty'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    fillcolor=fillcolor,
                    showlegend=False,
                    hoverinfo='skip'
                ))
    else:
        # No color grouping - single trace
        summary_data = tacs_df.groupby(time_column)['mu'].agg(['mean', 'std', 'count']).reset_index()
        summary_data = summary_data.sort_values(time_column)
        
        # Apply uncertainty calculation based on selected option
        if uncertainty_tacs == "95% Confidence Interval":
            summary_data['uncertainty'] = 1.96 * summary_data['std'] / summary_data["count"].apply(np.sqrt)
        elif uncertainty_tacs == "95% Prediction Interval":
            summary_data['uncertainty'] = 1.96 * summary_data['std']
        elif uncertainty_tacs == "1 Standard Deviation":
            summary_data['uncertainty'] = summary_data['std']
        else:  # "None"
            summary_data['uncertainty'] = 0
        
        summary_data['uncertainty'] = summary_data['uncertainty'].fillna(0)
        
        # Create hover template based on time column
        if use_actual_time:
            hover_template_single = (
                'Time: %{x:.0f}s<br>'
                'Value: %{y:.3f}<br>'
                '<extra></extra>'
            )
        else:
            hover_template_single = (
                'Frame: %{x:.0f}<br>'
                'Value: %{y:.3f}<br>'
                '<extra></extra>'
            )
        
        fig_tacs.add_trace(go.Scatter(
            x=summary_data[time_column],
            y=summary_data['mean'],
            name="All Data",
            mode='lines+markers',
            line=dict(color=colors[0], width=2),
            marker=dict(size=4),
            hovertemplate=hover_template_single
        ))
        
        # Add error bands if uncertainty is not "None"
        if uncertainty_tacs != "None" and len(summary_data) > 0:
            color_rgb = px.colors.hex_to_rgb(colors[0])
            fillcolor = f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 0.2)'
            
            fig_tacs.add_trace(go.Scatter(
                x=summary_data[time_column],
                y=summary_data['mean'] + summary_data['uncertainty'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_tacs.add_trace(go.Scatter(
                x=summary_data[time_column],
                y=summary_data['mean'] - summary_data['uncertainty'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                fillcolor=fillcolor,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout
    y_label = "Normalized Values" if norm_selector_tacs == "Multiple" else norm_selector_tacs
    x_label = "Time [s]" if use_actual_time else "Frame Number"

    fig_tacs.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified',
        height=485,
        template='plotly_white',
        legend=dict(
            traceorder="normal",
            title="Groups" if color_col_tacs else None,
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor='rgba(255, 255, 255, 0.7)',
        ),
        margin=dict(l=50, r=20, t=0, b=50)
    )
    
    st.plotly_chart(fig_tacs, use_container_width=True)

elif chosen_id == "tab2":
    # --- Static Organ Analysis Tab ---

    # Use the static means data
    static_df = means_data.copy()
    
    # Merge with demographic data from norm file
    demo_data = generate_norm_data().reset_index()
    static_df = static_df.merge(demo_data[['Subject', 'sex', 'demographic-group']], on='Subject', how='left')
    
    # Filter by selected organs (using region names directly)
    static_df = static_df[static_df.region.isin(organs_select_static)]
    
    # Apply erosion filter if erosion is not used for grouping
    if erosion_select_static is not None:
        # Map erosion option to actual value in data
        erosion_mapping = {
            "None": 0,
            "1 iteration": 1
        }
        erosion_value = erosion_mapping[erosion_select_static]
        static_df = static_df[static_df["Erosion Iterations"] == erosion_value]
    
    # Apply normalization(s) - create separate rows for each normalization when grouping by normalization
    if x_axis_group == "Normalization" or color_group == "Normalization":
        # Create multiple rows for different normalizations
        normalized_dfs = []
        for norm_name, norm_col in zip(selected_norm_names, selected_norms):
            df_norm = static_df.copy()
            norm_df = generate_norm_data()[norm_col]
            df_norm["mult"] = df_norm["Subject"].map(norm_df)
            df_norm["mu"] /= df_norm["mult"]
            df_norm["normalization"] = norm_name  # Add normalization as a grouping column
            normalized_dfs.append(df_norm)
        static_df = pd.concat(normalized_dfs, ignore_index=True)
    else:
        # Single normalization
        norm_df = generate_norm_data()[selected_norms[0]]
        static_df["mult"] = static_df["Subject"].map(norm_df)
        static_df["mu"] /= static_df["mult"]
    
    # Use region as organ name directly
    static_df["organ_name"] = static_df["region"]
    
    # Check if we have any data left after filtering
    if static_df.empty:
        st.warning("No data available for the selected organs and filters.")
        st.stop()
    
    # Map grouping variables to column names
    def get_column_name(group_var):
        if group_var == "Age":
            return 'demographic-group'
        elif group_var == "Sex":
            return 'sex'
        elif group_var == "Erosion":
            return 'Erosion Iterations'
        elif group_var == "Normalization":
            return 'normalization'
        elif group_var == "Organ":
            return 'organ_name'
        else:
            return None
    
    # Determine x-axis and color mappings
    x_col = get_column_name(x_axis_group)
    color_col = get_column_name(color_group)
    
    # Create title components
    title_parts = []
    if x_axis_group != "None":
        title_parts.append(f"by {x_axis_group}")
    if color_group != "None":
        title_parts.append(f"colored by {color_group}")
    
    title_suffix = f"Values" if norm_selector_static == "Multiple" else f"{norm_selector_static} Values"
    title = f"Static Organ {title_suffix}"
    if title_parts:
        title += " " + ", ".join(title_parts)
    
    # Create the box plot with two-dimensional grouping
    if x_col is None:
        # No x-axis grouping - use a default
        x_col = 'organ_name'
    
    if color_col:
        # Color grouping
        fig_box = px.box(
            static_df,
            x=x_col,
            y="mu",
            color=color_col
        )
    else:
        # No color grouping
        fig_box = px.box(
            static_df,
            x=x_col,
            y="mu"
        )
    
    y_label = "Normalized Values" if norm_selector_static == "Multiple" else norm_selector_static
    x_label = x_axis_group if x_axis_group != "None" else "Organ"
    fig_box.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=485,
        template='plotly_white',
        margin=dict(l=50, r=20, t=0, b=50)
    )
    
    # Update box plot to show mean values and improve hover formatting
    fig_box.update_traces(
        boxmean=True,
        hovertemplate='<b>%{x}</b><br>' +
                     'Value: %{y:.2f}<br>' +
                     '<extra></extra>'
    )
    
    # Rotate x-axis labels if many categories or long names
    if x_col == 'organ_name' and len(organs_select_static) > 3:
        fig_box.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Summary statistics table
    st.subheader("Summary Statistics")
    
    # Create grouping columns for summary stats
    group_cols_for_stats = []
    if x_col and x_col != 'organ_name':
        group_cols_for_stats.append(x_col)
    if color_col and color_col not in group_cols_for_stats:
        group_cols_for_stats.append(color_col)
    
    # Always include organ_name in grouping
    if 'organ_name' not in group_cols_for_stats:
        group_cols_for_stats = ['organ_name'] + group_cols_for_stats
    
    if group_cols_for_stats:
        summary_stats = static_df.groupby(group_cols_for_stats)['mu'].agg(['count', 'mean', 'std', 'min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75), 'max']).round(3)
    else:
        summary_stats = static_df.groupby('organ_name')['mu'].agg(['count', 'mean', 'std', 'min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75), 'max']).round(3)
    
    # Rename the lambda columns to proper names
    summary_stats.columns = ['count', 'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max']
    
    # Add coefficient of variation (CV = std/mean) as percentage
    summary_stats['Coefficient of Variation [%]'] = (summary_stats['std'] / summary_stats['mean'] * 100).round(1)
    
    st.dataframe(summary_stats, use_container_width=True)

elif chosen_id == "tab4":
    # --- Patlak Analysis Tab ---

    # Use the Patlak data
    patlak_df = patlak_data.copy()

    # Merge with demographic data from norm file
    demo_data = generate_norm_data().reset_index()
    patlak_df = patlak_df.merge(demo_data[['Subject', 'sex', 'demographic-group']], on='Subject', how='left')

    # Filter by frame count (single value or range)
    if selected_frames is not None:
        # Single frame value - not grouping by frames
        patlak_df = patlak_df[patlak_df['Regression Frames'] == selected_frames]
    elif frames_range is not None:
        # Frame range - grouping by frames
        patlak_df = patlak_df[(patlak_df['Regression Frames'] >= frames_range[0]) & (patlak_df['Regression Frames'] <= frames_range[1])]

    # Filter by selected organs (using region names directly)
    patlak_df = patlak_df[patlak_df.region.isin(organs_select_patlak)]

    # Apply erosion filter
    erosion_mapping = {
        "None": 0,
        "1 iteration": 1
    }

    if erosion_select_patlak is not None:
        # Single erosion value - not grouping by erosion
        erosion_value = erosion_mapping[erosion_select_patlak]
        patlak_df = patlak_df[patlak_df["Erosion Iterations"] == erosion_value]
    elif selected_erosions_patlak is not None and len(selected_erosions_patlak) > 0:
        # Multiple erosion values - grouping by erosion
        erosion_values = [erosion_mapping[e] for e in selected_erosions_patlak]
        patlak_df = patlak_df[patlak_df["Erosion Iterations"].isin(erosion_values)]

    # Helper function to parse abbreviated input function label back to original values
    def parse_input_function_label(label):
        """Parse abbreviated label like 'Asc. (1mL, 3px width)' back to original aorta_voi and voi_volume"""
        aorta_voi_abbrev = label.split(' (')[0]
        voi_volume = label.split('(')[1].rstrip(')')

        # Convert abbreviated name back to original
        aorta_voi = aorta_voi_abbrev.replace('Asc.', 'Ascending')
        aorta_voi = aorta_voi.replace('Des.', 'Descending')
        aorta_voi = aorta_voi.replace('bot.', 'bottom')

        return aorta_voi, voi_volume

    # Helper function to abbreviate aorta_voi for display
    def abbreviate_aorta_voi_display(name):
        """Abbreviate aorta_voi name for display"""
        name = name.replace('Ascending', 'Asc.')
        name = name.replace('Descending', 'Des.')
        name = name.replace('bottom', 'bot.')
        return name

    # Filter by input function(s)
    if x_axis_group_patlak != "Input Function" and color_group_patlak != "Input Function":
        # Single input function - parse the label back to voi_volume and aorta_voi
        if selected_input_functions:
            input_label = selected_input_functions[0]
            aorta_voi, voi_volume = parse_input_function_label(input_label)
            patlak_df = patlak_df[(patlak_df.aorta_voi == aorta_voi) & (patlak_df.voi_volume == voi_volume)]
            patlak_df['input_function'] = input_label
    else:
        # Multiple input functions - create label for grouping
        if selected_input_functions:
            # Filter to only include selected input functions
            filtered_dfs = []
            for input_label in selected_input_functions:
                aorta_voi, voi_volume = parse_input_function_label(input_label)
                df_temp = patlak_df[(patlak_df.aorta_voi == aorta_voi) & (patlak_df.voi_volume == voi_volume)].copy()
                df_temp['input_function'] = input_label
                filtered_dfs.append(df_temp)
            patlak_df = pd.concat(filtered_dfs, ignore_index=True)
        else:
            # If no input functions selected, create abbreviated label for all
            patlak_df['input_function'] = patlak_df.apply(
                lambda row: f"{abbreviate_aorta_voi_display(row['aorta_voi'])} ({row['voi_volume']})",
                axis=1
            )

    # Use region as organ name directly
    patlak_df["organ_name"] = patlak_df["region"]

    # Convert slope to Ki * 1000 for better readability
    patlak_df["ki_1000"] = patlak_df["slope"] * 1000

    # Check if we have any data left after filtering
    if patlak_df.empty:
        st.warning("No data available for the selected organs and filters.")
        st.stop()

    # Map grouping variables to column names
    def get_column_name_patlak(group_var):
        if group_var == "Age":
            return 'demographic-group'
        elif group_var == "Sex":
            return 'sex'
        elif group_var == "Erosion":
            return 'Erosion Iterations'
        elif group_var == "Input Function":
            return 'input_function'
        elif group_var == "Organ":
            return 'organ_name'
        elif group_var == "Frames":
            return 'Regression Frames'
        else:
            return None

    # Determine x-axis and color mappings
    x_col_patlak = get_column_name_patlak(x_axis_group_patlak)
    color_col_patlak = get_column_name_patlak(color_group_patlak)

    # Create title components
    title_parts = []
    if x_axis_group_patlak != "None":
        title_parts.append(f"by {x_axis_group_patlak}")
    if color_group_patlak != "None":
        title_parts.append(f"colored by {color_group_patlak}")

    # Title depends on whether we're grouping by frames or using a single frame value
    if selected_frames is not None:
        title = f"Patlak K<sub>i</sub> Analysis ({selected_frames} frames)"
    elif frames_range is not None:
        if frames_range[0] == frames_range[1]:
            title = f"Patlak K<sub>i</sub> Analysis ({frames_range[0]} frames)"
        else:
            title = f"Patlak K<sub>i</sub> Analysis ({frames_range[0]}-{frames_range[1]} frames)"
    else:
        title = "Patlak K<sub>i</sub> Analysis"

    if title_parts:
        title += " " + ", ".join(title_parts)

    # Create the box plot with two-dimensional grouping (similar to static tab)
    if x_col_patlak is None:
        # No x-axis grouping - use a default
        x_col_patlak = 'organ_name'

    if color_col_patlak:
        # Color grouping
        fig_patlak = px.box(
            patlak_df,
            x=x_col_patlak,
            y="ki_1000",
            color=color_col_patlak
        )
    else:
        # No color grouping
        fig_patlak = px.box(
            patlak_df,
            x=x_col_patlak,
            y="ki_1000"
        )

    x_label = x_axis_group_patlak if x_axis_group_patlak != "None" else "Organ"
    fig_patlak.update_layout(
        xaxis_title=x_label,
        yaxis_title="K<sub>i</sub> [10<sup>-3</sup> min<sup>-1</sup>]",
        height=485,
        template='plotly_white',
        margin=dict(l=50, r=20, t=0, b=50)
    )

    # Update box plot to show mean values and improve hover formatting
    fig_patlak.update_traces(
        boxmean=True,
        hovertemplate='<b>%{x}</b><br>' +
                     'K<sub>i</sub>: %{y:.3f} × 10<sup>-3</sup><br>' +
                     '<extra></extra>'
    )

    # Rotate x-axis labels if many categories or long names
    if x_col_patlak == 'organ_name' and len(organs_select_patlak) > 3:
        fig_patlak.update_xaxes(tickangle=45)

    st.plotly_chart(fig_patlak, use_container_width=True)

    # Summary statistics table
    st.subheader("Summary Statistics")

    # Create grouping columns for summary stats
    group_cols_for_stats = []
    if x_col_patlak and x_col_patlak != 'organ_name':
        group_cols_for_stats.append(x_col_patlak)
    if color_col_patlak and color_col_patlak not in group_cols_for_stats:
        group_cols_for_stats.append(color_col_patlak)

    # Always include organ_name in grouping
    if 'organ_name' not in group_cols_for_stats:
        group_cols_for_stats = ['organ_name'] + group_cols_for_stats

    if group_cols_for_stats:
        summary_stats = patlak_df.groupby(group_cols_for_stats)['ki_1000'].agg(['count', 'mean', 'std', 'min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75), 'max']).round(3)
    else:
        summary_stats = patlak_df.groupby('organ_name')['ki_1000'].agg(['count', 'mean', 'std', 'min', lambda x: x.quantile(0.25), 'median', lambda x: x.quantile(0.75), 'max']).round(3)

    # Rename the lambda columns to proper names
    summary_stats.columns = ['count', 'mean [×10⁻³]', 'std [×10⁻³]', 'min [×10⁻³]', 'Q1 [×10⁻³]', 'median [×10⁻³]', 'Q3 [×10⁻³]', 'max [×10⁻³]']

    # Add coefficient of variation (CV = std/mean) as percentage
    summary_stats['Coefficient of Variation [%]'] = (summary_stats['std [×10⁻³]'] / summary_stats['mean [×10⁻³]'].abs() * 100).round(1)

    st.dataframe(summary_stats, use_container_width=True)

elif chosen_id == "tab5":
    # --- About Tab ---
    st.markdown("""
    # About hedyPET

    hedyPET is a comprehensive data explorer for FDG PET/CT imaging data from 100 healthy humans.
    This tool provides interactive visualizations and statistical analysis of:

    - **Time Activity Curves (TACs)**: Explore organ uptake over time with various normalization methods
    - **Organ Means**: Analyze static organ uptake values across different demographics
    - **Patlak Analysis**: Examine glucose metabolic rates using Patlak kinetic modeling
    """)

    # Quick Links section prominently displayed
    st.markdown("## Resources")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("📘 Visit GitHub Repository", "https://github.com/your-repo-url", use_container_width=True)
    with col2:
        st.link_button("📖 How to Acquire Data", "https://your-acquisition-website.com", use_container_width=True)

    st.markdown("""
    ---

    ## Features

    - Multiple normalization methods (SUV, SUL variants)
    - Demographic grouping (age, sex)
    - Erosion iterations for organ mask refinement
    - Statistical uncertainty visualization (confidence intervals, prediction intervals)
    - Interactive filtering and comparison tools

    ## Citation

    If you use this data or tool in your research, please cite:

    > [Citation information to be added]

    ## Contact

    For questions or feedback, please open an issue on the GitHub repository.
    """)