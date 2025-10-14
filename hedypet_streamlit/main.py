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

@st.cache_data
def generate_mu_data():
    return pd.read_pickle(os.path.join(os.path.dirname(__file__),"v1.pkl.gz"))

@st.cache_data
def generate_norm_data():
    return pd.read_pickle(os.path.join(os.path.dirname(__file__),"norm.pkl")).set_index("sub")

@st.cache_data
def load_dist_data():
    """Load the temporal distribution data"""
    return np.load(os.path.join(os.path.dirname(__file__), "dist.npy"))

def calculate_summary_stats(df, group_cols, organs, erosion=1, norm="sul_ct",uncertainty="std"):
    """Calculate mean and std for grouped data"""
    # This function now simply USES the group_cols list it's given.
    # It no longer modifies it.
    assert erosion in [0, 1, 2, 3], erosion
    df = df[df.erosion == erosion].copy()
    
    # Filter by selected organs
    df = df[df.seg.isin(organs)]

    norm_df = generate_norm_data()[norm]
    df["mult"] = df["sub"].map(norm_df)
    df["mu"] /= df["mult"]
    
    # Group by the provided columns, which now correctly includes 'seg'
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
df = generate_mu_data()
dist_data = load_dist_data()  # Load the new temporal distribution data

_norm = generate_norm_data()
norm_options = {"SUV":"suv",
 "SUL (CT)": "sul_ct",
 "SUL (Janma)":"sul_janna",
 "SUL (James)": "sul_james"}

list(_norm.columns)

with open(os.path.join(os.path.dirname(__file__),"totalseg_ct_classes.json"), "r") as handle:
    ts_classes_orig = json.load(handle)

ts_classes = {v:int(k) for k,v in ts_classes_orig.items()}

# Title and description
st.title("📈 18F-FDG 100 Healthy Humans")

# Create tab bar with extra-streamlit-components
chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="📊 Time activity curves", description="Time series analysis"),
    stx.TabBarItemData(id="tab2", title="📈 Axial distribution", description="Spatial distribution"),
], default="tab1")

# Show appropriate controls in sidebar based on active tab
with st.sidebar:
    if chosen_id == "tab1":
        st.header("🎛️ Time Series Controls")
        
        # Grouping options
        grouping_options = {
            'None': [],
            'Gender': ['gender'],
            'Age': ['age-group'],
            'Gender + Age': ['gender', 'age-group']
        }
        group_by = st.selectbox("Group by:", options=list(grouping_options.keys()), index=1)

        # Organs multiselect
        organs_select = st.multiselect("Organ", list(ts_classes.keys()), default=["liver"])
        organs = [ts_classes[organ] for organ in organs_select]

        # Uncertainty options
        uncertainty_options = {
            "None": "none",
            "±1std": "std",
            "95% CI": "95CI",
            "95% PI": "95PI"
        }
        uncertainty_select = st.selectbox("Uncertainty", options=list(uncertainty_options.keys()), index=2)
        uncertainty = uncertainty_options[uncertainty_select]

        # Normalization selector
        norm_selector = st.selectbox("Normalization",options=list(norm_options.keys()),index=1,help="Select normalization method for the PET signal")
        norm_select = norm_options[norm_selector]

        # Erosion options
        erosion_options = ["None", "1 iteration", "2 iterations", "3 iterations"]
        erosion_select = st.selectbox("Erosion", options=erosion_options, index=1, help="Number of erosion iterations applied to organ mask")
        show_std = uncertainty_select != "None"

        # Time range filter
        time_range = st.slider(
            "Time (seconds):",
            min_value=0,
            max_value=4050,
            value=(0, 4050),
            help="Filter data to a specific time range"
        )
    
    elif chosen_id == "tab2":
        st.header("🎛️ Axial Distribution Controls")
        
        # Time slider for selecting which second to display
        selected_time = st.slider(
            "Select Time Point (seconds):",
            min_value=0,
            max_value=min(4049, dist_data.shape[0] - 1),  # Ensure we don't exceed array bounds
            value=0,
            help="Choose which time point to display the distribution for"
        )

# Ensure variables have default values for the non-active tab
if chosen_id != "tab1":
    # Default values for time series controls
    grouping_options = {
        'None': [],
        'Gender': ['gender'],
        'Age': ['age-group'],
        'Gender + Age': ['gender', 'age-group']
    }
    group_by = 'Gender'
    organs_select = ["liver"]
    organs = [ts_classes["liver"]]
    uncertainty_select = "95% CI"
    uncertainty = "95CI"
    norm_selector = "SUL (CT)"
    norm_select = "sul_ct"
    erosion_options = ["None", "1 iteration", "2 iterations", "3 iterations"]
    erosion_select = "1 iteration"
    show_std = True
    time_range = (0, 4050)

if chosen_id != "tab2":
    # Default value for axial distribution controls
    selected_time = 0

if chosen_id == "tab1":

    # --- Data Processing ---
    # Filter data by time range first
    filtered_df = df[(df['t'] >= time_range[0]) & (df['t'] <= time_range[1])]

    # ▼▼▼ STANDARDIZED GROUPING LOGIC ▼▼▼
    # 1. Start with the user's selection from the dropdown.
    base_group_cols = grouping_options[group_by]
    # 2. ALWAYS add 'seg' (organ) to the list of columns to group by.
    # This is now the single source of truth for grouping.
    final_group_cols = base_group_cols + ['seg']
    # ▲▲▲

    # Calculate summary statistics using the final, complete list of grouping columns.
    summary_stats = calculate_summary_stats(filtered_df, final_group_cols, organs, erosion_options.index(erosion_select),norm_select,uncertainty)
    summary_stats["seg"] = summary_stats["seg"].astype(str).map(ts_classes_orig)

    # --- Main Content and Visualization ---
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

    fig = go.Figure()

    # Define the order of precedence for visual mapping
    visual_mapping_priority = [col for col in ["gender","seg", "age-group"] if col in final_group_cols]

    # Identify which of these grouping variables are actually dynamic
    dynamic_vars = []
    for col in visual_mapping_priority:
        if summary_stats[col].nunique() > 1:
            dynamic_vars.append(col)

    # --- Create Mappings for Visual Properties ---
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star']
    linestyles = ['solid', 'dot', 'dash', 'longdash', 'dashdot']

    color_map, symbol_map, linestyle_map = {}, {}, {}

    if len(dynamic_vars) > 0:
        color_var = dynamic_vars[0]
        color_values = sorted(summary_stats[color_var].unique())
        color_map = {val: colors[i % len(colors)] for i, val in enumerate(color_values)}

    if len(dynamic_vars) > 1:
        symbol_var = dynamic_vars[1]
        symbol_values = sorted(summary_stats[symbol_var].unique())
        symbol_map = {val: symbols[i % len(symbols)] for i, val in enumerate(symbol_values)}

    if len(dynamic_vars) > 2:
        linestyle_var = dynamic_vars[2]
        linestyle_values = sorted(summary_stats[linestyle_var].unique())
        linestyle_map = {val: linestyles[i % len(linestyles)] for i, val in enumerate(linestyles)}

    # Define the list that dictates the sorting and legend text order
    legend_order = dynamic_vars + [v for v in visual_mapping_priority if v not in dynamic_vars]

    # Sort the DataFrame for both legend order and chronological plotting order
    if legend_order:
        summary_stats = summary_stats.sort_values(by=legend_order + ['t'])

    # --- Plotting Loop ---
    for group_keys, group_data in summary_stats.groupby(final_group_cols, sort=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        group_dict = dict(zip(final_group_cols, group_keys))

        trace_color = colors[0]
        trace_symbol = 'circle'
        trace_linestyle = 'solid'

        if len(dynamic_vars) > 0:
            trace_color = color_map.get(group_dict[dynamic_vars[0]])
        if len(dynamic_vars) > 1:
            trace_symbol = symbol_map.get(group_dict[dynamic_vars[1]])
        if len(dynamic_vars) > 2:
            trace_linestyle = linestyle_map.get(group_dict[dynamic_vars[2]])
        
        legend_name = ', '.join([str(group_dict[key]) for key in legend_order])
        
        group_data = group_data.sort_values('t')

        fig.add_trace(go.Scatter(
            x=group_data['t'],
            y=group_data['mean'],
            name=legend_name,
            mode='lines+markers',
            line=dict(color=trace_color, width=1.2, dash=trace_linestyle),
            marker=dict(symbol=trace_symbol, size=4),
            legendgroup=legend_name,
            hovertemplate=(
                f'<b>{legend_name}</b><br>'
                'Time: %{x:i}s <br>'
                f'{norm_selector}:' ' %{y:.2f} <br>'
                '<extra></extra>'
            )
        ))

        if show_std:
            color_rgb = px.colors.hex_to_rgb(trace_color)
            fillcolor = f'rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 0.2)'
            
            fig.add_trace(go.Scatter(
                x=group_data['t'],
                y=group_data['mean'] + group_data['std'],
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
                legendgroup=legend_name,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=group_data['t'],
                y=group_data['mean'] - group_data['std'],
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                fillcolor=fillcolor,
                showlegend=False,
                legendgroup=legend_name,
                hoverinfo='skip'
            ))

    # --- Figure Layout ---
    fig.update_layout(
        xaxis_title="Time [s]",
        yaxis_title=f"{norm_selector} Mean",
        hovermode='x unified',
        height=600,
        template='plotly_white',
        # Position legend inside the plot area
        legend=dict(
            traceorder="grouped",
            title="Groups",
            orientation="v",
            yanchor="top",
            y=0.98, # Position it just below the top
            xanchor="right",
            x=0.98, # Position it just inside the right edge
            bgcolor='rgba(255, 255, 255, 0.7)', # Make background semi-transparent
        ),
        # Reduce the right margin since the legend is no longer outside
        margin=dict(l=50, r=20, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

elif chosen_id == "tab2":


    
    # Get the distribution for the selected time point
    if selected_time < dist_data.shape[0]:
        current_dist = dist_data[selected_time, :]
        
        # Create the distribution plot
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Scatter(
            x=list(range(len(current_dist))),
            y=current_dist,
            mode='lines',
            name=f'Distribution at t={selected_time}s',
            line=dict(color='#636EFA', width=2),
            hovertemplate=(
                'Index: %{x}<br>'
                'Value: %{y:.4f}<br>'
                '<extra></extra>'
            )
        ))
        
        # Update layout
        fig_dist.update_layout(
            title=f"Normalized axial FDG distribution: {selected_time} seconds",
            xaxis_title="Distribution Index",
            yaxis_title="Normalized average density",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        

        fig_dist.update_layout(yaxis_range=[0, 4])
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Add the video below the graph in a centered, narrower column
        st.markdown("---")  # Add a separator line
        
        # Create columns to center and narrow the video
        vid_col1, vid_col2, vid_col3 = st.columns([1, 4, 1])  # Left spacer, video, right spacer
        
        with vid_col2:
            st.subheader("Animation")
            video_path = os.path.join(os.path.dirname(__file__), "animated_plot.mp4")
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.warning("Video file 'animated_plot.mp4' not found in the expected location.")
      
    else:
        st.error(f"Selected time point ({selected_time}) exceeds available data range.")
        st.info(f"Available time range: 0 to {dist_data.shape[0] - 1} seconds")