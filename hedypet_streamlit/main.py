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
    return pd.read_pickle(os.path.join(os.path.dirname(__file__),"norm2.pkl")).set_index("sub")

@st.cache_data
def load_dist_data():
    """Load the temporal distribution data"""
    return np.load(os.path.join(os.path.dirname(__file__), "dist.npy"))

@st.cache_data
def load_means_data():
    """Load the static organ means data"""
    return pd.read_pickle(os.path.join(os.path.dirname(__file__), "means.pkl"))

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
means_data = load_means_data()  # Load static organ means data

_norm = generate_norm_data()
norm_options = {"SUV":"suv",
 "SUL (Janma)": "sul_janma",
 "SUL (James)": "sul_james",
 "SUL (Decazes)": "sul_decazes",
 "SUV ID": "suv_id",
 "SUL ID": "sul_id"}

list(_norm.columns)

with open(os.path.join(os.path.dirname(__file__),"totalseg_ct_classes.json"), "r") as handle:
    ts_classes_orig = json.load(handle)

ts_classes = {v:int(k) for k,v in ts_classes_orig.items()}

# Title and description
st.title("📈 18F-FDG 100 Healthy Humans")

# Create tab bar with extra-streamlit-components
chosen_id = stx.tab_bar(data=[
    stx.TabBarItemData(id="tab1", title="Time Activity Curves", description=""),
    stx.TabBarItemData(id="tab2", title="Static Organ Readouts", description=""),
    stx.TabBarItemData(id="tab3", title="Axial Distribution", description=""),
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
        st.header("🎛️ Static Analysis Controls")
        
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
            default_norm_names = ["SUV", "SUV ID", "SUL (Decazes)", "SUL ID"]
            # Filter to only include available options
            available_defaults = [norm for norm in default_norm_names if norm in norm_options.keys()]
            selected_norm_names = st.multiselect("Normalizations", options=list(norm_options.keys()), default=available_defaults, help="Select normalization methods to compare")
            selected_norms = [norm_options[name] for name in selected_norm_names]
            norm_selector_static = "Multiple"  # For display purposes

        # Erosion options (for when erosion is NOT selected as grouping variable)
        erosion_options = ["None", "1 iteration", "2 iterations", "3 iterations"]
        if x_axis_group != "Erosion" and color_group != "Erosion":
            erosion_select_static = st.selectbox("Erosion", options=erosion_options, index=1, help="Number of erosion iterations applied to organ mask")
        else:
            erosion_select_static = None  # Will show all erosion levels
    
    elif chosen_id == "tab3":
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
    norm_selector = "SUL (Janma)"
    norm_select = "sul_janma"
    erosion_options = ["None", "1 iteration", "2 iterations", "3 iterations"]
    erosion_select = "1 iteration"
    show_std = True
    time_range = (0, 4050)

if chosen_id != "tab2":
    # Default values for static analysis controls
    x_axis_group = "Organ"
    color_group = "Normalization"
    organ_select_static = "Liver"
    organs_select_static = ["Liver", "Aorta"]
    norm_selector_static = "SUV"
    selected_norms = ["suv", "suv_id", "sul_decazes", "sul_id"]
    selected_norm_names = ["SUV", "SUV ID", "SUL (Decazes)", "SUL ID"]
    erosion_select_static = "1 iteration"

if chosen_id != "tab3":
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
    # --- Static Organ Analysis Tab ---
    
    # Use the static means data
    static_df = means_data.copy()
    
    # Merge with demographic data from norm file
    demo_data = generate_norm_data().reset_index()
    static_df = static_df.merge(demo_data[['sub', 'sex', 'demographic-group']], on='sub', how='left')
    
    # Filter by selected organs (using region names directly)
    static_df = static_df[static_df.region.isin(organs_select_static)]
    
    # Apply erosion filter if erosion is not used for grouping
    if erosion_select_static is not None:
        # Map erosion option to actual value in data
        erosion_mapping = {
            "None": "0",
            "1 iteration": "1", 
            "2 iterations": "2",
            "3 iterations": "3"
        }
        erosion_value = erosion_mapping[erosion_select_static]
        static_df = static_df[static_df.erosion == erosion_value]
    
    # Apply normalization(s) - create separate rows for each normalization when grouping by normalization
    if x_axis_group == "Normalization" or color_group == "Normalization":
        # Create multiple rows for different normalizations
        normalized_dfs = []
        for norm_name, norm_col in zip(selected_norm_names, selected_norms):
            df_norm = static_df.copy()
            norm_df = generate_norm_data()[norm_col]
            df_norm["mult"] = df_norm["sub"].map(norm_df)
            df_norm["mu"] /= df_norm["mult"]
            df_norm["normalization"] = norm_name  # Add normalization as a grouping column
            normalized_dfs.append(df_norm)
        static_df = pd.concat(normalized_dfs, ignore_index=True)
    else:
        # Single normalization
        norm_df = generate_norm_data()[selected_norms[0]]
        static_df["mult"] = static_df["sub"].map(norm_df)
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
            return 'erosion'
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
            color=color_col,
            title=title
        )
    else:
        # No color grouping
        fig_box = px.box(
            static_df,
            x=x_col,
            y="mu",
            title=title
        )
    
    y_label = "Normalized Values" if norm_selector_static == "Multiple" else norm_selector_static
    x_label = x_axis_group if x_axis_group != "None" else "Organ"
    fig_box.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=600,
        template='plotly_white'
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

elif chosen_id == "tab3":


    
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