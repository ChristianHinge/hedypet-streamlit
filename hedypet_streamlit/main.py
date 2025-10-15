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

@st.cache_data
def load_tacs_data():
    """Load the time activity curves data"""
    return pd.read_pickle(os.path.join(os.path.dirname(__file__), "tacs.pkl"))

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
df = generate_mu_data()  # Keep for backward compatibility if needed
dist_data = load_dist_data()  # Load the new temporal distribution data
means_data = load_means_data()  # Load static organ means data
tacs_data = load_tacs_data()  # Load time activity curves data

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

        # Get available regions from tacs data
        available_regions_tacs = sorted(tacs_data['region'].unique())

        # Color grouping for time activity curves
        grouping_variables = ["None", "Age", "Sex", "Erosion", "Normalization", "Organ"]
        color_group_tacs = st.selectbox("Color grouping:", grouping_variables, index=0, help="Group curves by color", key="tacs_color")

        # Organ selector (conditional based on grouping)
        if color_group_tacs != "Organ":
            # When not grouping by organ, select single organ
            default_organ_tacs = "Liver" if "Liver" in available_regions_tacs else available_regions_tacs[0]
            organ_select_tacs = st.selectbox("Organ", available_regions_tacs, index=available_regions_tacs.index(default_organ_tacs), help="Select organ to analyze")
            organs_select_tacs = [organ_select_tacs]
        else:
            # When grouping by organ, allow multiple organ selection
            default_organs_multi_tacs = ["Liver", "Aorta"] if all(organ in available_regions_tacs for organ in ["Liver", "Aorta"]) else available_regions_tacs[:2]
            organs_select_tacs = st.multiselect("Organs to compare", available_regions_tacs, default=default_organs_multi_tacs, help="Select organs to include in the comparison")

        # Normalization selector (conditional based on grouping)
        if color_group_tacs != "Normalization":
            norm_selector_tacs = st.selectbox("Normalization",options=list(norm_options.keys()),index=1,help="Select normalization method for the PET signal", key="tacs_norm")
            norm_select_tacs = norm_options[norm_selector_tacs]
            selected_norms_tacs = [norm_select_tacs]
            selected_norm_names_tacs = [norm_selector_tacs]
        else:
            # Allow multiple normalization selection when grouping by normalization
            default_norm_names_tacs = ["SUV", "SUV ID", "SUL (Decazes)", "SUL ID"]
            available_defaults_tacs = [norm for norm in default_norm_names_tacs if norm in norm_options.keys()]
            selected_norm_names_tacs = st.multiselect("Normalizations", options=list(norm_options.keys()), default=available_defaults_tacs, help="Select normalization methods to compare", key="tacs_norms")
            selected_norms_tacs = [norm_options[name] for name in selected_norm_names_tacs]
            norm_selector_tacs = "Multiple"

        # Erosion options (conditional based on grouping)
        erosion_options = ["None", "1 iteration", "2 iterations", "3 iterations"]
        if color_group_tacs != "Erosion":
            erosion_select_tacs = st.selectbox("Erosion", options=erosion_options, index=1, help="Number of erosion iterations applied to organ mask", key="tacs_erosion")
        else:
            erosion_select_tacs = None  # Will show all erosion levels

        # Confidence interval options
        uncertainty_options = ["None", "1 std", "95CI", "95PI"]
        uncertainty_tacs = st.selectbox("Confidence intervals:", uncertainty_options, index=1, help="Type of uncertainty bands to display", key="tacs_uncertainty")
        
        # Time axis toggle
        use_actual_time = st.toggle("Use actual time (seconds)", value=True, help="Toggle between frame numbers (tix) and actual time (t)")
        
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
                max_value=int(tacs_data['tix'].max()),
                value=(0, int(tacs_data['tix'].max())),
                help="Filter data to a specific frame range"
            )
            time_column = 'tix'
    
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
    color_group_tacs = "None"
    organ_select_tacs = "Liver"
    organs_select_tacs = ["Liver"]
    norm_selector_tacs = "SUL (Janma)"
    selected_norms_tacs = ["sul_janma"]
    selected_norm_names_tacs = ["SUL (Janma)"]
    erosion_select_tacs = "1 iteration"
    time_range = (0, 100)
    uncertainty_tacs = "1 std"
    use_actual_time = True
    time_column = 't'

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
    # --- Time Activity Curves Tab using TACS data ---
    
    # Use the tacs data
    tacs_df = tacs_data.copy()
    
    # Merge with demographic data from norm file
    demo_data = generate_norm_data().reset_index()
    tacs_df = tacs_df.merge(demo_data[['sub', 'sex', 'demographic-group']], on='sub', how='left')
    
    # Filter by time range using the selected time column
    tacs_df = tacs_df[(tacs_df[time_column] >= time_range[0]) & (tacs_df[time_column] <= time_range[1])]
    
    # Filter by selected organs (using region names directly)
    tacs_df = tacs_df[tacs_df.region.isin(organs_select_tacs)]
    
    # Apply erosion filter if erosion is not used for grouping
    if erosion_select_tacs is not None:
        # Map erosion option to actual value in data
        erosion_mapping = {
            "None": "0",
            "1 iteration": "1", 
            "2 iterations": "2",
            "3 iterations": "3"
        }
        erosion_value = erosion_mapping[erosion_select_tacs]
        tacs_df = tacs_df[tacs_df.erosion == erosion_value]
    
    # Apply normalization(s) - create separate rows for each normalization when grouping by normalization
    if color_group_tacs == "Normalization":
        # Create multiple rows for different normalizations
        normalized_dfs = []
        for norm_name, norm_col in zip(selected_norm_names_tacs, selected_norms_tacs):
            df_norm = tacs_df.copy()
            norm_df = generate_norm_data()[norm_col]
            df_norm["mult"] = df_norm["sub"].map(norm_df)
            df_norm["mu"] /= df_norm["mult"]
            df_norm["normalization"] = norm_name  # Add normalization as a grouping column
            normalized_dfs.append(df_norm)
        tacs_df = pd.concat(normalized_dfs, ignore_index=True)
    else:
        # Single normalization
        norm_df = generate_norm_data()[selected_norms_tacs[0]]
        tacs_df["mult"] = tacs_df["sub"].map(norm_df)
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
            return 'erosion'
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
            if uncertainty_tacs == "95CI":
                summary_data['uncertainty'] = 1.96 * summary_data['std'] / summary_data["count"].apply(np.sqrt)
            elif uncertainty_tacs == "95PI":
                summary_data['uncertainty'] = 1.96 * summary_data['std']
            elif uncertainty_tacs == "1 std":
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
        if uncertainty_tacs == "95CI":
            summary_data['uncertainty'] = 1.96 * summary_data['std'] / summary_data["count"].apply(np.sqrt)
        elif uncertainty_tacs == "95PI":
            summary_data['uncertainty'] = 1.96 * summary_data['std']
        elif uncertainty_tacs == "1 std":
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
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='x unified',
        height=600,
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
        margin=dict(l=50, r=20, t=50, b=50)
    )
    
    st.plotly_chart(fig_tacs, use_container_width=True)

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