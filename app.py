import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from data_manager import get_atm_data, get_distance_matrix
from optimizer import GurobiRouteSolver


# Page configuration
st.set_page_config(page_title="ATM Cash Logistics Optimizer", layout="wide")

# Initialize session state
if 'scenario_data' not in st.session_state:
    st.session_state.scenario_data = None
if 'distance_matrix' not in st.session_state:
    st.session_state.distance_matrix = None
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None


def generate_scenario():
    """
    Generate or retrieve scenario data from session state or Google Sheets.
    
    Methodology:
        Checks session state for cached data. If not present, fetches ATM data
        from Google Sheets and calculates distance matrix. Caches results in
        session state to avoid redundant API calls.
    
    Args:
        None (uses st.session_state)
    
    Returns:
        tuple: (pd.DataFrame, np.ndarray) containing:
            - DataFrame with ATM data including DEPOT
            - Distance matrix between all nodes in miles
    """
    if st.session_state.scenario_data is None:
        df = get_atm_data()
        st.session_state.scenario_data = df
        st.session_state.distance_matrix = get_distance_matrix(df)
    return st.session_state.scenario_data, st.session_state.distance_matrix


def calculate_route_distance(route_ids, df, distance_matrix):
    """
    Calculate total distance for a route given node IDs.
    
    Methodology:
        Iterates through consecutive pairs of nodes in the route and sums
        the corresponding distances from the distance matrix.
    
    Args:
        route_ids (list): List of node IDs in route order (e.g., ['DEPOT', 'ATM1', 'ATM2', 'DEPOT']).
        df (pd.DataFrame): DataFrame containing node information with 'ID' column.
        distance_matrix (np.ndarray): Square distance matrix where [i,j] is distance
            from node at index i to node at index j in miles.
    
    Returns:
        float: Total distance traveled along the route in miles. Returns 0 if
            route has fewer than 2 nodes.
    """
    if len(route_ids) < 2:
        return 0
    
    total_distance = 0
    for i in range(len(route_ids) - 1):
        node1_idx = df[df['ID'] == route_ids[i]].index[0]
        node2_idx = df[df['ID'] == route_ids[i + 1]].index[0]
        total_distance += distance_matrix[node1_idx, node2_idx]
    
    return total_distance


def generate_random_route(df, distance_matrix, depot_id='DEPOT'):
    """
    Generate a random route visiting all low-stock ATMs.
    
    Methodology:
        Creates a baseline comparison route by randomly shuffling ATM nodes
        (excluding depot) and inserting depot at the start and end. This
        provides a naive routing strategy for comparison with optimized routes.
    
    Args:
        df (pd.DataFrame): DataFrame containing node information with 'ID' column.
        distance_matrix (np.ndarray): Distance matrix (not used in this function
            but kept for API consistency).
        depot_id (str): ID of the depot node. Defaults to 'DEPOT'.
    
    Returns:
        list: Route as a list of node IDs, starting and ending with depot_id.
            If no ATMs exist, returns [depot_id].
    """
    # Get all nodes except depot
    nodes = df[df['ID'] != depot_id]['ID'].tolist()
    
    if len(nodes) == 0:
        return [depot_id]
    
    # Randomly shuffle
    np.random.shuffle(nodes)
    
    # Start and end at depot
    route = [depot_id] + nodes + [depot_id]
    return route


def create_map(df, route=None, threshold=20000):
    """
    Create a folium map with ATMs and route visualization.
    
    Methodology:
        Creates an interactive Folium map centered on the depot. Adds markers
        for each ATM (color-coded by cash level) and optionally draws a polyline
        representing the optimized route.
    
    Args:
        df (pd.DataFrame): DataFrame containing ATM information with columns:
            ID, lat, lon, current_cash, demand_forecast. Must include a DEPOT node.
        route (list, optional): List of node IDs in route order. If provided,
            draws a polyline connecting the nodes. Defaults to None.
        threshold (float): Cash threshold in dollars. ATMs below this threshold
            are marked as low-stock (red). Defaults to 20000.
    
    Returns:
        folium.Map: Interactive Folium map object with markers and optional route.
    """
    # Center map on depot
    depot = df[df['ID'] == 'DEPOT'].iloc[0]
    center_lat, center_lon = depot['lat'], depot['lon']
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add markers
    for idx, row in df.iterrows():
        if row['ID'] == 'DEPOT':
            # Depot marker (green)
            folium.Marker(
                [row['lat'], row['lon']],
                popup=f"DEPOT",
                icon=folium.Icon(color='green', icon='warehouse', prefix='fa')
            ).add_to(m)
        elif row['current_cash'] < threshold:
            # Low cash ATM (red)
            folium.Marker(
                [row['lat'], row['lon']],
                popup=f"{row['ID']}<br>Cash: ${row['current_cash']:,}<br>Demand: ${row['demand_forecast']:,}",
                icon=folium.Icon(color='red', icon='exclamation-circle', prefix='fa')
            ).add_to(m)
        else:
            # OK ATM (blue)
            folium.Marker(
                [row['lat'], row['lon']],
                popup=f"{row['ID']}<br>Cash: ${row['current_cash']:,}",
                icon=folium.Icon(color='blue', icon='check-circle', prefix='fa')
            ).add_to(m)
    
    # Draw route if provided
    if route and len(route) > 1:
        route_coords = []
        for node_id in route:
            node = df[df['ID'] == node_id].iloc[0]
            route_coords.append([node['lat'], node['lon']])
        
        # Draw polyline
        folium.PolyLine(
            route_coords,
            color='black',
            weight=4,
            opacity=0.8,
            popup='Optimized Route'
        ).add_to(m)
    
    return m


# Sidebar
st.sidebar.header("⚙️ Optimization Controls")

safety_threshold = st.sidebar.number_input(
    "Safety Threshold ($)",
    min_value=0,
    max_value=100000,
    value=20000,
    step=1000,
    help="ATMs with cash below this threshold will be included in the route"
)

truck_cost = st.sidebar.number_input(
    "Truck Cost ($/mile)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Cost per mile for truck operations"
)

optimize_button = st.sidebar.button("🚚 Optimize Logistics", type="primary", use_container_width=True)

# Main content
st.title("🏦 ATM Cash Logistics Optimizer")
st.markdown("---")

# Generate scenario data
df, distance_matrix = generate_scenario()

# Display data summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total ATMs", len(df[df['ID'] != 'DEPOT']))

with col2:
    low_stock_count = len(df[(df['ID'] != 'DEPOT') & (df['current_cash'] < safety_threshold)])
    st.metric("Low Stock ATMs", low_stock_count)

with col3:
    total_demand = df[df['current_cash'] < safety_threshold]['demand_forecast'].sum()
    st.metric("Total Demand", f"${total_demand:,}")

with col4:
    avg_cash = df[df['ID'] != 'DEPOT']['current_cash'].mean()
    st.metric("Avg Cash Level", f"${avg_cash:,.0f}")

st.markdown("---")

# Optimization process
if optimize_button or st.session_state.optimization_result is not None:
    with st.spinner("Optimizing route..."):
        # Filter data based on threshold
        filtered_mask = (df['ID'] == 'DEPOT') | (df['current_cash'] < safety_threshold)
        filtered_df = df[filtered_mask].copy().reset_index(drop=True)
        
        # Get filtered distance matrix
        original_indices = df[filtered_mask].index.tolist()
        n_filtered = len(filtered_df)
        filtered_distance_matrix = np.zeros((n_filtered, n_filtered))
        
        for i in range(n_filtered):
            for j in range(n_filtered):
                orig_i = original_indices[i]
                orig_j = original_indices[j]
                filtered_distance_matrix[i, j] = distance_matrix[orig_i, orig_j]
        
        # Validate data before optimization
        solver = None
        optimized_route = None
        
        if len(filtered_df) < 2:
            st.warning("⚠️ No ATMs to visit. All ATMs are above the safety threshold.")
            st.session_state.optimization_result = None
        else:
            try:
                # Solve optimization
                solver = GurobiRouteSolver(filtered_df, filtered_distance_matrix)
                optimized_route = solver.solve()
            except Exception as e:
                st.error(f"❌ Optimization error: {str(e)}")
                st.session_state.optimization_result = None
        
        if optimized_route:
            # Calculate optimized route distance
            opt_distance = calculate_route_distance(optimized_route, filtered_df, filtered_distance_matrix)
            
            # Calculate random route distance
            random_route = generate_random_route(filtered_df, filtered_distance_matrix)
            random_distance = calculate_route_distance(random_route, filtered_df, filtered_distance_matrix)
            
            # Calculate savings
            miles_saved = random_distance - opt_distance
            cost_saved = miles_saved * truck_cost
            
            # Store results
            st.session_state.optimization_result = {
                'route': optimized_route,
                'distance': opt_distance,
                'random_distance': random_distance,
                'miles_saved': miles_saved,
                'cost_saved': cost_saved
            }
        else:
            # Display detailed error message if available
            if solver is not None and hasattr(solver, 'error_message') and solver.error_message:
                # Format error message for Streamlit (convert \n to proper display)
                error_lines = solver.error_message.split('\n')
                st.error(f"❌ {error_lines[0]}")
                if len(error_lines) > 1:
                    for line in error_lines[1:]:
                        if line.strip():
                            st.write(f"   {line}")
            else:
                st.error("❌ Optimization failed. Please check the model constraints.")
            
            # Show diagnostic information
            with st.expander("🔍 Diagnostic Information"):
                st.write(f"**Nodes in optimization:** {len(filtered_df)}")
                st.write(f"- DEPOT: 1")
                st.write(f"- ATMs to visit: {len(filtered_df) - 1}")
                if len(filtered_df) > 1:
                    st.write(f"\n**Sample data:**")
                    st.dataframe(filtered_df[['ID', 'current_cash', 'demand_forecast']].head())
                    st.write(f"\n**Demand forecast stats:**")
                    st.write(f"- Min: ${filtered_df['demand_forecast'].min():,.2f}")
                    st.write(f"- Max: ${filtered_df['demand_forecast'].max():,.2f}")
                    st.write(f"- Mean: ${filtered_df['demand_forecast'].mean():,.2f}")
                    st.write(f"- Has NaN: {filtered_df['demand_forecast'].isna().any()}")
            st.session_state.optimization_result = None

# Display results
if st.session_state.optimization_result:
    result = st.session_state.optimization_result
    
    # Calculate capacity metrics
    truck_capacity = 100000  # Vehicle capacity from optimizer
    # Get all low-stock ATMs (red ATMs)
    low_stock_atms = df[(df['ID'] != 'DEPOT') & (df['current_cash'] < safety_threshold)]
    total_cash_required = low_stock_atms['demand_forecast'].sum()
    capacity_usage_pct = (total_cash_required / truck_capacity) * 100 if truck_capacity > 0 else 0
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Optimized Distance",
            f"{result['distance']:.2f} miles",
            delta=f"-{result['miles_saved']:.2f} miles vs random"
        )
    
    with col2:
        st.metric(
            "Miles Saved",
            f"{result['miles_saved']:.2f} miles",
            help="Compared to a random route"
        )
    
    with col3:
        st.metric(
            "Cost Savings",
            f"${result['cost_saved']:.2f}",
            help=f"Based on ${truck_cost}/mile"
        )
    
    with col4:
        # Capacity Usage metric with warning color
        if total_cash_required > truck_capacity:
            st.metric(
                "Capacity Usage",
                f"{capacity_usage_pct:.1f}%",
                delta=f"${total_cash_required:,} / ${truck_capacity:,}",
                delta_color="inverse"
            )
        else:
            st.metric(
                "Capacity Usage",
                f"{capacity_usage_pct:.1f}%",
                delta=f"${total_cash_required:,} / ${truck_capacity:,}"
            )
    
    # Warning message if capacity exceeded
    if total_cash_required > truck_capacity:
        st.warning(
            "⚠️ **Truck Capacity Exceeded**: Optimizer selected the best subset of ATMs to visit. "
            f"Total cash required (${total_cash_required:,}) exceeds truck capacity (${truck_capacity:,})."
        )
    
    st.markdown("---")
    
    # Map
    st.subheader("📍 Route Visualization")
    
    # Filter full dataframe for map display
    filtered_mask = (df['ID'] == 'DEPOT') | (df['current_cash'] < safety_threshold)
    filtered_df_map = df[filtered_mask].copy().reset_index(drop=True)
    
    map_obj = create_map(df, route=result['route'], threshold=safety_threshold)
    st_folium(map_obj, width=1200, height=600)
    
    # Route details
    st.markdown("---")
    st.subheader("🗺️ Route Sequence")
    route_text = " → ".join(result['route'])
    st.code(route_text, language=None)
    
    # Prescription
    st.markdown("---")
    st.subheader("💡 Prescription")
    st.info(
        "**Gurobi selected this route to minimize distance while ensuring all stockouts are covered.** "
        f"The optimized route visits {len(result['route']) - 2} low-stock ATMs in the most efficient order, "
        f"saving {result['miles_saved']:.2f} miles compared to a random route. "
        f"This translates to approximately ${result['cost_saved']:.2f} in operational cost savings."
    )
else:
    # Initial map without route
    st.subheader("📍 ATM Locations")
    map_obj = create_map(df, route=None, threshold=safety_threshold)
    st_folium(map_obj, width=1200, height=600)
    
    st.info("👆 Click 'Optimize Logistics' in the sidebar to generate an optimized route.")

# Mathematical formulation expander
st.markdown("---")
with st.expander("📐 How the Gurobi Optimizer Works"):
    st.markdown("### Decision Variables")
    st.latex(r"x_{ij} \in \{0, 1\} \quad \text{(Route segments)}")
    st.markdown("where $x_{ij} = 1$ if the truck travels from node $i$ to node $j$, and $x_{ij} = 0$ otherwise.")
    
    st.markdown("### Objective Function")
    st.latex(r"\min \sum_{i,j} d_{ij} \cdot x_{ij} \quad \text{(Minimize Cost)}")
    st.markdown("where $d_{ij}$ is the distance (in miles) between node $i$ and node $j$.")
    
    st.markdown("### Constraints")
    st.latex(r"\sum_j x_{ij} = 1 \quad \forall i \quad \text{(Flow conservation - enter/exit exactly once)}")
    st.markdown("This ensures each node is visited exactly once (one incoming and one outgoing edge).")
    
    st.markdown("### Subtour Elimination")
    st.markdown("**Miller-Tucker-Zemlin constraints are applied to prevent subtours (disconnected loops).**")
    st.markdown("These constraints ensure the solution forms a single connected route starting and ending at the depot.")

