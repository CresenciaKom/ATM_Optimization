"""
Script to generate synthetic ATM data that works with the optimizer.
Maintains the same locations (lat/lon) and number of ATMs from Google Sheets.
Generates valid values that satisfy optimization constraints.
"""

import pandas as pd
import numpy as np

# Google Sheets configuration (same as data_manager.py)
SHEET_ID = '1SKtDtj3KX9WvE0n9iETiPS2IcNvKxSWiqhhcbqbp2xs'
SHEET_NAME = 'Sheet1'

def generate_synthetic_data():
    """Generate synthetic data maintaining locations and number of ATMs."""
    
    # Fetch current data to get locations and count
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    
    try:
        current_df = pd.read_csv(url)
        current_df.columns = current_df.columns.str.lower()
        
        # Check if we have the required columns
        if 'lat' not in current_df.columns or 'lon' not in current_df.columns:
            raise ValueError("Current data must have 'lat' and 'lon' columns")
        
        # Get locations
        locations = current_df[['lat', 'lon']].copy()
        n_atms = len(locations)
        
        print(f"Found {n_atms} ATMs in current data")
        print(f"Location range: Lat {locations['lat'].min():.4f} to {locations['lat'].max():.4f}")
        print(f"Location range: Lon {locations['lon'].min():.4f} to {locations['lon'].max():.4f}")
        
    except Exception as e:
        print(f"Error fetching current data: {e}")
        print("Generating sample data with 10 ATMs in a reasonable area...")
        # Fallback: Generate sample locations around a central point
        np.random.seed(42)
        center_lat, center_lon = 40.7128, -74.0060  # NYC area
        n_atms = 10
        locations = pd.DataFrame({
            'lat': center_lat + np.random.uniform(-0.1, 0.1, n_atms),
            'lon': center_lon + np.random.uniform(-0.1, 0.1, n_atms)
        })
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate ATM IDs
    atm_ids = [f"ATM_{i+1:03d}" for i in range(n_atms)]
    
    # Generate location names based on coordinates
    # Create realistic location names
    location_names = []
    street_names = [
        "Main Street", "Oak Avenue", "Elm Street", "Park Boulevard", "First Street",
        "Second Street", "Market Street", "Broadway", "High Street", "Church Road",
        "Washington Avenue", "Lincoln Drive", "Riverside Drive", "Hilltop Road",
        "Cedar Lane", "Maple Avenue", "Pine Street", "Central Plaza", "Commerce Way",
        "Business Park", "Shopping Center"
    ]
    
    for i in range(n_atms):
        # Use a deterministic approach based on coordinates to get consistent names
        name_index = hash((locations.iloc[i]['lat'], locations.iloc[i]['lon'])) % len(street_names)
        street_name = street_names[name_index]
        # Add a street number
        street_num = (int(abs(locations.iloc[i]['lat']) * 100) % 900) + 100
        location_names.append(f"{street_num} {street_name}")
    
    # Calculate depot location (centroid)
    depot_lat = locations['lat'].mean()
    depot_lon = locations['lon'].mean()
    
    # Generate synthetic data with constraints that ensure feasibility
    
    # 1. Capacity: Random between 50000 and 100000 (typical ATM capacity)
    capacity = np.random.uniform(50000, 100000, n_atms).astype(int)
    
    # 2. Current cash: Mix of values to ensure some below threshold
    #     - About 40% below 20000 (low stock)
    #     - About 60% above 20000 (well-stocked)
    safety_threshold = 20000
    low_stock_pct = 0.4
    n_low_stock = int(n_atms * low_stock_pct)
    
    # Generate current_cash for each ATM individually to ensure it's within capacity
    current_cash = np.zeros(n_atms)
    
    # Determine which ATMs will be low-stock
    low_stock_indices = np.random.choice(n_atms, n_low_stock, replace=False)
    low_stock_mask_array = np.zeros(n_atms, dtype=bool)
    low_stock_mask_array[low_stock_indices] = True
    
    for i in range(n_atms):
        if low_stock_mask_array[i]:
            # Low stock: current_cash between 5000 and 19000, but not more than capacity
            max_cash = min(19000, capacity[i] * 0.9)
            min_cash = min(5000, capacity[i] * 0.1)
            current_cash[i] = np.random.uniform(min_cash, max_cash)
        else:
            # Well-stocked: current_cash between 25000 and capacity * 0.8
            min_cash = min(25000, capacity[i] * 0.4)
            max_cash = capacity[i] * 0.8
            current_cash[i] = np.random.uniform(min_cash, max_cash)
    
    current_cash = current_cash.astype(int)
    
    # 3. Demand forecast: 
    #     - For low-stock ATMs: demand = what they need to fill up (but reasonable)
    #     - For well-stocked: small positive values (maintenance)
    #     - Ensure all values are positive and reasonable
    #     - Total demand from low-stock ATMs should not exceed truck capacity (100000) by too much
    truck_capacity = 100000
    
    demand_forecast = np.zeros(n_atms)
    low_stock_mask = current_cash < safety_threshold
    
    # For low-stock ATMs: demand is what they need to fill up
    # But cap it reasonably to avoid infeasibility
    for i in range(n_atms):
        if low_stock_mask[i]:
            # Demand is the shortfall, but capped at a reasonable amount
            shortfall = capacity[i] - current_cash[i]
            # Cap demand at 50000 per ATM to avoid extreme values
            demand_forecast[i] = min(shortfall, 50000)
            # Ensure at least 5000 demand for low-stock ATMs
            demand_forecast[i] = max(demand_forecast[i], 5000)
        else:
            # Well-stocked ATMs: small maintenance demand
            demand_forecast[i] = np.random.uniform(1000, 5000)
    
    demand_forecast = demand_forecast.astype(int)
    
    # Check total demand from low-stock ATMs
    total_low_stock_demand = demand_forecast[low_stock_mask].sum()
    print(f"\nGenerated data summary:")
    print(f"  Total ATMs: {n_atms}")
    print(f"  Low-stock ATMs (< $20,000): {low_stock_mask.sum()}")
    print(f"  Total demand from low-stock ATMs: ${total_low_stock_demand:,}")
    print(f"  Truck capacity: ${truck_capacity:,}")
    
    if total_low_stock_demand > truck_capacity:
        print(f"  Warning: Total demand (${total_low_stock_demand:,}) exceeds truck capacity.")
        print(f"     The optimizer will select a subset of ATMs to visit.")
        
        # Adjust demands to be more reasonable - scale them down
        # This ensures the optimizer has a better chance of finding feasible solutions
        scale_factor = (truck_capacity * 0.9) / total_low_stock_demand  # Use 90% of capacity as target
        for i in range(n_atms):
            if low_stock_mask[i]:
                demand_forecast[i] = int(demand_forecast[i] * scale_factor)
                # Ensure minimum demand of 5000
                demand_forecast[i] = max(demand_forecast[i], 5000)
        
        total_low_stock_demand = demand_forecast[low_stock_mask].sum()
        print(f"     Adjusted total demand: ${total_low_stock_demand:,}")
    
    # Create dataframe for ATMs
    synthetic_df = pd.DataFrame({
        'atm_id': atm_ids,
        'location_name': location_names,
        'lat': locations['lat'].values,
        'lon': locations['lon'].values,
        'current_cash': current_cash,
        'capacity': capacity,
        'demand_forecast': demand_forecast
    })
    
    # Round lat/lon to reasonable precision (6 decimal places ~ 0.1m precision)
    synthetic_df['lat'] = synthetic_df['lat'].round(6)
    synthetic_df['lon'] = synthetic_df['lon'].round(6)
    
    # Create DEPOT row
    depot_row = pd.DataFrame({
        'atm_id': ['DEPOT'],
        'location_name': ['Central Distribution Warehouse'],
        'lat': [round(depot_lat, 6)],
        'lon': [round(depot_lon, 6)],
        'current_cash': [0],
        'capacity': [0],
        'demand_forecast': [0]
    })
    
    # Prepend DEPOT to the dataframe
    synthetic_df = pd.concat([depot_row, synthetic_df], ignore_index=True)
    
    # Verify no NaN values
    assert not synthetic_df.isna().any().any(), "Generated data contains NaN values!"
    
    # Verify all numeric columns are valid (excluding DEPOT)
    atm_df = synthetic_df[synthetic_df['atm_id'] != 'DEPOT']
    assert (synthetic_df['current_cash'] >= 0).all(), "Negative current_cash found!"
    assert (atm_df['capacity'] > 0).all(), "Non-positive capacity found in ATMs!"
    assert (synthetic_df['demand_forecast'] >= 0).all(), "Negative demand_forecast found!"
    # Check that ATM current_cash doesn't exceed capacity (DEPOT excluded)
    assert (atm_df['current_cash'] <= atm_df['capacity']).all(), "ATM current_cash exceeds capacity!"
    
    print(f"\nValidation passed: All constraints satisfied")
    
    return synthetic_df

if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_data()
    
    # Save to CSV
    output_file = 'synthetic_atm_data.csv'
    try:
        df.to_csv(output_file, index=False)
    except PermissionError:
        # If file is locked, try with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'synthetic_atm_data_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        print(f"Note: Original file was locked, saved as {output_file}")
    
    print(f"\nSynthetic data saved to: {output_file}")
    print(f"\nPreview of generated data:")
    print(df.head(12).to_string(index=False))
    print(f"\n... (showing first 12 rows including DEPOT)")
    
    print(f"\nDEPOT Information:")
    depot_row = df[df['atm_id'] == 'DEPOT'].iloc[0]
    print(f"   Location: {depot_row['location_name']}")
    print(f"   Coordinates: ({depot_row['lat']:.6f}, {depot_row['lon']:.6f})")
    
    print(f"\nReady to paste into Google Sheets!")
    print(f"   The CSV file has {len(df)} rows (1 DEPOT + {len(df)-1} ATMs)")
    print(f"   Copy the contents of {output_file} into your Google Sheet")

