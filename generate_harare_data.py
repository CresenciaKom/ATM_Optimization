"""
Generate synthetic ATM data for Harare, Zimbabwe.
Creates 20 ATMs and 1 Depot with real coordinates.
"""

import pandas as pd
import numpy as np

# Harare, Zimbabwe coordinates (city center)
HARARE_CENTER_LAT = -17.8292
HARARE_CENTER_LON = 31.0522

def generate_harare_data():
    """Generate synthetic ATM data for Harare, Zimbabwe."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    n_atms = 20
    
    # Generate ATM locations around Harare city center
    # Spread ATMs across the city area (approximately 10-15 km radius)
    # Harare spans roughly from -17.95 to -17.75 latitude and 30.95 to 31.15 longitude
    
    # Generate realistic locations within Harare
    locations = []
    
    # Key areas in Harare (approximate coordinates)
    harare_areas = [
        (-17.8292, 31.0522, "City Center"),  # Harare CBD
        (-17.8250, 31.0330, "Avondale"),
        (-17.8150, 31.0650, "Mount Pleasant"),
        (-17.8400, 31.0450, "Mbare"),
        (-17.8100, 31.0400, "Belvedere"),
        (-17.8350, 31.0600, "Highfield"),
        (-17.8000, 31.0500, "Borrowdale"),
        (-17.8500, 31.0500, "Glen View"),
        (-17.8200, 31.0700, "Eastlea"),
        (-17.8300, 31.0300, "Westgate"),
    ]
    
    # Generate locations around these areas
    atm_locations = []
    location_names = []
    
    for i in range(n_atms):
        # Select a base area
        area_idx = i % len(harare_areas)
        base_lat, base_lon, area_name = harare_areas[area_idx]
        
        # Add small random offset (within ~2km)
        offset_lat = np.random.uniform(-0.015, 0.015)  # ~1.5km
        offset_lon = np.random.uniform(-0.015, 0.015)  # ~1.5km
        
        lat = base_lat + offset_lat
        lon = base_lon + offset_lon
        
        # Generate location name
        street_names = [
            "Samora Machel Avenue", "Robert Mugabe Road", "First Street",
            "Second Street", "Julius Nyerere Way", "Herbert Chitepo Avenue",
            "Enterprise Road", "Fife Avenue", "Nelson Mandela Avenue",
            "Jason Moyo Avenue", "Angwa Street", "Speke Avenue",
            "Harare Street", "Cameron Street", "Bank Street"
        ]
        
        street_name = street_names[i % len(street_names)]
        street_num = np.random.randint(1, 200)
        location_name = f"{street_num} {street_name}, {area_name}"
        
        atm_locations.append((lat, lon))
        location_names.append(location_name)
    
    # Calculate depot location (centroid of all ATMs)
    depot_lat = np.mean([loc[0] for loc in atm_locations])
    depot_lon = np.mean([loc[1] for loc in atm_locations])
    
    # Generate ATM IDs
    atm_ids = [f"ATM_{i+1:03d}" for i in range(n_atms)]
    
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
        print(f"     Adjusting demands to fit within capacity...")
        
        # Adjust demands to be more reasonable - scale them down
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
        'name': location_names,
        'lat': [loc[0] for loc in atm_locations],
        'lon': [loc[1] for loc in atm_locations],
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
        'name': ['Central Distribution Warehouse, Harare CBD'],
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
    # Generate Harare data
    df = generate_harare_data()
    
    # Save to CSV
    output_file = 'harare_atm_data.csv'
    try:
        df.to_csv(output_file, index=False)
    except PermissionError:
        # If file is locked, try with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'harare_atm_data_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        print(f"Note: Original file was locked, saved as {output_file}")
    
    print(f"\nSynthetic data saved to: {output_file}")
    print(f"\nPreview of generated data:")
    print(df.head(12).to_string(index=False))
    print(f"\n... (showing first 12 rows including DEPOT)")
    
    print(f"\nDEPOT Information:")
    depot_row = df[df['atm_id'] == 'DEPOT'].iloc[0]
    print(f"   Name: {depot_row['name']}")
    print(f"   Coordinates: ({depot_row['lat']:.6f}, {depot_row['lon']:.6f})")
    
    print(f"\nReady to paste into Google Sheets!")
    print(f"   The CSV file has {len(df)} rows (1 DEPOT + {len(df)-1} ATMs)")
    print(f"   Copy the contents of {output_file} into your Google Sheet")
    print(f"\nLocation: Harare, Zimbabwe")
    print(f"   City Center: ({HARARE_CENTER_LAT}, {HARARE_CENTER_LON})")

