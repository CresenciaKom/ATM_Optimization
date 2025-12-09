import pandas as pd
import numpy as np


# Google Sheets configuration
SHEET_ID = '1SKtDtj3KX9WvE0n9iETiPS2IcNvKxSWiqhhcbqbp2xs'
SHEET_NAME = 'Sheet1'


def get_atm_data():
    """
    Read ATM data from Google Sheets and return a validated DataFrame.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ID, lat, lon, current_cash, capacity, demand_forecast
        Includes a DEPOT node at the centroid of all ATM locations
    """
    # Construct the URL for CSV export from Google Sheets
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    
    # Read data from Google Sheets
    df = pd.read_csv(url)
    
    # Convert column names to lowercase for consistent handling
    df.columns = df.columns.str.lower()
    
    # Expected columns (name/location_name is optional)
    required_columns = ['atm_id', 'lat', 'lon', 'current_cash', 'capacity', 'demand_forecast']
    
    # Validate that all required columns are present
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}. "
                        f"Found columns: {list(df.columns)}")
    
    # Select required columns and name/location_name if present
    columns_to_select = required_columns.copy()
    if 'name' in df.columns:
        columns_to_select.append('name')
    elif 'location_name' in df.columns:
        columns_to_select.append('location_name')
    
    df = df[columns_to_select].copy()
    
    # Force numeric types for numeric columns (handle dirty data)
    numeric_columns = ['lat', 'lon', 'current_cash', 'capacity', 'demand_forecast']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing critical values (NaNs from coercion)
    df = df.dropna(subset=['atm_id', 'lat', 'lon'])
    
    # Rename atm_id to ID for compatibility with rest of codebase
    df = df.rename(columns={'atm_id': 'ID'})
    
    # Check if DEPOT already exists in the data
    if 'DEPOT' in df['ID'].values:
        # DEPOT already exists, ensure it's at the beginning
        depot_mask = df['ID'] == 'DEPOT'
        depot_row = df[depot_mask].copy()
        atm_rows = df[~depot_mask].copy()
        df = pd.concat([depot_row, atm_rows], ignore_index=True)
    else:
        # Add DEPOT node at the centroid of all ATM locations
        # DEPOT represents the central warehouse/starting point
        atm_df = df  # All rows are ATMs if no DEPOT exists
        avg_lat = atm_df['lat'].mean()
        avg_lon = atm_df['lon'].mean()
        
        # Build depot row with same columns as df
        depot_data = {
            'ID': ['DEPOT'],
            'lat': [avg_lat],
            'lon': [avg_lon],
            'current_cash': [0],  # Depot has no cash (it's a warehouse)
            'capacity': [0],  # Not applicable for depot
            'demand_forecast': [0]  # Depot has no demand (it supplies cash)
        }
        
        # Add name/location_name if it exists in the dataframe
        if 'name' in df.columns:
            depot_data['name'] = ['Central Distribution Warehouse']
        elif 'location_name' in df.columns:
            depot_data['location_name'] = ['Central Distribution Warehouse']
        
        depot_row = pd.DataFrame(depot_data)
        
        # Prepend DEPOT to the beginning of the dataframe
        df = pd.concat([depot_row, df], ignore_index=True)
    
    return df


def get_distance_matrix(df):
    """
    Calculate distance matrix between all nodes using Haversine formula.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing node information with columns: lat, lon
    
    Returns:
    --------
    np.ndarray
        Distance matrix where element [i, j] is the great-circle distance
        between node i and node j in miles
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty or None.")
    
    if 'lat' not in df.columns or 'lon' not in df.columns:
        raise ValueError("DataFrame must contain 'lat' and 'lon' columns.")
    
    n = len(df)
    distance_matrix = np.zeros((n, n))
    
    # Earth's radius in miles
    R = 3959.0
    
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                lat1, lon1 = df.iloc[i]['lat'], df.iloc[i]['lon']
                lat2, lon2 = df.iloc[j]['lat'], df.iloc[j]['lon']
                
                # Convert latitude and longitude from degrees to radians
                phi1 = np.radians(lat1)
                phi2 = np.radians(lat2)
                delta_phi = np.radians(lat2 - lat1)
                delta_lambda = np.radians(lon2 - lon1)
                
                # Haversine formula
                a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                
                # Distance in miles
                distance_matrix[i, j] = R * c
    
    return distance_matrix
