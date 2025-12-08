import pandas as pd
import numpy as np
import random


class ScenarioGenerator:
    """
    Generates ATM locations and depot for optimization scenarios.
    """
    
    def __init__(self, center_lat=42.36, center_lon=-71.05, num_atms=15, seed=None):
        """
        Initialize the ScenarioGenerator.
        
        Parameters:
        -----------
        center_lat : float
            Latitude of the center point (default: 42.36)
        center_lon : float
            Longitude of the center point (default: -71.05)
        num_atms : int
            Number of ATMs to generate (default: 15)
        seed : int, optional
            Random seed for reproducibility
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.num_atms = num_atms
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.df = None
        self._generate_locations()
    
    def _generate_locations(self):
        """
        Generate ATM locations and depot, then create DataFrame.
        """
        locations = []
        
        # Generate ATMs around the center point
        for i in range(1, self.num_atms + 1):
            # Generate random offsets (in degrees, roughly 0-0.1 degrees ~ 0-11km)
            # Using a radius of approximately 0.05 degrees (~5.5km)
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0, 0.05)  # Roughly 0-5.5km from center
            
            lat = self.center_lat + radius * np.cos(angle)
            lon = self.center_lon + radius * np.sin(angle)
            
            # Generate random attributes
            current_cash = random.randint(5000, 50000)
            capacity = 100000
            demand_forecast = random.randint(2000, 5000)
            
            locations.append({
                'ID': f'ATM_{i:02d}',
                'lat': lat,
                'lon': lon,
                'current_cash': current_cash,
                'capacity': capacity,
                'demand_forecast': demand_forecast
            })
        
        # Add DEPOT at the center
        locations.append({
            'ID': 'DEPOT',
            'lat': self.center_lat,
            'lon': self.center_lon,
            'current_cash': 0,  # Depot typically doesn't have cash stored
            'capacity': 0,
            'demand_forecast': 0
        })
        
        # Create DataFrame
        self.df = pd.DataFrame(locations)
        
        # Reorder so DEPOT is first
        depot_idx = self.df[self.df['ID'] == 'DEPOT'].index[0]
        other_indices = self.df[self.df['ID'] != 'DEPOT'].index.tolist()
        self.df = self.df.loc[[depot_idx] + other_indices].reset_index(drop=True)
    
    def get_dataframe(self):
        """
        Return the locations as a Pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: ID, lat, lon, current_cash, capacity, demand_forecast
        """
        return self.df.copy()
    
    def get_distance_matrix(self):
        """
        Calculate distance matrix between all nodes (ATMs + DEPOT) using Haversine formula.
        
        Returns:
        --------
        np.ndarray
            Distance matrix where element [i, j] is the great-circle distance
            between node i and node j in miles
        """
        if self.df is None:
            raise ValueError("Locations not generated. Call _generate_locations() first.")
        
        n = len(self.df)
        distance_matrix = np.zeros((n, n))
        
        # Earth's radius in miles
        R = 3959.0
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    lat1, lon1 = self.df.iloc[i]['lat'], self.df.iloc[i]['lon']
                    lat2, lon2 = self.df.iloc[j]['lat'], self.df.iloc[j]['lon']
                    
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

