import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


class GurobiRouteSolver:
    """
    Solves the cash logistics routing problem using Gurobi optimizer.
    """
    
    def __init__(self, dataframe, distance_matrix):
        """
        Initialize the solver with ATM data and distance matrix.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            DataFrame containing ATM information with columns:
            ID, lat, lon, current_cash, capacity, demand_forecast
        distance_matrix : np.ndarray
            Distance matrix between all nodes (in miles)
        """
        self.original_df = dataframe.copy()
        self.original_distance_matrix = distance_matrix.copy()
        
        # Preprocessing: Filter to DEPOT + ATMs with current_cash < 20,000
        self._preprocess()
        
        self.model = None
        self.x = None  # Binary variables x[i,j]
        self.u = None  # Continuous variables u[i]
        self.solution_route = None
        self.error_message = None  # Store error messages
    
    def _preprocess(self):
        """
        Filter data to include only DEPOT + ATMs with current_cash < 20,000.
        """
        # Keep DEPOT and ATMs with low stock
        mask = (self.original_df['ID'] == 'DEPOT') | (self.original_df['current_cash'] < 20000)
        self.df = self.original_df[mask].copy().reset_index(drop=True)
        
        # Validate: Must have DEPOT
        if 'DEPOT' not in self.df['ID'].values:
            raise ValueError("DEPOT node not found in dataframe")
        
        # Validate: Need at least one ATM to visit (DEPOT + at least 1 ATM)
        if len(self.df) < 2:
            raise ValueError("Need at least one ATM to visit (only DEPOT found)")
        
        # Validate: Check for NaN values in demand_forecast
        if self.df['demand_forecast'].isna().any():
            # Fill NaN values with 0 for ATMs (depot should already be 0)
            self.df['demand_forecast'] = self.df['demand_forecast'].fillna(0)
        
        # Ensure demand_forecast is numeric and non-negative
        self.df['demand_forecast'] = pd.to_numeric(self.df['demand_forecast'], errors='coerce').fillna(0)
        # Ensure non-negative (negative demand doesn't make sense)
        self.df['demand_forecast'] = self.df['demand_forecast'].clip(lower=0)
        
        # Get indices of filtered nodes in original dataframe
        original_indices = self.original_df[mask].index.tolist()
        
        # Create filtered distance matrix
        n_filtered = len(self.df)
        self.distance_matrix = np.zeros((n_filtered, n_filtered))
        
        for i in range(n_filtered):
            for j in range(n_filtered):
                orig_i = original_indices[i]
                orig_j = original_indices[j]
                self.distance_matrix[i, j] = self.original_distance_matrix[orig_i, orig_j]
        
        # Find depot index in filtered data
        self.depot_idx = self.df[self.df['ID'] == 'DEPOT'].index[0]
        self.n = len(self.df)
    
    def build_model(self):
        """
        Build the Gurobi optimization model.
        """
        # Initialize model
        self.model = gp.Model('Cash_Logistics')
        self.model.setParam('OutputFlag', 1)  # Show solver output
        
        # Variables
        # x[i,j]: Binary variable, 1 if truck goes from node i to j
        self.x = self.model.addVars(
            self.n, self.n,
            vtype=GRB.BINARY,
            name='x'
        )
        
        # u[i]: Continuous variable for MTZ subtour elimination
        self.u = self.model.addVars(
            self.n,
            vtype=GRB.CONTINUOUS,
            name='u'
        )
        
        # Objective: Minimize total distance (in miles)
        self.model.setObjective(
            gp.quicksum(
                self.distance_matrix[i, j] * self.x[i, j]
                for i in range(self.n)
                for j in range(self.n)
            ),
            GRB.MINIMIZE
        )
        
        # Constraints
        
        # 0. Prevent self-loops: x[i,i] = 0 for all i
        for i in range(self.n):
            self.model.addConstr(self.x[i, i] == 0, name=f'no_self_loop_{i}')
        
        # 1. Assignment: Each chosen ATM must be visited exactly once
        # (excluding depot from the "exactly once" constraint)
        for j in range(self.n):
            if j != self.depot_idx:  # Skip depot
                # Exactly one incoming edge
                self.model.addConstr(
                    gp.quicksum(self.x[i, j] for i in range(self.n) if i != j) == 1,
                    name=f'incoming_{j}'
                )
                # Exactly one outgoing edge
                self.model.addConstr(
                    gp.quicksum(self.x[j, i] for i in range(self.n) if i != j) == 1,
                    name=f'outgoing_{j}'
                )
        
        # 2. Depot: Truck must leave the depot and return to the depot
        # Exactly one edge leaving depot
        self.model.addConstr(
            gp.quicksum(self.x[self.depot_idx, j] for j in range(self.n) if j != self.depot_idx) == 1,
            name='depot_out'
        )
        # Exactly one edge entering depot
        self.model.addConstr(
            gp.quicksum(self.x[i, self.depot_idx] for i in range(self.n) if i != self.depot_idx) == 1,
            name='depot_in'
        )
        
        # 3. Subtour Elimination (MTZ): u[i] - u[j] + Q * x[i,j] <= Q - q[j]
        # Q is capacity (using a reasonable vehicle capacity)
        # q[j] is demand_forecast at node j
        Q = 100000  # Vehicle capacity (can be adjusted)
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j and j != self.depot_idx:  # Skip self-loops and depot as destination
                    q_j = self.df.iloc[j]['demand_forecast']
                    self.model.addConstr(
                        self.u[i] - self.u[j] + Q * self.x[i, j] <= Q - q_j,
                        name=f'mtz_{i}_{j}'
                    )
        
        # Set bounds for u variables (for MTZ)
        # u[depot] = 0
        self.model.addConstr(self.u[self.depot_idx] == 0, name='u_depot')
        
        # u[i] >= q[i] for all nodes (except depot)
        for i in range(self.n):
            if i != self.depot_idx:
                q_i = self.df.iloc[i]['demand_forecast']
                self.model.addConstr(self.u[i] >= q_i, name=f'u_lower_{i}')
    
    def solve(self):
        """
        Solve the optimization model.
        
        Returns:
        --------
        list
            Sequence of stops (route) as a list of ATM IDs
        """
        if self.model is None:
            self.build_model()
        
        # Optimize
        self.model.optimize()
        
        # Extract solution
        if self.model.status == GRB.OPTIMAL:
            self.solution_route = self._extract_route()
            return self.solution_route
        else:
            # Provide detailed error information
            status_map = {
                GRB.LOADED: "Model is loaded but not yet optimized",
                GRB.OPTIMAL: "Optimal solution found",
                GRB.INFEASIBLE: "Model is infeasible - constraints conflict with each other",
                GRB.INF_OR_UNBD: "Model is infeasible or unbounded",
                GRB.UNBOUNDED: "Model is unbounded",
                GRB.CUTOFF: "Optimal objective was worse than cutoff",
                GRB.ITERATION_LIMIT: "Optimization stopped at iteration limit",
                GRB.NODE_LIMIT: "Optimization stopped at node limit",
                GRB.TIME_LIMIT: "Optimization stopped at time limit",
                GRB.SOLUTION_LIMIT: "Optimization stopped at solution limit",
                GRB.INTERRUPTED: "Optimization was interrupted",
                GRB.NUMERIC: "Numerical issues encountered",
                GRB.SUBOPTIMAL: "Suboptimal solution found",
                GRB.INPROGRESS: "Optimization is in progress"
            }
            error_msg = status_map.get(self.model.status, f"Unknown status: {self.model.status}")
            self.error_message = f"Optimization failed: {error_msg}"
            if self.model.status == GRB.INFEASIBLE:
                self.error_message += "\n\nPossible causes:\n"
                self.error_message += "  - No valid route exists with current constraints\n"
                self.error_message += "  - Data inconsistencies (check for NaN or invalid values)\n"
                self.error_message += "  - Only DEPOT node present (need at least one ATM to visit)\n"
                self.error_message += f"  - Current nodes: {len(self.df)} (DEPOT + {len(self.df) - 1} ATMs)"
            print(self.error_message)
            return None
    
    def _extract_route(self):
        """
        Extract the route sequence from the solution.
        
        Returns:
        --------
        list
            List of ATM IDs in the order of the route
        """
        if self.model.status != GRB.OPTIMAL:
            return None
        
        # Find the route starting from depot
        route = []
        current = self.depot_idx
        route.append(self.df.iloc[current]['ID'])
        
        # Follow the path until we return to depot
        while True:
            found_next = False
            for j in range(self.n):
                if j != current and self.x[current, j].X > 0.5:  # Edge is used
                    current = j
                    route.append(self.df.iloc[current]['ID'])
                    found_next = True
                    break
            
            # If we're back at depot, we've completed the route
            if current == self.depot_idx and len(route) > 1:
                break
            
            if not found_next:
                break
        
        return route
    
    def get_route(self):
        """
        Get the solution route.
        
        Returns:
        --------
        list
            Sequence of stops (route) as a list of ATM IDs
        """
        return self.solution_route

