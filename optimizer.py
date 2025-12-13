import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


class GurobiRouteSolver:
    """
    Solves the cash logistics routing problem using Gurobi optimizer.
    
    This class formulates and solves a Traveling Salesman Problem (TSP) variant
    for ATM cash replenishment. The optimization minimizes total travel distance
    while ensuring all low-stock ATMs are visited exactly once, starting and
    ending at the depot.
    
    Methodology:
        Uses Mixed Integer Linear Programming (MILP) with Miller-Tucker-Zemlin
        (MTZ) constraints for subtour elimination. The model uses binary decision
        variables to represent route segments and continuous variables for MTZ
        formulation.
    """
    
    def __init__(self, dataframe, distance_matrix):
        """
        Initialize the solver with ATM data and distance matrix.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing ATM information with
                columns: ID, lat, lon, current_cash, capacity, demand_forecast.
                Must include a DEPOT node.
            distance_matrix (np.ndarray): Square distance matrix between all nodes
                in miles. Element [i, j] represents the distance from node i to
                node j.
        
        Returns:
            None
        
        Raises:
            ValueError: If DEPOT node is missing or insufficient nodes exist.
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
        
        Methodology:
            Filters the original dataframe to include only the depot and ATMs
            with cash levels below the threshold. Creates a corresponding filtered
            distance matrix. Validates data integrity and handles missing values
            in demand forecasts.
        
        Args:
            None (uses self.original_df and self.original_distance_matrix)
        
        Returns:
            None (modifies self.df, self.distance_matrix, self.depot_idx, self.n)
        
        Raises:
            ValueError: If DEPOT is missing or insufficient nodes exist.
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
        Build the Gurobi optimization model with decision variables and constraints.
        
        Methodology:
            Constructs a Mixed Integer Linear Programming (MILP) model for the
            Traveling Salesman Problem. Uses binary variables for route decisions
            and continuous variables for Miller-Tucker-Zemlin (MTZ) subtour
            elimination constraints. The objective minimizes total travel distance.
        
        Args:
            None (uses self.df, self.distance_matrix, self.depot_idx, self.n)
        
        Returns:
            None (creates self.model, self.x, self.u)
        """
        # Initialize model
        self.model = gp.Model('Cash_Logistics')
        self.model.setParam('OutputFlag', 1)  # Show solver output
        
        # Decision Variables
        # x[i,j]: Binary variable, 1 if truck travels from node i to node j, 0 otherwise
        self.x = self.model.addVars(
            self.n, self.n,
            vtype=GRB.BINARY,
            name='x'
        )
        
        # u[i]: Continuous variable for MTZ subtour elimination
        # Represents the cumulative "load" or "position" in the route sequence
        self.u = self.model.addVars(
            self.n,
            vtype=GRB.CONTINUOUS,
            name='u'
        )
        
        # Objective Function: Minimize total distance traveled (in miles)
        # Sum of all edge distances multiplied by binary route variables
        self.model.setObjective(
            gp.quicksum(
                self.distance_matrix[i, j] * self.x[i, j]
                for i in range(self.n)
                for j in range(self.n)
            ),
            GRB.MINIMIZE
        )
        
        # ========================================================================
        # CONSTRAINTS
        # ========================================================================
        
        # Constraint 0: Prevent self-loops
        # Mathematical: x[i,i] = 0 for all i
        # Logic: Truck cannot travel from a node to itself
        for i in range(self.n):
            self.model.addConstr(self.x[i, i] == 0, name=f'no_self_loop_{i}')
        
        # Constraint 1: Flow conservation for ATMs (excluding depot)
        # Mathematical: Σ_i x[i,j] = 1 and Σ_i x[j,i] = 1 for all ATM nodes j
        # Logic: Each ATM must have exactly one incoming edge and one outgoing edge
        #        This ensures every ATM is visited exactly once
        for j in range(self.n):
            if j != self.depot_idx:  # Skip depot
                # Exactly one incoming edge (flow into node j)
                self.model.addConstr(
                    gp.quicksum(self.x[i, j] for i in range(self.n) if i != j) == 1,
                    name=f'incoming_{j}'
                )
                # Exactly one outgoing edge (flow out of node j)
                self.model.addConstr(
                    gp.quicksum(self.x[j, i] for i in range(self.n) if i != j) == 1,
                    name=f'outgoing_{j}'
                )
        
        # Constraint 2: Depot flow constraints
        # Mathematical: Σ_j x[depot,j] = 1 and Σ_i x[i,depot] = 1
        # Logic: Truck must leave depot exactly once and return to depot exactly once
        #        This ensures the route starts and ends at the depot
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
        
        # Constraint 3: Subtour Elimination using Miller-Tucker-Zemlin (MTZ) method
        # Mathematical: u[i] - u[j] + Q * x[i,j] <= Q - q[j] for all i,j (i≠j, j≠depot)
        #               where Q = vehicle capacity, q[j] = demand at node j
        # Logic: If edge (i,j) is used (x[i,j]=1), then u[j] >= u[i] + q[j]
        #        This creates an ordering that prevents disconnected subtours
        #        The u variables represent cumulative load/position in the route
        Q = 100000  # Vehicle capacity (can be adjusted)
        
        for i in range(self.n):
            for j in range(self.n):
                if i != j and j != self.depot_idx:  # Skip self-loops and depot as destination
                    q_j = self.df.iloc[j]['demand_forecast']
                    # MTZ constraint: ensures proper sequencing and prevents subtours
                    self.model.addConstr(
                        self.u[i] - self.u[j] + Q * self.x[i, j] <= Q - q_j,
                        name=f'mtz_{i}_{j}'
                    )
        
        # Constraint 4: MTZ variable bounds
        # Mathematical: u[depot] = 0
        # Logic: Depot is the starting point, so its position/load is zero
        self.model.addConstr(self.u[self.depot_idx] == 0, name='u_depot')
        
        # Mathematical: u[i] >= q[i] for all nodes (except depot)
        # Logic: Each node's position must be at least its demand value
        #        This ensures proper ordering and prevents negative loads
        for i in range(self.n):
            if i != self.depot_idx:
                q_i = self.df.iloc[i]['demand_forecast']
                self.model.addConstr(self.u[i] >= q_i, name=f'u_lower_{i}')
    
    def solve(self):
        """
        Solve the optimization model using Gurobi solver.
        
        Methodology:
            Calls Gurobi's optimizer to solve the MILP model. If optimal solution
            is found, extracts the route sequence. Otherwise, provides detailed
            error information for debugging infeasible or unbounded models.
        
        Args:
            None (uses self.model)
        
        Returns:
            list: Sequence of stops (route) as a list of ATM IDs, starting and
                ending with 'DEPOT'. Returns None if optimization fails.
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
        Extract the route sequence from the optimal solution.
        
        Methodology:
            Traverses the solution graph starting from the depot, following
            active edges (where x[i,j] = 1) until returning to the depot.
            Constructs the ordered list of node IDs visited.
        
        Args:
            None (uses self.model, self.x, self.df, self.depot_idx)
        
        Returns:
            list: List of ATM IDs in the order of the route, starting and ending
                with 'DEPOT'. Returns None if solution is not optimal.
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
        Get the solution route from the solved model.
        
        Args:
            None (uses self.solution_route)
        
        Returns:
            list: Sequence of stops (route) as a list of ATM IDs, or None if
                no solution exists.
        """
        return self.solution_route

