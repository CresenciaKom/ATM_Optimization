# CashRoute: Intelligent ATM Logistics Optimizer

> A prescriptive analytics engine that optimizes armored truck logistics by balancing cash inventory risks against transportation costs using Gurobi optimization.

## 🎯 The Problem

Banks and Independent ATM Deployers (IADs) face a costly dilemma known as the "Cash Management Trade-off."
1.  **Too much cash in ATMs** leads to "dead capital" (high opportunity cost) and increased insurance liability.
2.  **Too little cash** causes stockouts, damaging customer trust.
3.  **Frequent refills** incur massive logistics costs (armored trucks, guards, fuel), often exceeding $500 per stop.

Most operators currently rely on static schedules (for example, "Visit every Tuesday") rather than actual demand. This results in inefficient routes where trucks visit full ATMs or miss empty ones, leading to operational waste and lost revenue.

## 💡 The Solution

**CashRoute** moves beyond simple reporting (Descriptive) and forecasting (Predictive) to **Prescriptive Analytics**. It tells operations managers *exactly* what action to take.

I used **Mixed-Integer Linear Programming (MILP)** to solve the **Capacitated Vehicle Routing Problem (CVRP)**. By connecting to a live data source, the system calculates the mathematical global optimum for a route that:
1.  Prioritizes only critical-status ATMs.
2.  Respects the physical cash carrying capacity of the armored truck.
3.  Minimizes the total travel distance using Haversine (Great Circle) physics.

## Live Googlesheet Dataset
**(https://docs.google.com/spreadsheets/d/1SKtDtj3KX9WvE0n9iETiPS2IcNvKxSWiqhhcbqbp2xs/edit?usp=sharing)**

## 🚀 Live Demo

**[Try the Optimizer Here →](https://atmoptimization-nw5qb73e4mw4pp4gvdmvfw.streamlit.app/)**

[Dashboard Screenshot]  <img width="732" height="413" alt="Dashboard Screenshot" src="https://github.com/user-attachments/assets/090fca64-3aba-4ab6-9843-21e67d15f80d" />


## ⚙️ How It Works

1.  **Data Ingestion:** The app connects to a **Live Google Sheet** serving as the centralized inventory database.
2.  **Constraint Filtering:** The system identifies ATMs where `Cash_Level < Safety_Threshold` and calculates the total refill amount required against the Truck Capacity ($1M).
3.  **Optimization Engine:** A **Gurobi** solver instance initializes a network graph of the qualified locations.
4.  **Prescription:** The model outputs a sequence of stops that minimizes mileage while adhering to all physical constraints.

### The Analytics Behind It

*   **Live Data:** Integrated via API (Pandas/Google Cloud) to simulate real-time inventory changes.
*   **The Model:** A **Traveling Salesperson Problem (TSP)** formulation with subtour elimination.
    *   **Decision Variables:** $x_{ij} \in \{0,1\}$ (Binary decision to travel edge $i \to j$).
    *   **Objective Function:** $\min \sum d_{ij}x_{ij}$ (Minimize total distance).
    *   **Constraints:** Miller-Tucker-Zemlin (MTZ) constraints ensure a valid, single continuous loop without teleportation.
*   **Geospatial:** Distances are calculated using the Haversine formula to account for the curvature of the earth (in Miles).

## 📊 Example Output

In a recent test run simulating operations in Harare:
*   **Input:** 20 ATMs with varying demand; 8 flagged as "Critical Low Stock."
*   **Optimization:** The algorithm correctly ignored 12 "Healthy" ATMs.
*   **Efficiency:** The route utilized **90.0% of Truck Capacity**, proving high asset utilization.
*   **Savings:** The optimized route reduced travel distance by **~12 miles (40% reduction)** compared to a random or static route sequence.

## 🛠️ Technology Stack

*   **Frontend:** Streamlit (Python)
*   **Optimization Engine:** Gurobi (Mixed-Integer Programming)
*   **Data Layer:** Google Sheets API (Real-time DB)
*   **Geospatial:** Folium & Haversine
*   **Environment:** Python 3.9+

## 🎓 About This Project

Built for **ISOM 839 (Prescriptive Analytics)** at Suffolk University. This project demonstrates Optimization capabilities by applying operations research to a financial services context.

**Author:** [Cresencia Komboni]
**LinkedIn:** [https://www.linkedin.com/in/cresencia-kudzai-komboni/]
**Email:** [cresenciakomboni@gmail.com]

## 🔮 Future Possibilities

With more development time, this product could expand to:
1.  **Multi-Vehicle Support:** expanding from TSP to m-VRP (managing a fleet of 5+ trucks).
2.  **Time Windows:** Adding constraints for ATMs that are only accessible during business hours (e.g., inside malls).

## 🎬 Demo Video: [Watch the 5-minute walkthrough →](https://www.loom.com/share/c1777fdf481d453b93024b5e9877fa37)
