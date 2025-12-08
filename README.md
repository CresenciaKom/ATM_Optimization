# ATM Cash Logistics Optimizer

A Streamlit web application that optimizes cash logistics routes for ATM networks using Gurobi optimization solver.

## Features

- **Scenario Generation**: Generates 15 fictitious ATMs around a center point with random cash levels and demand forecasts
- **Route Optimization**: Uses Gurobi solver to find the optimal route minimizing distance
- **Interactive Map**: Visualizes ATM locations and optimized routes using Folium
- **Capacity Analysis**: Monitors truck capacity usage and warns when capacity is exceeded
- **Cost Analysis**: Compares optimized routes against random routes to show savings

## Requirements

- Python 3.7+
- Gurobi Optimizer (requires license - free academic licenses available)

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd ATM_Optimization
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up Gurobi license:
   - Get a free academic license from [Gurobi's website](https://www.gurobi.com/academia/academic-program-and-licenses/)
   - Follow Gurobi's installation instructions for your platform

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository and branch
6. Set the main file path to `app.py`
7. Click "Deploy"

**Note**: Make sure your `requirements.txt` includes all dependencies. Streamlit Cloud will automatically install them.

## Project Structure

```
ATM_Optimization/
├── app.py                 # Main Streamlit application
├── data_manager.py        # ScenarioGenerator class for generating ATM data
├── optimizer.py          # GurobiRouteSolver class for route optimization
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## How It Works

The optimizer uses a Traveling Salesman Problem (TSP) formulation with:
- **Decision Variables**: Binary variables indicating route segments
- **Objective**: Minimize total distance traveled
- **Constraints**: Flow conservation and Miller-Tucker-Zemlin subtour elimination

See the "How the Gurobi Optimizer Works" section in the app for detailed mathematical formulation.

## License

This project requires a Gurobi license for the optimization solver. Academic licenses are available for free.

