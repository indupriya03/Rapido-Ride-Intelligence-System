import pandas as pd
import os

# def load_data(data_dir="../data/raw"):
#     """
#     Load all Rapido datasets, path-independent from where notebook/script is run.
#     """
#     cwd = os.getcwd()  # e.g., .../Rapido-Ride-Intelligence-System/notebooks
#     project_root = os.path.abspath(os.path.join(cwd, ".."))  # go up to project root

#     # 2. Build path to raw data folder
#     data_dir = os.path.join(project_root, "data", "raw")

#     # 3. Load CSVs
#     bookings = pd.read_csv(os.path.join(data_dir, "bookings.csv"))
#     customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
#     drivers = pd.read_csv(os.path.join(data_dir, "drivers.csv"))
#     location_demand = pd.read_csv(os.path.join(data_dir, "location_demand.csv"))
#     time_features = pd.read_csv(os.path.join(data_dir, "time_features.csv"))
#     return bookings, customers, drivers, location_demand, time_features


# def _get_project_root():
#     """
#     Get project root directory regardless of where script is run.
#     """
#     cwd = os.getcwd()
#     return os.path.abspath(os.path.join(cwd, ".."))


def _get_project_root():
    """
    Get project root directory (works for both .py and .ipynb)
    """
    try:
        # When running as .py
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(current_file))
    except NameError:
        # When running in Jupyter notebook
        cwd = os.getcwd()
        project_root = os.path.abspath(os.path.join(cwd, ".."))

    return project_root


def _get_data_dir(folder):
    """
    Build path to a data folder (raw / cleaned / processed).
    """
    project_root = _get_project_root()
    return os.path.join(project_root, "data", folder)


# -----------------------
# RAW DATA
# -----------------------
def load_raw_data():
    data_dir = _get_data_dir("raw")

    bookings = pd.read_csv(os.path.join(data_dir, "bookings.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    drivers = pd.read_csv(os.path.join(data_dir, "drivers.csv"))
    location_demand = pd.read_csv(os.path.join(data_dir, "location_demand.csv"))
    time_features = pd.read_csv(os.path.join(data_dir, "time_features.csv"))

    return bookings, customers, drivers, location_demand, time_features



# -----------------------
# CLEANED DATA
# -----------------------
def load_cleaned_data():
    data_dir = _get_data_dir("cleaned")

    bookings = pd.read_csv(os.path.join(data_dir, "bookings_cleaned.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "customers_cleaned.csv"))
    drivers = pd.read_csv(os.path.join(data_dir, "drivers_cleaned.csv"))
    location_demand = pd.read_csv(os.path.join(data_dir, "location_demand_cleaned.csv"))
    time_features = pd.read_csv(os.path.join(data_dir, "time_features_cleaned.csv"))

    return bookings, customers, drivers, location_demand, time_features
