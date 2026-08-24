# import data Manipulation libraries
import numpy as np
import pandas as pd

def data_loader():
    """
    Function to load the data from the CSV file.
    Returns:
        df (DataFrame): Loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(r"https://raw.githubusercontent.com/tulsip26-hub/CementStrength_PredictionModel/refs/heads/main/data/Concrete_Data.csv")
    return df