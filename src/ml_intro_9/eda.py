import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport


def main():
    df = pd.read_csv('data/train.csv')
    profile = ProfileReport(df, title="Pandas Profiling Report")
    profile.to_file("reports/eda.html")

