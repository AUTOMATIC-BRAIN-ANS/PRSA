import numpy as np
import pandas as pd

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame."""
    df_copy = df.copy()
    df_copy['Values'].ffill(inplace=True)
    return df_copy
    results['NN50'] = np.sum(np.abs(np.diff(rr)) > 50)  # Count of successive RR interval differences greater than 50 ms
    results['pNN50 (%)'] = 100 * results['NN50'] / len(rr)  # Percentage of NN50 divided by total number of RR intervals

    return results