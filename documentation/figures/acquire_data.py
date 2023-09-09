from glob import glob
import pandas as pd
from numpy import ndarray
from typing import List

RESULT_PATH = "/home/uhrich/Bachelorthesis/results/"

def acquire_curves(dir: str, prefix: str, key: str) -> List[ndarray]:
    search_regex = RESULT_PATH + f"{dir}/{prefix}*/results.csv"
    paths = glob(search_regex)
    if not len(paths):
        raise ValueError(
            f"No matching experiment results with: '{search_regex}' found")
    keys = pd.read_csv(paths[0])["entity"].unique()
    if key not in keys:
        print(keys)
        return

    data = []
    for path in paths:
        df = pd.read_csv(path)
        key_df = df[df["entity"] == key]

        data.append(key_df[["x0", "y"]].to_numpy())

    # try:
    #     data = np.stack(data)
    # except:
    #     raise ValueError(f"Inconsistent shaped data found: {[d.shape for d in data]}")

    return data