import pandas as pd

for index in range(5):
    path = f"/home/so87pot/n0w0f/structllm/data/canonical_360/csv/test_matbench_log_gvrh_{index}.csv"
    df = pd.read_csv(path)
    df=df.dropna()
    df.to_csv(path)