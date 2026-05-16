import os
import pandas as pd

# pastikan folder ada
os.makedirs("data/prepared", exist_ok=True)

# contoh data dummy
data = {
    "col1": [1, 2, 3],
    "col2": ["a", "b", "c"]
}

df = pd.DataFrame(data)

# simpan file
df.to_csv("data/prepared/test.csv", index=False)

print("Data berhasil dibuat!")