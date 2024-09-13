import pandas as pd

d = {
    "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
    "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
}

print(f'Series {d}')

df = pd.DataFrame(d)

print(f'DataFrame {df}')

dfi = pd.DataFrame(d, index=["d", "b", "a"])

print(f'DataFrame Index {dfi}')

cols = pd.DataFrame(d, index=["d", "b", "a"], columns=["two", "three"])

print(f'DataFrame Index {cols}')
