import pandas as pd
import os

DATA_DIR = "data/raw"

files = [
    "MTeams.csv",
    "MSeasons.csv",
    "MRegularSeasonDetailedResults.csv",
    "MNCAATourneyDetailedResults.csv",
    "MNCAATourneySeeds.csv",
    "MNCAATourneySlots.csv",
]

for filename in files:

    path = os.path.join(DATA_DIR, filename)

    print("\n" + "=" * 70)
    print(filename)
    print("=" * 70)

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3))