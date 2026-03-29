import numpy as np
import pandas as pd

from flowers import FlowersModel

# Create generic layout
D = 126.0
layout_x = D * np.array(
    [0.0, 0.0, 0.0, 7.0, 7.0, 7.0, 14.0, 14.0, 14.0, 21.0, 21.0, 21.0, 28.0, 28.0, 28.0]
)
layout_y = D * np.array(
    [0.0, 7.0, 14.0, 0.0, 7.0, 14.0, 0.0, 7.0, 14.0, 0.0, 7.0, 14.0, 0.0, 7.0, 14.0]
)

# Load in wind data
df = pd.read_csv("../data/HKW_wind_rose.csv")

# Setup FLOWERS model
flowers_model = FlowersModel(df, layout_x, layout_y)

# Calculate the AEP
aep = flowers_model.calculate_aep()
print(aep)
