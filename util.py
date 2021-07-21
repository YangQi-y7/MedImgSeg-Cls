import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
base_dir = 'out/loss'

# show_plt
df = pd.read_csv(os.path.join(base_dir, 'seg_3D%3A2021-07-19 20%3A58%3A52.552529.csv'), header=0, index_col=0)
df2 = df.replace('-', 0.0)
df2 = df2.astype(float)

df.plot()
df2.plot()
plt.show()
