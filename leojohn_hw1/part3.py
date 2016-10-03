data =[{"Data":"Heart","m":2 ,"accuracy":0.6796116504854369},
       {"Data":"Heart","m":5 ,"accuracy":0.6990291262135923},
       {"Data":"Heart","m":10 ,"accuracy":0.7184466019417476},
       {"Data":"Heart","m":20 ,"accuracy":0.7281553398058253},
       {"Data":"Diabetes","m":2 ,"accuracy":0.68},
       {"Data":"Diabetes","m":5 ,"accuracy":0.69},
       {"Data":"Diabetes","m":10 ,"accuracy":0.70},
       {"Data":"Diabetes","m":20 ,"accuracy":0.72},
      
      ]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks")
sns.set_context("paper")
plt.figure(figsize=(8, 6))
df = pd.DataFrame(data)
# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(df, col="Data", hue="Data", col_wrap=5, size=1.5)
# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "m", "accuracy", marker="o", ms=5)
# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0, ls=":", c=".5")
# Adjust the tick positions and labels
grid.set(xticks=[0,2,5,10,20,25], yticks=np.arange(0.0,1.0,0.1))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)                     
plt.show()                     
