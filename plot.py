import matplotlib.pyplot as plt
import numpy as np
import math




private_scores = [
0.04012, 
0.04176,
0.04194,
0.04467,
0.04491,
]

public_scores = [
0.04422,
0.04456,
0.04681,
0.04760,
0.04912,
]


plt.plot([0, 1, 2, 3, 4], private_scores, label = "private")
plt.plot([0, 1, 2, 3, 4], public_scores, label = "public")
plt.title("scores based on different latent dimension")
plt.xlabel("number of dimensions")
plt.ylabel("MAP score on test")
plt.xticks([0, 1, 2, 3, 4], ['16', '32', '64', '128', '256'], rotation=20)
plt.legend()
plt.show()