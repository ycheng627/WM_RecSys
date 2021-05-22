import matplotlib.pyplot as plt
import numpy as np
import math








private_scores = [
0.02948,
0.03034,
0.03457,
0.03641,
0.03757,

]

public_scores = [
0.02891,
0.03237,
0.03109,
0.03582,
0.04244,
]


plt.plot([0, 1, 2, 3, 4], private_scores, label = "private")
plt.plot([0, 1, 2, 3, 4], public_scores, label = "public")
plt.title("scores based on different negative Ratio in BCE")
plt.xlabel("Negative Ratio")
plt.ylabel("MAP score on test")
plt.xticks([0, 1, 2, 3, 4], ['1/4', '1/2', '1', '2', '4'])
plt.legend()
plt.show()