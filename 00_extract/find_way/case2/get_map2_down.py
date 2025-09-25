from PIL import Image
import numpy as np

img = Image.open("map2.png").convert("L")  
arr = np.array(img)
grid = (arr > 128).astype(int) 
print(grid.shape) # (730, 1398)

# (703, 363) to (780,664)
x = [703, 780]
y = [363, 664]

import matplotlib.pyplot as plt

plt.imshow(grid, cmap="gray", origin="upper")

plt.scatter(x, y, c="red", s=50, marker="o", label="Points")
plt.title("Grid with points")
plt.legend()
plt.show()
