from PIL import Image
import numpy as np

img = Image.open("map1.png").convert("L")  # 0=black、255=white
arr = np.array(img)
grid = (arr > 128).astype(int)  # white=1 free, black=0 obstacle
print(grid.shape) # (2364, 3330)

# (1750, 200) to (3100, 1400) 
x = [1750, 3100]
y = [200,1400]

import matplotlib.pyplot as plt

plt.imshow(grid, cmap="gray", origin="upper")

plt.scatter(x, y, c="red", s=50, marker="o", label="Points")
plt.title("Grid with points")
plt.legend()
plt.show()
