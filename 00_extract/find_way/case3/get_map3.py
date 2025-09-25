from PIL import Image
import numpy as np

img = Image.open("map3.png").convert("L")  # 0=black、255=white
arr = np.array(img)
grid = (arr > 128).astype(int)  # white=1 free, black=0 obstacle
print(grid.shape) # (1099, 1365)

# (735, 601) to (85, 850) 
x = [735, 85]
y = [601,850]

import matplotlib.pyplot as plt

plt.imshow(grid, cmap="gray", origin="upper")

plt.scatter(x, y, c="red", s=50, marker="o", label="Points")
plt.title("Grid with points")
plt.legend()
plt.show()
