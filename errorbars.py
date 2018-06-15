import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7])

ys = np.array([7871.499899, 7940.275694, 8019.555684, 7985.078138, 8009.834683, 8020.959054, 8032.814742, 8011.950408])
es = np.array([404.535802, 384.288900, 366.886279, 364.818985, 364.947113, 360.696366, 352.541192, 364.770285])

plt.errorbar(x, ys, es/10, linestyle='None', marker='*')

yh = np.array([28782.956174, 30164.749594, 29168.079375, 29208.968371, 29236.466415, 29164.344911, 29356.190202, 30037.852945])
eh = np.array([4324.029081, 4176.648975, 4188.362478, 4234.522426, 4180.213250, 4244.271056, 4233.507766, 4097.757079])

# plt.errorbar(x, yh, eh / 10, linestyle='None', marker='*')

plt.show()

# synthetic (3*x - 2*y - z) 0 3000
# 7940.275694, 384.288900 - 0
# 8019.555684, 366.886279 - 10
# 7985.078138, 364.818985 - 15
# 8009.834683, 364.947113 - 20
# 8020.959054, 360.696366 - 25
# 8032.814742, 352.541192 - 30
# 8011.950408, 364.770285 - 50

# reference 7871.499899, 404.535802

# heart -3 3
# 30164.749594, 4176.648975 - 0
# 29168.079375, 4188.362478 - 10
# 29208.968371, 4234.522426 - 15
# 29236.466415, 4180.213250 - 20
# 29164.344911, 4244.271056 - 25
# 29356.190202, 4233.507766 - 30
# 30037.852945, 4097.757079 - 50

# 28782.956174, 4324.029081 - reference
