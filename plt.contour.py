import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF
clf = RF()
clf.fit(X, y)

dx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), resolution)
dy = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), resolution)
dx, dy = np.meshgrid(dx, dy)
test_x = np.c_[dx.flatten(), dy.flatten()]
z = clf.predict(test_x)
z = z.reshape(dx.shape)

plt.contour(dx, dy, z, alpha = 0.2)
