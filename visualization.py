import matplotlib.pyplot as plt
import numpy as np

'''
W = np.load("user_movie_rating.npy")
W = W/5
W = W*12
(a, b) = W.shape
v1 = np.zeros((1, b))
v2 = np.zeros((1, b))
v3 = np.zeros((1, b))
v4 = np.zeros((1, b))
V = np.load("V.npy")
for i in range(a):
    if V[i][0] and not V[i][1]:
        v1 = np.append(v1, [W[i]], 0)
    elif V[i][0] and V[i][1]:
        v2 = np.append(v2, [W[i]], 0)
    elif not V[i][0] and V[i][1]:
        v3 = np.append(v3, [W[i]], 0)
    elif not V[i][0] and not V[i][1]:
        v4 = np.append(v4, [W[i]], 0)
v1 = np.delete(v1, 0, 0)
v2 = np.delete(v2, 0, 0)
v3 = np.delete(v3, 0, 0)
v4 = np.delete(v4, 0, 0)
print(v1.shape)
print(v2.shape)
print(v3.shape)
print(v4.shape)
w = np.append(v1, v2, 0)
w = np.append(w, v3, 0)
w = np.append(w, v4, 0)
np.save("w.npy", w)
plt.matshow(w)
plt.savefig('fig.png', bbox_inches='tight')
'''

w = np.load("w.npy")
U = np.load("U.npy")
(a, b) = w.shape
(c, d) = U.shape
for j in range(d):
    x = np.zeros((a, 1))
    for i in range(c):
        if U[i][j]:
            y = w[:, i]
            y = y.reshape((a, 1))
            x = np.append(x, y, 1)
    x = np.delete(x, 0, 1)
    plt.matshow(x)
    plt.savefig('fig%s.png' % j, bbox_inches='tight')
