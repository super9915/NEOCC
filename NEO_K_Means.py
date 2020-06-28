import numpy as np
# from matplotlib import pyplot as plt
# import seaborn as sns
from sklearn.cluster import KMeans
# from sklearn.metrics import calinski_harabasz_score
import time

user_movie_rating = np.load("user_movie_rating.npy")
movie_user_rating = user_movie_rating.T
# user_num = user_movie_rating.shape[0]
# movie_num = user_movie_rating.shape[1]

# alpha_r = 0.5503
# beta_r = 0.0001

# alpha_c = 0.1541
# beta_c = 0.0


# 估算α，β
def estimate_alpha_beta(Matrix, classes_num):
    """
    with open("dataset/movies.csv") as f:
        movies = np.loadtxt(f, str, skiprows=1, delimiter=',')
    classes = []
    for movie in movies:
        c = movie[1].split('|')
        for i in c:
            if i not in classes:
                classes.append(i)

    print(classes)
    print(len(classes))
    """

    K = classes_num

    num_of_row = Matrix.shape[0]
    num_of_col = Matrix.shape[1]

    # print(num_of_row)

    y_pred = KMeans(n_clusters = K).fit_predict(Matrix)
    # print(len(y_pred))
    # print(calinski_harabasz_score(movie_user_rating, y_pred))

    # sns.heatmap(movie_user_rating, cmap = 'Reds')
    # plt.show()

    d = {i: y_pred[i] for i in range(num_of_row)}
    Matrix_modify = [[] for i in range(K)]
    for i in range(num_of_row):
        c = d[i]
        if not Matrix_modify[c]:
            Matrix_modify[c].append(Matrix[i])
        else:
            Matrix_modify[c][0] = np.vstack((Matrix_modify[c][0], Matrix[i]))

    for i in range(K):
        if Matrix_modify[i][0].shape[0] == num_of_col:
            Matrix_modify[i][0] = Matrix_modify[i][0].reshape(1, -1)
    # print("---" * 20)
    # 估计α, β
    dis = np.zeros(num_of_row)
    cnt = 0
    mean_k = []
    for i in range(K):
        mean = np.mean(Matrix_modify[i][0], axis=0)
        mean_k.append(mean)
        row = Matrix_modify[i][0].shape[0]
        # print(row)
        for j in range(row):
            dis[cnt] = np.linalg.norm(Matrix_modify[i][0][j] - mean)
            cnt += 1

    miu = np.mean(dis)
    sigma = np.std(dis)
    beta_threshold = miu + sigma * 6
    cnt = 0
    for i in range(num_of_row):
        if dis[i] > beta_threshold:
            cnt += 1

    beta = cnt / num_of_row
    print("beta: ", beta)

    cnt = 0
    alpha_threshold = 1 / (K + 1)
    for i in range(num_of_row):
        dis_k = np.zeros(K)
        for j in range(K):
            dis_k[j] = np.linalg.norm(Matrix[i] - mean_k[j])
        dis_k /= np.sum(dis_k)
        for d in range(K):
            if dis_k[d] < alpha_threshold:
                cnt += 1
    alpha = cnt / (num_of_row * K)
    print("alpha: ", alpha)

    # tmp = movie_user_rating_modify[0][0]
    # for i in range(1, 20):
    #    tmp = np.vstack((tmp, movie_user_rating_modify[i][0]))

    # movie_user_rating_modify = tmp
    # sns.heatmap(movie_user_rating_modify, cmap='Reds')
    # plt.show()
    return alpha, beta


# 找到矩阵中距离最远的两个向量的距离，用于初始化聚类中心
def find_max_dis(Matrix):
    num_of_row = Matrix.shape[0]
    num_of_col = Matrix.shape[1]
    dis_point2point = np.zeros((num_of_row, num_of_row))
    for i in range(0, num_of_row):
        print(i)
        for j in range(i, num_of_row):
            dis_point2point[i][j] = np.linalg.norm(Matrix[i] - Matrix[j])

    max_dis = np.max(dis_point2point)
    return max_dis


# 二维矩阵的行向量方向NEO聚类算法
def NEO_K_Means(Matrix, classes_num, alpha, beta, iter = 10, max_dis = 1):
    K = classes_num

    num_of_row = Matrix.shape[0]
    num_of_col = Matrix.shape[1]

    # print(111)

    # mean_M = np.random.uniform(0, 2, (K, user_num))

    # """
    cnt = 0
    mean_M = np.zeros((K, num_of_col))
    mean_M[0] = Matrix[0]

    # print(111)

    for i in range(num_of_row):
        flag = True
        for j in range(cnt):
            if np.linalg.norm(Matrix[i] - mean_M[j]) < max_dis * 0.9:
                flag = False
        if flag:
            cnt += 1
            mean_M[cnt] = Matrix[i]
        if cnt + 1 == K:
            break
    # """

    t = 0

    U = np.zeros((num_of_row, K), dtype=bool)
    start_time = time.time()
    his_M = 0
    while t < iter:
        p = 0
        print("iter: ", t)
        U = np.zeros((num_of_row, K), dtype=bool)

        dis = np.zeros((num_of_row, K))
        for i in range(num_of_row):
            for j in range(K):
                d = np.linalg.norm(Matrix[i] - mean_M[j])
                dis[i][j] = d
        # T = []
        # S = np.zeros(movie_num, dtype = bool)
        C = [[] for i in range(K)]
        # print(np.max(dis))
        # exit(0)
        dis_copy_T = dis.copy()
        dis_copy_S = dis.copy()
        while p < (1 + alpha) * num_of_row:
            if p < (1 - beta) * num_of_row:
                row = int(np.where(dis_copy_S == np.min(dis_copy_S))[0][0])
                col = int(np.where(dis_copy_S == np.min(dis_copy_S))[1][0])

                for i in range(K):
                    dis_copy_S[row][i] = 100000

                C[col].append(Matrix[row])
                U[row][col] = 1
            else:
                row = int(np.where(dis_copy_T == np.min(dis_copy_T))[0][0])
                col = int(np.where(dis_copy_T == np.min(dis_copy_T))[1][0])

                if U[row][col] != 1:
                    C[col].append(Matrix[row])
                    U[row][col] = 1
            dis_copy_T[row][col] = 100000
            p += 1
        t += 1

        l = [len(C[i]) for i in range(K)]
        print(l)
        for i in range(K):
            if len(C[i]) <= int(np.ceil(beta * num_of_row)):
                mean_M[i] = np.random.uniform(0, 2, size = num_of_col)
                continue
            mean_M[i] = np.mean(np.array(C[i]), axis = 0)

        this_M = np.mean(mean_M)
        print("mean of C: ", this_M)
        print("time of this iter: ", time.time() - start_time)
        start_time = time.time()
        if np.abs(this_M - his_M) < 0.00001:
            break
        his_M = this_M

    return U


if __name__ == "__main__":
    start_time = time.time()
    # V = NEO_K_Means(user_movie_rating, 2, alpha_c, beta_c, 10, 150)
    # np.save("V.npy", V)

    # U = NEO_K_Means(movie_user_rating, 2, 0.3587, 0.0013, 10, 180)
    # np.save("U_2.npy", U)

    # estimate_alpha_beta(user_movie_rating, 2)

    # U = np.load("U.npy")
    # print(U.shape)

    print("cost time: ", time.time() - start_time)
