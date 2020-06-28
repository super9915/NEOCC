import numpy as np
import time
import csv

user_movie_rating = np.load("user_movie_rating.npy")
movie_user_rating = user_movie_rating.T

# U = np.load("U.npy")
# V = np.load("V.npy")

# save_path = "Object_value/obj_value_20.csv"


# MSSR_cal计算中使用，实现向量到矩阵对角线值分布
def D_ui(ui):
    D = np.zeros((ui.shape[0], ui.shape[0]))
    for i in range(ui.shape[0]):
        D[i, i] = 1
    return D


# 评价NEO-CC算法，U、V，目标函数值的计算
def MSSR_cal(Matrix, U, V):
    num_of_row = Matrix.shape[0]
    num_of_col = Matrix.shape[1]

    K_row = U.shape[1]
    K_col = V.shape[1]

    U_hat = U.copy().astype(np.float)
    V_hat = V.copy().astype(np.float)
    for i in range(K_row):
        U_hat[:, i] /= np.sqrt(np.sum(U_hat[:, i]))
    for j in range(K_col):
        V_hat[:, j] /= np.sqrt(np.sum(V_hat[:, j]))
    
    res = 0
    for i in range(K_row):
        for j in range(K_col):
            tmp1 = np.dot(D_ui(U[:, i]), Matrix)
            tmp1 = np.dot(tmp1, D_ui(V[:, j]))
            tmp2 = np.dot(U_hat[:, i].reshape(-1, 1), U_hat[:, i].reshape(1, -1))
            tmp2 = np.dot(tmp2, Matrix)
            tmp2 = np.dot(tmp2, V_hat[:, j].reshape(-1, 1))
            tmp2 = np.dot(tmp2, V_hat[:, j].reshape(1, -1))
            res += np.linalg.norm(tmp1 - tmp2)
    return res


# NEO-CC中距离的计算
def Dis_Cal(Matrix, U, V):
    num_of_row = Matrix.shape[0]
    num_of_col = Matrix.shape[1]

    K_row = U.shape[1]
    K_col = V.shape[1]

    # """
    U_hat = U.copy().astype(np.float)
    V_hat = V.copy().astype(np.float)
    for i in range(K_row):
        U_hat[:, i] /= np.sqrt(np.sum(U_hat[:, i]))
    for j in range(K_col):
        V_hat[:, j] /= np.sqrt(np.sum(V_hat[:, j]))

    dis_r = np.zeros((num_of_row, K_row))
    for p in range(num_of_row):
        # if p % 100 == 0:
        #     print(p, "/", num_of_row)
        for q in range(K_row):
            d = 0
            for j in range(K_col):
                tmp1 = Matrix[p] * V[:, j]
                # Ip_ = np.zeros((1, num_of_row))
                # Ip_[0, p] = 1
                # D_vj = np.zeros((num_of_col, num_of_col))
                # for i in range(num_of_col):
                #     D_vj[i, i] = V[i, j]
                # tmp1 = np.dot(Ip_, Matrix)
                # tmp1 = np.dot(tmp1, D_vj)
                tmp2 = np.dot(U_hat[:, q].reshape(1, -1), Matrix)
                tmp2 = np.dot(tmp2, V_hat[:, j].reshape(-1, 1))
                tmp2 = np.dot(tmp2, V_hat[:, j].reshape(1, -1)) / np.sqrt(np.sum(U[:, q]))
                d += (np.linalg.norm(tmp1 - tmp2)) ** 2
                # d += np.linalg.norm(tmp1)
            dis_r[p, q] = d
    # """

    # dis_r = np.zeros(())

    return dis_r


# 二维矩阵NEO联合聚类算法
def NEO_CC(Matrix, U, V, alpha_r = 0.0, beta_r = 0.0, alpha_c = 0.0, beta_c = 0.0, iter = 10):
    num_of_row = Matrix.shape[0]
    num_of_col = Matrix.shape[1]

    K_row = U.shape[1]
    K_col = V.shape[1]

    '''
    print("calculate object value...")
    obj_value = MSSR_cal(Matrix, U, V)
    f = open(save_path, "a+")
    csv_writer = csv.writer(f)
    csv_writer.writerow([obj_value])
    f.close()
    print("Objective value: ", obj_value)
    '''

    t = 0
    while t < iter:
        start_time = time.time()
        print("iter: ", t)
        print("calculate distance matrix...")
        dis = Dis_Cal(Matrix, U, V)
        w = 0
        dis_copy_T = dis.copy()
        dis_copy_S = dis.copy()
        U = np.zeros((num_of_row, K_row))
        print("iterate U, V...")
        while w < (1 + alpha_r) * num_of_row:
            if w < (1 - beta_r) * num_of_row:
                row = int(np.where(dis_copy_S == np.min(dis_copy_S))[0][0])
                col = int(np.where(dis_copy_S == np.min(dis_copy_S))[1][0])
                for i in range(K_row):
                    dis_copy_S[row][i] = 100000

                U[row][col] = 1
            else:
                row = int(np.where(dis_copy_T == np.min(dis_copy_T))[0][0])
                col = int(np.where(dis_copy_T == np.min(dis_copy_T))[1][0])
                U[row][col] = 1
            dis_copy_T[row][col] = 100000
            w += 1

        dis = Dis_Cal(Matrix.T, V, U)
        w = 0
        dis_copy_T = dis.copy()
        dis_copy_S = dis.copy()
        V = np.zeros((num_of_col, K_col))
        while w < (1 + alpha_c) * num_of_col:
            if w < (1 - beta_c) * num_of_col:
                row = int(np.where(dis_copy_S == np.min(dis_copy_S))[0][0])
                col = int(np.where(dis_copy_S == np.min(dis_copy_S))[1][0])
                for i in range(K_col):
                    dis_copy_S[row][i] = 100000

                V[row][col] = 1
            else:
                row = int(np.where(dis_copy_T == np.min(dis_copy_T))[0][0])
                col = int(np.where(dis_copy_T == np.min(dis_copy_T))[1][0])
                V[row][col] = 1
            dis_copy_T[row][col] = 100000
            w += 1
        t += 1

        '''
        print("calculate object value...")
        obj_value = MSSR_cal(Matrix, U, V)
        f = open(save_path, "a+")
        csv_writer = csv.writer(f)
        csv_writer.writerow([obj_value])
        f.close()
        print("Objective value: ", obj_value)
        '''
        print("this iteration cost time: ", time.time() - start_time)
        start_time = time.time()
        print("--------" * 10)

    return U, V


if __name__ == "__main__":
    start_time = time.time()

    # U, V = NEO_CC(movie_user_rating, U, V, alpha_r=0.5503, beta_r=0.0001, alpha_c=0.1541, beta_c=0.0, iter=10)
    # np.save("NEO-CC-UV/U.npy", U)
    # np.save("NEO-CC-UV/V.npy", V)
    # dis = Dis_Cal(user_movie_rating, V, U)
    # print(dis.shape)

    print("cost time: ", time.time() - start_time)
