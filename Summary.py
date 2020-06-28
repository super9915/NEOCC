from NEO_K_Means import NEO_K_Means, estimate_alpha_beta
from NEO_CC import NEO_CC
import numpy as np
import time


def Summary(Matrix, K_row, K_col, iter = 10, init_iter = 1):
    print("estimate alpha_r, beta_r...")
    alpha_r, beta_r = estimate_alpha_beta(movie_user_rating, K_row)
    print("--------" * 10)
    print("estimate alpha_c, beta_c...")
    alpha_c, beta_c = estimate_alpha_beta(user_movie_rating, K_col)
    print("--------" * 10)

    print("initial U...")
    U = NEO_K_Means(movie_user_rating, K_row, alpha_r, beta_r, init_iter, 150)
    print("--------" * 10)
    print("initial V...")
    V = NEO_K_Means(user_movie_rating, K_col, alpha_c, beta_c, init_iter, 150)
    print("--------" * 10)

    print("NEO CC algorithm...")
    U, V = NEO_CC(movie_user_rating, U, V, alpha_r, beta_r, alpha_c, beta_c, iter=iter)

    return U, V


if __name__ == "__main__":
    start_time = time.time()

    user_movie_rating = np.load("user_movie_rating.npy")
    movie_user_rating = user_movie_rating.T

    K_row = 2
    K_col = 2

    U, V = Summary(movie_user_rating, K_row, K_col, iter = 5)
    np.save("UV/U.npy", U)
    np.save("UV/V.npy", V)

    print("cost time: ", time.time() - start_time)