import numpy as np
import time
import random
import csv

movie_path = "dataset/ml-latest-small/movies.csv"
rating_path = "dataset/ml-latest-small/ratings.csv"
movie_modify_path = "dataset/movies.csv"
rating_modify_path = "dataset/ratings.csv"


# 数据集的预处理
if __name__ == "__main__":
    count = -1
    f = open(movie_path, encoding = "utf8")
    movies = csv.reader(f)

    d = {}
    movie_modify = [["movieId", "genres"]]
    for movie in movies:
        if count == -1:
            count = 0
            continue
        movieID = movie[0]
        movieGenres = movie[2]
        movie_modify.append([count, movieGenres])
        d[movieID] = count
        count += 1

    f.close()

    movie_num = count
    print("movie num: ", movie_num)

    with open(movie_modify_path, 'w', newline='', encoding = "utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(movie_modify)

    count = 0
    with open(rating_path) as f:
        ratings = np.loadtxt(f, str, skiprows=1, delimiter=',')

    user_num = int(ratings[-1][0])
    print("user num: ", user_num)

    user_rating_matrix = np.zeros((user_num, movie_num), dtype = float)

    rating_modify = [["userId", "movieId", "rating"]]
    for line in ratings:
        userID = line[0]
        movieID = line[1]
        rating = line[2]
        rating_modify.append([int(userID) - 1, d[movieID], rating])
        user_rating_matrix[int(userID)-1][d[movieID]] = float(rating)

    with open(rating_modify_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rating_modify)

    print(user_rating_matrix[:10][:10])
    np.save("user_movie_rating.npy", user_rating_matrix)
