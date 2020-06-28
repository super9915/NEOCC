# Non-Exhaustive, Overlapping Co-Clustering:Implementation, Improvement and DataVisualization

# Code Structure

* [`dataset/`](dataset/): 数据集存储文件夹
    * [`ml-latest-small/`](dataset/ml-latest-small/): [MovieLen](https://grouplens.org/datasets/movielens/)网站下载的原数据集，[README.txt](dataset/ml-latest-small/README.txt)中有具体的csv文件解释
    * [`movies.csv`](dataset/movies.csv): 预处理数据集得到的csv文件，第一列为movieID，第二列为电影种类
    * [`ratings.csv`](dataset/ratings.csv): 预处理数据集得到的csv文件，第一列为userID，第二列为movieID，第三列为用户评分，0.5 - 5，半分一档

* [`UV/`](UV/): 存储算法得到的U，V矩阵

* [`NEO_CC.py`](NEO_CC.py): NEO-CC算法实现及一些辅助函数

* [`NEO_K_Means.py`](NEO_K_Means.py): NEO-K-Means算法实现，estimate_alpha_beta算法实现，及一些辅助函数

* [`Summary.py`](Summary.py): 最终算法的实现，总结上两个py文件中的算法

* [`user_movie_preprocess.py`](user_movie_preprocess.py): 对数据集的预处理，及得到[movies.csv](dataset/movies.csv)，[ratings.csv](dataset/ratings.csv)和[user_movie_rating.npy](user_movie_rating.npy)

* [`user_movie_rating.npy`](user_movie_rating.npy): $610 \times 9742$矩阵， 行为userID，列为movieID，值为用户评分，无评分置零
