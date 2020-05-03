import numpy as np
import pandas as pd
import math
from scipy.sparse.linalg import svds
import csv
from itertools import zip_longest


class MatrixFactorization:

    def __init__(self, n, k, learning_rate=0.008, regularization_param=0.1, epochs=30):
        self.top_n_recommendation = n
        self.rank = k
        self.train_dataset = None
        self.test_dataset = None
        self.pivot_table = None
        self.feature_matrix = None
        self.user_rating_mean = None
        self.normalized_feature_matrix = None
        self.predicted_ratings = None
        self.rating_predict_matrix = {}
        self.P = None
        self.Q = None
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.epochs = epochs
        self.train_movie_index_map = {}

    def generate_dataset(self, training_set_file, testing_set_file):
        self.train_dataset = pd.read_csv(training_set_file, usecols=['userId', 'movieId', 'rating'],
                              dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        self.test_dataset = pd.read_csv(testing_set_file, usecols=['userId', 'movieId', 'rating'],
                              dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        self.pivot_table = self.train_dataset.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        for index, id in enumerate(self.pivot_table.columns.astype(int).tolist()):
            self.train_movie_index_map[id] = index
        # print(self.train_movie_index_map)
        self.feature_matrix = self.pivot_table.iloc[:, :].values
        self.user_rating_mean = np.mean(self.feature_matrix, axis=1)
        self.normalized_feature_matrix = self.feature_matrix - self.user_rating_mean.reshape(-1, 1)

    def svd(self):
        U, sig, Vt = svds(self.normalized_feature_matrix, k=self.rank)
        self.P = U
        sig = np.diag(sig)
        self.Q = sig.dot(Vt)
        # print(self.P)
        # print(self.Q)

    def optimaize(self):
        for epoch in range(self.epochs):
            # print('epoch', epoch)
            for _, row in self.train_dataset.iterrows():
                user = int(row['userId'])
                movie = int(row['movieId'])
                rating = row['rating']
                for k in range(self.rank):
                    error = 2 * self.error(user, movie, rating) * self.P[user - 1, k]
                    regularization = -2 * self.regularization_param * self.Q[k, self.train_movie_index_map[movie]]
                    q_gradient = self.learning_rate * (error + regularization)
                    self.Q[k, self.train_movie_index_map[movie]] += q_gradient
                    p_gradient = self.learning_rate * (2 * self.error(user, movie, rating) *
                                                       self.Q[k, self.train_movie_index_map[movie]] - 2 *
                                                       self.regularization_param * self.P[user - 1, k])
                    self.P[user - 1, k] += p_gradient

    def error(self, user, movie, rating):
        predicted_rating = self.predict_rating(user, movie)
        return rating - predicted_rating

    def predict_rating(self, user, movie):
        row = self.P[user - 1, :]
        col = self.Q[:, self.train_movie_index_map[movie]]
        return np.dot(row, col)

    def test(self, precision_list, recall_list, f_measure_list, NDCG_list, MAE_list, RMSE_list):
        matched_count = 0
        recommended_total = 0
        test_total = 0
        predict_total = 0
        NDCG = 0
        MAE = 0
        RMSE = 0
        valid_count = 0
        user_set = set(self.train_dataset['userId'].tolist())
        predicted_rating = np.dot(self.P, self.Q) + self.user_rating_mean.reshape(-1, 1)
        self.predicted_ratings = pd.DataFrame(predicted_rating, columns=self.pivot_table.columns)
        # print(self.predicted_ratings)

        for user_id in user_set:
            # print('Recommending for user %d' % user_id)
            user_test_data = self.test_dataset[self.test_dataset['userId'] == user_id]
            test_movies = user_test_data['movieId'].tolist()

            for movie in test_movies:
                # print(user_test_data[self.test_dataset['movieId'] == movie]['rating'].iloc[0])
                # print(self.predicted_ratings.iloc[user_id - 1], type(self.predicted_ratings.iloc[user_id - 1]))
                # print(self.predicted_ratings.iloc[user_id - 1].get(movie), type(self.predicted_ratings.iloc[user_id - 1].get(movie)))
                diff = user_test_data[self.test_dataset['movieId'] == movie]['rating'].iloc[0] - self.predicted_ratings.iloc[user_id - 1].get(movie)
                MAE += abs(diff)
                RMSE += pow(diff, 2)
                predict_total += 1
                # print(user_test_data[self.test_dataset['movieId'] == movie]['rating'].iloc[0], self.predicted_ratings[user_id - 1, self.train_movie_index_map[movie]])

            recommended_movies = self.recommend(user_id)
            pos = 0
            i = 0
            DCG = 0
            IDCG = 0

            for recommendation in recommended_movies:
                pos += 1
                if recommendation['movieId'] in test_movies:
                    matched_count += 1
                    i += 1
                    DCG += 1 / math.log(1 + pos, 2)
                    IDCG += 1 / math.log(1 + i, 2)
                    # diff = recommendation['predictedRating'] - \
                    #        user_test_data[self.test_dataset['movieId'] == recommendation['movieId']]['rating'].iloc[0]
                    # MAE += abs(diff)
                    # RMSE += pow(diff, 2)
                    # predict_total += 1
            test_total += len(test_movies)
            recommended_total += self.top_n_recommendation
            if IDCG > 0:
                NDCG += DCG / IDCG
                valid_count += 1

        precision = matched_count / (recommended_total * 1.0)
        recall = matched_count / (test_total * 1.0)
        f_measure = 2 * precision * recall / (precision + recall)
        NDCG /= valid_count
        precision_list.append(precision)
        recall_list.append(recall_list)
        f_measure_list.append(f_measure)
        NDCG_list.append(NDCG_list)
        print('Precision=%f, Recall=%f F-measure=%f, NDCG=%f' % (precision, recall, f_measure, NDCG))

        MAE /= predict_total
        RMSE = math.sqrt(RMSE / predict_total)
        MAE_list.append(MAE)
        RMSE_list.append(RMSE_list)
        print('MAE=%f, RMSE=%f' % (MAE, RMSE))
        return RMSE

    def recommend(self, user_id):
        recommendation = []
        user_row_num = user_id - 1
        sorted_prediction = self.predicted_ratings.iloc[user_row_num].sort_values(ascending=False)
        user_data = self.train_dataset[self.train_dataset.userId == user_id]
        rated_movie = user_data['movieId'].tolist()
        for movie_id, rating in sorted_prediction.iteritems():
            if movie_id in rated_movie:
                continue
            recommendation.append({'movieId': movie_id, 'predictedRating': rating})
            if len(recommendation) == self.top_n_recommendation:
                break
        return recommendation


precision_list = []
recall_list = []
f_measure_list = []
NDCG_list = []
MAE_list = []
RMSE_list = []
e = 100
previous_rmse = float('inf')
current_rmse = float('inf')
diff = float('inf')
while diff > 0.005:
    print("Epoch equals", e)
    matrixFactorization = MatrixFactorization(10, 20, epochs=e)
    # matrixFactorization.generate_dataset()
    matrixFactorization.generate_dataset('./data/trainset_4.csv', './data/testset_4.csv')
    matrixFactorization.svd()
    matrixFactorization.optimaize()
    # matrixFactorization.predict_rating()
    current_rmse = matrixFactorization.test(precision_list, recall_list, f_measure_list, NDCG_list, MAE_list, RMSE_list)
    diff = abs(current_rmse - previous_rmse)
    print(diff)
    previous_rmse = current_rmse
    e += 20

d = [precision_list, recall_list, f_measure_list, NDCG_list, MAE_list, RMSE_list]
data = zip_longest(*d, fillvalue='')
with open('result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(("precision", "recall", "f-measure", "NDCG", "MAE", "RMSE"))
    writer.writerows(data)
f.close()
