import numpy as np
import pandas as pd
import math
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split


class MatrixFactorization:

    def __init__(self, n, k):
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

    def generate_dataset(self):
        ratings = pd.read_csv('./ml-latest-small/ratings.csv', usecols=['userId', 'movieId', 'rating'],
                              dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
        self.train_dataset, self.test_dataset = train_test_split(ratings, train_size=0.8)
        self.pivot_table = self.train_dataset.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.feature_matrix = self.pivot_table.iloc[:, :].values
        self.user_rating_mean = np.mean(self.feature_matrix, axis=1)
        self.normalized_feature_matrix = self.feature_matrix - self.user_rating_mean.reshape(-1, 1)

    def predict_rating(self):
        U, sig, Vt = svds(self.normalized_feature_matrix, k=self.rank)
        sig = np.diag(sig)
        predicted_ratings = np.dot(np.dot(U, sig), Vt) + self.user_rating_mean.reshape(-1, 1)
        self.predicted_ratings = pd.DataFrame(predicted_ratings, columns=self.pivot_table.columns)

    def test(self):
        matched_count = 0
        recommended_total = 0
        test_total = 0
        predict_total = 0
        NDCG = 0
        MAE = 0
        RMSE = 0
        invalid_count = 0
        user_set = set(self.train_dataset['userId'].to_list())

        for user_id in user_set:
            print('Recommending for user %d' % user_id)
            user_test_data = self.test_dataset[self.test_dataset['userId'] == user_id]
            test_movies = user_test_data['movieId'].to_list()
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
                    diff = recommendation['predictedRating'] - \
                           user_test_data[self.test_dataset['movieId'] == recommendation['movieId']]['rating'].iloc[0]
                    MAE += abs(diff)
                    RMSE += pow(diff, 2)
                    predict_total += 1
            test_total += len(test_movies)
            recommended_total += self.top_n_recommendation
            if IDCG > 0:
                NDCG += DCG / IDCG
                invalid_count += 1

        precision = matched_count / (recommended_total * 1.0)
        recall = matched_count / (test_total * 1.0)
        f_measure = 2 * precision * recall / (precision + recall)
        NDCG /= (recommended_total - invalid_count)
        print('Precision=%f, Recall=%f F-measure=%f, NDCG=%f' % (precision, recall, f_measure, NDCG))

        MAE /= predict_total
        RMSE = math.sqrt(RMSE / predict_total)
        print('MAE=%f, RMSE=%f' % (MAE, RMSE))

    def recommend(self, user_id):
        recommendation = []
        user_row_num = user_id - 1
        sorted_prediction = self.predicted_ratings.iloc[user_row_num].sort_values(ascending=False)
        user_data = self.train_dataset[self.train_dataset.userId == user_id]
        rated_movie = user_data['movieId'].to_list()
        for movie_id, rating in sorted_prediction.iteritems():
            if movie_id in rated_movie:
                continue
            recommendation.append({'movieId': movie_id, 'predictedRating': rating})
            if len(recommendation) == self.top_n_recommendation:
                break
        return recommendation


matrixFactorization = MatrixFactorization(10, 20)
matrixFactorization.generate_dataset()
matrixFactorization.predict_rating()
matrixFactorization.test()