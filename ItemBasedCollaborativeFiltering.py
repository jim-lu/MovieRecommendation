import numpy as np
import random
import math
import collections


class ItemBasedCollaborativeFiltering:

    def __init__(self, top_k_similar_movie=20, top_n_recommendation=10):
        self.training_dataset = {}
        self.testing_dataset = {}
        self.top_k_similar_movie = top_k_similar_movie
        self.top_n_recommendation = top_n_recommendation
        self.similar_matrix = {}
        self.popularity_matrix = {}

    def generate_dataset(self, filename, split_param=0.05):
        ratings = np.loadtxt(filename, dtype=np.str, delimiter=',')[1:].tolist()
        for line in ratings:
            user_id = line[0]
            movie_id = line[1]
            rating = line[2]
            if random.random() < split_param:
                self.training_dataset.setdefault(user_id, {})
                self.training_dataset[user_id][movie_id] = float(rating)
            else:
                self.testing_dataset.setdefault(user_id, {})
                self.testing_dataset[user_id][movie_id] = float(rating)

    def calculate_movie_similarity(self):
        for user, movies in self.training_dataset.items():
            for movie in movies:
                self.similar_matrix.setdefault(movie, collections.defaultdict(int))
                if movie not in self.popularity_matrix:
                    self.popularity_matrix[movie] = 0
                self.popularity_matrix[movie] += 1
                for other_movie in movies:
                    if movie == other_movie:
                        continue
                    self.similar_matrix[movie][other_movie] += 1
        for movie, related_movies in self.similar_matrix.items():
            for other_movie, count in related_movies.items():
                # Calculate the similarity based on the co-occurrence
                self.similar_matrix[movie][other_movie] = count / math.sqrt(self.popularity_matrix[movie]
                                                                            * self.popularity_matrix[other_movie])

    def test(self):
        matched_count = 0  # Number of the recommended movies matching the true test movie
        test_total = 0  # Total number of test movie
        recommended_total = 0  # Total number of recommended movie
        predict_total = 0
        NDCG = 0
        MAE = 0
        RMSE = 0

        for user in self.training_dataset:
            test_movies = self.testing_dataset.get(user, {})
            recommended_movies = self.recommend(user)
            pos = 0
            i = 0
            DCG = 0
            IDCG = 0
            for movie, rating in recommended_movies:
                pos += 1
                if movie in test_movies:
                    matched_count += 1
                    i += 1
                    DCG += 1 / math.log(1 + pos, 2)
                    IDCG += 1 / math.log(1 + i, 2)

                    diff = rating - self.testing_dataset[user][movie]
                    MAE += abs(diff)
                    RMSE += pow(diff, 2)
                    predict_total += 1
            test_total += len(test_movies)
            recommended_total += self.top_n_recommendation
            if IDCG > 0:
                NDCG += DCG / IDCG

        # Evaluate recommendation
        precision = matched_count / (recommended_total * 1.0)
        recall = matched_count / (test_total * 1.0)
        f_measure = 2 * precision * recall / (precision + recall)
        NDCG /= recommended_total
        print('Precision=%f, Recall=%f F-measure=%f, NDCG=%f' % (precision, recall, f_measure, NDCG))

        # Evaluate predicted rating
        MAE /= predict_total
        RMSE = math.sqrt(RMSE / predict_total)
        print('MAE=%f, RMSE=%f' % (MAE, RMSE))

    def recommend(self, user):
        recommendation_list = {}
        rated_movies = self.training_dataset[user]
        # More efficient by just recommend the movie in the training dataset
        for movie, rating in rated_movies.items():
            top_k_similar_movies = sorted(self.similar_matrix[movie].items(), key=lambda x: x[1],
                                          reverse=True)[0: self.top_k_similar_movie]
            for related_movie, similarity in top_k_similar_movies:
                if related_movie in rated_movies:
                    continue
                recommendation_list.setdefault(related_movie, 0)
                recommendation_list[related_movie] += similarity * rating
        return sorted(recommendation_list.items(), key=lambda x: x[1], reverse=True)[0: self.top_n_recommendation]


item_based_cf = ItemBasedCollaborativeFiltering()
item_based_cf.generate_dataset('./ml-latest-small/ratings_dealt.csv')
item_based_cf.calculate_movie_similarity()
item_based_cf.test()
