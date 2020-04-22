import numpy as np
import random
import collections
import math


class UserBasedCollaborativeFiltering:

    def __init__(self, top_k_similar_user=10, top_n_recommendation=10):
        self.training_dataset = {}
        self.testing_dataset = {}
        self.top_k_similar_user = top_k_similar_user
        self.top_n_recommendation = top_n_recommendation
        self.similarity_matrix = {}
        self.rating_predict_matrix = {}

    def generate_dataset(self, filename, split_param=0.8):
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

    def calculate_user_similarity(self):
        """
        Since users might never watch the same movie, user-to-movie matrix is sparse and consume a lot of resources
        to traverse. User movie-to-user matrix can easily see who watch the same movie.

            A B C D         A  u1 u3
        u1  *   *           B  u2 u3
        u2    *   *    =>   C  u1
        u3  * *             D  u2
        """

        movie_to_user_dict = {}
        for user, movies in self.training_dataset.items():
            for movie in movies:
                if movie not in movie_to_user_dict:
                    movie_to_user_dict[movie] = set()
                movie_to_user_dict[movie].add(user)
        for movie, users in movie_to_user_dict.items():
            for user in users:
                self.similarity_matrix.setdefault(user, collections.defaultdict(int))
                # Brute froce to construct the matrix
                for other_user in users:
                    if user == other_user:
                        continue
                    self.similarity_matrix[user][other_user] += 1
        for user, related_users in self.similarity_matrix.items():
            for other_user, count in related_users.items():
                # cosine similarity
                self.similarity_matrix[user][other_user] = count / math.sqrt(len(self.training_dataset[user]) *
                                                                             len(self.training_dataset[other_user]))

    def test(self):
        matched_count = 0  # Number of the recommended movies matching the true test movie
        test_total = 0  # Total number of test movie
        recommended_total = 0  # Total number of recommended movie
        NDCG = 0
        MAE = 0
        RMSE = 0
        predict_total = 0
        valid_count = 0

        for user in self.training_dataset:
            test_movies = self.testing_dataset.get(user, {})
            recommended_movies, similar_users = self.recommend(user)
            pos = 0
            i = 0
            DCG = 0
            IDCG = 0
            for movie, _ in recommended_movies:
                pos += 1
                if movie in test_movies:
                    matched_count += 1
                    i += 1
                    DCG += 1 / math.log(1 + pos, 2)
                    IDCG += 1 / math.log(1 + i, 2)
                    self.rating_predict_matrix.setdefault(user, {})
                    self.rating_predict_matrix[user][movie] = self.predict_rating(movie, similar_users)
                    predict_total += 1
            test_total += len(test_movies)
            recommended_total += self.top_n_recommendation
            if IDCG > 0:
                NDCG += DCG / IDCG
                valid_count += 1

        # Evaluate recommendation
        precision = matched_count / (recommended_total * 1.0)
        recall = matched_count / (test_total * 1.0)
        f_measure = 2 * precision * recall / (precision + recall)
        NDCG /= valid_count

        # Evaluate predicted rating
        for user, movies in self.rating_predict_matrix.items():
            for movie in movies:
                diff = self.rating_predict_matrix[user][movie] - self.testing_dataset[user][movie]
                MAE += abs(diff)
                RMSE += pow(diff, 2)
        MAE /= predict_total
        RMSE = math.sqrt(RMSE / predict_total)

        print('Precision=%f, Recall=%f F-measure=%f, NDCG=%f' % (precision, recall, f_measure, NDCG))
        print('MAE=%f, RMSE=%f' % (MAE, RMSE))

    def recommend(self, user):
        rated_movies = self.training_dataset[user]  # These movie should be excluded from the recommendation
        recommendation_list = {}
        top_k_similar_users = sorted(self.similarity_matrix[user].items(), key=lambda x: x[1],
                                     reverse=True)[0: self.top_k_similar_user]
        for similar_user, similarity in top_k_similar_users:
            for movie in self.training_dataset[similar_user]:
                if movie in rated_movies:
                    continue
                recommendation_list.setdefault(movie, 0)
                recommendation_list[movie] += similarity
        return sorted(recommendation_list.items(), key=lambda x: x[1], reverse=True)[0: self.top_n_recommendation], \
               top_k_similar_users

    def predict_rating(self, movie, users):
        rating_count = 0
        rating_total = 0
        user_list = [u[0] for u in users]
        for user in user_list:
            if movie in self.training_dataset[user].keys():
                rating_count += 1
                rating_total += self.training_dataset[user][movie]
        return rating_total / (rating_count * 1.0)


user_based_cf = UserBasedCollaborativeFiltering()
user_based_cf.generate_dataset('./data/ml-latest-small/ratings.csv')
user_based_cf.calculate_user_similarity()
user_based_cf.test()