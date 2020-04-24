import numpy as np
import math


class ItemBasedCollaborativeFiltering:

    def __init__(self, top_k_similar_movie=10, top_n_recommendation=10):
        self.training_dataset = {}
        self.testing_dataset = {}
        self.top_k_similar_movie = top_k_similar_movie
        self.top_n_recommendation = top_n_recommendation
        self.similar_matrix = None
        self.popularity_matrix = {}
        self.user_number = 0
        self.movie_number = 0
        self.user_list = []
        self.movie_list = []
        self.user_id_dict = {}
        self.movie_id_dict = {}

    def generate_dataset(self, training_set_file, testing_set_file):
        training_ratings = np.loadtxt(training_set_file, dtype=np.str, delimiter=',')[1:].tolist()
        testing_ratings = np.loadtxt(testing_set_file, dtype=np.str, delimiter=',')[1:].tolist()
        user_set = set()
        movie_set = set()
        for line in training_ratings:
            user_id = int(line[1])
            movie_id = int(line[2])
            rating = float(line[3])
            self.training_dataset.setdefault(user_id, {})
            self.training_dataset[user_id][movie_id] = rating
            user_set.add(user_id)
            movie_set.add(movie_id)
        for line in testing_ratings:
            user_id = int(line[1])
            movie_id = int(line[2])
            rating = float(line[3])
            self.testing_dataset.setdefault(user_id, {})
            self.testing_dataset[user_id][movie_id] = rating
        self.user_number = len(user_set)
        self.movie_number = len(movie_set)
        self.user_list = sorted(list(user_set))
        self.movie_list = sorted(list(movie_set))
        for i in range(len(self.user_list)):
            self.user_id_dict[self.user_list[i]] = i
        for i in range(len(self.movie_list)):
            self.movie_id_dict[self.movie_list[i]] = i

    def calculate_movie_similarity(self):
        movie_to_user_matrix = np.zeros([self.movie_number, self.user_number])
        self.similar_matrix = np.zeros([self.movie_number, self.movie_number])
        np.fill_diagonal(self.similar_matrix, 1)
        for user, movies in self.training_dataset.items():
            for movie in movies:
                movie_to_user_matrix[self.movie_id_dict[movie]][self.user_id_dict[user]] = self.training_dataset[user][movie]
        row1 = 0
        while row1 < self.movie_number - 2:
            if row1 % 10 == 0:
                print(row1)
            denorm_sum = 0
            norm = 1
            row2 = row1 + 1
            while row2 < self.movie_number - 1:
                col = -1
                while col < self.user_number - 1:
                    col += 1
                    if movie_to_user_matrix[row1][col] == 0 or movie_to_user_matrix[row2][col] == 0:
                        continue
                    denorm_sum += movie_to_user_matrix[row1][col] * movie_to_user_matrix[row2][col]
                    norm += math.log(pow(movie_to_user_matrix[row1][col], 2) + pow(movie_to_user_matrix[row2][col], 2))
                if denorm_sum == 0:
                    self.similar_matrix[row1][row2] = 0
                    self.similar_matrix[row2][row1] = 0
                    row2 += 1
                    continue
                self.similar_matrix[row1][row2] = math.log(denorm_sum) / norm
                self.similar_matrix[row2][row1] = math.log(denorm_sum) / norm
                row2 += 1
            row1 += 1

    def test(self):
        matched_count = 0  # Number of the recommended movies matching the true test movie
        test_total = 0  # Total number of test movie
        recommended_total = 0  # Total number of recommended movie
        predict_total = 0
        NDCG = 0
        MAE = 0
        RMSE = 0

        for user in self.training_dataset:
            if user % 10 == 0:
                print("Recommending for ", user)
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
        for movie in self.movie_list:
            if movie in rated_movies:
                continue
            movie_index = self.movie_id_dict[movie]
            row = self.similar_matrix[movie_index]
            denorm_sum = 0
            norm = 0
            i = 1
            count = 0
            while count < self.top_k_similar_movie - 1 and i < self.movie_number - 1:
                if row[i] == 0 or row[i] == 1 or self.movie_list[i] not in self.training_dataset[user]:
                    i += 1
                    continue
                denorm_sum += row[i] * self.training_dataset[user][self.movie_list[i]]
                norm += row[i]
                i += 1
                count += 1
            if norm == 0:
                continue
            recommendation_list.setdefault(movie, 0)
            recommendation_list[movie] += denorm_sum / norm
        return sorted(recommendation_list.items(), key=lambda x: x[1], reverse=True)[0: self.top_n_recommendation]


item_based_cf = ItemBasedCollaborativeFiltering()
# item_based_cf.generate_dataset('./data/ml-latest-small/ratings.csv')
item_based_cf.generate_dataset('../data/trainset_4.csv', '../data/testset_4.csv')
item_based_cf.calculate_movie_similarity()
item_based_cf.test()
