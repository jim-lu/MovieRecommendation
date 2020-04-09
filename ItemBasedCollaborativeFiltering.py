import numpy as np
import random
import math
import copy
import csv


class ItemBasedCollaborativeFiltering:

    def __init__(self, top_k_similar_movie=20, top_n_recommendation=10):
        self.training_dataset = {}
        self.testing_dataset = {}
        self.top_k_similar_movie = top_k_similar_movie
        self.top_n_recommendation = top_n_recommendation
        self.similar_matrix = {}
        self.sorted_similar_matrix = {}
        self.adjusted_rating = {}
        self.rating_predict_matrix = {}
        self.movie_list = {}

    def generate_dataset(self, filename, split_param=0.8):
        ratings = np.loadtxt(filename, dtype=np.str, delimiter=',')[1:].tolist()
        for line in ratings:
            user_id = line[0]
            movie_id = line[1]
            rating = line[2]
            if random.random() < split_param:
                self.training_dataset.setdefault(movie_id, {})
                self.training_dataset[movie_id][user_id] = float(rating)
            else:
                self.testing_dataset.setdefault(movie_id, {})
                self.testing_dataset[movie_id][user_id] = float(rating)

    def load_all_movie(self):
        with open('./ml-latest-small/movies.csv', encoding='utf8') as movies:
            reader = csv.reader(movies, delimiter=',', quotechar='"', skipinitialspace=True)
            next(reader)
            for line in reader:
                # print(line)
                self.movie_list[line[0]] = line[1]

    def calculate_movie_similarity(self):
        rating_mean = {}
        norm = {}
        self.adjusted_rating = copy.deepcopy(self.training_dataset)
        for movie, users in self.adjusted_rating.items():
            self.similar_matrix[movie] = {}
            rating_mean[movie] = sum(users.values()) / len(users)
            norm[movie] = 0
            for user in users:
                self.adjusted_rating[movie][user] -= rating_mean[movie]
                norm[movie] += pow(self.adjusted_rating[movie][user], 2)
            # print('The adjusted rating for movie %s is ' % movie, self.adjusted_rating[movie])
            norm[movie] = math.sqrt(norm[movie])
            # print('The norm for movie %s is %f' % (movie, norm[movie]))
        for movie, users1 in self.adjusted_rating.items():
            for other_movie, users2 in self.adjusted_rating.items():
                if movie == other_movie or (other_movie in self.similar_matrix.keys()
                                            and movie in self.similar_matrix[other_movie].keys()):
                    continue
                vector_sum = 0
                for user in set(users1.keys()).intersection(set(users2.keys())):
                    vector_sum += self.adjusted_rating[movie][user] * self.adjusted_rating[other_movie][user]
                if norm[movie] == 0 or norm[other_movie] == 0:
                    cosine_similarity = 0
                else:
                    cosine_similarity = vector_sum / (norm[movie] * norm[other_movie])
                self.similar_matrix[movie][other_movie] = cosine_similarity
                self.similar_matrix[other_movie][movie] = cosine_similarity
                print("Set up similarity between movie %s and %s. The similarity is %f." % (movie, other_movie, cosine_similarity))
            self.sorted_similar_matrix[movie] = sorted(self.similar_matrix[movie].items(), key=lambda x: x[1], reverse=True)

    def test(self):
        matched_count = 0  # Number of the recommended movies matching the true test movie
        test_total = 0  # Total number of test movie
        recommended_total = 0  # Total number of recommended movie
        NDCG = 0
        MAE = 0
        RMSE = 0

        for user in self.training_dataset:
            test_movies = self.testing_dataset.get(user, {})
            recommended_movies = self.recommend(user)
            print('Finished recommending for user %s. Rating list: ' % user)
            print(recommended_movies)
            pos = 0
            i = 0
            DCG = 0
            IDCG = 0
            for recommendation in recommended_movies:
                pos += 1
                if recommendation['movie'] in test_movies:
                    matched_count += 1
                    i += 1
                    DCG += 1 / math.log(1 + pos, 2)
                    IDCG += 1 / math.log(1 + i, 2)
                    self.rating_predict_matrix.setdefault(recommendation['movie'], {})
                    self.rating_predict_matrix[recommendation['movie']][user] = recommendation['rating']
            print('Found %d match for user %s.' % (i, user))
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
        for user, movies in self.rating_predict_matrix.items():
            for movie in movies:
                diff = self.rating_predict_matrix[movie][user] - self.testing_dataset[movie][user]
                MAE += abs(diff)
                RMSE += pow(diff, 2)
        MAE /= recommended_total
        RMSE = math.sqrt(RMSE / recommended_total)
        print('MAE=%f, RMSE=%f' % (MAE, RMSE))

    def recommend(self, user):
        recommendation_list = []
        rated_movies = self.training_dataset[user]  # These movie should be excluded from the recommendation
        for movie in self.movie_list.keys():
            if movie in rated_movies or movie not in self.similar_matrix.keys() or len(self.similar_matrix[movie]) == 0:
                continue
            # sorted_similar_movies = self.similar_matrix[movie]
            weighted_rating = 0
            similarity_sum = 0
            count = 0
            for item in self.sorted_similar_matrix[movie]:
                similar_movie = item[0]
                similarity = item[1]
                if user in self.training_dataset[similar_movie].keys():
                    weighted_rating += similarity * self.training_dataset[similar_movie][user]
                    similarity_sum += similarity
                    count += 1
                if count == 2 or count == self.top_k_similar_movie:
                    if similarity_sum != 0:
                        recommendation_list.append({'movie': movie, 'rating': round(weighted_rating / similarity_sum, 2)})
                        print('Finish predict rating for %s. The rating is %f' % (movie, weighted_rating / similarity_sum))
                    break
        return sorted(recommendation_list, key=lambda x: x['rating'], reverse=True)[0: self.top_n_recommendation]


item_based_cf = ItemBasedCollaborativeFiltering()
item_based_cf.generate_dataset('./ml-latest-small/ratings.csv')
item_based_cf.load_all_movie()
item_based_cf.calculate_movie_similarity()
item_based_cf.test()