import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
import logging
import math
import argparse


logging.basicConfig(filename='model_contentbased.log',
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=-1, type=int)
argparams = parser.parse_args()


class TFIDF:
    def __init__(self):
        self.movies = {}
        self.w2i = {}
        self.movieId2i = {}
        self.i2movieId = {}
        self.mat = None
        self.vocab = set()
        self.idf = None
        self.similarity_matrix = None
        self.sorted_sim_mat = None
        self.n_movies = 0

    def add_movie(self, movieId, words):
        movieId = int(movieId)
        content = self.movies.get(movieId, [])
        content.extend(words)
        self.movies[movieId] = content
        self.vocab = self.vocab.union(content)

    def TFIDF(self):
        self.n_movies = len(self.movies)
        self.n_vocab = len(self.vocab)
        self.mat = np.zeros((self.n_movies, self.n_vocab), dtype=np.float32)
        for movieId in self.movies:
            if movieId not in self.movieId2i:
                self.movieId2i[movieId] = len(self.movieId2i)
                self.i2movieId[self.movieId2i[movieId]] = movieId
            for word in self.movies[movieId]:
                if word not in self.w2i:
                    self.w2i[word] = len(self.w2i)
                self.mat[self.movieId2i[movieId], self.w2i[word]] += 1
        self.mat = np.log(self.mat + 1)
        df = np.count_nonzero(self.mat, axis=0)
        #self.idf = np.log(self.n_movies / (1 + df)) + 1
        self.idf = np.log(self.n_movies / df)
        self.mat *= self.idf[None,:]

    def build_similarity_matrix(self):
        norm_mat = (self.mat * self.mat).sum(1, keepdims=True) ** .5
        self.similarity_matrix = self.mat @ self.mat.T / norm_mat / norm_mat.T
        self.sorted_sim_mat = np.apply_along_axis(np.argsort, 1, self.similarity_matrix)

    def movie_sim(self, a, b):
        veca = self.mat[self.movieId2i[a]]
        vecb = self.mat[self.movieId2i[b]]
        return self.cos_sim(veca, vecb)

    def cos_sim(self, u, v):
        return np.dot(u, v) / self.norm(u) / self.norm(v)

    def norm(self, u):
        return np.sqrt(np.sum(u*u))

    def query_sent(self, words):
        self.vec = np.zeros((self.n_vocab,), dtype=np.float32)
        for word in words:
            if word not in self.w2i:
                continue
            self.vec[self.w2i[word]] += 1
        self.vec = np.log(self.vec + 1)
        self.vec *= self.idf
        scores = np.apply_along_axis(self.cos_sim, 1, arr=self.mat, v=self.vec)
        pairs = [(self.i2movieId[idx], scores[idx]) for idx in np.argsort(scores)[::-1]]
        return pairs

    def query_vec(self, vec, mode = 0, k = 10, watched_list = None):
        # watched_list: a list of movieId which a user has watched
        # while watched_list is not None
        # ├── mode = 0: return k movies that is similar to the vec that a user hasn't watched
        # └── mode = 1: return k movies that a user has watched
        scores = np.apply_along_axis(self.cos_sim, 1, arr=self.mat, v=vec)
        if watched_list is None:
            pairs = [(self.i2movieId[idx], scores[idx]) for idx in np.argsort(scores)[::-1]]
        else:
            if mode == 0:
                pairs = [(self.i2movieId[idx], scores[idx]) for idx in np.argsort(scores)[::-1] if not self.i2movieId[idx] in watched_list]
            else:
                pairs = [(self.i2movieId[idx], scores[idx]) for idx in np.argsort(scores)[::-1] if self.i2movieId[idx] in watched_list]
        return pairs

    def query_movie(self, movieId, mode = 0, k = 10, watched_list = None):
        ret = self.sorted_sim_mat[self.movieId2i[movieId], :]
        return ret


class UserProfile:
    def __init__(self):
        self.profiles = {}
        self.ratings = {}
        self.count = {}

    def build(self, df, tfidf):
        rating_user_sum = df.rating.groupby(df.userId).sum().to_dict()
        for index, row in df.iterrows():
            userId = int(row['userId'])
            movieId = int(row['movieId'])
            rating = row['rating']
            vec_movie = tfidf.mat[tfidf.movieId2i[movieId]]
            vec_user = self.profiles.get(userId, np.zeros(vec_movie.shape))
            vec_user += rating / rating_user_sum[userId] * vec_movie
            self.profiles[userId] = vec_user
            rating_user = self.ratings.get(userId, {})
            rating_user[movieId] = rating
            self.ratings[userId] = rating_user

    def build_2(self, df, tfidf, k=20):
        # Another user profile created by top k rated movies
        #rating_user_sum = df.rating.groupby(df.userId).sum().to_dict()
        for index, row in df.sort_values(by='rating', ascending=False).iterrows():
            userId = int(row['userId'])
            movieId = int(row['movieId'])
            rating = row['rating']
            if self.count.get(userId, 0) > k:
                continue
            self.count[userId] = self.count.get(userId, 0) + 1
            vec_movie = tfidf.mat[tfidf.movieId2i[movieId]]
            vec_user = self.profiles.get(userId, np.zeros(vec_movie.shape))
            vec_user += vec_movie
            self.profiles[userId] = vec_user
            rating_user = self.ratings.get(userId, {})
            rating_user[movieId] = rating
            self.ratings[userId] = rating_user
        for userId in self.profiles.keys():
            self.profiles[userId] /= self.count[userId]


def compute_tfidf(path, tokenizer = RegexpTokenizer(r'\w+')):
    # Compute TFIDF scores using movie.csv and tags.csv
    tfidf = TFIDF()
    stopwords_set = set(stopwords.words('english'))
    df_movies = pd.read_csv(path + 'movies.csv')
    df_tags = pd.read_csv(path + 'tags.csv')
    for index, row in df_movies.iterrows():
        movieId = row['movieId']
        content = row['genres'] + ' ' + row['title']
        tokens = [tok.lower() for tok in tokenizer.tokenize(content) if tok not in stopwords_set]
        tfidf.add_movie(movieId, tokens)
    for index, row in df_tags.iterrows():
        movieId = row['movieId']
        content = row['tag']
        tokens = [tok.lower() for tok in tokenizer.tokenize(content) if tok not in stopwords_set]
        tfidf.add_movie(movieId, tokens)
    tfidf.TFIDF()
    return tfidf


def recommend(user_list, user_profile, tfidf, k=10):
    results = {}
    for userId in user_list:
        logger.info('Recommending for user {}'.format(userId))
        vec_user = user_profile.profiles[userId]
        watched_list = set(user_profile.ratings[userId].keys())
        recommended_movies = tfidf.query_vec(vec_user, mode = 0, k = k, watched_list = watched_list)
        #logger.info('Recommend movies: {}'.format(recommended_movies))
        for (movieId, score) in recommended_movies[:k]:
            most_similar_rated_movies = tfidf.query_movie(movieId, mode = 1, k = 10, watched_list = watched_list)
            i = 0
            j = 0
            predicted_rating = 0.0
            for i in range(tfidf.n_movies):
                m = tfidf.i2movieId[most_similar_rated_movies[i]]
                if m in watched_list:
                    predicted_rating += user_profile.ratings[userId][m]
                    j += 1
                    if j > 10:
                        break
            predicted_rating /= j
            #predicted_rating = np.mean([user_profile.ratings[userId][tfidf.i2movieId[idx]] for idx in most_similar_rated_movies])
            #logger.info('Predicted rating for {}: {}'.format(movieId, predicted_rating))
            predictions = results.get(userId, [])
            predictions.append((movieId, predicted_rating))
            results[userId] = predictions
    return results


def evaluate(predictions, df):
    matched_count = 0  # Number of the recommended movies matching the true test movie
    test_total = 0  # Total number of test movie
    recommended_total = 0  # Total number of recommended movie
    predict_total = 0
    NDCG = 0
    MAE = 0
    RMSE = 0

    ratings = df.groupby(df.userId)[['movieId', 'rating']].apply(lambda x: x.set_index('movieId').to_dict(orient='index')).to_dict()

    for userId in predictions:
        prediction_user = predictions[userId]
        test_movies = ratings[userId]
        pos = 0
        i = 0
        DCG = 0
        IDCG = 0
        #logger.info(test_movies)
        for movieId, predicted_rating in prediction_user:
            pos += 1
            if movieId in test_movies:
                matched_count += 1
                i += 1
                DCG += 1 / math.log(1 + pos, 2)
                IDCG += 1 / math.log(1 + i, 2)

                diff = predicted_rating - test_movies[movieId]['rating']
                MAE += abs(diff)
                RMSE += pow(diff, 2)
                predict_total += 1
        recommended_total += len(prediction_user)
        test_total += len(test_movies)
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


if __name__ == '__main__':
    kFold = 5
    path = 'data/'
    path_movielens = path + 'ml-latest-small/'
    tfidf = compute_tfidf(path_movielens)
    tfidf.build_similarity_matrix()
    if argparams.id == -1:
        for i in range(kFold):
            df_train = pd.read_csv(path + 'trainset_{}.csv'.format(i))
            df_test = pd.read_csv(path + 'testset_{}.csv'.format(i))
            logger.info('Loaded trainset_{}.csv, shape:{}'.format(i, df_train.shape))
            logger.info('Loaded testset_{}.csv, shape:{}'.format(i, df_test.shape))
            user_profile = UserProfile()
            #user_profile.build(df_train, tfidf)
            user_profile.build_2(df_train, tfidf, k=20)
            user_list = set(df_test.userId.to_list())
            recommendations = recommend(user_list, user_profile, tfidf, 15)
            evaluate(recommendations, df_test)
    else:
        logger.info('Training and testing on id {}'.format(argparams.id))
        i = argparams.id
        df_train = pd.read_csv(path + 'trainset_{}.csv'.format(i))
        df_test = pd.read_csv(path + 'testset_{}.csv'.format(i))
        logger.info('Loaded trainset_{}.csv, shape:{}'.format(i, df_train.shape))
        logger.info('Loaded testset_{}.csv, shape:{}'.format(i, df_test.shape))
        user_profile = UserProfile()
        user_profile.build(df_train, tfidf)
        user_list = set(df_test.userId.to_list())
        recommendations = recommend(user_list, user_profile, tfidf, 10)
        evaluate(recommendations, df_test)


