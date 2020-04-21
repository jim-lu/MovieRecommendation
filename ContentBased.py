import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer

class TFIDF:
    def __init__(self):
        self.movies = {}
        self.w2i = {}
        self.movieId2i = {}
        self.i2movieId = {}
        self.mat = None
        self.vocab = set()
        self.idf = None

    def add_movie(self, movieId, words):
        content = self.movies.get(movieId, [])
        content.extend(words)
        self.movies[movieId] = content
        self.vocab = self.vocab.union(content)

    def TFIDF(self):
        self.mat = np.zeros((len(self.movies), len(self.vocab)), dtype=np.float32)
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
        self.idf = np.log(len(self.movies) / (1 + df)) + 1
        self.mat *= self.idf[None,:]

    def cos_sim(self, u, v):
        return np.dot(u, v) / self.norm(u) / self.norm(v)

    def norm(self, u):
        return np.sqrt(np.sum(u*u))

    def query(self, words):
        self.vec = np.zeros((len(self.vocab),), dtype=np.float32)
        for word in words:
            if word not in self.w2i:
                continue
            self.vec[self.w2i[word]] += 1
        self.vec = np.log(self.vec + 1)
        self.vec *= self.idf
        scores = np.apply_along_axis(self.cos_sim, 1, arr=self.mat, v=self.vec)
        pairs = [(self.i2movieId[idx], scores[idx]) for idx in np.argsort(scores)[::-1]]
        return pairs


class UserProfile:
    def __init__(self):
        self.profiles = {}

    def build(self, df, tfidf):
        rating_user_sum = df.rating.groupby(df.userId).sum().to_dict()
        for index, row in df.iterrows():
            userId = row['userId']
            movieId = row['movieId']
            rating = row['rating']
            vec_movie = tfidf.mat[tfidf.movieId2i[movieId]]
            vec_user = self.profiles.get(userId, np.zeros(vec_movie.shape))
            vec_user += rating / rating_user_sum[userId] * vec_movie
            self.profiles[userId] = vec_user


def compute_tfidf(path, tokenizer = RegexpTokenizer(r'\w+')):
    # Compute TFIDF scores using movie.csv and tags.csv
    tfidf = TFIDF()
    stopwords_set = set(stopwords.words('english'))
    df_movies = pd.read_csv(path + 'movies.csv')
    df_tags = pd.read_csv(path + 'tags.csv')
    for index, row in df_movies.iterrows():
        movieId = row['movieId']
        content = row['genres']
        tokens = [tok.lower() for tok in tokenizer.tokenize(content) if tok not in stopwords_set]
        tfidf.add_movie(movieId, tokens)
    for index, row in df_tags.iterrows():
        movieId = row['movieId']
        content = row['tag']
        tokens = [tok.lower() for tok in tokenizer.tokenize(content) if tok not in stopwords_set]
        tfidf.add_movie(movieId, tokens)
    tfidf.TFIDF()
    return tfidf


if __name__ == '__main__':
    kFold = 5
    path = 'data/'
    path_movielens = path + 'ml-latest-small/'
    tfidf = compute_tfidf(path_movielens)
    for i in range(kFold):
        df_train = pd.read_csv(path + 'trainset_{}.csv'.format(i))
        df_test = pd.read_csv(path + 'testset_{}.csv'.format(i))
        print('Loaded trainset_{}.csv, shape:{}'.format(i, df_train.shape))
        print('Loaded testset_{}.csv, shape:{}'.format(i, df_test.shape))
        user_profile = UserProfile()
        user_profile.build(df_train, tfidf)
        #recommend(df_train, DataFrame)
        break

