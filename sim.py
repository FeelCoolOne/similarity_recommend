# encoding:utf-8

from numpy.linalg import norm
from pandas import DataFrame
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import scale


class Sim(object):

    def __init__(self, X, y, index):
        '''data float array'''
        self.data = X
        self.target = y
        self.index = index
        self.num = 20
        self.similar_frame = None

    def select_feature(self, data, y):
        '''array(n_classes, n_features)'''
        # from sklearn.model_selection import train_test_split
        # alphas = [10, 5, 2, 1, 0.5, 0.1, 0.01]
        # X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
        # scores = [Lasso(alpha=alpha).fit(X_train, y_train).score(X_test, y_test)for alpha in alphas]
        # alpha = alphas[scores.index(max(scores))]
        # regr = Lasso(alpha=alpha).fit(data, y)
        # return regr.coef_
        # lasso_cv = LassoCV(alphas=alphas, positive=True, random_state=0)
        lasso_cv = LassoCV(positive=True, random_state=0)
        lasso_cv.fit(data, y)
        return lasso_cv.coef_

    def preprocessing(self):
        self.data = scale(self.data)

    def calculate_cosine_similarity(self, weight):
        '''
        Parameters:
            X: numpy.ndarray, shape: (num_sample, num_feature)
            weight: numpy.ndarray, shape: (num_feature,)
        Return：
            array with shape (num_sample, num_sample)
        '''
        X = self.data * weight
        sample_norms = norm(X, axis=1)
        XX_norm = sample_norms * sample_norms.reshape(sample_norms.size, 1)
        sim_array = (X * X.T / XX_norm) * 0.5 + 0.5
        self.similar_frame = DataFrame(sim_array, columns=self.index, index=self.index)

    def process(self):
        self.preprocessing()
        weight = self.select_feature(self.data, self.target)
        self.calculate_cosine_similarity(weight)
        for id in self.index:
            yield id, self.calculate_output(id)

    def calculate_output(self, cover_id):
        sorted_result = self.similar_frame.sort_values(by=cover_id, ascending=False)[cover_id]
        result = {}
        for index in range(len(sorted_result)):
            if sorted_result.index[index] == cover_id:
                continue
            if index == self.num + 1:
                break
            result[sorted_result.index[index]] = sorted_result[index]
        format_result = {'Results': result, 'V': '2.0.0'}
        return format_result
