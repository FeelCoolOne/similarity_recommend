# encoding=utf-8
import matplotlib.pyplot as plt
import matplotlib
import sys
from sklearn.linear_model import LassoCV, Lasso
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')


def analysis_data(data):
    matplotlib.rc('xtick', labelsize=16)
    # dataframe = data.fillna('null')
    dataframe = data
    for index in dataframe.columns:
        plt.figure(index)
        plt.title('{0}统计分布'.format(index))
        plt.xlabel(index, fontsize=16)
        plt.ylabel('num')
        c = dataframe[index].value_counts(dropna=True)
        c.sort_values(ascending=False)[:50].plot('bar')
        try:
            dataframe[index].fillna('null', inplace=True)
            print '{0}":"{1}'.format(index, float(dataframe[index].value_counts(dropna=False)['null']) / dataframe[index].size)
        except:
            print index, 'no nan'
    plt.show()


def select_feature(data, y):
    '''array(n_classes, n_features)'''
    # from sklearn.model_selection import train_test_split
    alphas = [10, 5, 2, 1, 0.5, 0.1, 0.01]
    # X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
    # scores = [Lasso(alpha=alpha).fit(X_train, y_train).score(X_test, y_test)for alpha in alphas]
    # alpha = alphas[scores.index(max(scores))]
    # regr = Lasso(alpha=alpha).fit(data, y)
    # return regr.coef_
    lasso_cv = LassoCV(alphas=alphas, random_state=0)
    lasso_cv.fit(data, y)
    return lasso_cv.coef_


def preprocessing(X):
    from sklearn.preprocessing import scale
    X_scaled = scale(X)
    return X_scaled


def calculate_cosine_similarity(X, weight):
    '''
    Parameters:
        X: numpy.ndarray, shape: (num_sample, num_feature)
        weight: numpy.ndarray, shape: (num_feature,)
    Return：
        array with shape (num_sample, num_sample)
    '''
    from numpy.linalg import norm
    X *= weight
    norms = norm(X, axis=0)
    tmp = X.dot(X.T)
    tmp = tmp.astype(float) / norms
    tmp.T /= norms
    return tmp


if __name__ == '__main__':
    import cPickle as pickle
    from get_id_model import Video
    data = {}
    with open('../data/id.dat', 'rb') as f:
        data = pickle.load(f)
    if len(data) == 0:
        print 'data source error'
        exit()
    handler = Video()
    dataframe = data['tv']
    analysis_data(dataframe)
    y = dataframe['grade_score'].values
    dataframe.drop('grade_score', inplace=True, axis=1)
    X = dataframe.values
    weight = select_feature(X, y)
    for index in range(dataframe.columns.size):
        print dataframe.columns[index], weight[index]
# dataframe = handler.clean_data(dataframe)
