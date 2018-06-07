# encoding=utf-8
from urllib2 import urlopen
from pandas import Series
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')


def pullword(context, threshold=0.5, debug=False, num=3):
    '''
    pull word from context
    using online free service by liangbo

    Parameters
    ----------
    context : string
    threshold : float [0, 1]
    debug : True or False
    num : max num of words

    Returns
    -------
    result : Series
    '''
    pullword = r'http://api.pullword.com'
    debug_flag = 1 if debug is True else 0
    url = ('{0}/get.php?source={1}&param1={2}&param2={3}'
           .format(pullword, context.strip(), threshold, debug_flag))
    # print url
    raw = urlopen(url).read()
    w = raw.strip().split(u'\r\n')
    if debug is True:
        data = {}
        for s in w:
            [word, probability] = s.strip().split(u':')
        data[word] = float(probability)
        features = Series(data).sort_values(ascending=False)[:num]
    elif debug is False:
        features = Series(w)[:num]
    else:
        raise ValueError(debug)
    return features.values


def extract_text_feature(dataset, threshold=0.9, maxnum=3):
    '''
    extract documents tf-idf features

    Parameters
    ----------
    dataset : list of string with blackspace as separator
    threshold : float [0, 1]
    maxnum : max num of words for each document

    Returns
    -------
    data : sparse matrix
    features : list
    '''
    data_list = []
    if not isinstance(dataset, (list, Series)):
        raise TypeError('data must be list')
    for document in dataset:
        data_list.append(' '.join(pullword(document, threshold=threshold, num=maxnum)))
    vectorizer = TfidfVectorizer(smooth_idf=True)
    data = vectorizer.fit_transform(data_list)
    features = vectorizer.get_feature_names()
    return data, features


if __name__ == '__main__':
    context = '''正义红师'''
    for w in pullword(context):
        print(w)

    # data, features = extract_text_feature([s1, s2])
    # print features
    # print data.toarray()
