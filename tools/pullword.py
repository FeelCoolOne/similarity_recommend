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
    url = '{0}/get.php?source={1}&param1={2}&param2={3}'.format(pullword,
                                                                context.strip(),
                                                                threshold,
                                                                debug_flag)
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
    if not isinstance(dataset, list):
        raise TypeError('data must be list')
    for document in dataset:
        data_list.append(' '.join(pullword(document, threshold=threshold, num=maxnum)))
    vectorizer = TfidfVectorizer(smooth_idf=True)
    data = vectorizer.fit_transform(data_list)
    features = vectorizer.get_feature_names()
    return data, features


if __name__ == '__main__':
    context = '''北京遇上西雅图他们不喊累、不闹情绪、'''
    print pullword(context)

    s1 = u'''人工智能将在未来十年取代一半人的工作。在需要考虑少于5秒的领域，人根本不是机器的对手，他们不喊累、不闹情绪、犯错率极低。但也正是人工智能这样的优点，证明我们还有机会，
     '''
    s2 = u'''是一种用于信息检索与数据挖掘的常用加权技术.简介TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中'''
    data, features = extract_text_feature([s1, s2])
    print data.toarray()
