# encoding=utf-8
from urllib2 import urlopen
from pandas import Series
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

    Returns
    -------
    result : Series
    '''
    pullword = r'http://api.pullword.com'
    debug_flag = 1 if debug is True else 0
    url = '{0}/get.php?source={1}&param1={2}&param2={3}'.format(pullword,
                                                                context,
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


if __name__ == '__main__':
    context = u'北京遇上西雅图'
    print pullword(context)
