# encoding:utf-8
"""
============================================================
Fetch and clean media data before calculation of similarity
============================================================

fetch data from mongodb.
use charactor  'language', 'country', 'writer', 'director', 'tag', 'actor', 'year', 'score'

Script output:
    data/{model_name}.dat
        local clean data for calculation to reading directly from locals
        make local cache since frequence of similarity calculation
        be greater than of the scripts that fetch and clean media data

Created by:
    yonggang Huang
In:
    05-25-2017
"""


import re
import traceback
from pandas import DataFrame
from datetime import datetime, date


class Formatter:

    def __init__(self):
        self.pat = re.compile(r'\s*[/\\|]\s*')

    def _filter_id(self, document):
        record = document
        id = None
        if record['tencent'] == '1' and 'pp_tencent' in record:
            id = record['pp_tencent']['tencentId']
        elif record['youpeng'] == '1' and 'pp_youpeng' in record:
            id = record['pp_youpeng']['yp_id']
        '''
        elif record['iqiyi'] == '1':
            id = '666666'
        '''
        return id

    def _filter_pay_status(self, document):
        record = document
        vip = 0
        if record['tencent'] == '1' and record['pp_tencent']['pay_status'].strip() in [u'用券', u'会员点播', u'会员免费']:
            vip = 1
        elif record['youpeng'] == '1':
            vip = 1
        else:
            vip = 0
        return vip

    def _filter_year(self, document):
        try:
            if 'd_issue' in document and document.get('d_issue').strip() != '':
                d_issue = document.get('d_issue').strip()
                year = datetime.strptime(d_issue, "%Y-%m-%d").year
            elif 'issue' in document and document.get('issue').strip() != '':
                issue = document.get('issue').strip()
                year = datetime.strptime(issue, "%Y-%m-%d").year
            elif 'year' in document:
                year = str(document.get('year'))
                year = datetime.strptime(year, "%Y").year
            if year > date.today().year or year < 1900:
                raise Exception("year error : {0}".format(year))
            year = int(year)
        except Exception:
            year = None
        return year

    def _filter_tag(self, document):
        tag = document.get('d_type', '').strip()
        return self.pat.split(tag)

    def _filter_language(self, document):
        language = document.get('language', '').strip()
        return self.pat.split(language)

    def _filter_country(self, document):
        country = document.get('country', '').strip()
        return self.pat.split(country)

    def _filter_categorys(self, document):
        categorys = list()
        for item in document.get('categorys', []):
            categorys.append(item['name'])
        return categorys

    def _filter_score(self, document):
        if 'd_grade_score' in document and document.get('d_grade_score').strip() != '':
            score = float(document.get('d_grade_score'))
        elif 'grade_score' in document:
            score = float(document.get('grade_score'))
        else:
            score = None
        return score

    def _filter_duration(self, document):
        duration = 0
        if document['tencent'] == 1 and 'duration' in document['pp_tencent']:
            duration = document['pp_tencent']['duration']
        elif document['youpeng'] == 1 and 'duration' in document['pp_youpeng']:
            duration = document['pp_youpeng']['duration']
        elif document['iqiyi'] == 1 and 'duration' in document['pp_iqiyi']:
            duration = document['pp_iqiyi']['duration']
        else:
            duration = document['duration'] if 'duration' in document else None
        return duration

    def _handle_all_attr(self, document):
        data = dict()
        document = dict(document)
        data['cover_id'] = self._filter_id(document)
        # data['model'] = document.get('model', none_label)
        # data['alias'] = document.get('alias', none_label)
        # data['episodes'] = document.get('episodes', -1)
        # data['enname'] = document.get('enName', none_label)
        # data['name'] = document.get('name', none_label).strip()
        data['duration'] = self._filter_duration(document)
        data['director'] = self.pat.split(document.get('director').strip())  # list
        data['actor'] = self.pat.split(document.get('cast', '').strip())[:7]  # list
        data['writer'] = self.pat.split(document.get('writer', '').strip())  # list
        data['score'] = self._filter_score(document)  # float
        data['tag'] = self._filter_tag(document)  # list
        data['country'] = self._filter_country(document)  # list
        data['language'] = self._filter_language(document)  # list
        # data['definition'] = document.get('definition', None)  # int
        data['year'] = self._filter_year(document)  # int
        # data['vip'] = self._filter_pay_status(document)  # bool
        # 上架
        # data['enable'] = document.get('enable', None)  # str
        # data['isClip'] = document.get('isClip', '-1')
        data['categorys'] = self._filter_categorys(document)
        # data['pp_tencent'] = document.get('pp_tencent', '{}')
        # data['pp_iqiyi'] = document.get('pp_iqiyi', '{}')
        # data['pp_youpeng'] = document.get('pp_youpeng', '{}')
        # data['tencent'] = document.get('tencent', '-1')
        # data['iqiyi'] = document.get('iqiyi', '-1')
        # data['youpeng'] = document.get('youpeng', '-1')
        # data['focus'] = document.get('focus', '').strip()
        for label in data:
            if isinstance(data[label], list):
                data[label] = '|'.join(data[label])
        return data

    def process(self, documents):
        data_df = list()
        for doc in documents:
            try:
                data = self._handle_all_attr(doc)
            except Exception:
                print('clean label get error')
                raise Exception('{0}'.format(traceback.print_exc()))
            data_df.append(data)
        return DataFrame(data=data_df)
