#### recommendation for TV

##### step 1
edit ./etc/config.ini

##### step 2
``` 
python ./tools/preprocess_data.py
``` 
> get data from mongodb and format

##### step 3
```
python ./tools/standard_douban_recommend.py
```
> get standard recommend result of douban from mongodb

##### step 4
```
    python ./rec_content_based.py
```
> calculate recommend data result and write into redis

