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
python ./tools/get_prefix.py
```
> get prefix map from mongo

##### step 5
```
[work mode]
python ./rec_content_based.py work

[train mode]
python ./rec_content_based.py train

[prefix mode]
python ./rec_content_based.py prefix
```
> calculate recommend data result and write into redis. 
> if you run mode 'train', the weight of train will be work in this time 
> and history weight should be modify manually if you like the 'train' weight and want to affect next 'work' mode.
> prefix mode like work mode ,but result add prefix ahead of value.
