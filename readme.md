#### recommendation for TV

##### step 1
edit ./etc/config.ini

##### step 2
``` 
python ./tools/preprocess.py
``` 
> get data from mongodb and format

##### step 3 [optional]
```
python ./tools/standard_douban_recommend.py
```
> get standard recommend result of douban from mongodb

##### step 4 [optional]
```
python ./tools/get_prefix.py
```
> get prefix map from mongo

##### step 5
```
[predict mode]
python ./rec_content_based.py predict

[train mode]
python ./rec_content_based.py train
```
> calculate recommend data result and write into redis. 
> if you run mode 'train', the weight of train will be work in this time 
> and history weight should be modify manually if you like the 'train' weight and want to affect next 'work' mode.
> prefix mode like work mode ,but result add prefix ahead of value.


crontab  
per week

### Since the change of database "chiq_video_converge" not support douban, the mode "train" not work.
