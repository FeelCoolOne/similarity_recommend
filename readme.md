#### recommendation for TV
##### step 1
    edit ./etc/config.ini
##### step 2
    run ./tools/get_format_raw_data.py to get data from mongodb and format
#### step 3
    run ./rec_content_based.py to get recommend data result and write into redis

#### Bug list:
- slow rate for every record.
- the matrix is too heavy.
- split local file data.dat according to model.
