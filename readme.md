#### recommendation for TV
##### step 1
    edit ./etc/config.ini
##### step 2
    run ./tools/get_format_raw_data.py to get data from mongodb and format
#### step 3
    run ./rec_content_based.py to get recommend data result and write into redis

#### to be improved[list]:
- slow rate for record because of comparison with all records.
- calculation of records can be changed to others.
