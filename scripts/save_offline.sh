#!/bin/bash
#循环执行 gh_filter.py 10 次,并附带参数i
for i in {1..25}
do
    python3 ./src/offline_data_collector.py $i
done