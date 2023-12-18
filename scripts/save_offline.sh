#!/bin/bash
#循环执行 gh_filter.py 10 次,并附带参数i
for i in {1..10}
do
    python3 ../filters/gh_filter.py $i
done