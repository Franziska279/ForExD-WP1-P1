#!/bin/bash

# for i in 684 #{7..976} # 684   # {99..977}   #977
# do
#   python 05_sentle_preprocessing.py $i
# done


seq 0 976 | parallel -j 5 python 05_sentle_preprocessing.py {}


