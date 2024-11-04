#!/bin/bash

for i in {46..977} # for i in 114 115 125 126 242 479 856 #for i in {66..977}   
do
  python 05_sentle_download_parallel.py $i
done


