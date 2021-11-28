#!bin/bash
docker run -it -v ${PWD}:/usr/src/app --rm --name ml-hw1 ntut-ml-hw1 python train.py