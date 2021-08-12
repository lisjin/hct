#!/usr/bin/env bash

for domain in 'calling' 'messaging' 'music' 'news' 'reminders' 'weather'; do
    wget -O mudoco_${domain}.json https://github.com/apple/ml-cread/blob/main/MuDoCo-QR-dataset/mudoco_${domain}.json?raw=true
done
