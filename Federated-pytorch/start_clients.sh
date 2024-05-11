#!/bin/bash

json_file="./utils/conf.json"
client_threads=$(python3 -c "import json; conf = json.load(open('$json_file'));print(conf['no_models'])")
echo "客户端数量为:$client_threads"
for ((id=1; id<=client_threads;id++))
do 
    python3 run_clients_pro.py -i $id &
done