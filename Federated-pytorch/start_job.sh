#!/bin/bash
python3 server.py &
sleep 5
echo "开始执行"
python3 client.py &
python3 client.py &
