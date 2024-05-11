import socket
import os
from tqdm import tqdm
import time
import math

def file_send(sk,file_path):
    file_size = os.path.getsize(file_path)
    sk.send(f"{file_size}".encode())
    sendProgress = tqdm(range(file_size),f"发送{file_path}",unit="B",unit_divisor=1024,unit_scale=True)
    with open(file_path,"rb") as file:
        for _ in range(math.ceil(file_size/1024)):
            bytesData = file.read(1024)
            if not bytesData:
                break
            sk.sendall(bytesData)
            sendProgress.update(len(bytesData))


def file_recv(sk,file_path):
    received = sk.recv(1024)
    if not received:
        print("关闭连接")
        sk.close()
        exit(0)
    received = received.decode()
    print(received)
    filesize = int(received)
    progress = tqdm(range(filesize),f"接受{file_path}",unit="B",unit_divisor=1024,unit_scale=True)
    
    with open(file_path,'wb') as file:
        for _ in range(math.ceil(filesize/1024)):
            bytesData = sk.recv(1024)
            if not bytesData:
                break
            file.write(bytesData)
            progress.update(len(bytesData))
    