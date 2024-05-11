import argparse, json
import datetime
import os
import logging
import torch, random
import sys

import socket
import queue
import threading
import pickle

from server import *
from client import *
import models, datasets

def is_server_reachable(host, port):
    try:
        # 创建套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 设置超时时间
        sock.settimeout(2)  # 设置为2秒
        
        # 连接服务器
        result = sock.connect_ex((host, port))
        
        # 判断连接结果
        if result == 0:
            # 连接成功
            return True
        else:
            # 连接失败
            return False
    except Exception as e:
        # 发生异常，连接失败
        return False
    finally:
        # 关闭套接字
        sock.close()


def create_server(server_address,eval_dataset,conf):
	server = Server(conf,eval_dataset)

	
	server.socket_init(server_address)
	
	q = queue.Queue() #存储客户端发回的更新信息

	for e in range(conf["global_epochs"]):
	
		candidates = random.sample(server.clients, conf["k"])
		
		weight_accumulator = {}
		
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		
		server_threads = []
		index = 1
		for client_address,client_socket in candidates: #并行获取客户端数据
			server_thread = threading.Thread(target=server.client_handler_file,args=(server.global_model,client_socket,q,index))
			index = index+1
			server_threads.append(server_thread)
			server_thread.start()
			
			
		for server_thread in server_threads:
			server_thread.join() #等待所有线程返回结果

		if q.qsize()>0:
			print("客户端模型获取成功")
			print("实际发回数量为",q.qsize(),"预期发回数量为",conf["k"])
				#exit(-1)
		else:
			print("客户端模型获取失败")
			exit(-1)
		
		length = q.qsize()
		
		print("开始进行模型聚合")
		for i in range(q.qsize()):
			diff = q.get()
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
				
		
		server.model_aggregate(weight_accumulator,length)

		print("模型聚合完毕")
		print("开始进行模型评估")
		acc, loss = server.model_eval()
		
		
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

	print("模型训练结束")
	for client_address,client_socket in server.clients:
		client_socket.close()
		print("关闭与",client_address,"的连接")
	server.socket_close()



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	

	with open('utils/conf.json', 'r') as f:
		conf = json.load(f)	
	
	eval_datasets = datasets.get_dataset("../dataset/Data_CIFAR10",0)

	server_address = ('127.0.0.1', 8888)

	server_thread = threading.Thread(target=create_server,args=(server_address,eval_datasets,conf))
	server_thread.start()

	print("等待服务器端就绪")
	while True:
		if is_server_reachable(server_address[0],server_address[1]):
			break
	print("服务器端准备就绪")

	server_thread.join()
	print("服务器线程返回")

				
			
		
		
	
		
		
	
