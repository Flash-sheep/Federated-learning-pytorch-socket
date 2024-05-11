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
import warnings
warnings.filterwarnings("ignore")


def create_client_file(server_address,conf,client_dataset,id):
	client = Client(conf,client_dataset,id)
	client.socket_init(server_address)
	while True:

		model = client.message_recv_file()

		if not model:
			client.socket_close()
			print(client.local_address,"断开连接")
			break
		print(f"客户端{id}开始训练")
		diff = client.local_train(model)

		client.message_passing_file(diff)

# def create_client(server_address,conf,client_dataset,id):
# 	client = Client(conf,client_dataset,id)
# 	client.socket_init(server_address)
# 	while True:
# 		serialized_model = client.message_recv()
# 		if not serialized_model:
# 			client.socket_close()
# 			print(client.local_address,"断开连接")
# 			break
# 		model = pickle.loads(serialized_model)
# 		diff = client.local_train(model)
# 		client.message_passing(diff)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-i', '--id', dest='id')
	args = parser.parse_args()
	

	with open('utils/conf.json', 'r') as f:
		conf = json.load(f)	
	
	server_address = ('127.0.0.1', 8888)
	
	client_threads = []
	
	print(args.id)

	train_datasets = datasets.get_dataset("../dataset/Data_CIFAR10",int(args.id))
	# create_client_file(server_address,conf,train_datasets,1)
	print(f"创建客户端{args.id}")
	create_client_file(server_address,conf,train_datasets,int(args.id))
	print(f"客户端{args.id}返回")

				
			
		
		
	
		
		
	
