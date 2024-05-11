
import models, torch
import socket
import queue
import threading
import random
import pickle
import sys
import datasets
import json
import os
import file


class Server(object):
	
	def __init__(self, conf, eval_dataset):
		self.conf = conf 
		self.global_model = models.get_model(self.conf,self.conf["model_name"]) 
		
		self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)
		
	def socket_init(self,server_address):
		self.server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		self.server_address = server_address
		self.server_socket.bind(server_address)
		self.server_socket.listen(1)
		self.clients = []

		count = 0 #标志第一个测试客户端

		while True:
			client_socket, client_address = self.server_socket.accept()
			count = count+1
			if count>1: 
				self.clients.append([client_address,client_socket]) #记录客户端地址
				print('接收到来自 %s 的连接' % str(client_address))
			else:
				print('测试连接成功')

			if len(self.clients)==self.conf["no_models"]:
				print("客户端全部连接")
				break

	def message_passing_file(self,model,client_socket,id):
		#将模型转换为文件进行传输
		model_path = './trace/Server/model{}.pth'.format(id)

		# model.cpu() #保存前转移到cpu

		torch.save(model,model_path)
		file.file_send(client_socket,model_path)


	def message_recv_file(self,client_socket,id):
		diff_path = './trace/Server/update_diff{}.pth'.format(id)
		if os.path.exists(diff_path):
			try:
				os.remove(diff_path)
			except OSError as e:
				print(f"文件删除失败：{e}")

		print("正在接收梯度文件")
		
		close_flag = 0 #判断是否接收到关闭信号

		file.file_recv(client_socket,diff_path)

		try:
			diff = torch.load(diff_path)
		except:
			raise

		# if torch.cuda.is_available():
		# 	diff = diff.cuda()

		return diff

	# def message_passing(self,model,client_socket):
	# 	#client_socket.send(model.encode())#发送全局模型给指定的客户端
	# 	serialized_model = pickle.dumps(model)
	# 	print("正在发送全局模型，模型大小：",sys.getsizeof(serialized_model),"字节")
	# 	client_socket.sendall(serialized_model) #通过序列化方式传输
	# 	print("发送完毕")
	
	# def message_recv(self,client_socket):
	# 	print("正在接收更新梯度")
	# 	serialized_diff = b''
	# 	while True:
	# 		chunk = client_socket.recv(4096)
	# 		if not chunk:
	# 			break
	# 		serialized_diff+=chunk
	# 	print("梯度接收完毕，大小：",sys.getsizeof(serialized_diff),"字节")
	# 	return serialized_diff
	
	def client_handler_file(self,model,client_socket,q,id):
		self.message_passing_file(model,client_socket,id)
		try:
			diff = self.message_recv_file(client_socket,id)
			if not diff:
				print("异常:客户端断开")
				exit(-1)
			q.put(diff)
		except:
			print("diff传输错误,丢弃")

		

	# def client_handler(self,model,client_socket,q):
	# 	self.message_passing(model,client_socket)
	# 	serialized_diff = self.message_recv(client_socket)
	# 	if not serialized_diff:
	# 		print("异常:客户端断开")
	# 		exit(-1)
			
			
	# 	diff = pickle.loads(serialized_diff)
	# 	q.put(diff)

	def socket_close(self):
		self.server_socket.close()
		print("服务器关闭")


	
	def model_aggregate(self, weight_accumulator,k):
		for name, data in self.global_model.state_dict().items():
			
			# update_per_layer = weight_accumulator[name] / float(self.conf["k"])
			update_per_layer = weight_accumulator[name] / float(k)	
			if data.type() != update_per_layer.type():
				data.add_(update_per_layer.to(torch.int64))
			else:
				data.add_(update_per_layer)
				
	def model_eval(self):
		self.global_model.eval()
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		for batch_id, batch in enumerate(self.eval_loader):
			data, target = batch 
			dataset_size += data.size()[0]
			
			if self.conf['gpu']:
				if torch.cuda.is_available():
					data = data.cuda()
					target = target.cuda()
				
			
			output = self.global_model(data)
			
			total_loss += torch.nn.functional.cross_entropy(output, target,
											  reduction='sum').item() # sum up batch loss
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

		acc = 100.0 * (float(correct) / float(dataset_size))
		total_l = total_loss / dataset_size

		return acc, total_l
if __name__ == '__main__':
	with open('utils/conf.json', 'r') as f:
		conf = json.load(f)
	server_address = ('127.0.0.1',8888)

	train_datasets = datasets.get_dataset("../dataset/Data_CIFAR10",1)
	server = Server(conf,train_datasets)
	server.socket_init(server_address)
	torch.save(server.global_model,'model.pth')

	print('开始发送文件')
	file.file_send(server.clients[0][1],'model.pth')
	print('发送完毕')

	# with open('model.pth', 'rb') as file:
	# 	# 读取文件内容并发送给客户端
	# 	print('开始发送文件')
	# 	data = file.read(4096)
	# 	while data:
	# 		server.clients[0][1].send(data)
	# 		data = file.read(4096)
		
	# 	print('发送完毕')


#server.message_passing(server.global_model,server.clients[0][1])
