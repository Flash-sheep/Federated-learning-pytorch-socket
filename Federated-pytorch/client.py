import models, torch, copy
import socket
import pickle
import sys
import datasets
import json
import os
import file
import threading

class Client(object):
	local_model = threading.local()
	train_dataset = threading.local()

	def __init__(self, conf,train_dataset, id = -1):

		self.conf = conf
		
		self.local_model = models.get_model(self.conf,self.conf["model_name"]) 
		
		self.client_id = id
		
		self.train_dataset = train_dataset
		
		all_range = list(range(len(self.train_dataset)))
		data_len = int(len(self.train_dataset) / self.conf['no_models'])
		train_indices = all_range[id * data_len: (id + 1) * data_len]

		# self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
		# 							sampler=torch.utils.data.sampler.SubsetRandomSampler(all_range))
		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=conf["batch_size"], 
									sampler=torch.utils.data.sampler.SubsetRandomSampler(all_range))
									
		
	def local_train(self, model):
		
		print("开始进行本地模型训练")

		for name, param in model.state_dict().items():
			self.local_model.state_dict()[name].copy_(param.clone())
	
		#print(id(model))
		optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'],
									momentum=self.conf['momentum'])
		# print(id(self.local_model))
		print("预处理结束")
		self.local_model.train()
		for e in range(self.conf["local_epochs"]):
			
			for batch_id, batch in enumerate(self.train_loader):
				data, target = batch
				
				if self.conf['gpu']:
					if torch.cuda.is_available():
						data = data.cuda()
						target = target.cuda()
				optimizer.zero_grad()
				output = self.local_model(data)
				loss = torch.nn.functional.cross_entropy(output, target)
				loss.backward()
				optimizer.step()
			print("Epoch %d done." % e)	
		diff = dict()
		for name, data in self.local_model.state_dict().items():
			diff[name] = (data - model.state_dict()[name])
			#print(diff[name])
		print("本地模型训练完毕")
		return diff
	
	def socket_init(self,server_address):
		print("客户端开始连接")
		self.server_address = server_address
		self.client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
		self.client_socket.connect(server_address)
		self.local_address = self.client_socket.getsockname()
		#self.client_socket.settimeout(1)
		print(self.local_address,'连接到',server_address)
	
	def message_passing_file(self,diff):
		#将模型转换为文件进行传输
		diff_path = './trace/Client/diff{}.pth'.format(self.client_id)

		# diff.cpu()
		print('开始发送梯度文件')
		torch.save(diff,diff_path)
		file.file_send(self.client_socket,diff_path)
		# 读取文件内容并发送给客户端
		print('梯度文件发送完毕')

	def message_recv_file(self):
		
		model_path = './trace/Client/update_model{}.pth'.format(self.client_id)
		if os.path.exists(model_path):
			try:
				os.remove(model_path)
			except OSError as e:
				print(f"文件删除失败：{e}")

		print("正在接收全局模型文件")
		
		close_flag = 0 #判断是否接收到关闭信号

		file.file_recv(self.client_socket,model_path)
		print("全局模型文件接收完毕")
		model = torch.load(model_path)

		print("模型文件加载完毕")
		# if torch.cuda.is_available():
		# 	model = model.cuda()

		return model
		

	# def message_passing(self,diff):
		
	# 	#self.client_socket.send(diff.encode())
	# 	serialized_diff = pickle.dumps(diff)
	# 	print("正在传输更新梯度，梯度大小：",sys.getsizeof(serialized_diff),"字节")
	# 	self.client_socket.sendall(serialized_diff)
		
	
	# def message_recv(self):
	# 	#model = self.client_socket.recv(1024)
	# 	print("正在接收全局模型")
	# 	serialized_model = b''
	# 	while True:
	# 		chunk = self.client_socket.recv(4096)
	# 		if not chunk:
	# 			break
	# 		serialized_model+=chunk
	# 	print("全局模型接收完毕，模型大小为",sys.getsizeof(serialized_model),"字节")
	# 	return serialized_model
	
	def socket_close(self):
		self.client_socket.close()
		print(self.local_address,"关闭连接")

if __name__ == '__main__':
	with open('utils/conf.json', 'r') as f:
		conf = json.load(f)
	server_address = ('127.0.0.1',8888)
	train_datasets = datasets.get_dataset("../dataset/Data_CIFAR10",1)
	client = Client(conf,train_datasets,1)
	# model = torch.load('update.pth')
	# print(model)
	client.socket_init(server_address)
	#serialized_model = client.message_recv()
	print('开始接收文件')
	file.file_recv(client.client_socket,'update.pth')
	print('文件接收完成')
	print(torch.load('update.pth'))
	# with open('update.pth', 'wb') as file:
	# 	print('开始接收文件')
	# 	while True:
	# 		data = client.client_socket.recv(4096)
	# 		if data == 'quit'.encode():
	# 			break
	# 		file.write(data)
	# 	print('文件接收完成')





		