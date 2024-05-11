import dill
import os
import torch 
from torchvision import datasets, transforms

def get_dataset(dir,number):
	if number==0:
		dataset_path = os.path.join(dir,'Test.pkl')
	else:
		dataset_path = os.path.join(dir,'Client'+str(number)+'.pkl')
	print(dataset_path)
	with open(dataset_path,'rb') as f:
		dataset = dill.load(f)
	return dataset

if __name__ == '__main__':
	for c in range(10):
			train_datasets = get_dataset("../dataset/Data_CIFAR10",c)	
			print(len(train_datasets))