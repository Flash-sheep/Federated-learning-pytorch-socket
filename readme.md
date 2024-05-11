# 横向联邦学习-pytorch
使用pytorch，mnist、cifar数据集实现基础的联邦学习（平局聚合、不含加密过程）
## 需要安装
python

pytorch

下载mnist、cifar10数据集

如果需要使用自己的数据即替换dataset即可

## 无socket通信

### 代码运行

```
cd Federated-basic
python3 main.py
```

### 修改参数utils/conf.json
在utils/conf.json中修改参数设置

* model_name：模型名称
* no_models：客户端数量
* type：数据集信息
* global_epochs：全局迭代次数，即服务端与客户端的通信迭代次数
* local_epochs：本地模型训练迭代次数
* k：每一轮迭代时，服务端会从所有客户端中挑选k个客户端参与训练。
* batch_size：本地训练每一轮的样本数
* lr，momentum，lambda：本地训练的超参数设置

### 服务端server.py
服务端的主要功能是将被选择的客户端上传的本地模型进行模型聚合（如果需要完善其他的复杂功能如同态加密、服务端需要对各个客户端节点进行网络监控、对失败节点发出重连信号等等功能，可以采用FATE平台）

这里的模型是在本地模拟的，不涉及网络通信细节和失败故障等处理，因此不讨论这些功能细节，仅涉及模型聚合功能。

服务端的工作包括：

第一，将配置信息拷贝到服务端中；

第二，按照配置中的模型信息获取模型，这里我们使用torchvision 的models模块内置的ResNet-18模型。

第三，这里的模型定义模型聚合函数。采用经典的FedAvg 算法。

第四，定义模型评估函数。

### 客户端client.py
客户端主要功能是接收服务端的下发指令和全局模型，利用本地数据进行局部模型训练。

### 整合main.py
首先，读取配置文件信息。

每一轮的迭代，服务端会从当前的客户端集合中随机挑选一部分参与本轮迭代训练，被选中的客户端调用本地训练接口local_train进行本地训练，最后服务端调用模型聚合函数model_aggregate来更新全局模型

## socket通信

在该模块中，为客户端和服务端添加了socket通信机制，并增添了服务端的多线程以及客户端的多线程处理。

### 代码运行

需要两个终端，其中一个终端运行：

```
cd Federated-pytorch
python3 run_server.py
```

待出现“测试连接成功”输出后，在另一个终端运行：

```
cd Federated-pytorch
bash start_clients.sh
```

