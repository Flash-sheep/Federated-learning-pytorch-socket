import threading
import json
import torch
class Server:
    shared_variable = 0
    def handler(self):
        lock = threading.Lock()  # 创建线程锁
          # 局部变量

        self.shared_variable += 1

        print(self.shared_variable)

# server = Server()
# threads = []
# for _ in range(10):
#     thread = threading.Thread(target=server.handler)
#     threads.append(thread)
#     thread.start()

# with open('utils/conf.json', 'r') as file:
#     data = json.load(file)
# data['gpu'] = True

# with open('utils/conf.json', 'w') as file:
#     json.dump(data, file, indent=4)
def test(id):
    try:
      model = torch.load('./trace/Server/update_diff{}.pth'.format(id))
    except:
      raise

for id in range(1,21):
  try:
    test(id)
    print(id,"成功打开")
  except:
    print(id)
    



