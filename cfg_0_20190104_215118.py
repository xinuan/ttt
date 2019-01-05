import torch
# 测试
learning_rate = 1e-4
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# Additional Info when using cuda
if device.type == 'cuda':
	print(torch.cuda.get_device_name(0))
	print('Memory Usage:')
	print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
	print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu:0")
d_dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU
l_dtype = torch.cuda.LongTensor # Uncomment this to run on GPU
# ---- Define Loss and Train Para
# loss_fn = torch.nn.MSELoss(reduction='sum')
loss_fn = torch.nn.CrossEntropyLoss()
onehot = 0
BatchSize = 100
Epoch = 1

dataPath = r"D:\deep learning\zzzzzzzzzdata\mnist\fashionMNIST"
TrainPath = r"E:\Haloes\src\pytorch\result\Weight_fashionMNIST.dict"
TestPath = r"E:\Haloes\src\pytorch\result\Weight_fashionMNIST.dict"
Test_CONF = r"E:\Haloes\src\pytorch\result\confuseMatrixfashionMNIST.txt"
Test_Y = r"E:\Haloes\src\pytorch\result\Y_fashionMNIST.txt"
Weight_dict = r"E:\Haloes\src\pytorch\result\Weight_fashionMNIST.txt"
