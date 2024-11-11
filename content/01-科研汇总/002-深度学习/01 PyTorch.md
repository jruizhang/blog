
### 一、基于conda prompt创建pytorch环境
- 打开conda prompt
- conda create -n pytorch python=3.11 (环境名是pytorch，以及对应的版本)
- y
- conda activate pytorch
- ![[Pasted image 20240825151107.png]]
- pip list
- 缺少pytorch，添加pytorch
- ![[Pasted image 20240825151308.png]]
- nvidia-smi查看cuda版本号
- 适要时可更新显卡
- 安装pytorch
> - https://blog.csdn.net/weixin_67957872/article/details/133801531
- 检验是否安装，1、输入python；2、import torch ；3、torch.cuda.is_available()
##### 附：Anaconda3镜像源修改
> https://blog.csdn.net/qq_37344125/article/details/103099267?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-103099267-blog-140966878.235%5Ev43%5Econtrol&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-103099267-blog-140966878.235%5Ev43%5Econtrol&utm_relevant_index=2

### 二、基于创建环境设置初始模型(Pytorch Jupyter)
**1、Pytorch:**（基于pytorch虚拟环境）
- 使用现有conda环境
- 1、在conda prompt下设置conda info --envs，查找各个虚拟变量的位置，选择
- 2、如果在对应文件下找不到python.exe时，寻找conda下的condabin文件夹下的combat，参考如下
>https://zhuanlan.zhihu.com/p/699069168

**2、Jupyter:**（基于pytorch虚拟环境）
- 1、conda activate pytorch；在conda prompt下进入pytorch虚拟环境
- 2、conda list；查看是否有ipython包
- 3、conda install jupyter notebook；如果没有，则进行安装
- 4、安装完毕后，cmd中输入jupyter notebook
- 5、
### 三、基于创建环境设置初始模型
**1、基本函数:（dir函数和help函数）**
- dir函数是查看有什么，能一直查看，但当查看到函数的固有属性时（__name__等），则只显示固有属性，不能再继续查看（有什么东西）
- help用于查找编程者对于该函数的输入与输出的设定，（如何使用）
![[Pasted image 20240825214019.png]]

**2、python文件、python控制台与Jupyter的对比**
假定代码运行是以块为一个整体的话

- python文件的所有行的代码是一个块
- python控制台以每一个代码行为代码块，但可阅读性较差
- Jupyter以任意行为块运行
![[Pasted image 20240826163657.png]]

### 四、数据处理（DataSet、Dataloader、Tensorboard）
![[Pasted image 20240826182313.png]]
- 数据经历：垃圾—DataSet—Dataloader，Dataset相当于一摞扑克牌，Dataloader相当于生成 打牌后的拿牌数量、拿牌方式、牌顺序
##### 1、Dataset实现
- 介绍：主要告诉数据的位置以及第一张数据是什么
- os包主要应用于路径的提取
	- os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
	- os.path.join() 方法用于路径拼接文件路径。
- MyData类继承Dataset类，主要在主函数上输入文件目录、标签路径，并生成对应路径下的文件名称
	- 通过结合主函数产生__getitem__与__len__函数，用于提取操作者系统下的系统数据信息
	- 由于是继承，因此形式与Dataset的形式相同，例如通过+实现


**2、Tensorboard实现**
- 介绍：主要用于可视化，从而在网页上展示数据变化情况
- **注意：cv2.imread(img_path) 产生的是np类型 ；  PIL.Image.open(img_path)产生的是PIL类型**
- **add_scalar**
``` python
# Tensorboard 的使用（add_scalar）  
from torch.utils.tensorboard import SummaryWriter  
# pip install tensorboard -U  
writter = SummaryWriter('logs')  
# 添加图像  
# writter.add_image('train/ants', img)  
# 添加数  
# y=x  
for i in range(100):  
    writter.add_scalar('y=2x', 2*i, i)  
writter.close()  
  
# cmd下运行  
# tensorboard --logdir=logs --port=6007
```
- 使用说明：
	- 基于SummaryWriter函数，运行前需要安装tensorboard包
	- logs为生成文件的文件夹保存名称，主要有两个功能，分别为添加数与添加图像
	- writter.add_scalar中，'y=2x'为图像标题，相当于图像的id，是唯一的，scalar_value为y值，global_step相当于x轴
	- 在python文件中运行完毕后，通过在对应环境下，通过tensorboard --logdir=logs --port=6007打开对应的展示链接，从而进行查看
		- 其中logdir为事件文件所在文件夹名
- **add_image**
```python
# Tensorboard 的使用（add_image） 
from torch.utils.tensorboard import SummaryWriter  
from PIL import Image  
import numpy as np  
# 添加图像  
img_path = 'dataset/train/ants/20935278_9190345f6b.jpg'  
img_pil = Image.open(img_path)  
print(type(img_pil))  
img_np = np.array(img_pil)  
writter = SummaryWriter('logs')  
writter.add_image('train/ants', img_np, 2, dataformats='HWC')  
writter.close()
```
- 使用说明：
	- 也可以添加图片，add_image只允许几种图片格式，PIL格式不允许，因此需要首先转换为numpy格式或者cv格式
	- dataformats是可选参数，不选择时报错，这是因为numpy的三维数据不满足函数要求，因此需要额外设定数据形式，可参考示例
	- 'train/ants'提供了大文件夹下的小文件命名
	- ![[Pasted image 20240827161107.png]]

##### 3、Transform
- Transforms包整体为一个文件，文件中有许多工具，主要用于实现数据与tensor的转换，即提供工具箱方便编程使用，编程的任务是使用Transforms提供的工具选择并构建一个针对自身问题的针对性工具，主要目标为图片，以下为结构理解。
- ![[Pasted image 20240827165303.png]]
- pycharm提供结构方便用户查询文件中各个代码，了解代码分布情况。
- 注意：
	- ![[Pasted image 20240827220248.png]]
	- 注意函数的输入和输出，即需要导入与需要产生的类型要求
	- 多看官方文档
	-  关注方法需要什么参数，尤其是没有设置默认值的参数（Args）
- **Useful Transform** ![[Pasted image 20240827213447.png]]
- 
1. transforms.ToTensor（将PIL与narrays转换为tensor数据类型）
2. transforms.ToPILImage
3. transforms.Normalize（数据归一化，输入数据需要是Tensor，提供平均值与标准差，维度由输入数据的维度决定）![[Pasted image 20240827214838.png]]
4. transforms.Resize（重新调整尺寸，输入是PIL，结果等比缩放（一个数）或者按尺寸缩放（一对数））![[Pasted image 20240827215709.png]]
5. transforms.Compose（结合transforms中的多个函数，依次运行）
6. transforms.RandomCrop（随机裁剪，结果裁剪为正方形（一个数），结果矩阵（一对数））
![[Pasted image 20240827220144.png]]

##### 4、常用深度学习数据下载
- torch官网中，官方文档的torchvision网站提供许多数据集，通过使用函数进行下载，其中通过运行函数即可实现下载
- 代码中数据从下载，到保存，并结合transform进行变化
```python
import torchvision  
from torch.utils.tensorboard import SummaryWriter  
  
dataset_transform = torchvision.transforms.Compose([  
    torchvision.transforms.ToTensor()  
])  
  
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)  
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)  
  
  
# train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True,  download=True)  
# test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)  
# print(test_set[0])  
# print(test_set.classes)  
#  
# img, target = test_set[0]  
# print(img)  
# print(target)  
# print(test_set.classes[target])  
# img.show()  
#  
print(test_set[0])  
  
writer = SummaryWriter("p10")  
for i in range(10):  
    img, target = test_set[i]  
    writer.add_image("test_set", img, i)  
  
writer.close()
```
- 一般都是默认download=True，没有坏处。
- 查看数据情况，其中包含数据集的编码处理等
##### 5、Dataloader实现
- 介绍：基于Dataset为神经网络提供数据选择，生成imgs
![[Pasted image 20240828175841.png]]
- torch.utils.data.DataLoader包含许多参数，例如每batch的选择数量，每次迭代是否随机，服务器数量，以及最后是否删除
- 需要通过for循环来查看Dataloader中的数据
- Dataloader将原始数据集的部分转换为一个batch，即imgs和对应的targets，并产生了许多imgs与tagerts
- 代码如下：
```python
import torch  
import torchvision  
from torch.utils.tensorboard import SummaryWriter  
  
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())  
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)  
  
writer = SummaryWriter("p10")  
step= 0  
for data in testloader:  
    imgs, targets = data  
    print(imgs.shape)  
    print(targets.shape)  
    writer.add_images("testloader", imgs, step)  
    step += 1  
writer.close()
```


### 五、神经网络结构
![[Pasted image 20240831203457.png]]
##### 1、torch.nn构造神经网络
神经网络简易流程
- 基于nn.Module父类，构造神经网络框架，均需构造__init__与forward函数。
- 下列代码定义网络框架、并使用网络结构运行
- Datarui为网络名称，需要super(Datarui, self).__init__()
```python
import torch  
from torch import nn  
  
  
class Datarui(nn.Module):  
    def __init__(self):  
        super(Datarui, self).__init__()  
    def forward(self, input):  
        output = input + 1  
        return output  
  
datarui = Datarui()  
x = torch.tensor(1.0)  
output = datarui(x)  
print(output)
```
示例：输入、卷积、非线性、卷积、非线性、输出
![[Pasted image 20240829162004.png]]
##### 2、torch.nn.functional
- 基于torch.nn.functional展示神经网络的卷积操作，functional相当于组成众多nn.layers的工具。
![[Pasted image 20240829164038.png]]
- 首先通过torch.tensor创建tensor矩阵，[[]]代表二维，每个行为一个[]。
- 由于F.conv2d需要输入量为四维结构，因此基于reshape函数进行结构转化
- stride表示下次跳几个单元格，也可以上下不同；padding代表是否给矩阵四周加几层空白格；
```python
import torch  
import torch.nn.functional as F  
tensor_matix = torch.tensor([[1, 2, 0, 3, 1],  
                             [0, 1, 2, 3, 1],  
                             [1, 2, 1, 0, 0],  
                             [5, 2, 3, 1, 1],  
                             [2, 1, 0, 1, 1]])  
tensor_conv = torch.tensor([[1, 2, 1],  
                            [0, 1, 0],  
                            [2, 1, 0]])  
tensor_matix1 = tensor_matix.reshape(1, 1, 5, 5)  
tensor_matix = torch.reshape(tensor_matix,(1, 1, 5, 5))  
tensor_conv = torch.reshape(tensor_conv,(1, 1, 3, 3))
output1 = F.conv2d(tensor_matix, tensor_conv, stride=1)  
print(output1)  
  
output2 = F.conv2d(tensor_matix, tensor_conv, stride=2)  
print(output2)  
  
output3 = F.conv2d(tensor_matix, tensor_conv, stride=1, padding=1)  
print(output3)
```

##### 3、卷积核（CONV）
CONV2D参数
![[Pasted image 20240831192240.png]]
- 输入通道数与输出通道数，相当于2D矩阵的层数。
- kernel_size代表卷积核的尺寸，即2D矩阵的行数与列数；stride代表卷积核移动的步频；padding代表输入矩阵的扩张维数；padding_mode代表什么形式填入扩张单元格；dilation代表矩阵是否扩张成网；bias代表增加误差，参数解释与动画在torch官网中有。![[Pasted image 20240831193739.png]]
- 图片中由于希望展示卷积后的图片，但卷积之后成为了6通道的tensor，而生成图片只需要3通道，因此基于**reshape**对6通道进行转变为3通道，但reshape过程中矩阵的数字个数是不变的，从而batch_size（图片数）自然增多。![[Pasted image 20240831194506.png]]![[Pasted image 20240831194932.png]]

- pading和stride需要通过以下公式进行推到，![[Pasted image 20240831195012.png]]
##### 4、池化核（MaxPool2D）
池化主要用于减少数据量，通常跟在卷积后边，把数据图片马赛克化。![[Pasted image 20240831195224.png]]
- kernel_size为核的窗口尺寸选取；池化层stride的默认值是与kernel_size相同；ceil_mode为真时为ceil模式，否则为floor模式，其中ceil代表向上取整，floor代表向下取整，即ceil_mode不放弃每一个元素，floor就是只采区确定拥有的元素。![[Pasted image 20240831195730.png]]
##### 5、其他层
- **padding layers**：主要为对输入数据进行填充的各种方式。
- **非线性layers**：主要引入非线性特征，提高泛化能力
	- Relu：需要输入batchsize，其余不限制![[Pasted image 20240831200943.png]]![[Pasted image 20240831201145.png]]inplace代表是否将改变应用到原始数据上，一般设置为False，保留原始数据，防止数据丢失
	- Sigmoid：![[Pasted image 20240831201027.png]]
- **正则化layers**：
	- BatchNorm2D:相当于对输入数据，会对各个通道进行归一化等操作。![[Pasted image 20240831201613.png]]![[Pasted image 20240831201713.png]]
- **Recurrentlayers**：![[Pasted image 20240831201947.png]]
- **Transfomer Layesrs**：![[Pasted image 20240831202100.png]]
- **Linear Layers:**![[Pasted image 20240831202209.png]]
	- ![[Pasted image 20240831202936.png]]
- **Dropout Layers**:随机把一些input随机变为0，从而防止过拟合![[Pasted image 20240831202259.png]]
- **Sparse Layers**：主要用于自然语言处理![[Pasted image 20240831202456.png]]
##### 6、layers_sequential
- 方便隐藏层的管理
![[Pasted image 20240831204114.png]]
- 

| ![[Pasted image 20240831204901.png]] | ![[Pasted image 20240831205606.png]] |
| ------------------------------------ | ------------------------------------ |


##### 7、现成网络结构：
- torch提供一些现成的搭建好的网络结构
	- tourchvision![[Pasted image 20240831203811.png]]
- 网络一般也会使用一些公开的数据进行试验，也都可以在torch中找到数据源。

### 六、网络评价、优化器、保存与复制
##### 1、loss function
![[Pasted image 20240831210256.png]]
- lossfunction目标是为了评价模型与理想模型之间的差距，如果需要根据这些差距进行模型修改时，则使用backward的进行方向传播，但是需要注意的是针对计算出的损失函数值进行back，而不是针对损失函数，同时backward的目的是计算方向的梯度![[Pasted image 20240831220315.png]]
- **L1 loss**
	- ![[Pasted image 20240831214249.png]]
	- ![[Pasted image 20240831214217.png]]
- **MSE loss**:
	- ![[Pasted image 20240831214357.png]]
- **交叉熵loss**
	- ![[Pasted image 20240831214516.png]]![[Pasted image 20240831214928.png]]![[Pasted image 20240831215649.png]]输入的input，即预测概率，有batchsize=N个样本,以及分类的类别或者概率，以及输入的target需要是对应的N个
	- 目标是希望整体上概率都小（不要全是0.8，0.9），同时预测正确的那一类的概率尽可能高
	- ![[Pasted image 20240831214818.png]]
	- loss function应该根据需求去使用，并且使用时要注意要求的输入形状与输出形状。
- **Accuarcy**:
	- ![[Pasted image 20240901203714.png]]
##### 2、Optimzer
![[Pasted image 20240901132205.png]]
设定好优化器后，通过step运行优化器，实现对网络参数的调整，此外注意每次梯度的清零![[Pasted image 20240901132333.png]]
- **Adam优化算法**：
	- 模型参数在不同的优化器中参数不一样，此外参数数值可以尝试寻找或者参考借鉴优秀文献。![[Pasted image 20240901132832.png]]
##### 3、模型复制
- vgg分类神经网络：vgg16和vgg19较为常用![[Pasted image 20240901142454.png]]
	- pretrained=True代表下载使用模型最优参数，=False代表下载模式时，模型采用原始随机参数
- 迁移学习，copy的模型和我们需要的模型可能网络实现的结果不一样，比如vgg是1000分类，而我只是需要10分类（利用优秀网络作为前置网络实现结构修改）
- 方法1：尾部加一个1000->10的线性连接层，通过model.add_module("网络名称"，网络层，)，或者model.1级标题.add_module，在二级标题里加
- 方法2：对于尾部网络进行重新修改，指定网络后，直接重新赋值新网络
- ![[Pasted image 20240901144213.png]]
##### 4、模型保存与读取
- 两种保存方式，一种直接保存模型（模型结构＋参数），一种以字典格式保存模型（模型参数）
- 方式一保存与加载使用save与load。但方式一在加载网络结构时，应该在加载的时候将原来load的网络结构（class）定义在load文件中，但不需要创建（只导入类不需要运行）
- 方式二保存与加载使用save：保存模型的参数与model.load_state_dict()：实现加载后的参数加载到模型中，与load:实现参数加载，加载后的是个tensor字典。因此方式二需要提前设置好网络结构并创建运行好的网络从而实现加载，（既导入类也需要运行）
- 

| ![[Pasted image 20240901163101.png]] | ![[Pasted image 20240901163636.png]] |
| ------------------------------------ | ------------------------------------ |
##### 5、CPU与GPU
- **调用方式1**
	- 只能对三种类型调用cuda（GPU）
	- ![[Pasted image 20240901205758.png]]
	- 
	- ![[Pasted image 20240901205944.png]]
- **调用方式2**
	- :0代表设备的第一个gpu，:1代表设备的第二个GPU，对设备设定完毕后，通过.to(device)，对三种类型数据指定处理器
	- ![[Pasted image 20240901210723.png]]
- **通用**
	- 网络模型和损失函数不需要通过= 赋值进行设置，可以直接  .   ，数据必须通过赋值进行设置
### 七、神经网络搭建
##### 1、模型实例
- 基于CIFAR10数据的神经网络设计
![[Pasted image 20240831204345.png]]
- 复现过程中，padding与stride需要结合公式进行计算，可以假设stride求padding

##### 5、网络搭建
准备数据，加载数据，准备模型，设置损失函数，设置优化器，开始训练，最后验证，结果聚合展示
准备数据集，dataloader加载数据集，搭建网络模型，创建网络模型实例，定义损失函数，定义优化器，设置网络训练的参数，开始训练，验证模型，最后保存模型。可以将训练结果展示
##### 3、网络搭建
- 期间，例如flatten后的大小不确定，可以通过先运行模型，查看输入数据的输出数据的大小，从而确定后来的线性层的输入维度。
- 搭建完毕后，仅仅运行网络，无法判断网络是否成功搭建，应使用数据放入模型，测试能否输出
- 可以基于tensorboard创建网络的可视化形象。
	- ![[Pasted image 20240831205652.png]]

##### 4、网络训练
- 注意梯度清零
- 采用variable.item()可以让tensor数据形式呈现出数字形式
- 加入with torch.no_grad（）:，其中注意加（）和:
- model.train()和model.eval在训练与测试前最好先调用，对某些层有作用。固定所有参数 避免测试集或者验证集影响参数。![[Pasted image 20240901204018.png]]
##### 5、网络预测
- 使用CUDA建立的预测的模型，应用的数据等也需要使用加一个CUDA指示
- 或者对于CUDA模型，加载时通过
```python
model = torch.load("tudui_model/tudui20.pth", map_location=torch.device("cpu"))
```
- 将模型转换为CPU模型，对cpu数据进行预测
- 预测时，eval和no_grad不要忘记添加
```python
model.eval()  
with torch.no_grad():
	output = model(image)
```



### 八、Github建议
- 一定要读Readme
- 结构很相似，只是有些是通过cmd语言进行运行，其中python打开python编辑器，train.py指示运行这个文件，后续的--代表的是变量名，括号后的值为对变量赋予的值。![[Pasted image 20240901212807.png]]
	- 处理这个问题的办法是，把这个输入语句中运行模型所必须的参数值（变量），中必要的部分的对应的源代码，将其代码改为default=“xxxx”，其中改为你实际的值，然后直接在pycharm中运行即可