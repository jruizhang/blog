
| 论文     | 论文使用数据集                                                  | 代码使用数据集        | 格式             |                                      |
| ------ | -------------------------------------------------------- | -------------- | -------------- | ------------------------------------ |
| NCDM   | Math dataset <br>ASSIST (2009-2010)                      | A09-10         | json           | ![[Pasted image 20241105170849.png]] |
| ICD    | Math dataset <br>ASSIST (2009-2010)                      | a09-10<br>Math | csv格式<br>csv格式 | ![[Pasted image 20241105170118.png]] |
| ID-CDF | ASSIST 09-10<br>Algebra 2006-2007<br>Math1 and Math2<br> | math1          | csv格式          | ![[Pasted image 20241105171132.png]] |
| ICDM   | FrcSub<br>EdNet-1,<br>Assist17<br>NeurIPS20              | EdNet-1        | csv格式          | ![[Pasted image 20241105171830.png]] |
| DCD    | Matmat<br>Junyi <br>NIPS2020EC                           | matmat         | csv格式          | ![[Pasted image 20241105172136.png]] |

### 一、ID-CDF模型
##### 1.1 导入文件要求
- **学生日志情况**：
	- 将train、valid、test作为三个文件进行导入，其中每个数据集情况如下：
	- user_id从0开始，将会直接用于后续创建学生答题矩阵中的行值
	- item_id从0开始，同样用于创建答题矩阵中的列值
	- score二分值，其中0代表答错，1代表答对
	- ![[Pasted image 20241102205134.png]]
- **Q矩阵**：
	- Q矩阵20行11列，其中行代表item id，列代表知识概念。
	- 后续直接被加到df_tarin中
	- ![[Pasted image 20241102205628.png]]

##### 1.2 模型数据要求
- 输入数据前提：在原始导入数据的基础，通过Q_mat额外增加一列，为知识维度列表
- ![[Pasted image 20241102210331.png]]
- 模型结构构建：通过model.IDCD调用模型结构
- 模型训练：放入所有训练数据与验证数据，调用train文件train方法训练模型
- 模型验证：模型训练调用train文件test类，输出模型在测试集的表现情况
- ![[Pasted image 20241102210557.png]]

### 二、ICDM模型
按照模型要求进行配置
- **pytorch（通过pytorch官网找不到指定版本或者控制台下载较慢时）**
	- pytorch安装conda或pip命令报错或较慢的情况下，可以根据[pip报类似No matching distribution found for torch-scatter== 2.1.0+pt113cu116的一种解决方案_no matching distribution found for pyqt6-plugins-CSDN博客](https://blog.csdn.net/qysh123/article/details/143368493)直接下载，下载链接和方法如下
	- 通过[pytorch安装解决报错全流程-卡在solving environment后采取离线安装 ERROR:Ignored the following versions、Could not find-CSDN博客](https://blog.csdn.net/m0_74253352/article/details/133942794)中的第四部分内容中分享的下载链接[download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html)通过迅雷进行下载
	- 其中pytorch中的各个命名含义参考：[Python本地安装whl文件详解与高级pip命令技巧_pip install whl-CSDN博客](https://blog.csdn.net/weixin_45812624/article/details/140582797)
	- 根据[Python本地安装whl文件详解与高级pip命令技巧_pip install whl-CSDN博客](https://blog.csdn.net/weixin_45812624/article/details/140582797)中的pip install D:\Downloads\torch_cluster-1.6.0+pt113cu117-cp38-cp38-win_amd64.whl通过pycharm对应环境的终端进行下载
- dgl（安装问题）
	- dgl 官网主页展示的版本不全，可以参考[dgl安装指南-CSDN博客](https://blog.csdn.net/Simplepig/article/details/125826643)，评论区中有作者发的链接[Deep Graph Library](https://www.dgl.ai/pages/start.html)，下载对应的whl文件，然后同时参考pytorch方法安装对应的文件。
	- 安装后遇到以下问题，[pycharm中dgl安装出错（FileNotFoundError: Could not find module ‘E:\XXXX\XXXX\lib\site-packages\dgl\dgl.dl）_ filenotfounderror: could not find module 'e:-CSDN博客](https://blog.csdn.net/weixin_40659546/article/details/122017184)  ，其中文中存在一部分错误，本质上是由于pytorch与dgl的安装版本不配套，而不是文中说的需要考虑显卡的cuda版本，所以一般根据别人的dgl版本及对应的pytorch版本的对应配置要求进行安装。
- 环境安装问题：
	- 安装过慢可以参考，进行改善：[【已解决】Anaconda中conda 某个包之后Solving environment: \一直转 卡住不动解决办法（图文教程）-CSDN博客](https://blog.csdn.net/weixin_51484460/article/details/134424715)
	- pip端可以采用`pip install EduCDM==0.0.13 --index-url https://pypi.tuna.tsinghua.edu.cn/simple` ，即指定一个镜源--index-url，结果会快很多。


##### 1.1 导入文件要求
- **学生日志情况**：
	- 将训练数据整体进行导入，通过train_splite进行划分，数据集情况如下：
		- ![[Pasted image 20241104104643.png]]
	- 这一步没有将Val数据集进行单独提取
	- 'stu_num': 1827,  'prob_num': 11996,  'know_num': 189,   'batch_size': 1024
	- 第0列为学生id（范围为：0-1826），第1列为item id（范围为：0-11995），第2列为答题情况，二分值，其中0代表答错，1代表答对。
	- ![[Pasted image 20241104104919.png]]
- **Q矩阵**：
	- Q矩阵1196行189列，其中行代表item id，列代表知识概念。之后转化未tensor类型
		- ![[Pasted image 20241104110651.png]]
	- ![[Pasted image 20241104110021.png]]

##### 1.2 模型数据要求
- 输入数据前提：模型数据与导入数据基本一致，基本不需要在模型外对数据集进行变换
- 数据处理：采用config字典直接包含前期的导入数据，不需要进行后续处理
	- ![[Pasted image 20241104110501.png]]
	- 以下为config情况，![[Pasted image 20241104110411.png]]
- 模型结构构建：exe文件中首先通过build_graph4SE、build_graph4CE、build_graph4SC构建图神经网络，并纳入config字典中，随后网络架构见文件icdm.py中的icdm类网络架构。
- 模型训练：runner根据 icdm_runner.py中的icdm_runner调用ICDM类模型生成icdm类，通过icdm.train进行模型训练
- 模型验证：icdm.train训练中每个epoch会调用icdm.eval来输出在测试集中的数据情况
- ![[Pasted image 20241104112452.png]]

![[Pasted image 20241104154832.png]]

### 三、NCDM模型
这部分参考的NCDM论文，但实现也可通过EduCDM实现完成，EduCDM中的NCDM与KaNCD实现部分相似，可直接参考KaNCD实现。
##### 1.1 导入文件要求
导入文件的过程包含在data_loader.py提供的TrainDataLoader中
- **学生日志情况**：
	- 将train、valid、test作为三个文件进行导入，同时需要config.txt文件对学生人数、项目数以及知识概念数量进行声明，其中每个数据集情况如下：
	- train等数据集为json字典格式，其中外层为列表，每个列表值为一个字典，字典键值包括学生id、itemid、知识概念id以及得分情况
		- ![[Pasted image 20241104152827.png]]
	- user_id不限制学生的id从0开始，该数据的值范围包含于[1-4163]之间，其中规定的学生人数为4163
	- exer_id不限制item的id从0开始，该数据的值范围包含于[1-17746]之间，其中规定的练习题数量为17746
	- 'knowledge_code'为与其他情况不一致，为列表，其中0代表答错，该数据的值范围包含于[1-123]之间，其中规定的知识概念数量为123。
- **Q矩阵**：
	- 这个数据集中的知识概念情况包含在训练数据中，因此不设置Q矩阵

##### 1.2 模型数据要求
- 输入数据前提：通过dataloader转化为tensor数据，其中exer_id、student_id、labels均为单值，input_knowledge则包含对应的知识维度。
- ![[Pasted image 20241104155324.png]]![[Pasted image 20241104155705.png]]
- 模型结构构建：通过Net(student_n, exer_n, knowledge_n)调用模型结构
- 模型训练：根据train函数中，放入所有训练数据，通过net.forward与optim.Adam训练模型
- 模型验证：train函数在每个epoch后调用validate，其中由ValTestDataLoader类调用Val数据或者test数据输出模型的结果

### 四、KaNCD模型
##### 1.1 导入文件要求
基于EduCDM进行实现
- **学生日志情况**：
	- 将train、valid、test作为三个文件进行导入，其中每个数据集情况如下：
	- train等数据集为csv文件，包含user_id, item_id,score，其中user_id、item_id数值范围为[1-max]之间，score中0代表答错，1代表答对。
	- ![[Pasted image 20241105103542.png]]
- **Q矩阵**：
	- Q矩阵以df_item的形式进行出现，其中第一列代表项目id，第二列代表知识点情况，使用列表进行表示。
	- ![[Pasted image 20241105103930.png]]

##### 1.2 模型数据要求
- 输入数据：数据结构为Dataloader返回的数据集，数据集包含学生id、项目id、知识点情况（列表）、得分。
- ![[Pasted image 20241105110319.png]]
- 模型结构构建：通过KanNCD调用模型结构
- ![[Pasted image 20241105110410.png]]
- 模型训练：根据train函数，放入所有Train数据与Val数据。
- 模型验证：根据eval函数进行评估，放入test数据进行训练。

### 五、IRT模型
##### 5.1 导入文件要求
基于EduCDM进行实现
- **学生日志情况**：
	- 将train、valid、test作为三个文件进行导入，其中每个数据集情况与KaNCD一致：
	- train等数据集为csv文件，包含user_id, item_id,score，其中user_id、item_id数值范围为[1-max]之间，score中0代表答错，1代表答对。
	- ![[Pasted image 20241105112321.png]]
- **Q矩阵**：
	- 无论是EM思想还是GD思想，该模型使用时均没有使用Q矩阵信息
##### 1.2 模型数据要求
- 输入数据：最好在输入数据部分进行更改，方便使用（大概逻辑与KaCDM一致，不想看了）
- ![[Pasted image 20241105112807.png]]
- 模型训练：根据train函数，放入所有Train数据，**模型没有使用Val数据**。
- 模型验证：根据eval函数进行评估，放入test数据进行验证。

### 六、DINA模型
##### 6.1 导入文件要求
基于EduCDM进行实现
- **学生日志情况**：
	- GD思想的DINA模型中，将train、valid、test作为三个文件进行导入，其中每个数据集情况与KaNCD一致，而EM思想的DINA模型使用的数据根据与GD思想不一致，暂时放弃EM思想：
	- train等数据集为csv文件，包含user_id, item_id,score，其中user_id、item_id数值范围为[1-max]之间，score中0代表答错，1代表答对。
	- ![[Pasted image 20241105112321.png]]
- **Q矩阵**：
	- EM思想使用Q矩阵信息，Q矩阵情况与其他EduCDM模型一致。
##### 6.2 模型数据要求
- 输入数据：最好在输入数据部分进行更改，方便使用（大概逻辑与KaCDM一致，不想看了）
- 模型训练：根据train函数，放入Train数据与Val数据。
- 模型验证：根据eval函数进行评估，放入test数据进行验证。

