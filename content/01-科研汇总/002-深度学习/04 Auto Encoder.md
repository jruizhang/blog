> [!NOTE] [一文看懂AutoEncoder模型演进图谱 - 知乎](https://zhuanlan.zhihu.com/p/68903857)
> 本篇根据以上文章进行学习，其中将终点摘要如下


![[Pasted image 20241030193233.png]]
- AutoEncoder框架包含两大模块：编码过程和解码过程。通过encoder（g）将输入样本x映射到特征空间z，即编码过程；然后再通过decoder（f）将抽象特征z映射回原始空间得到重构样本x'，即解码过程。优化目标则是通过最小化重构误差来同时优化encoder和decoder，从而学习得到针对样本输入x的抽象特征表示z。
- 对于基于神经网络的AutoEncoder模型来说，则是encoder部分通过逐层降低神经元个数来对数据进行压缩；decoder部分基于数据的抽象表示逐层提升神经元数量，最终实现对输入样本的重构。
- 由于AutoEncoder通过神经网络来学习每个样本的唯一抽象表示，这会带来一个问题：当神经网络的参数复杂到一定程度时AutoEncoder很容易存在过拟合的风险。

> 以下为Auto Encoder 的一些案例
#### Denoising AutoEncoder
为降低模型的过拟合，分别可以采用两种思路进行降低：
- 引入随机噪声：在传统AutoEncoder输入层加入随机噪声或者随机将某些值取为0，来增强模型的鲁棒性
- 正则化保证：结合正则化思想，通过在AutoEncoder目标函数中加上encoder的Jacobian矩阵范式来约束使得encoder能够学到具有抗干扰的抽象特征。
![[Pasted image 20241030194016.png]]
### Variational AutoEncoder
VAE比较大的不同点在于：VAE不再将输入x映射到一个固定的抽象特征z上，而是假设样本x的抽象特征z服从（μ，σ^2）的正态分布，然后再通过分布生成抽象特征z。最后基于z通过decoder得到输出。
由于抽象特征z是从正态分布采样生成而来，因此VAE的encoder部分是一个生成模型，然后再结合decoder来实现重构保证信息没有丢失。VAE是一个里程碑式的研究成果，倒不是因为他是一个效果多么好的生成模型，主要是提供了一个结合概率图的思路来增强模型的鲁棒性。后续有很多基于VAE的扩展，包括infoVAE、betaVAE和factorVAE等。
![[Pasted image 20241030194432.png]]