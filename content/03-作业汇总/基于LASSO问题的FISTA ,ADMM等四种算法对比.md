###### 注：统计计算期末作业  2023112069   张金睿      

#### 一、问题重述：

**问题设计**：使用Proximal Gradient Descent method、Subgradient Method、FISTA以及ADMM等方法解决以下最优化问题，并比较各个模型的优化结果。

给定矩阵A∈Rm×n, 向量b∈Rm 和正数μ，求解以下优化问题：
$$ \min_{\frac{1}{2}}{\|Ax-b\|^2+\mu\|x\|_1} $$

测试参数设置如下：
- $n = 1024$
- $m = 512$
- $A = \text{randn}(m,n)$
- $u = \text{sprandn}(n,1,0.1)$
- $b = Au+0.1*\text{randn}(n,1)$
- $\mu = 1e^{-3}$

**任务设计**：分别应用上述四种方法来求解该优化问题，并对比分析各模型的优化效果。

#### 二、模型介绍：

下面分别阐述Proximal Gradient Descent（PGD）、Subgradient Method、Alternating Direction Method of Multipliers (ADMM)、和Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)的算法流程设计。
##### 2.1  Proximal Gradient Descent (PGD)

PGD适用于优化形式为$f(x) + g(x)$的问题，其中$f(x)$是光滑的（具有Lipschitz连续梯度），而$g(x)$是凸的但可能非光滑的。

**算法流程设计：**

1. 初始化：选取初始点$x_0$和步长参数$\alpha > 0$。

2. 对于每一步$k = 1, 2, 3, ...$

   a. 计算梯度：$\nabla f(x_k)$。

   b. 进行梯度下降和近似操作：$$x_{k+1} = \arg\min_x \left\{ g(x) + \langle \nabla f(x_k), x - x_k \rangle + \frac{1}{2\alpha}\|x - x_k\|^2 \right\}$$
   c. 更新$x_k$为$x_{k+1}$。

   d. 检查收敛条件，如果不满足则继续迭代。

##### 2.2  Subgradient Method

Subgradient Method适用于非光滑但凸的优化问题。

**算法流程设计：**

- 1. 初始化：选取初始点$x_0$和步长序列$\alpha_k > 0$。

- 2. 对于每一步$k = 1, 2, 3, ...$

   a. 计算$f(x_k)$的一个次梯度$\partial f(x_k)$。

   b. 更新点：$x_{k+1} = x_k - \alpha_k \cdot s_k$，其中$s_k \in \partial f(x_k)$。

   c. 检查收敛条件，如果不满足则继续迭代。

##### 2.3  Alternating Direction Method of Multipliers (ADMM)

ADMM适用于分解型优化问题，形式为$\min_x f(x) + g(z)$ s.t. $Ax + Bz = c$。

**算法流程设计：**

1. 初始化：选取初始点$x_0, z_0$和拉格朗日乘子$u_0$，以及惩罚参数$\rho > 0$。

2. 对于每一步$k = 1, 2, 3, ...$

   a. 更新$x$：$\hat{x}_{k+1} = \arg\min_x \left\{ f(x) + (\rho/2)\|Ax + Bz_k - c + u_k\|^2 \right\}$。

   b. 更新$z$：$\hat{z}_{k+1} = \arg\min_z \left\{ g(z) + (\rho/2)\|A\hat{x}_{k+1} + Bz - c + u_k\|^2 \right\}$。

   c. 更新拉格朗日乘子：$u_{k+1} = u_k + \rho(A\hat{x}_{k+1} + B\hat{z}_{k+1} - c)$。

   d. 检查收敛条件，如果不满足则继续迭代。

##### 2.4  Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

FISTA是PGD的一个加速版本，用于解决形式为$min_x f(x) + g(x)$ $的问题，其中$$f(x)$是光滑的，而$g(x)$是非光滑的。

**算法流程设计：**

1. 初始化：选取初始点$x_0$，以及步长参数$\alpha > 0$和Lipschitz常数$L$。

2. 设置$y_0 = x_0$和$t_0 = 1$。

3. 对于每一步$k = 1, 2, 3, ...$

   a. 计算梯度：$\nabla f(y_k)$。

   b. 近似操作：$x_{k+1} = \arg\min_x \left\{ g(x) + \langle \nabla f(y_k), x - y_k \rangle + \frac{1}{2\alpha}\|x - y_k\|^2 \right\}$。

   c. 更新$y$：$y_{k+1} = x_{k+1} + \left( \frac{t_k - 1}{t_{k+1}} \right)(x_{k+1} - x_k)$，其中$t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$。

   d. 检查收敛条件，如果不满足则继续迭代。

#### 三、结果展示：
本作业设定不同的惩罚项系数，分别为 $\mu = 0.5、\mu = 0.1、\mu = 1e^{-2}、\mu = 1e^{-3}$，从而比较在解决不同的lasso问题中的四种优化算法的优化效果，其中设定最大迭代次数为2000次，图片中仅展示到1600次，以方便观察不同优化算法图像的区别。

> [!NOTE] Title
注意：由于log的图像更方便比较不同方法的函数曲线，且要求函数值大于零，因此本节中的函数值为损失函数的值，而损失函数值相较于真实解的误差值将在第四节进行展示。

##### 3.1  惩罚项系数$\mu = 0.5$下的lasso优化结果：
![[loss Comparison of different methods with 0.5 2.png]]
**模型优化结果展示如下：**
- SGD：未达到收敛条件，损失函数的最终优化值为80.99。
- ADMM：未达到收敛条件，损失函数的最终优化值为45.08。
- FISTA：在第234次迭代时达到收敛条件，损失函数的最终优化值为44.72。
- PGD：在第41次迭代时达到收敛条件，损失函数的最终优化值为12389.82。

**模型分析结果呈现如下：**
- FISTA优化值与ADMM优化值较为接近，其中FISTA率先达到收敛条件，且损失函数值最小；
- SGD劣于FISTA与ADMM算法，优化值远远不如前两种算法；
- PGD则陷入了局部最优解。

##### 3.2  惩罚项系数$\mu = 0.1$下的lasso优化结果：
![[loss Comparison of different methods with 0.1.png]]
**模型优化结果展示如下：**
- SGD：未达到收敛条件，损失函数的最终优化值为16.72。
- ADMM：未达到收敛条件，损失函数的最终优化值为8.99。
- FISTA：在第409次迭代时达到收敛条件，损失函数的最终优化值为8.97。
- PGD：在第77次迭代时达到收敛条件，损失函数的最终优化值为839.89。

**模型分析结果呈现如下：**
- FISTA优化值与ADMM优化值较为接近，其中FISTA率先达到收敛条件，优化值也更小，但ADMM的下降速度更快；
- SGD劣于FISTA与ADMM算法，优化值不如前两种算法，模型不收敛；
- PGD则依然陷入了局部最优解。

##### 3.3  惩罚项系数$\mu = 0.01$下的lasso优化结果：
![[loss Comparison of different methods with 0.01.png]]
**模型优化结果展示如下：**
- SGD：未达到收敛条件，损失函数的最终优化值为1.7。
- ADMM：在第435次迭代时达到收敛条件，损失函数的最终优化值为0.9。
- FISTA：在第421次迭代时达到收敛条件，损失函数的最终优化值为0.9。
- PGD：在第218次迭代时达到收敛条件，损失函数的最终优化值为10.97。

**模型分析结果呈现如下：**
- FISTA优化值与ADMM优化值相同，其中FISTA与ADMM的收敛迭代次数相近，FISTA所需迭代次数更小，但ADMM的下降速度更快。
- SGD劣于FISTA与ADMM算法，优化值不如前两种算法，模型依然不收敛；
- PGD则依然陷入了局部最优解。

##### 3.4  惩罚项系数$\mu = 0.001$下的lasso优化结果：
![[loss Comparison of different methods with 0.001.png]]
**模型优化结果展示如下：**
- SGD：未达到收敛条件，损失函数的最终优化值为0.19。
- ADMM：在第239次迭代时达到收敛条件，损失函数的最终优化值为0.09。
- FISTA：在第400次迭代时达到收敛条件，损失函数的最终优化值为0.09。
- PGD：未达到收敛条件，损失函数的最终优化值为0.73。

**模型分析结果呈现如下：**
- FISTA优化值与ADMM优化值相同，其中ADMM超越了FISTA算法实现了超越，所需收敛迭代次数更少，下降速度也更快。
- SGD劣于FISTA与ADMM算法，优化值不如前两种算法，模型依然不收敛；
- PGD则依然弱于以上三种方法，容易陷入局部最优解，PGD下降速度快于SGD，但优化值不如SGD。
#### 四、模型比较：
本届对四种的迭代值、误差值（相较于解析解的差值）、迭代次数、迭代耗时进行统计，模型结果如下表：

| 方法                | 迭代记录  | μ=0.5    | μ=0.1   | μ=0.01 | μ=1e−3 |
| ----------------- | ----- | -------- | ------- | ------ | ------ |
| Proximal gradient | 迭代值：  | 12389.82 | 839.89  | 10.97  | 0.73   |
|                   | 误差值：  | 12387.48 | 837.55  | 8.63   | -1.62  |
|                   | 迭代次数： | 41       | 77      | 218    | 2000   |
|                   | 迭代耗时： | 0.06s    | 0.11s   | 0.28s  | 1.70s  |
| Subgradient       | 迭代值：  | 80.99    | 16.72   | 1.7    | 0.19   |
|                   | 误差值：  | 78.65    | 14.38   | -0.64  | -2.16  |
|                   | 迭代次数： | 2000     | 2000    | 2000   | 2000   |
|                   | 迭代耗时： | 4.41s    | 4.34s   | 4.28s  | 4.30s  |
| ADMM              | 迭代值：  | 45.08    | 8.99    | 0.9    | 0.09   |
|                   | 误差值：  | 42.74    | 6.65    | -1.44  | -2.25  |
|                   | 迭代次数： | 2000     | 2000    | 435    | 239    |
|                   | 迭代耗时： | 1.576s   | 0.872s  | 0.247s | 0.072s |
| FISTA             | 迭代值：  | 44.72    | 8.97    | 0.9    | 0.09   |
|                   | 误差值：  | 42.38    | 6.63    | -1.44  | -2.25  |
|                   | 迭代次数： | 234      | 409     | 421    | 400    |
|                   | 迭代耗时： | 0.033s   | 0.0375s | 0.043s | 0.044s |
**模型结果对比结果总结如下**：
- ADMM与FISTA在处理LASSO问题时，相较于PGD与SGD表现出了强烈的优势;
- ADMM与FISTA在寻找最优解的过程中，FISTA的耗时更短且迭代解略优于FIST;
- SGD优于PGD方法，SGD寻找到的迭代解均优于PGD，但SGD的所需迭代次数与迭代耗时较长;
- 在不同的惩罚项系数求解LASSO问题中，FISTA在迭代值、迭代次数、迭代耗时三个方面均优于其他三种算法。
综上，FISTA在解决LASSO问题中表现出了明显的优越性，因此在解决LASSO问题求解时，应优先选择FISTA进行计算

#### 五、附录
**代码展示如下：**

``` python
import numpy as np  
import os  
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  
import torch  
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
import time  
# 定义prox操作  
def prox(x, μ=0.5):  
    return np.sign(x) * np.maximum(0, np.abs(x) - μ)  
  
def proximal_gradient_hui_descent(A, b, x0, max_iter=1000, tol=1e-6, μ=0.5, lr_init=1e-3, alpha=0.9, c1=0.5):  
    x = x0.copy()  
    values = []  
  
    for i in range(max_iter):  
        # 计算损失  
        loss = 0.5 * np.linalg.norm(np.dot(A, x) - b) ** 2  
        value = loss + μ * np.linalg.norm(x, ord=1)  
        values.append(value)  
  
        # 计算梯度  
        grad = A.T @ (np.dot(A, x) - b)  
  
        lr = lr_init  
        for j in range(1000):  
            new_x = x - lr * grad  
            new_loss = 0.5 * np.linalg.norm(np.dot(A, new_x) - b) ** 2  
  
            # Armijo条件  
            if new_loss <= loss + c1 * lr * np.dot(grad.flatten(), (new_x - x).flatten()):  
                x_new = prox(new_x, μ)  
                x_new[np.abs(x_new) < 1e-4] = 0  
                break  
            else:  
                lr *= alpha  
            if j == 999:  
                x_new = x  
  
        # 检查收敛条件  
        if np.linalg.norm(x_new - x) < tol:  
            print(f"proximal_gradient（回溯步长）算法收敛于第{i}次迭代")  
            break  
  
        # 更新参数  
        x = x_new  
  
    # 返回最终解和损失序列  
    return x, values  
  
  
  
#次梯度法  
def set_step(k, opts):  
    """  
    根据给定的迭代次数k和选项opts，返回步长alpha。  
    """    if opts['step_type'] == 'fixed':  
        alpha = opts['alpha0']  
    elif opts['step_type'] == 'diminishing':  
        alpha = opts['alpha0'] / np.sqrt(k)  
    elif opts['step_type'] == 'diminishing2':  
        # 注意：这里MATLAB代码没有明确给出diminishing2的具体公式，我假设它与diminishing类似  
        # 但可能有一个不同的衰减因子或形式。这里我们简单地使用1/k作为例子。  
        alpha = opts['alpha0'] / k  
    else:  
        raise ValueError("Unknown step_type: {}".format(opts['step_type']))  
    return alpha  
  
def l1_subgrad(x0, A, b, mu, opts):  
    """  
    L1正则化的次梯度法优化函数。  
    """    # 设置默认参数  
    if 'maxit' not in opts: opts['maxit'] = 2000  
    if 'thres' not in opts: opts['thres'] = 1e-4  
    if 'step_type' not in opts: opts['step_type'] = 'fixed'  
    if 'alpha0' not in opts: opts['alpha0'] = 0.01  
    if 'ftol' not in opts: opts['ftol'] = 0  
    # mu = 0.5  
    # 初始化变量  
    x = np.array(x0, dtype=float)  
    out = {  
        'f_hist': np.zeros(opts['maxit']),  
        'f_hist_best': np.zeros(opts['maxit']),  
        'g_hist': np.zeros(opts['maxit']),  
    }  
    f_best = np.inf  
  
    # 迭代主循环  
    for k in range(1, opts['maxit'] + 1):  
        r = A.dot(x) - b  
        g = A.T.dot(r)  
        norm_r_squared = np.linalg.norm(r, 2) ** 2  
        if np.isfinite(norm_r_squared):  # 检查是否为有限数  
            f_now = 0.5 * norm_r_squared + mu * np.linalg.norm(x, 1)  
        else:  # 发生溢出时设置为np.nan  
            f_now = np.nan  
  
        # 记录可微部分的梯度的范数  
        out['g_hist'][k - 1] = np.linalg.norm(r, 2)  
        # 记录当前目标函数值  
        f_now = 0.5 * np.linalg.norm(r, 2) ** 2 + mu * np.linalg.norm(x, 1)  
        out['f_hist'][k - 1] = f_now  
  
        # 更新历史最优目标函数值  
        f_best = min(f_best, f_now)  
        out['f_hist_best'][k - 1] = f_best  
  
        # 将接近0的值设为0  
        x[np.abs(x) < opts['thres']] = 0  
        sub_g = g + mu * np.sign(x)  
  
        # 计算步长并执行迭代  
        alpha = set_step(k, opts)  
        # 检查停机准则，如果x小于阈值，则停止  
        if k > 1 and np.linalg.norm(alpha * sub_g) < opts['ftol']:  
            if opts['step_type'] == 'fixed':  
                print(f"Converged after {k} iterations with fixed step size{opts['alpha0']}")  
            else:  
                print(f"Converged after {k} iterations with{opts['step_type']} ")  
                break  
        x = x - alpha * sub_g  
  
        # 记录迭代终止时的信息  
    out['itr'] = k  
    out['f_hist'] = out['f_hist'][:k]  
    out['f_hist_best'] = out['f_hist_best'][:k]  
    out['g_hist'] = out['g_hist'][:k]  
  
    # 返回结果  
    return x, out  
  
  
  
  
  
  
  
diedai = 1  
for mu in [0.1, 0.01, 0.001 ,0.5]:  
    # 设置随机种子  
    torch.manual_seed(1234)  
    n = 1024  
    m = 512  
    A = torch.randn(m, n)  
    # 生成n行1列的稀疏向量，其中90%的元素为0.  
    u = torch.randn(n, 1)  
    u[torch.randperm(n)[:int(0.9 * n)]] = 0  
    b = A @ u + 0.1 * torch.randn(m, 1)  
    μ = 1e-3  
    x0 = torch.zeros(n, 1)  
    iter = 2000  
  
    A = A.numpy()  
    b = b.numpy()  
    x0 = x0.numpy()  
    u = u.numpy()  
  
    #导出b  
    np.savetxt('b.csv', b, delimiter=',')  
    value_final=0.5 * np.linalg.norm(np.dot(A, u) - b) ** 2  
    print(f"Final value: {value_final}")  
    # b=b.flatten()  
    #计时  
    start = time.time()  
    x_opt_hui, value_pro= proximal_gradient_hui_descent(A, b, x0,lr_init=1,c1=0.01,max_iter=iter,μ=mu)  
    end = time.time()  
    # print(f"Proximal gradient time cost: {end - start:.2f}s")  
  
    opts = {  
            'maxit': iter,  
            'alpha0': 0.0005,  
            'step_type': 'diminishing',  
            'ftol': 1e-6  
        }  
    start = time.time()  
    x_fix, value_sub = l1_subgrad(x0, A, b, mu, opts)  
    end = time.time()  
    # print(f"Subgradient method time cost: {end - start:.2f}s")  
    # 读取admm_data1.csv文件  
    value_admm = np.loadtxt(f'admm_data{diedai}.csv', delimiter=',')  
  
    # 读取output_data1.xlsx文件  
    fista = pd.read_excel(f'output_data{diedai}.xlsx', sheet_name='Sheet1', header=None)  
    # 读取第三列  
    value_fista = fista.iloc[:, 2].values  
  
    #删除nan值  
    value_fista = value_fista[~np.isnan(value_fista)]  
    value_admm = value_admm[~np.isnan(value_admm)]  
  
    plt.figure(figsize=(12, 8))  
    plt.plot(value_sub['f_hist']-value_final, label=f'Subgradient method result {value_sub["f_hist"][-1]-value_final:.2f}')  
    plt.plot(value_admm[0:1599]-value_final, label=f'ADMM result {value_admm[-1]-value_final:.2f}')  
    plt.plot(value_fista-value_final, label=f'FISTA result {value_fista[-1]-value_final:.2f}')  
    plt.plot(value_pro-value_final, label=f'Proximal gradient  result {value_pro[-1]-value_final:.2f}')  
    plt.yscale('log')  
    plt.xlabel('Iteration')  
    plt.ylabel('Objective value')  
    plt.title(f'Comparison of different methods with μ={mu}')  
    plt.legend()  
    plt.grid()  
    plt.savefig(f'Comparison of different methods with {mu}.png')  
    plt.show()  
    print(f"Comparison of different methods with {mu}/n")  
    print(f"Proximal gradient：{len(value_pro)-1}")  
    print(f"Subgradient method：{len(value_sub['f_hist'])-1}")  
    print(f"ADMM：{len(value_admm)-1}")  
    print(f"FISTA：{len(value_fista)-1}")  
    # print(f'Proximal gradient result {value_pro[-1]:.2f}')  
    # print(f'Subgradient method result {value_sub["f_hist"][-1]:.2f}')    # print(f'ADMM result {value_admm[-1]:.2f}')    # print(f'FISTA result {value_fista[-1]:.2f}')  
    diedai += 1  
  
  
diedai = 1  
for mu in [0.1, 0.01, 0.001 ,0.5]:  
    # 设置随机种子  
    torch.manual_seed(1234)  
    n = 1024  
    m = 512  
    A = torch.randn(m, n)  
    # 生成n行1列的稀疏向量，其中90%的元素为0.  
    u = torch.randn(n, 1)  
    u[torch.randperm(n)[:int(0.9 * n)]] = 0  
    b = A @ u + 0.1 * torch.randn(m, 1)  
    μ = 1e-3  
    x0 = torch.zeros(n, 1)  
    iter = 2000  
  
    A = A.numpy()  
    b = b.numpy()  
    x0 = x0.numpy()  
    u = u.numpy()  
    value_final=0.5 * np.linalg.norm(np.dot(A, u) - b) ** 2  
    print(f"Final value: {value_final}")  
    # b=b.flatten()  
    #计时  
    start = time.time()  
    x_opt_hui, value_pro= proximal_gradient_hui_descent(A, b, x0,lr_init=1,c1=0.01,max_iter=iter,μ=mu)  
    end = time.time()  
    print(f"Proximal gradient time cost: {end - start:.2f}s")  
  
    opts = {  
            'maxit': iter,  
            'alpha0': 0.0005,  
            'step_type': 'diminishing',  
            'ftol': 1e-6  
        }  
    start = time.time()  
    x_fix, value_sub = l1_subgrad(x0, A, b, mu, opts)  
    end = time.time()  
    print(f"Subgradient method time cost: {end - start:.2f}s")  
    # 读取admm_data1.csv文件  
    value_admm = np.loadtxt(f'admm_data{diedai}.csv', delimiter=',')  
  
    # 读取output_data1.xlsx文件  
    fista = pd.read_excel(f'output_data{diedai}.xlsx', sheet_name='Sheet1', header=None)  
    # 读取第三列  
    value_fista = fista.iloc[:, 2].values  
  
    #删除nan值  
    value_fista = value_fista[~np.isnan(value_fista)]  
    value_admm = value_admm[~np.isnan(value_admm)]  
  
    plt.figure(figsize=(12, 8))  
    plt.plot(value_sub['f_hist'][0:1599], label=f'Subgradient method result {value_sub["f_hist"][-1]:.2f}')  
    plt.plot(value_admm[0:1599], label=f'ADMM result {value_admm[-1]:.2f}')  
    plt.plot(value_fista[0:1599], label=f'FISTA result {value_fista[-1]:.2f}')  
    plt.plot(value_pro[0:1599], label=f'Proximal gradient  result {value_pro[-1]:.2f}')  
    plt.yscale('log')  
    plt.xlabel('Iteration')  
    plt.ylabel('Objective value')  
    plt.title(f'Comparison of different methods with μ={mu}')  
    plt.legend()  
    plt.grid()  
    plt.savefig(f'loss Comparison of different methods with {mu}.png')  
    plt.show()  
    # print(f"Comparison of different methods with {mu}/n")  
    # print(f"Proximal gradient：{len(value_pro)-1}")  
    # print(f"Subgradient method：{len(value_sub['f_hist'])-1}")  
    # print(f"ADMM：{len(value_admm)-1}")  
    # print(f"FISTA：{len(value_fista)-1}")  
    # print(f'Proximal gradient result {value_pro[-1]:.2f}')    # print(f'Subgradient method result {value_sub["f_hist"][-1]:.2f}')    # print(f'ADMM result {value_admm[-1]:.2f}')    # print(f'FISTA result {value_fista[-1]:.2f}')  
    diedai += 1
```

