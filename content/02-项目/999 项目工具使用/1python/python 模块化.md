函数是串代码
模块其实就是.py文件
包其实就是实现某个方法的函数的全体
- 默认必须有__init__的文件，一般用作导入包下的其他模块，最好不要乱写这部分
- ![[Pasted image 20240419203955.png]]
- ![[Pasted image 20240419211649.png]]
- ![[Pasted image 20240420101116.png]]
# 编程
``` python
add = lambda x, y: x + y result = add(5, 3) # result将会是8
#`add`就是一个Lambda函数，它接受两个参数`x`和`y`，并返回它们的和。然后我们可以像调用普通函数一样调用这个Lambda函数。
```

