---
aliases:
  - SQL练习
创建日期: 2024-09-15
related:
  - "[[SQL-Tables-基于bilibili]]"
tags:
  - SQL
  - 牛客网
---
### 1、[SQL29 计算用户的平均次日留存率](https://www.nowcoder.com/practice/126083961ae0415fbde061d7ebbde453?tpId=199&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fintelligent)
题目：==左连接（自连接）的使用==
![[Pasted image 20240915161932.png]]

解题：
![[Pasted image 20240915162553.png]]
- FROM与连接采用DISTINCT，是保证同一天答过多道题的信息不纳入date1表中，这是因为DISTINCT是只查询对于id和date的双重组合的DISTINCT，从而同一id同一天的多道答题记录只会被记录一次。
- 连接时LEFT JOIN，保证不满足条件的信息也不会被删除，依然会根据date1的排列进行排序，此外由于date2相对于date1是不重复的，所以连接时，date1不会增加。
- 采用date2中存在id相同，且date2的日期比date1中的日期多一个单位的条件，进行连接，满足条件，则date1右侧多出date2的信息，不满足条件则为空值。
- 因此date2与date1的信息显示情况![[Pasted image 20240915163613.png]]
- 从而右侧代表满足条件的数量，左侧代表每天登录的用户数量，注意此时计数的时候不能使用DISTINCT，因为即使同一个用户多天连续登录，也应该分多天进行统计。

### 2、[SQL30 统计每种性别的人数](https://www.nowcoder.com/practice/f04189f92f8d4f6fa0f383d413af7cb8?tpId=199&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fintelligent)
题目：考察字符串提取函数
![[Pasted image 20240915164746.png]]

解析：
方法1：CASE+模糊查询
![[Pasted image 20240915164816.png]]

方法2：SUBSTR
![[4EF7B5DAE69AD54DE81C49D58020D7A4.png]]

![[Pasted image 20240915164946.png]]
![[Pasted image 20240915165629.png]]

### 3、[SQL33 找出每个学校GPA最低的同学](https://www.nowcoder.com/practice/90778f5ab7d64d35a40dc1095ff79065?tpId=199&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fintelligent)
题目：使用子查询
![[Pasted image 20240915172700.png]]

解法：
- 1、子查询（自己写的）（自连接）
```mysql
SELECT device_id, university, gpa
FROM user_profile up1
WHERE up1.gpa = (
    SELECT min(up2.gpa)
    FROM user_profile up2
    WHERE up2.university = up1.university
)
ORDER BY university;
```
- 2、子查询，WHERE语句
```mysql
-- 子查询进阶
SELECT a.device_id
      ,a.university
      ,a.gpa
from user_profile a
where (university,gpa) in (
              select b.university  -- 不是纯聚合字段，需要配合group by
                    ,min(b.gpa)
              from user_profile b 
              group by b.university )
order by a.university

-- 使用all子查询（自连接）
SELECT a.device_id
      ,a.university
      ,a.gpa
from user_profile a
where gpa <= all (
              select b.gpa
              from user_profile b 
              where a.university = b.university )
order by a.university
```
- 3、JOIN连接，与WHERE的概念基本相同
```mysql
SELECT a.device_id,a.university,a.gpa FROM user_profile a
JOIN (SELECT university,min(gpa) gpa FROM user_profile GROUP BY university) b
on a.university=b.university and a.gpa=b.gpa
ORDER BY university;
```

### 4、[SQL124 统计作答次数](https://www.nowcoder.com/practice/90778f5ab7d64d35a40dc1095ff79065?tpId=199&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fintelligent)
题目：SELECT中的使用
![[Pasted image 20240919202801.png]]

解法：
- 1、自己写的使用子查询
- ![[Pasted image 20240919202921.png]]
- 2、别人写的
- ![[Pasted image 20240919203001.png]]
### 5、[SQL126 平均活跃天数和月活人数](https://www.nowcoder.com/practice/9e2fb674b58b4f60ac765b7a37dde1b9?tpId=240&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fcompany)
题目：考察字符串提取函数
![[Pasted image 20240919210629.png]]

解析：
- 方法1：我写的
![[Pasted image 20240919210710.png]]
	- 行1：格式有问题，采用`date_format(submit_time,` `'%Y%m'``)`更好
	- line 6：GROUP BY报错，并且必须与line1同时修改正确，才能运行
	- line2：第一项应该双重 去重（uid和日期到天），以保证同一用户同一天多次登录，也只被记录一次

- 方法2：标准答案
	- ![[Pasted image 20240919212051.png]]

### 6、[SQL127 月总刷题数和日均刷题数](https://www.nowcoder.com/practice/f6b4770f453d4163acc419e3d19e6746?tpId=240&tqId=2183006&ru=%2Fpractice%2Ff6b4770f453d4163acc419e3d19e6746&qru=%2Fta%2Fsql-advanced%2Fquestion-ranking&sourceUrl=%2Fexam%2Fcompany)
题目：考察`COALESCE`与`WITH ROLLUP`
![[Pasted image 20240920170755.png]]

解析：
- **方法1**：CASE+模糊查询
![[Pasted image 20240920170934.png]]
- COALESCE是由于汇总行，并不对submit_mont中的2021汇总进行添加，因此引入COALESCE
- WITH ROLLUP对GROUP进行汇总
- 出现问题：加上COALESCE后，出现GROUP与SELECT出现矛盾的问题，如何解决？
- 聚合函数目的在于对聚合后的多个数，进行归纳，尽管多个数都是31，但也需要归纳
![[Pasted image 20240920171638.png]]

### 7、[SQL128 未完成试卷数大于1的有效用户](https://www.nowcoder.com/practice/46cb7a33f7204f3ba7f6536d2fc04286?tpId=240&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fcompany)
题目：考察字符串连接以及分组内的字符串连接
![[Pasted image 20240920190514.png]]

解析：
- **方法1**：GROUP_CONCAT+CONCAT_WS
![[Pasted image 20240920190955.png]]
- GROUP_CONCAT旨在将组内的值，按照字符串分隔进行连接
- CONCAT_WS旨在将数据表内的值按照分隔符进行连接

### 8、[SQL134 满足条件的用户的试卷完成数和题目练习数](https://www.nowcoder.com/practice/5c03f761b36046649ee71f05e1ceecbf?tpId=240&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fcompany)
题目：考察对两个表如何放入同一个COUNT表
![[Pasted image 20240922192016.png]]

解析：
- **方法1**：错误：自己编写（先生成前两列，再连接生成后一列）
![[Pasted image 20240922192230.png]]
- 思路：首先对exam表中的信息针对uid进行分组统计，将分组统计表作为一个新表，通过查询进行应用，相当于前两列，并连接question表，注意应该为外连接，防止某些question为0的uid被剔除，并对question进行计数，但是计数前需要限制用户与计数的年份
- 问题：
	- 由于要求限制年份为2021年，使用WHERE时会直接将没有2021question的经历的uid剔除，但是实际要求输出为0，因此不能应用于WHERE语句
	- 这是由于WHERE直接作用于样本，
	- 同时尝试将限制性放在HAVING中，但出现问题在YEAR部分，疑似HVING语句只能对分组特征进行处理
	- ![[Pasted image 20240922193629.png]]
	- **更新：问题的解决可以参考9中方法2**
- **方法2**：正确（先生成前两列，再生成一列和三列，连接两个表，最后对用户进行限制，其中对于question的年份要求直接在一三列中产生，）
![[Pasted image 20240922193815.png]]

### 9、[SQL135 每个6/7级用户活跃情况](https://www.nowcoder.com/practice/a32c7c8590324c96950417c57fa6ecd1?tpId=240&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fcompany)
题目：考察字符串提取函数
![[Pasted image 20240922202139.png]]

解析：
- **方法1**：又臭又长，肯定，我写的
![[Pasted image 20240922202328.png]]
![[Pasted image 20240922202359.png]]
- 注意，这个问题处理时不能按照前一个问题处理，**应该首先在用户信息表中进行连接**，这是由于如果按照1、2列的数据作为FROM表，会导致2列为0的用户直接被剔除，但是结果要求我们输出6，7级用户，即使对应用户没有活跃消息。
- **方法2**：增加一列信息表示所属表
- 一表代替四表，实现了代码的简化
- ``count(distinct if(YEAR(start_time)=2021,date_format(start_time, '%Y%m%d'),null)) act_days_2021,``**代码实现了不同查询要求的WHERE条件不一致，带来的问题，通过COUNT和IF以及NULL的关系，应用于计数中提供了优秀的思路**
![[Pasted image 20240922203023.png]]
![[Pasted image 20240922203041.png]]

### 10、[SQL141 试卷完成数同比2020年的增长率及排名变化](https://www.nowcoder.com/practice/13415dff75784a57bedb6d195262be7b?tpId=240&tags=&title=&difficulty=0&judgeStatus=0&rp=0&sourceUrl=%2Fexam%2Fcompany)
题目：考察年份之间的连接与计算
![[Pasted image 20240928134630.png]]

解析：
- **方法1**：自己的问题
	- 构造出下方总示例表，但问题在于将2020年与2021年进行连接
	- 因此使用两次子循环，会导致代码过于冗余，且需要对连接后的表进行条件筛选
	- ==解决办法2：为构造时仅对tag分组，不使用WHERE语句，对年份的分组在SELECT中通过聚合函数与IF进行构造，存在问题为需要再套依次循环来计算排名==，
	- ==解决办法3：为分别构造2020表与2021表，通过连接来寻找同比结果==
![[Pasted image 20240928134751.png]]

- **方法2**：子循环实现
	- 第一步通过SUM（IF）来实现GROUP BY的效果，优势是能生成两列不一致的数据
	- 百分数利用CONCAT实现
	- 第二部中加入IFNULL，防止exam_cnt_20出现0时，数据库输出NONE，统一默认0，以方便第四部筛选

![[Pasted image 20240928140734.png]]
- **方法3**：两表的连接
	- CAST的实现主要是因为两数加减，SQL一般是unsigned的（无负值），因此需要CAST对数据转一下数据类型，从而使得可以输出负数。
![[Pasted image 20240928141510.png]]
### 11、[SQL144 每月及截止当月的答题情况](https://www.nowcoder.com/practice/1ce93d5cec5c4243930fc5e8efaaca1e?tpId=240&tqId=2183423&ru=/exam/oj&qru=/ta/sql-advanced/question-ranking&sourceUrl=%2Fexam%2Foj%3Fpage%3D1%26tab%3DSQL%25E7%25AF%2587%26topicId%3D240)
题目：新增用户统计
![[Pasted image 20240929102800.png]]

解析：
- **方法1**：为新增用户设置额外0,1变量
	- 利用窗口函数，对于每一行，由窗口函数生成该行uid、start_time对应的首次考试时间，并进行匹配，如果满足条件则返回1不满足则返回0.
![[Pasted image 20240929102910.png]]


### 12、[SQL171 零食类商品中复购率top3高的商品]([零食类商品中复购率top3高的商品_牛客题霸_牛客网 (nowcoder.com)](https://www.nowcoder.com/practice/9c175775e7ad4d9da41602d588c5caf3?tpId=268&tqId=2285711&ru=%2Fpractice%2F65de67f666414c0e8f9a34c08d4a8ba6&qru=%2Fta%2Fsql-factory-interview%2Fquestion-ranking&sourceUrl=%2Fexam%2Fcompany))
题目：产品与用户结合后，最优秀的产品
![[Pasted image 20241011173229.png]]

解析：
- **方法1**：统计满足要求的复购率，
- ![[Pasted image 20241011173826.png]]

- 注意，最大时间是全局最大时间，不是各个产品最近销售的最大时间，t1中添加：MAX(event_time) OVER(PARTITION BY product_id ) AS max_date反而会出错
- 错误1：
	- 本身想在t1中使窗口函数计算各产品product的总用户数，然后直接在t2进行比值并输出，结果报错
	- 报错：程序异常退出, 请检查代码"是否有数组越界等异常"或者"是否有语法错误"，SQL_ERROR_INFO: "This version of MySQL doesn't yet support '
	- 原因在于： 1、t1窗口函数在使用窗口函数时，不能使用COUNT(DISTINCT )，解决办法：可以使用DENSErank替代2、t1成功后，由于t2需要GROUP，因此在GROUP中也不可以直接使用uid_product，考虑可以加上GROUP BY product_id，uid_product
		- ![[Pasted image 20241011190932.png]]
	- 实现：显然本文没有使用这些修改方法，这是后期想到的
- 错误2：
	- 由于t2步需要剔除没有复购的样本，因此t2中不存在产品复购率为0的样本，但结果要求即使没有复购的也需要输出复购率为0，因此使用LEFT JOIN

### 13、[第二高的薪水](https://leetcode.cn/problems/second-highest-salary/)
题目：不存在数据的情况下如何输出NULL
![[Pasted image 20241022171725.png]]
解析：
- **方法1**：使用子查询来避免空值
	- 将第二名的工资作为子查询，如果不存在第二名，则子查询中没有结果，使用IFNULL返回指定值，但是耗时较多，需要进行优化
	- 后来发现IFNULL没有必要，因为子查询为空值时会自动输出NULL
![[Pasted image 20241022171807.png]]

- **方法2**：子查询+LIMIT
	- OFFSET n 代表剔除n个值，集取排名中n+1及以后的值
	- ![[Pasted image 20241022172227.png]]
- 方法3：
	- 为查询添加IFNULL进行干预的执行用时小于非干预用时，即方法三表现优于方法二
	- ![[Pasted image 20241022172517.png]]

### 14、[180. 连续出现的数字 - 力扣（LeetCode）](https://leetcode.cn/problems/consecutive-numbers/?envType=problem-list-v2&envId=database&favoriteSlug=&difficulty=MEDIUM%2CHARD)
题目：考察连续出现问题
![[Pasted image 20241026151938.png]]
解析：
- **方法1**：窗口函数+WHERE
	- 利用窗口函数选择连续三次出现的数字
![[Pasted image 20241023211913.png]]

- **方法2**：==窗口函数+GROUP 绝妙==
	- 由于id依次递增，构造rn，如果数字内部是连续出现，由于两者连续递增的差值都是1，因此相减后会出现相同的值，且不连续的相同数字之间，由于id的存在，会差值之间不一致，从而根据两者进行分组后，筛选组内出现三次数（数都相同）以上的组，满足条件的组对应的num满足条件。
![[Pasted image 20241023211740.png]]
### 15、[给定数字的频率查询中位数](https://leetcode.cn/problems/find-median-given-frequency-of-numbers/)
题目：中位数如何提取（没有中位数函数）
![[Pasted image 20241026152158.png]]
解析：
- **方法1**：窗口函数
	- 正序排序大于总数的一半  且  逆序的排序大于等于总数的一半
![[Pasted image 20241026152300.png]]
- **方法2**：窗口函数
	- 正序排序大于总数的一半   且  正序排序小于等于总数的一半+1

### 16、[2016年的投资](https://leetcode.cn/problems/investments-in-2016/description/?envType=problem-list-v2&envId=database&favoriteSlug=&difficulty=HARD%2CMEDIUM)
题目：问题理解
![[Pasted image 20241027133305.png]]

解析：
- **方法1**：CASE+模糊查询
	- 通过子查询筛选样本
![[Pasted image 20241027133352.png]]
- **方法2**：SUBSTR
	- 第一个筛选条件其实就是要求有重复系数大于等于2的样本
	- 第二个筛选条件就是筛选同一位置只存在一个的样本
	- 结果就是通过计数函数进行统计，侧面完成问题
- ![[Pasted image 20241027133453.png]]





