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

