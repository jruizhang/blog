>【中字】SQL进阶教程 | 史上最易懂SQL教程！10小时零基础成长SQL大师！！
>SQL的功能是实现数据库的查添删改

### SQL 基本整理 （Base bilibili）
#### SQL-查
##### 1、SELECT
- **1、DISTINCT**:
	- distinct必须放在开头，select id, distinct name from A; --也会提示错误
	- distinct语句中select显示的字段只能是distinct指定的字段，其他字段是不可能出现的。例如，假如表A有“备注”列，如果想获取distinc name，以及对应的“备注”字段，想直接通过distinct是不可能实现的。
	- ![[Pasted image 20240911164752.png]]
- **2、聚合函数**:
	- 说明：聚合函数只运行非空数值，COUNT(* ) 计算所有的数量，无论
	- ![[Pasted image 20240912163856.png]]
	- 使用聚合函数不要求必须使用GROUP BY
	- COUNT( * )代表根据类别进行计数，特别的点是，他不会忽略空缺值，对行进行求和
	- 注意：这些函数会应用于一列来产生单个结果，不能和列进行放在，如果想可以考虑使用子查询。
- **3、数值函数**：
	- 说明：对数值进行四舍五入、取整、绝对值、随机数的操作
	- ![[Pasted image 20240913175757.png]]
- **4、字符串函数**：
	- 对字符串进行求长、==大小写转换==、取字、提出等操作
	- ![[Pasted image 20240913180626.png]]
	- **字符串提取函数**![[Pasted image 20240915165835.png]]
		- ![[4EF7B5DAE69AD54DE81C49D58020D7A4.png]]
	- 截取法可靠性最差，替换法的删除法可靠性较差，字段切割较好
	- 注意：空格在字符串中也算字符
	- **CONCAT_WS**：通过字符串连接函数
		- ![[Pasted image 20240920190134.png]]
- **5、日期函数**：
	- 直接显示的函数：
		- ![[Pasted image 20240913193229.png]]
		- 其中，ECTRACT为提取函数，比较通用
		- CURDATE和CURTIME相当于把NOW拆分开
	- 用户端日期函数：
		- ![[Pasted image 20240913194219.png]]
	- 日期运算函数(DATEDIFF只计算天数之差，不计算分钟)
		- ![[Pasted image 20240913194400.png]]
	- 时间运算函数：
		- ![[Pasted image 20240916144410.png]]
	- ==日期时间函数总结==
		- ![[Pasted image 20240926165253.png]]
		- 其中前者是对日记加时间（精确时间）进行运算，后三个均为对纯日期（精确日期）进行运算，对前两者进行更加详细的比较
		- datediff()函数的作用是求日期差，也就是把一个时间的日期部分取出来求差。例如：'2021-09-05 11:00:00'和'2021-09-04 12:00:00'这两个日期，datediff只取2021-09-05和2021-09-04求日期差，并不会管后面的时间部分，结果为1。
		- timestampdiff()函数的作用则是求时间戳的差，例如：'2021-09-05 11:00:00'和'2021-09-04 12:00:00'这两个日期，datediff只会先求出这个日期的时分秒差，之后再转换成天数来求日期差，结果为0。
- **6、空值函数**：
	- IFNULL函数：如果空，则返回自定义字符
		- ![[Pasted image 20240913194559.png]]
	- COALESCE函数：如果空，可选择同表格另一个特征进行替代，如果另一个也空，则返回自定义字符
		- ![[Pasted image 20240913195144.png]]
- **7、条件函数**：
	- IF函数：
		- ![[Pasted image 20240913195335.png]]
	- CASE-END函数：
		- ![[Pasted image 20240913195453.png]]
- **8、窗口函数**：
	- 窗口函数能实现窗口的多样性，但窗口函数的结果再次调用需要再嵌套一个子循环
		- 例如GROUP BY后可以通过HAVING对GROUP窗口内的数据进行筛选，但想要对窗口函数的输出结果进行筛选，则需要再嵌套一个SELECT语句。
	- [SQL开窗函数（窗口函数）详解_sql开窗函数详解-CSDN博客](https://blog.csdn.net/qq_31183727/article/details/107023293)
	- ![[Pasted image 20240927213342.png]]
	- 
- **9、转换函数**：
	- 形式：CAST(  varable AS type)
	- ![[Pasted image 20240928142139.png]]![[Pasted image 20240928142203.png]]
##### 2、JOIN，UNION
- **INNER JOIN** ：连接两个表，但只返回满足条件的结果
	- 形式：JOIN  table  t  ON    equal
	- 两表连接SELECT变量时，对于特定的变量不需要加表.，对于共有的必须指定表
	- **内连接**：也可以自连接，连接自己的表但改名即可。
		- ![[Pasted image 20240911200653.png]]
	- **多重连接**：
		- ![[Pasted image 20240911200843.png]]
	- **隐式连接**：
		- ![[Pasted image 20240911200917.png]]
- **OUTER JOIN** ：连接两个表，根据指令保存两表中对应的所有值（相当于取并集）
	- 形式：LEFT JOIN 保留单词左边的表，右边的表也需要安排，所以左边的表的信息会被安排多遍来与右边的表在一行或者没有适配的则右排为空左侧依然保留；右连接保留单词右边的表，内连接则是取交集
		- LEFT JOIN结果展示
			- ![[Pasted image 20240911201826.png]]
		- RIGHT JOIN结果展示
			- ![[Pasted image 20240911201907.png]]
	- 尽量不适用右连接，尽量使用右连接
	- **自连接（外）**：与自连接内相似
	- **USING**：连接时两个表有相同列名时，才可以使用using，
		- - USING的多种形势，连续使用（与多表连接相似连接一个则已经形成一个表，只是连接表的所属列名不一致，此时再次使用是，由于USING只关注列名，因此前两次合并的表的列名与后边的列名再对应识别，尽管这时使用的可能是第一次连接的表的列名），一次使用多个（两个表格通过多个表格进行连接）
		- ![[Pasted image 20240911202806.png]]
		
		- ![[Pasted image 20240911202841.png]]
- **NATURAL JOIN**：模型自动选择连接，一般不用。
- **CROSS JOIN**：交叉连接，不需要等式，直接对表格进行连接。
	- ![[Pasted image 20240911204242.png]]
	- 两种形式都是交叉连接，后者与内连接的隐式连接相似，但缺少WHERE，从而隐式连接相当于交叉连接的条件结果。
- **UNION**：对表的列的合并，JOIN主要用于列的内容的增添
	- UNION只是对列的合并，对数据的要求：列的数量一致，对数据的列名是否匹配没有要求。
		- ![[Pasted image 20240911205214.png]]
	- 要求不同查询的列的数量一致，变量名根据第一个查询的列名而设定。
		- ![[Pasted image 20240911205137.png]]![[Pasted image 20240911205154.png]]
- 
##### 3、WHERE
- **比较运算符**：>,>=,<,<=,=,!=,<>,仍然不区分大小写
- **关系组合方式**：AND、OR、NOT及其组合，（）用于明确指定运算符的优先级
- **in 运算符**：精确查找，替代了多个or
	- ![[Pasted image 20240911170301.png]]
- **BETWEEN 运算符**：精确查找，替代了多个AND
	- ![[Pasted image 20240911170529.png]]
- **LIKE 运算符**：模糊查找；
	- % 表示任何字符串，可以代表没有也可以代表无数多个字符；
	- _ 代表一个字符
	- ![[Pasted image 20240911170941.png]]
- **REGEXP 运算符**：正则表达式模糊查找，LIKE运算符的替代
	- '^str'表示需要找以str为开头的的数据
	- 'str$'表示需要找以str为结尾的数据
	- 'str1|str2多个条件的并，其中str1后的空格等因素也被认定为字符串，类似LIKE '%str1%' OR '%str2%'
	- []代表与字母相邻的多个选项的并
	- ![[Pasted image 20240911172510.png]]
- **ALL、ANY、SOME关键字**：用于一群值之前，ANY（，，）
	- ALL 表示使得ALL后的一群值都一次被放入不等式，并满足条件
		- ![[Pasted image 20240912202404.png]]![[Pasted image 20240912202418.png]]
	- ANY表示任何一个满足条件就可以
		- ![[Pasted image 20240912202840.png]]![[Pasted image 20240912202820.png]]
	- SOME 是 ANY的一个同义词，功能相同。
- **EXISTS**： EXISTS运算符通常与子查询一起使用，子查询返回一个布尔值（TRUE 或 FALSE）来确定是否存在满足给定条件的行
	- EXISTS是一个布尔运算符返回true或false，返回的是TRUE 和FALESE，而不是整个子查询，以测试一个“存在”状态。如果子查询返回任何行，则EXISTS运算符返回true，否则返回false
	- 从前面几个例子来看，子查询不存在时，则自身查询也不存在；子查询存在时，则自身查询内容将显示。使用EXISTS运算符要么就是匹配返回表中所有数据，要么就是不匹配不返回任何数据，好像EXISTS运算符并没有太大意义，其实上面这两个例子在实际中并不常用，**EXISTS运算符的真正意义只有和相关子查询一起使用才更有意义**。相关子查询中引用外部查询中的这个字段，这样在匹配外部子查询中的每行数据的时候相关子查询就会根据当前行的信息来进行匹配判断了，这样就可以完成非常丰富的功能。
	- SQL流程类似依次SELECT数据，每次判断是否满足WHERE，然后满足的话则选择
	- **性能优化**：仅当子查询返回任何行时，EXISTS 才执行主查询。
	- **简洁性**：与 NOT IN 或 LEFT JOIN 等其他方法相比，EXISTS 语法更简洁。
	- **可扩展性**：EXISTS 可用于各种查询，包括嵌套查询和联合查询。
	- ![[Pasted image 20240913113139.png]]
	- 
##### 4、GROUP BY
- **说明**：GROUP BY分组统计，位置永远在FROM和WHERE之后，在ORDER_BY之前
- **形式**：`GROUP BY variable`
	- 单个分组条件
		- ![[Pasted image 20240912172111.png]]
	- 多个分组条件
		- ![[Pasted image 20240912172250.png]]
- HAVING语句，对于分类统计后的查询，无法通过WHERE语句对数据进行选择（因为此时还没有total_sales变量）
- **ROLLUP**：`WITH ROLLUP `
	- 用于对聚合函数的值进行汇总，having不改变值RPOLLUP的结果显示
		- ![[Pasted image 20240912173143.png]]![[Pasted image 20240912173204.png]]
	- 多条件分组时，显示每组以及整个结果集的汇总值
		- ![[Pasted image 20240912173359.png]]![[Pasted image 20240912173419.png]]
	- 运行ROLLUP时，不能在GROUP BY中使用别名
		- ![[Pasted image 20240912173509.png]]
-  **GROUP_CONCAT**：
	- `GROUP_CONCAT` 函数用于将查询结果按指定顺序连接成一个字符串。通常结合 `GROUP BY` 子句一起使用，可以将同一组的多个字段值连接成一个字符串。
	- ![[Pasted image 20240920185800.png]]
##### 5、HAVING
- **说明**：用于对GROUP BY语句后的数据进行筛选（并不是删除，只是呈递满足条件的部分用于显示），WHERE语句在分组前筛选语句，可以使用原始数据的任何列，HAVING在分组后筛选语句
- 形式：`HAVING crition`
	- ![[Pasted image 20240912172900.png]]
- **HAVING用到的列必须是SELECT语句中存在的**


##### 6、ORDER BY
- **DESC**：varable DESC，代表降序
- 快捷方式：1，2代表按照SELECT的第一个和第二个变量进行排序
- ![[Pasted image 20240911172914.png]]

##### 7、LIMIT
- 提取选择语句的前6行，LIMIT 6;
- 提取选择数据第6行后的，三行，LIMIT 6,3;
##### 8、子查询
- **说明**：WHERE、FROM、SELECT子句中都可以使用子查询
- **WHERE使用子查询**：
	- ![[Pasted image 20240912200511.png]]
- **相关子查询**：外查询与子查询是相关的，示例目标是查询得到大于公司内大于同一个部门平均工资的人
	- ![[Pasted image 20240912203233.png]]
	- EXISTS经常使用相关子查询
- **SELECT使用子查询**：
	- ![[Pasted image 20240913114215.png]]
	- 使用目的是为了摆脱聚合函数返回的单值限制，但自然常数产生的单值没有限制。
- **FROM使用子查询**：
	- FROM子查询：希望使用子查询建立得到的表，并进行处理
	- 要求：必须起别名
	- 问题：导致代码过于冗长，可以通过试图进行改善
	- ![[Pasted image 20240913114632.png]]
- **tips**：
	- tip1：有时子查询和连接的实现结果一致，子查询表达有时更清晰，但有时运算更慢；
		- 数据量多用连接，数据量少用子查询
			- ![[Pasted image 20240912200837.png]]
		- 此时，连接的表达反而更清晰
			- ![[Pasted image 20240912201113.png]]
	- tip2：如果子查询的结果只有列表时，请加DISTINCT
#### SQL-添
##### 1、INSERT
- **形式**：INSERT INTO table VALUES ()
	- 1、直接插入单行值：
		- ![[Pasted image 20240912154332.png]]
	- 2、选择变量后直接插入单行值，（未选择的变量必须具备默认值）
		- ![[Pasted image 20240912154703.png]]
	- 3、插入多行值
		- ![[Pasted image 20240912155020.png]]
	- 4、插入子查询：
		- ![[Pasted image 20240912160327.png]]
		- ![[Pasted image 20240916140957.png]]
		- ![[Pasted image 20240916141016.png]]
- **LAST_INSERT_ID**：
	- 多个表中进行插入，通过LAST_INSERT_ID()学习母表插入的主键，并放入子表的信息中
	- ![[Pasted image 20240912155215.png]]
	- 一次插入只能对应的插入一次，一次可以有多个值，第一个插入中order_id为主键，因此自动叠加。
- **REPLACE**:
	- 作用与INSERT相似，但会提前删除与插入目标一致的数据，再插入
	- ![[Pasted image 20240916141840.png]]
	- 
##### 2、CREATE TABLE
- **说明**：CREATE TABLE代表表复制，只复制值，不复制主键属性以及其他属性
- **形式**：CREATE TABLE name AS
	- ![[Pasted image 20240912160617.png]]
	- 直接创建表：
		- ![[Pasted image 20240917132324.png]]
		- ![[Pasted image 20240917132449.png]]
	- 从另一张表复制表结构创建表：
		- ``CREATE TABLE tb_name LIKE tb_name_old``
	- 从另一张表的查询结果创建表：
		- ``CREATE TABLE tb_name AS SELECT * FROM tb_name_old WHERE options``
	- 修改表：
		- ![[Pasted image 20240917132748.png]]
		- ![[Pasted image 20240917134233.png]]
		- 添加列，修改列名等使用CHANGE（新列类型不继承旧列类型，均需重新定义，比如修改后comment为空值），不修改列名，只修改列类型使用MODIFY（修改后，列类型也均需重新定义，比如修改后comment为空值）

##### 3、CREATE VIEW
- **说明**：视图用来简化选择语句，为虚拟表，不占用存储空间，可以作为TABLE使用，其中的数值会随原始表进行动态更新。
- **形式**：CREATE VIEW name AS，name：结果_by_源表
	- 直接创建（只要名字一致，无法运行第二次，即使内容结构不一致）
		- ![[Pasted image 20240914125152.png]]
	- 删除后再创建（组合后无限制运行）
		- ![[Pasted image 20240914125407.png]]
	- 创建或重构（无限制运行）
		- ``CREATE OR REPLACE VIEW``
		- ![[Pasted image 20240914125458.png]]
- **WITH CHECK OPTION**：
	- 应用条件：使用的聚合的则没必要使用，但对于提供通过视图插入或更新数据途径的可以使用。
	- 作用在于，对于试图进行插入与更新时，会作用到原始表，即原始表也会插入与更新相关的内容。出现问题，试图中只展示统计学支部的人员，但对该视图插入了一个数学支部的人，这种插入作用到原始表，但其实并没有对统计支部试图进行影响；因此加入WITH CHECK OPTION时，对统计支部试图插入统计支部的人，则原始支部会出现更新，插入非统计支部的人则原始支部也不更新
	- ![[Pasted image 20240914133033.png]]
- **tips**：
	- tip1：创建试图的代码文件最好整理为单独文件，且最好放置在同名文件夹下的views文件夹，以方便管理。
	- tip2：![[Pasted image 20240914133146.png]]
- **意义**：
	- 基于视图编写代码能增强稳定性。
	- view为想让看到表中的某些可向外提供的数据，限制原始数据的访问权限。

##### 4、CREATE PROCEDURE
- 参考：[存储过程参考链接](https://www.kuangstudy.com/bbs/1485074232206888962)
- **说明**：存储过程的思想很简单，就是 SQL 语句的封装。一旦存储过程被创建出来，使用它就像使用函数一样简单，我们直接通过调用存储过程名即可。我在前面讲过，存储过程实际上由 SQL 语句和流控制语句共同组成。
	- 创建存储过程，从而更方便提取数据，其中过程名为小写加下划线，BEGIN和END之间是主体。
- **形式**：
	- ==创建 PROCEDURE== ：使用DELIMITER更改默认分配符，使得存储过程整体为一个语句，也可以右键，通过MYSQL系统提供的方法进行创建。
		- ![[Pasted image 20240914140645.png]]
	- ==创建 PROCEDURE（包含参数）==：参数必须加属性，加什么属性可以参考源数据
		- ![[Pasted image 20240914141423.png]]
	- ==创建 PROCEDURE（包含默认参数设置）==：（）内仍然需要有NULL，只要设置了参数，应用时就不能是空值，以下两种形式等价。
		- ![[Pasted image 20240914141720.png]]![[Pasted image 20240914142219.png]]
	- ==创建 PROCEDURE（获取过程的参数值）==：
		- 存储过程的 OUT 参数在调用时不需要（也不应该）被赋值，因为它们的值将在存储过程执行后被返回
		- ![[Pasted image 20240914145543.png]]
- 存储过程示例
	- ![[Pasted image 20240914143457.png]]
- **DECLARE**
	- `DECLARE`语句用于在存储过程的BEGIN...END块内部声明局部变量。这些变量只能在声明它们的BEGIN...END块内部访问和使用
- **SET**
	- `SET`语句用于
	- 给变量赋值，无论是全局变量、会话变量、用户定义的变量，还是存储过程中声明的局部变量。`SET`也可以用于更新表中的数据，但在这里我们主要关注它如何用于变量赋值。
	- ![[Pasted image 20240914150221.png]]
##### 5、CREATE FUNCTION
- **说明**：函数与存储过程相似，但函数仅能返回单一值，不能返回多行，多列数据集。
- **形式**：比存储过程增加：RETURNS INTEGER（返回类型）、READS SQL DATA（函数权限）、RETURN risk_factor（返回值）
	- ![[Pasted image 20240914151300.png]]
- **READS SQL DATA（函数属性）**：
	- ![[Pasted image 20240914154644.png]]
##### 6、CREATE TRIGEER
- **说明**：触发器为插入、更新和删除语句前后自动执行的一堆SQL代码，以增强数据一致性。
- **形式**：``CREATE TRIGGER name AFTER/BEFORE  INSERT/UPDATE ON table``，触发器名称采用table名+after/before+激活的SQL语句类型
	- ![[Pasted image 20240914165307.png]]
	- new代表表内插入的新的值，即payments中的新插入值，作用于invoiments
- **SHOW TRIGGERS**：
	- ![[Pasted image 20240914165627.png]]
- **触发器可以用于数据同步，也可用于审计调查，即记录每次的使用记录**
	- ![[Pasted image 20240914165734.png]]
##### 7、CREATE EVENTS
- **说明**：事件主要用于帮助数据库维护任务实现自动处理，例如事件定期发生，以实现自动剔除、更新某些数据
- **形式**：``CREATE EVENT 频率_执行_表格_行 ``
	- 使用EVENTS前需要，查看EVENTS变量是否关闭，如果已关闭则需要开始后，才能使用事件
		- ![[Pasted image 20240914170619.png]]
	- ![[Pasted image 20240914170528.png]]
- SHOW/DROP/ALTER EVENTS：
	- 查看、删除、修改事件
	- ![[Pasted image 20240914170905.png]]
##### 8、START TRANSACTION
- **说明**：事务由一组代表单个工作单元的SQL语句，一个单元不执行，便还原最初的样子，一起成功一起失败。
- ![[Pasted image 20240914172802.png]]![[Pasted image 20240914171951.png]]
- commit和rollback都是事务的结束，一个是我觉得做的对，就这样执行并且提交，另一个是我事务中正在做，发现做错了，那就rollback，直接回到事务开始以前![[Pasted image 20240914172836.png]]
#### SQL-改
##### 1、UPDATE
- **说明**：用于更新SQL现有值
- **形式**：UPDATE	table SET    WHERE equal
	- 更新单行（WHERE只取一个样本）
		- ![[Pasted image 20240912161012.png]]
	- 更新多行（WHERE包含多个样本）
		- ![[Pasted image 20240912161107.png]]
	- 更新整张表（可忽略WHER句）
		- ![[Pasted image 20240912161452.png]]
- **tips**：
	- tip1：更新需要指定对象，有时具体的id不知道，但知道其他特征，借助子查询进行实现
		- ![[Pasted image 20240912162507.png]]
		- 注意：结果中有多个情况时，不继续使用=来表示见下图，因为 `WHERE client_id = ...` 期望一个单一的值，面对一个变量与另一个集合比较，需要使用IN等来表示这种包含关系或者在后边加ANY。
		- ![[Pasted image 20240912161628.png]]
	- tip2：更改前，先使用SELECT查看WHERE选定的人群是否正确，确保一切无误后，再进行更新。
	- tip3:
		- ![[Pasted image 20240916142638.png]]

#### SQL-删
##### 1、DELETE
- 说明：按条件删除表记录
- 形式：`DELETE FROM table WHERE equal;`
	- ![[Pasted image 20240912163322.png]]
##### 2、TRUNCATE TABLE
- ![[Pasted image 20240916151843.png]]
- 说明：删除表中所有记录，重置该表，但表的结构保持不变
- 形式：``TRUNCATE TABLE table;``
	- ![[Pasted image 20240916151306.png]]
	- TRUNCATE TABLE必须是前者，不能是TRUNCATE ，也不能是TRUNCATE table![[Pasted image 20240916151603.png]]
##### 2、DROP
- ``DROP TABLE/VIEW/PROCEDURE/FUNCTION IF EXISTS name;``