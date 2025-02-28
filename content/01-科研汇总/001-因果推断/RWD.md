### RWD的类型和来源

**数据特征：**
- 观察性
- 非结构化：存在不一致的情况，杂乱无章
- 高频式：海量、动态数据
- 不完整：缺少分析终点
- 偏差与测量误差：由某些专业设备测量的数据具有选择偏差的影响

**数据类型（医学）：**
![[Pasted image 20240524162411.png]]
完整：Makady A, de Boer A, Hillege H, Klungel O, Goettsch W, et al. What is
real-world data? A review of definitions based on literature and stake-
holder interviews. Value Health. 2017;20(7):858–65.

### RWD的分析和应用
**1、项目临床实验**：
**2、目标试验模拟**：
目标试验模拟是将（目标）随机试验的试验设计和分析原则应用于观察数据的分析。通过精确指定目标试验的入选/排除标准、治疗策略、治疗分配、因果对比、结局、随访期和统计分析，可以从RWD中得出关于干预的有效因果推断。典型的是倾向得分匹配。
**3、作对照试验的历史对照和参考组**：
RWD也可用作对照试验的历史对照和参考组，评估RWD的质量和适当性，并采用适当的统计方法分析数据
**4、ML技术：**
现代ML技术非常能够处理大量的、混乱的、多模态的和各种非结构化的数据类型，而无需对数据的分布进行强有力的假设。
深度学习可以学习大型、复杂和非结构化数据的抽象表示;自然语言处理（NLP）和嵌入方法可用于处理EHR中的文本和临床笔记，并将其转换为实值向量，用于下游学习任务。
ML技术主要用于预测和分类（例如，疾病诊断）、变量选择（例如，生物标志物筛选）、数据可视化等，而不是生成监管级别的RWE;
无论RWD项目侧重于哪个领域-因果推断或预测和分类，RWD项目结论将推广到的人群的RWD代表性至关重要。否则，估计或预测可能会误导甚至有害。RWD中的信息可能不足以验证数据是否适合推广;
### RWD的挑战与机遇
**数据质量：**
RWD现在通常用于其他目的，而不是最初收集的目的，因此可能缺乏关键终点的信息，并不总是用于生成监管级证据。最重要的是，RWD是混乱的，异质的，并且受到各种测量误差的影响，所有这些都导致RWD的质量低于对照试验的数据。因此，基于RWD的结果的准确度和精密度受到负面影响，可能产生误导性结果或错误结论。虽然这些并不排除在证据生成和决策中使用RWD，但需要一致地记录数据质量问题，并通过数据清理和预处理（例如，填补缺失值的插补、不平衡数据的过采样、去噪、跨数据库组合不同的信息片段等）。如果问题可以在预处理阶段得到解决，则应在数据分析期间努力纠正该问题，或者在解释结果时应谨慎。关键利益攸关方的早期参与（例如，监管机构（如需要）、研究机构、行业等）鼓励建立数据质量标准，减少不可预见的风险和问题。

**高效实用的ML和统计程序：**
数字医疗数据的快速增长以及劳动力和投资涌入该领域的事实也推动了现代统计程序和ML算法的快速开发和采用，以分析数据。开放源码平台和软件的存在大大便利了这些程序的实际应用。另一方面，RWD的噪声、异质性、不完整性和不平衡性可能导致现有统计和ML过程的相当大的性能不佳，并且需要专门针对RWD并且可以有效地部署在真实的世界中的新过程。此外，开源平台和软件的可用性以及随之而来的便利性，虽然是出于良好的意图提供的，但也增加了从业人员滥用程序的机会，如果在将其应用于现实世界的情况之前没有适当的培训或理解技术的原则。此外，为了在从RWD生成RWE的过程中保持科学严谨性，统计和ML程序的结果在用于真实的决策之前，需要使用专业知识或进行再现性和可复制性研究进行医学验证

**可解释性和可解释性：**
现代ML方法通常以黑盒方式使用，并且缺乏对输入和输出之间的关系以及因果关系的理解。模型选择、参数初始化和超参数调整也经常以试错的方式进行，而没有领域专家的输入。这与医疗保健领域形成鲜明对比，在医疗保健领域，可解释性对于建立患者/用户信任至关重要，医生不太可能使用他们不理解的技术。关于这个主题的有希望和令人鼓舞的研究工作已经开始[106-111]，但需要更多的研究。

**复制性和可复制性：**
复制性和可复制性是科学研究的主要原则，包括RWD。如果一个分析方法不稳定，其输出结果不可重现或复制，公众会质疑工作的科学严谨性，并怀疑基于RWD的研究结论[113-115]。结果验证、可重复性和可复制性可能具有挑战性，因为它们的混乱、不完整、非结构化数据，但需要建立，特别是考虑到生成的证据可以用于监管决策并影响数百万人的生活。假设在此过程中没有隐私受到损害，可以通过共享原始和处理后的数据和代码来减轻不可再现性。对于可复制性，考虑到RWD不是从对照试验中生成的，并且每个数据集都可能具有自己独特的数据特征，因此完全可复制性可能是困难的，甚至是不可行的。尽管如此，数据特征和预处理的详细文件，分析程序的预登记，以及对开放科学原则的遵守（例如，代码存储库[116]）对于复制不同RWD数据集上的发现至关重要，假设它们来自相同的底层人群。读者可以参考[117-119]以获得关于此主题的更多建议和讨论。

**隐私：**
RWD项目实施时存在伦理问题，其中隐私是一个经常讨论的话题。RWD中的信息通常是敏感的，例如病史、疾病状态、财务状况和社会行为等。当不同的数据库（例如，EHR、可穿戴设备、索赔）联系在一起，这是RWD分析中的常见做法。数据用户和政策制定者应尽一切努力确保RWD的收集、存储、共享和分析遵循既定的数据隐私原则（即，合法性、公平性、目的限制和数据最小化）。此外，可以部署隐私增强技术和隐私保护数据共享和分析，其中已经存在大量有效且广为接受的最先进概念和方法，例如差异隐私3 [120]和联邦学习4 [121，122]。在收集和分析RWD并传播RWD的结果和RWE时，研究者和决策者可考虑整合这些概念和技术。

**多样性、公平性、伦理公平性和透明度（DEAT）：**
DEAT是RWD项目中需要考虑的另一个重要伦理问题。RWD可能包含来自各种人口统计组的信息，与在受控环境中收集的数据相比，这些信息可用于生成具有改进的概括性的RWE。另一方面，某些类型的RWD可能严重偏向于某个群体，不具有多样性或包容性，并且在某些情况下，甚至加剧差异（例如，可穿戴设备以及使用设施和治疗的机会可能仅限于某些人口群体）。需要做出更大的努力，使代表性不足的群体获得RWD，并有效地考虑RWD的异质性，同时注意多样性/公平性的限制。本主题还涉及算法公平性，旨在理解和防止ML模型中的偏见。在文献[123- 127]中，伦理公平是一个越来越受欢迎的研究课题。如果训练的模型系统地使某个群体处于不利地位，则可能得出错误和误导性的结论（例如，经过训练的算法可能比白色患者更不可能检测到黑人患者的癌症，或者比女性更不可能检测到男性的癌症）。透明度意味着有关个人数据处理的信息和通信必须易于访问和理解。透明度确保数据贡献者了解他们的数据是如何被使用的，以及用于什么目的，决策者可以评估方法的质量和生成的RWE的适用性[128-131]。在与RWD合作时保持透明对于在RWD生命周期中建立关键利益相关者（提供数据的个人、收集和管理数据的人、设计研究和分析数据的数据管理者以及决策者和政策制定者）之间的信任至关重要。
![[Pasted image 20240524171217.png]]

如图2所示，上述挑战不是孤立的，而是相互关联的。数据质量影响统计和ML程序的性能;数据源和清理及预处理过程与结果的再现性和可复制性有关。如何分析数据以及使用哪些统计和ML程序对可再现性和可复制性产生影响，在数据收集和分析过程中是否使用隐私保护程序以及如何共享和发布信息与数据隐私，DEAT以及可解释性和可解释性有关，这反过来又会影响应用哪些ML程序以及新ML技术的开发。

在机器学习中，随机森林(Random Forests)算法不仅是一个强大的预测模型，它还提供了额外的分析工具，其中之一就是**邻近度图**(Proximity Plots)。这些图可以帮助理解数据集内部的结构，特别是在高维空间中的数据点之间的关系。

### 随机森林的邻近度矩阵

当训练一个随机森林时，算法会在每次迭代中创建一棵决策树。每棵树都是通过对训练数据集进行自助抽样(bootstrap sampling)来构建的。在树的生长过程中，对于每一个终端节点(leaf node)，如果两个样本点被划分到了同一个终端节点内，它们的邻近度就会增加。这样，在整个随机森林训练完成后，会得到一个\(N \times N\)的邻近度矩阵，其中\(N\)是训练样本的数量。这个矩阵记录了数据集中任意两个样本之间的相似程度。

### 多维缩放(MDS)

为了可视化这个高维邻近度矩阵，通常会使用多维缩放(Multidimensional Scaling, MDS)技术将矩阵投影到二维平面上。MDS试图保持原始邻近度矩阵中的距离关系，从而在二维图上展示出样本点间的相对接近程度。

### 邻近度图的解释

邻近度图通常呈现出星状结构，每类数据形成一个臂，分类性能越好，这种星状结构越明显。具体来说：
- **纯区域**中的样本点，即那些属于同一类且周围也是同类样本的点，倾向于映射到星形的尖端。这是因为纯区域的样本点在随机森林的决策树中经常会被分在同一终端节点。
- **决策边界**附近的样本点，即那些类别混合的区域，更可能映射到星形的中心附近。因为这类点有时会被不同类别的样本影响，导致它们可能在某些树中被分到同一终端节点，但不是总是如此。

### 总结

虽然邻近度图可以提供直观的数据点间相似性的视觉化表示，但是它们在不同数据集上的表现往往很相似，这可能限制了它们作为诊断工具的有效性。不过，对于理解随机森林如何分割数据以及识别数据中的类群结构，邻近度图仍然是一项有价值的辅助工具。