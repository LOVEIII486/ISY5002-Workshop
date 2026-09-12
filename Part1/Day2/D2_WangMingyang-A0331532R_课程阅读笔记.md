# Day 2 课程与笔记本阅读笔记

> 生成日期：2026-09-08
> 阅读对象：
> - 课程 PPT：`C:\Users\75643\Desktop\ISY5002\info\Day 2\S-PSUPR Day2.pdf`
> - 配套笔记本：`C:\Users\75643\Desktop\ISY5002\Part1\Day2\D2_WangMingyang-A0331532R.ipynb`
> - 数据文件：`C:\Users\75643\Desktop\ISY5002\Part1\Day2\diabetes.csv`

---

## 一、Day 2 课程主题

PPT 标题为 **PROBLEM SOLVING USING PATTERN RECOGNITION - DAY 2**，内容分为两节：

1. `2.1 Solving Pattern Recognition Problems Using Supervised Learning Techniques (II)`
   - Decision Trees（决策树）
   - Neural Networks（神经网络）
   - Support Vector Machines（支持向量机）
2. `2.2 Pattern Recognition Workshop 2`（Workshop 2 任务说明）

核心思想：在 Day 1 的有监督学习基础上，继续学习三种常用分类模型，并用 Pima Indians Diabetes 数据集实际建模和比较性能。

---

## 二、课程 PPT 要点

### 2.1 决策树（Decision Trees, PPT 第 5–30 页）

- 决策树是一种流程图式树结构：
  - 内部节点：对某个属性做测试
  - 分支：测试结果
  - 叶节点：类别标签
  - 每个节点选择一个特征来划分训练样本
- 典型应用：客户关系管理、欺诈检测、流失预测、信用风险、购买行为预测、故障检测、情感分析、投资方案等。
- 基本算法：Quinlan 的 ID3 / C4.5 / C5.0。
- 建树策略：自顶向下，贪心搜索，通常优先选择能产生“更纯”子集的属性。
- 常用划分准则：
  - Information Gain / 信息增益：`Gain(S,A) = E(S) - Σ(|Sv|/|S|) * E(Sv)`
  - Gain Ratio / 信息增益率：用于降低对取值较多属性的偏好
  - Gini Index：CART 的划分准则
- 熵：`E(S) = -Σ pc * log2(pc)`；熵越高表示不确定性越高。
- 连续属性：先排序，再以相邻值中点作为候选切分点，选择信息增益最大的切分点。
- 停止条件示例：
  - 所有样本属于同一类
  - 所有属性已用完
  - `min_samples_split`、`min_samples_leaf`、`max_depth` 等超参数限制
- 高分支属性问题：信息增益会偏向取值很多的特征（极端例子是 ID 编号），可能导致过拟合。
- 过拟合与剪枝：
  - Pre-pruning / 预剪枝：提前停止生长
  - Post-pruning / 后剪枝：先长成完整树，再用验证集选择最优剪枝树
- Scikit-learn 示例：
  - `DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)`
- 决策树优缺点：
  - 优点：易理解、易解释，数据准备和计算开销小，能给出特征重要性
  - 缺点：不保证全局最优，多类别且样本少时表现差，过于复杂会过拟合

### 2.2 神经网络（Neural Networks, PPT 第 31–48 页）

- 神经网络受生物神经结构启发，能从数据中学习模式，对数据分布假设较少，可处理数值型和类别型目标，但属于“黑箱”模型。
- 一般结构：
  - 输入层 + 隐藏层 + 输出层
  - 权重 `W`
  - 激活函数 `f(net)`
- 权重更新一般形式：`Wi(t+1) = Wi(t) + ΔWi(t)`。
- 训练流程：
  1. 初始化权重
  2. 输入训练样本
  3. 计算输出
  4. 计算误差
  5. 根据误差调整权重
  6. 判断停止条件
- MLP + 反向传播：
  - 信号前向传播，误差反向传播
  - Backpropagation 本质上是梯度下降学习
- 反向传播步骤：
  1. 用小随机数初始化权重
  2. 随机选取训练样本 `(xp, tp)`，前向计算输出
  3. 计算误差
  4. 反向传播误差并更新权重
  5. 检查损失函数（MSE、交叉熵等）
  6. 必要时测试泛化性能
- 梯度下降更新：
  - `Δwji(t+1) = -η * ∂E/∂wji(t) + α * Δwji(t)`
  - `η` 是学习率，`α` 是动量率
- 过训练 / 过拟合：
  - 网络可能“记住”训练集，导致泛化能力下降
  - 需要观察训练误差和验证/测试误差，在合适位置提前停止训练
- 数据预处理：
  - 数据编码、平滑、变换
  - 对数变换：`y = log(x)`
  - 差分：`Δxi = xi - xi-1`
  - Min-Max 归一化
  - Z-score 标准化：`z = (x - μ) / σ`
- 测试与评估：
  - 使用验证集和测试集评估泛化能力
  - 训练后可进行网络剪枝，去除冗余节点和权重
  - 应定期测试模型，防止数据环境变化导致性能下降
- Scikit-learn 示例：
  - `MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)`
  - 通常先用 `StandardScaler` 标准化数据

### 2.3 支持向量机（SVM, PPT 第 49–72 页）

- SVM 是另一类前馈网络，可分类，也可做非线性回归。
- 一般架构：
  - 输入层
  - 由 inner-product kernels 构成的隐藏层
  - 线性输出神经元
- 核心思想：
  - 对非线性问题，通过非线性映射 `φ(x)` 把数据映射到高维特征空间
  - 在高维空间中寻找线性最优分离超平面
- 决策面：`wᵀx + b = 0`
- 间隔（margin）：
  - `margin = 2 / ||w||`
  - SVM 的目标是最大化间隔
- Hard Margin / 硬间隔：
  - 适用于线性可分数据
  - 优化目标：最小化 `1/2 * wᵀw`，约束 `yi(wᵀxi + b) ≥ 1`
- Soft Margin / 软间隔：
  - 适用于线性不可分数据
  - 引入松弛变量 `ξi`，允许一定误分类
  - 优化目标：最小化 `1/2 * wᵀw + C * Σξi`
  - `C` 是惩罚参数：
    - `C` 小：间隔更宽，容忍更多误分类，支持向量可能较多
    - `C` 大：间隔更窄，更少训练误差，支持向量可能较少
    - `C → ∞`：近似硬间隔
- 非线性核：
  - 用核函数 `K(x, x') = <φ(x), φ(x')>` 直接计算特征空间内积，避免显式计算高维映射
  - 示例：把二维输入映射到三维 `(x1, x2, x1*x2)`
- 常用核函数（PPT 列出典型形式，包括线性、多项式、RBF 等）。
- SVM 示例：一维数据 + 多项式核 `K(xi,xj) = (xi·xj + 1)²`，`C=100`，得到支持向量和判别函数。
- 多分类 SVM：
  - One-vs-others：每个类别训练一个“该类 vs 其他类”的 SVM
  - One-vs-one：每两类训练一个 SVM，测试时投票
- SVM 应用：生物信息学、机器视觉、文本分类、手写字符识别等。
- 实践流程：准备数据 → 选核函数 → 选核参数和 `C` → 训练得到 `αi` → 用支持向量分类新样本。
- Scikit-learn 示例：
  - `SVC(kernel="rbf", gamma=5, C=1)`
  - 数据先 `StandardScaler` 标准化

### 2.4 Workshop 2 要求（PPT 第 73–74 页）

- 打开 Workshop 2 提供的 Jupyter notebook。
- 构建 Decision Tree、Neural Network、SVM 三个模型。
- 理解每个模型的构建方式，可以在 notebook 中用 Markdown 记录笔记。
- 比较三个模型的性能。
- 尝试调整不同参数设置。
- 可以尝试使用自己的数据集。
- 保存 notebook 及 cell 输出，上传到 Canvas 指定文件夹。

---

## 三、配套 Notebook 阅读结果

### 3.1 数据与任务

- 数据集：Pima Indians Diabetes（`diabetes.csv`）。
- 形状：`(768, 9)`，即 768 条样本、9 列。
- 特征列：`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`。
- 目标列：`Outcome`（0 或 1，表示是否糖尿病）。
- 划分方式（cell 05）：
  - `test_size = 1/3`
  - `random_state = 42`
  - `stratify = y`，保持训练/测试集中类别比例一致

### 3.2 决策树部分

- 第一次建模（cell 07）：
  - `DecisionTreeClassifier(criterion='entropy', random_state=0)`
  - 训练集准确率：`1.000`
  - 测试集准确率：`0.703`
- notebook 中的判断（cell 08）：训练集 100%、测试集明显更低，说明决策树过拟合，需要预剪枝。
- 改进建模（cell 09）：
  - `max_depth=3`
  - 训练集准确率：`0.793`
  - 测试集准确率：`0.730`
- 测试集混淆矩阵（cell 11）：
  - `[[145, 22], [47, 42]]`
- 分类报告要点：
  - 类别 0：precision `0.76`, recall `0.87`, F1 `0.81`
  - 类别 1：precision `0.66`, recall `0.47`, F1 `0.55`
  - accuracy：`0.73`
- cell 12 使用 `sklearn.tree.plot_tree` 可视化剪枝后的决策树。
- cell 13 已补全：绘制 Decision Tree 的 ROC 曲线，基于 `predict_proba` 和 `roc_curve`/`auc`。
- cell 14 已补全：对 `criterion`、`max_depth`、`min_samples_leaf` 做小规模网格搜索；在该数据集上最优组合约为 `entropy, max_depth=6, min_samples_leaf=5`，测试准确率约 `0.773`，优于当前 `max_depth=3` 的 `0.730`。
- cell 15 已补全：输出特征重要性，并用横向条形图可视化。
- 原 cell 14、cell 17 等空白分隔单元格已被删除；当前 DT 作业已集中在 cell 13–15。

### 3.3 神经网络部分

- cell 17：用 `StandardScaler` 对训练集和测试集做 Z-score 标准化。
- cell 18：
  - `MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, verbose=2)`
  - 两个隐藏层，每层 10 个节点
- cell 19：
  - 测试集准确率：`0.71484375`
  - 混淆矩阵：`[[129, 38], [35, 54]]`
  - 类别 0 F1：`0.78`；类别 1 F1：`0.60`
- cell 20：
  - 训练集准确率：`0.861`
  - 测试集准确率：`0.715`
  - 训练集高于测试集约 15 个百分点，说明存在一定过拟合/泛化差距
- cell 22：绘制 `mlp.loss_curve_` 损失曲线。
- cell 23–27：查看第一层参数：
  - `mlp.intercepts_[0]` 形状：`(10,)`
  - `mlp.coefs_[0]` 形状：`(8, 10)`
  - 这与 8 个输入特征、第一隐藏层 10 个节点一致
- cell 28 已补全：比较 `(5,)`、`(10,)`、`(20,)`、`(10,10)`、`(20,20)`、`(10,10,10)` 等结构；当前数据下最优为单隐藏层 `(10,)`，训练准确率约 `0.811`、测试准确率约 `0.742`，优于原 `(10,10)` 结构，并设置 `mlp = best_mlp` 供后续部署使用。
- cell 29 为空白。

### 3.4 SVM 部分

- cell 31 使用 RBF 核，并做了小规模参数网格：
  - `gamma ∈ {0.1, 1}`
  - `C ∈ {1, 10}`
- 四种参数组合的测试集准确率：

| gamma | C | Accuracy |
|---:|---:|---:|
| 0.1 | 1 | **0.75390625** |
| 0.1 | 10 | 0.7109375 |
| 1 | 1 | 0.70703125 |
| 1 | 10 | 0.69921875 |

- 在当前划分和参数范围内，`gamma=0.1, C=1` 表现最好。
- 该组合的混淆矩阵为 `[[143, 24], [39, 50]]`，类别 1 的 recall 为 `0.56`，说明对少数类的召回仍有提升空间。

### 3.5 部署部分

- cell 33 使用 `pickle`：
  - 保存：`finalized_model.sav`
  - 加载并预测测试集
- **需要注意**：虽然这个 section 位于 SVM 之后，但代码保存的是 `mlp`（MLP 模型），而不是 SVM。
  - 如果本意是保存“当前最优模型”，那么当前笔记本中 DT 的 `0.773` 和 SVM 的 `0.754` 都高于调优后 NN 的 `0.742`，应确认这里是否写错。
  - 如果本意就是保存 MLP 做部署示例，则逻辑成立，但注释/章节位置容易造成误解。

### 3.6 其他观察

- cell 21 重新导入了一批库，但 `execution_count` 为 `None`，说明该 cell 未运行。
- cell 34 为新增的模型性能对比与学习笔记 Markdown。
- cell 35、37 为空白；cell 36 前有 Markdown 链接指向 scikit-learn 分类器比较示例。
- notebook 中部分 cell 的执行编号不是严格递增顺序，说明编辑过程中可能有重新执行、覆盖执行等情况；若继续填写作图代码，建议重新 `Run All`，确保变量顺序一致。

---

## 四、课程与 Notebook 的对应关系

| 课程内容 | Notebook 对应单元格 |
|---|---|
| 决策树、熵、预剪枝 | cell 06–12 |
| DT 可视化、ROC、参数调整、特征重要性 | cell 12–15（已补全） |
| NN、标准化、MLP、损失曲线、权重 | cell 16–27 |
| NN 调参 | cell 28（已补全） |
| SVM、RBF、C/gamma 参数 | cell 30–31 |
| 模型保存与部署 | cell 32–33 |
| Workshop 2 要求“比较模型性能” | 已部分完成：DT `0.730`、NN `0.715`、SVM `0.754` |

---

## 五、结论与建议

- Day 2 的理论重点是：三种有监督分类器的原理、超参数含义、优缺点，以及它们共同的“训练/验证/测试、防止过拟合、调参比较”流程。
- 当前 notebook 已跑通数据加载、DT、NN、SVM 和 pickle 部署，且已有模型结果。
- 按 Workshop 2 要求，**还需补完的关键任务**是：
  1. 三个模型的统一性能比较总结（可用 Markdown 写清楚）
- 建议在补完上述代码后，从顶部执行 `Run All`，确认所有输出一致后保存并上传 Canvas。