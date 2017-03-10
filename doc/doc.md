#### 数据源：
chiq.chiq_video_converge
#### 流程：
1. 从mongo获取数据
> function: Video._get_documents
2. 对数据进行整理
> function: Video._handle_all_attr
3. 数据标准化
> function: Video._process_documents \
> sim._filter_label
4. 分类别计算两两相似度
> function: sim._init_data \
> sim.process
5. 结果标准化
> function: sim._calculate_output
6. codis写入
> rec_content_based
#### tags处理
媒体库字段：
- category： 所有类别权重为1
- tag：剔除category中标签，之后安装所处位置设置权重
```math
w(i,x)=\dfrac{e^{1-\sqrt{i}}} {\sum_{j=1}^nsign(x_j)}
```
此处，```i```标识类别标签所处位置，```x```标识演员名称，```n```标识样本个数。
> language、country字段处理方式和以上方式等同。

#### actor处理
```math
w(i)=e^{1-\sqrt{i}}
```
此处，```i```标识类别标签所处位置.
> 其他字段和该处理方式相同

#### 相似度计算
```math
sim(x, y) = \sum_{i=1}^m w_i\sum_{j=1}^n \frac{x_{i,j}*y_{ij}} {x_{i,j}+y_{i,j}}
```
此处，```m```表示属性数目， ```n```标识每个属性值的数目,```w_i```为各个属性权重
> ```w_i```当前根据业务人为给定，可于建立评价指标后，自动适当调整。

#### 输出格式
key前缀：

```PATTERN = 'AlgorithmsCommonBid_Cchiq3Test:SIM:ITI:'```

输出样例：

```
key: 
AlgorithmsCommonBid_Cchiq3Test:SIM:ITI:zzafbbhsakxms11
value:
{ 
  'Results': {
      't19riu4cwhatfsd': 0.825, 
      'f72got05px34ohi': 0.941, 
      'k0hyd3gpxattsd6': 0.825, 
      'j94c3bv2bt8zr66': 0.941, 
      '5fcscniamyzu1k3': 0.825, 
      'b57tu60lvuohj2k': 0.825, 
      'fp84d23m2ctg725': 0.889, 
      '38b1486hvgkdogw': 0.825, 
      'g2nkordawghenlm': 0.839, 
      '3mvvheu5ntpe9k1': 0.888, 
      },
      "V": "1.0.0"
}
```