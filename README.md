# ICA-SVM
# 提取了26个特征，基于ICA方法对4通道EEG的去除EOG
### get_rawdata.py:进行ICA获得成分的模式空间和时间序列，同时手动制作label.csv
### get_trait.py:输入模式空间和时间序列，获得特征空间，合并label和trait
### train.py:训练
### test.py:模拟实时的测试，输入测试数据，依次间隔一秒，取前五秒进行分析。图片显示ICA自动分类并过滤的前后差距。
