# LinearRegression


# 1.简单线性回归

## （1）建立数据集


```python
#导入pandas包 
#导入 collections 中的 OrderedDict包
from collections import OrderedDict
import pandas as pd
```


```python
#数据集
examDic={'学习时间':[0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,
            2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50],
        '分数':[10,  22,  13,  43,  20,  22,  33,  50,  62,  
              48,  55,  75,  62,  73,  81,  76,  64,  82,  90,  93]}
```


```python
#定义有序字典
examOrderedDict=OrderedDict(examDic)
#定义数据框
examDf=pd.DataFrame(examOrderedDict)
```


```python
examDf.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>学习时间</th>
      <th>分数</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.50</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.75</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.00</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.25</td>
      <td>43</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.50</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



## （2）相关系数：两个变量每单位的相关程度


```python
#提取特征和标签
#特征 fetures
exam_X=examDf.loc[:,'学习时间']
exam_Y=examDf.loc[:,'分数']
```


```python
#绘制散点图
import  matplotlib.pyplot as plt
#散点图
plt.scatter(exam_X,exam_Y,color="b",label="exam data")
#添加图标标签
plt.legend(loc=1)
plt.xlabel("Hours")
plt.ylabel("Score")
#显示图像
plt.show()
```


![png](output_8_0.png)



```python
#相关系数 corr 返回结果是一个数据框，存放的是相关系数矩阵
rDf=examDf.corr()
print('相关系数矩阵：')
rDf
```

    相关系数矩阵：
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>学习时间</th>
      <th>分数</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>学习时间</th>
      <td>1.000000</td>
      <td>0.923985</td>
    </tr>
    <tr>
      <th>分数</th>
      <td>0.923985</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## （3）线性回归的实现

### 1.提取特征值和标签


```python
#特征值 features
exam_X=examDf.loc[:,'学习时间']
#标签 labels
exam_Y=examDf.loc[:,'分数']
```

### 2.建立训练数据和测试数据


```python
'''
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取训练数据（train）和测试数据（test）
第一个参数：所要划分的样本特征
第2个参数：所要划分的样本标签
train_size：训练数据占比，如果是整数的话就是样本的数量
sklearn包0.8版本以后，需要将之前的sklearn.cross_validation 换成sklearn.model_selection
'''
# 导入sklearn 中的 train_test_split包
from sklearn.model_selection import train_test_split
```


```python
#建立测试数据和训练数据
X_train , X_test , Y_train , Y_test = train_test_split(exam_X ,
                                                       exam_Y ,
                                                       train_size = .8)
```

    D:\software\ANACONDA\lib\site-packages\sklearn\model_selection\_split.py:2026: FutureWarning: From version 0.21, test_size will always complement train_size unless both are specified.
      FutureWarning)
    


```python
print ('原始特征数据：',exam_X.shape,'训练特征数据：',X_train.shape,'测试特征数据：',X_test.shape)
print ('原始标签数据：',exam_Y.shape,'训练标签数据：',Y_train.shape,'测试标签数据：',Y_train.shape)
```

    原始特征数据： (20,) 训练特征数据： (16,) 测试特征数据： (4,)
    原始标签数据： (20,) 训练标签数据： (16,) 测试标签数据： (16,)
    


```python
#绘制散点图
plt.scatter(X_train,Y_train,color="blue",label="train data")
plt.scatter(X_test,Y_test,color="red",label="test data")
#添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
#显示图像
plt.show()
```


![png](output_17_0.png)


### 3.训练模型


```python
# 第一步：导入线性回归
from sklearn.linear_model import LinearRegression
# 第二步：创建模型：线性回归
model=LinearRegression()
#第三步：训练模型
model.fit(X_train,Y_train)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-13-da8a20faee6b> in <module>()
          4 model=LinearRegression()
          5 #第三步：训练模型
    ----> 6 model.fit(X_train,Y_train)
    

    D:\software\ANACONDA\lib\site-packages\sklearn\linear_model\base.py in fit(self, X, y, sample_weight)
        480         n_jobs_ = self.n_jobs
        481         X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
    --> 482                          y_numeric=True, multi_output=True)
        483 
        484         if sample_weight is not None and np.atleast_1d(sample_weight).ndim > 1:
    

    D:\software\ANACONDA\lib\site-packages\sklearn\utils\validation.py in check_X_y(X, y, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, warn_on_dtype, estimator)
        571     X = check_array(X, accept_sparse, dtype, order, copy, force_all_finite,
        572                     ensure_2d, allow_nd, ensure_min_samples,
    --> 573                     ensure_min_features, warn_on_dtype, estimator)
        574     if multi_output:
        575         y = check_array(y, 'csr', force_all_finite=True, ensure_2d=False,
    

    D:\software\ANACONDA\lib\site-packages\sklearn\utils\validation.py in check_array(array, accept_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, warn_on_dtype, estimator)
        439                     "Reshape your data either using array.reshape(-1, 1) if "
        440                     "your data has a single feature or array.reshape(1, -1) "
    --> 441                     "if it contains a single sample.".format(array))
        442             array = np.atleast_2d(array)
        443             # To ensure that array flags are maintained
    

    ValueError: Expected 2D array, got 1D array instead:
    array=[ 2.    3.5   2.5   3.    2.75  4.75  4.25  2.25  0.75  4.5   1.5   5.5
      1.75  0.5   4.    1.75].
    Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.



```python
'''
上面的报错内容，最后一行是这样提示我们的：
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
上面报错的内容翻译过来就是：
如果你输入的数据只有1个特征，需要用array.reshape(-1, 1)来改变数组的形状
'''
'''
reshape行的参数是-1表示什么呢？例如reshape(-1,列数)
如果行的参数是-1，就会根据所给的列数，自动按照原始数组的大小形成一个新的数组，
例如reshape(-1,1)就是改变成1列的数组，这个数组的长度是根据原始数组的大小来自动形成的。
原始数组总共是2行*3列=6个数，那么这里就会形成6行*1列的数组
'''
```




    '\nreshape行的参数是-1表示什么呢？例如reshape(-1,列数)\n如果行的参数是-1，就会根据所给的列数，自动按照原始数组的大小形成一个新的数组，\n例如reshape(-1,1)就是改变成1列的数组，这个数组的长度是根据原始数组的大小来自动形成的。\n原始数组总共是2行*3列=6个数，那么这里就会形成6行*1列的数组\n'




```python
import numpy as np
#定义2行*3列的数组
aArr = np.array([
    [1, 2, 3],
    [5, 6, 7]
])
aArr.shape
```




    (2, 3)




```python
aArr
```




    array([[1, 2, 3],
           [5, 6, 7]])




```python
#改变数组形成为3行*2列
bArr=aArr.reshape(3,2)
bArr.shape
```




    (3, 2)




```python
bArr
```




    array([[1, 2],
           [3, 5],
           [6, 7]])




```python
#改变数组形成为1行*6列
bArr=bArr.reshape(-1,1)
```


```python
bArr
```




    array([[1],
           [2],
           [3],
           [5],
           [6],
           [7]])




```python
#改变数组形成为6行*1列
bArr=bArr.reshape(1,-1)
```


```python
bArr
```




    array([[1, 2, 3, 5, 6, 7]])




```python
#将训练数据特征转换成二维数组XX行*1列
X_train=X_train.values.reshape(-1,1)
#将测试数据特征转换成二维数组XX行*1列
X_test=X_test.values.reshape(-1,1)
# 第一步：导入线性回归
from sklearn.linear_model import LinearRegression
# 第二步：创建模型：线性回归
model=LinearRegression()
#第三步：训练模型
model.fit(X_train,Y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)




```python
'''
最佳拟合线：z=𝑎+𝑏x
截距intercept：a
回归系数coef：b
'''
#截距
a=model.intercept_
#回归系数
b=model.coef_
print('最佳拟合线:截距a=',a,'截距b=',b)
```

    最佳拟合线:截距a= 8.31692937071 截距b= [ 16.19732884]
    


```python

'''
第1步：绘制训练数据散点图
'''

#训练数据散点图
plt.scatter(X_train,Y_train,color='blue',label='train data')

'''
第2步：用训练数据绘制最佳线
'''
#训练数据的预测值
Y_train_pred = model.predict(X_train)

#用训练数据绘制最佳拟合线
plt.plot(X_train,Y_train_pred,color='black',linewidth=3,label='best line')

'''
第3步：绘制测试数据的散点图
'''

#测试数据散点图
plt.scatter(X_test, Y_test, color='red', label="test data")

#添加图标签
plt.legend(loc=2)
plt.xlabel('Hours')
plt.ylabel('Score')
plt.show()
```


![png](output_31_0.png)


### 4.模型评估


```python
#线性回归模型的scroe得到的决定系数R平方
#评估模型：决定系数R平方

model.score(X_test,Y_test)
```




    0.89111305771129079


