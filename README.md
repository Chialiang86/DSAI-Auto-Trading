# DSAI AutoTrading

## Overview

* In this HW, we will implement a very aged prediction problem from the financial field. Given a series of stock prices, including daily open, high, low, and close prices, decide your daily action and make your best profit for the future trading. Can you beat the simple “buy-and-hold” strategy?
## 虛擬環境
### 使用 Anoconda 進行套件管理
* Create virtual environment
```
$ conda create --name DSAI-HW2
```
* Activate virtual environment
```
$ conda activate DSAI-HW2
```
## 套件版本及安裝資訊
### python 版本
* python version : 3.8.8
### 套件版本
* numpy==1.19.2
* pandas==1.2.3
* scikit-learn==0.24.1
* matplotlib==3.3.4
* tensorflow==2.4.1
* keras==2.4.3
### 安裝方法
```bash 
$ conda install --yes --file requirements.txt
```

### 執行方式（以助教提供之資料集當範例）
1. 訓練
```shell 
$ python app.py --training training.csv --testing testing.csv --output output.csv
```
- 訓練完後會產生時間戳記搭配 loss 為檔名的 h5 模型檔
```python
    # save model
    save_name = '{}_{}.h5'.format(int(time.time()), np.around(np.min(history.history['val_mse']), decimals=4))
    lstm_model.save(save_name)
```
- example : `1618239997_0.0002.h5`
2. 使用現有模型(ex:training.h5)來產出股票預測及操作操作
```shell 
$ python test_model.py --model [model name] --training [trainging data] --testing [testing data]
```
- example
```shell 
$ python test_model.py --model training.h5 --training training.csv --testing testing.csv
```
3. 計算 profit，為訓練 model 所使用的對應測試 csv 檔
```shell 
$ python profit_calculator.py testing.csv output.csv
```

## 實做細節
### app.py 參數
```python=
parser.add_argument("--training", default="training_data.csv", help="input training data file name")
parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
parser.add_argument("--output", default="output.csv", help="output file name")
```
- 訓練資料集預設為 training_data.csv
- 測試資料集預設為 testing_data.csv
- 預設輸出檔名為 output.csv
### Training Data

- 使用兩種資料集

1. 助教提供的 IBM 資料集
- [training.csv](https://www.dropbox.com/s/uwift61i6ca9g3w/training.csv?dl=0)

```csv=1
1	186.73	188.71	186.0	186.3
2	185.57	186.33	184.94	185.54
3	184.81	185.03	183.1	184.66
4	184.39	184.48	182.31	182.54
5	182.2	182.27	180.27	181.59
6	183.23	183.72	181.2	181.31
7	180.73	182.81	180.5	182.32
...
```
- [testing.csv](https://www.dropbox.com/s/duqiffdpcadu6s7/testing.csv?dl=0)
```csv=1
1	154.4	155.02	152.91	154.76
2	155.96	156.8	155.07	156.46
3	156.45	156.74	154.68	155.35
4	154.1	156.22	154.09	154.1
5	153.59	154.45	153.26	153.57
6	154.81	155.03	153.55	154.81
7	155.46	155.89	154.57	155.41
...
```
- 由於沒有資料欄位，因此使用 `pandas.read_csv` 時會增加 `'open', 'high', 'low', 'close'` 欄位
```python=
df = pd.read_csv(args.training, header=None, usecols=[0,1,2,3], names=['open', 'high', 'low', 'close'])
```

2. [Carriage Services](https://www.nasdaq.com/market-activity/stocks/csv/historical) 資料集
- 將欄位及格式設計成與助教提供的一樣，減少訓練的操作

### Data Preprocessing
- 對各欄位做 1D Convolution 平滑化，減少雜訊
```python 
def convolution(data, ksize=3)
```
#### 1. kernel 大小需為奇數、並將原始資料邊界拓寬，使 convolution 後 size 一樣
```python=
    bias = (ksize - 1) // 2
    pb, pe = [], []

    # padding begin
    for i in range(bias):
        pb.append(data.iloc[0].values)
    # padding end
    for i in range(bias):
        pe.append(data.iloc[-1].values)
    pb, pe = np.array(pb), np.array(pe)
    data_pad = np.vstack((pb, data.values, pe))
```
#### 2. 進行 convolution

1. 平均平滑化：將自己與鄰居的值等權重加總求平均
```python=
    for i in range(data.shape[0]):
        tmp = data_pad[i:i+ksize]
        for j in range(data.shape[1]):
            data.iloc[i,j] = np.sum(tmp[:,j]) / ksize
    return data
```
- 比較，上面為原始曲線、下面為平滑化曲線
![](https://i.imgur.com/ndByUW9.png)

2. Gaussian filter : 使用簡化過的高斯分佈來當作 kernel ，優點是保留較多原始的資訊
```python=
    x = []
    for i in range(ksize):
        x.append([-bias + i])
    x = np.array(x)
    x = np.exp(-x**2)
    kernel = x / x.sum()
```
- ksize = 7
![](https://i.imgur.com/shIHrIq.png)
- ksize = 9
![](https://i.imgur.com/JiXt9Xl.png)

- ksize = 11
![](https://i.imgur.com/08iocI6.png)

- 比較，上面為原始曲線、下面為平滑化曲線
ksize = 5
![](https://i.imgur.com/7EIWJQr.png)
ksize = 7
![](https://i.imgur.com/6ut9RX9.png)
ksize = 9
![](https://i.imgur.com/7PEOwEQ.png)

### Maching Learning Model
- input : 30 天的股市資料、包含 open price, high price, low price, close price
- output : 延續 input 30 天資料隔天的 open price
- LSTM

```python
    BATCH_SIZE = 32
    REF_DAY = 30
    PREDICT_DAY = 1
    TRAIN_RATIO = 0.8
    EPOCH = 200
    PATIENCE = 20
    KERNEL_SIZE = 9
```

```shell=
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 30, 64)            17664     
_________________________________________________________________
dropout (Dropout)            (None, 30, 64)            0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 30, 64)            33024     
_________________________________________________________________
dropout_1 (Dropout)          (None, 30, 64)            0         
_________________________________________________________________
time_distributed (TimeDistri (None, 30, 1)             65        
_________________________________________________________________
flatten (Flatten)            (None, 30)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                1984      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65        
=================================================================
Total params: 52,802
Trainable params: 52,802
Non-trainable params: 0
_________________________________________________________________
```
- Callback : 使用 early_stopping 、 patience 為 20
- 訓練上限 200 epoch

### Result
#### loss history (epoch = 200)
![](https://i.imgur.com/RyQiMsU.png)
#### predict policy
- 依據 model 預測隔天的 open price 來決定明天的買賣
    - policy
    - 1. 預測明天漲
        - 持有股票 ： -1(賣掉)
        - 沒持有股票 ： -1(賣空)
        - 賣空狀態 ： 0(繼續持有)
    - 2. 預測明天跌
        - 持有股票 ： 0(持有)
        - 沒持有股票 ： 1(賣進)
        - 賣空狀態 ： 1(買空)
    ```python=
    # state 股票持有的狀態 1:持有一張 0:沒持有 -1:賣空
    # dp1 今日預測跟昨日預測之 open price 的差值
    def makeDecision(state, dp1) -> int:
        decision = 0
        if dp1 < 0:
            if state == 1:
                decision = 0
            elif state == 0:
                decision = 1
            else :
                decision = 1
        elif dp1 > 0:
            if state == 1:
                decision = -1
            elif state == 0:
                decision = -1
            else :
                decision = 0
    return decision
    ```
- 若測試的 csv 有 20 天，會將此 20 天資料之前的指定天數（在此為 30 天）設為初始預測 input 、預測完第一天後第二天再將第一天的資料加入預測 input 中、並去掉最舊的資料，因此每天都會有一筆新的資料加入預測 input 中
- 保證不會使用到預測目標日期之後（含）的資料當作訓練 input 
- 使用助教提供的 `profit_calculator.py` 來運算
- 程式碼
```python=
state = 0
dp1 = 0
with open(args.output, "w") as output_file:
        for i in range(testing_data.shape[0] - 1):
            # add today's data into predict input and drop the oldest day's data
            predict_input_raw = np.vstack((predict_input_raw, testing_data.iloc[i]))[1:]
            
            # setting model input for prediction
            tmp = []
            tmp.append(predict_input_raw)
            predict_input = np.array(tmp)
            
            # predict
            predict_output = lstm_model.predict(predict_input)[0]
            predict_output = recoverNormalize(predict_output, data_min, data_diff)
            predict_res.append(predict_output[0])
            print(predict_output)

            # condition val 
            dp1 = predict_output[0] - cache
            cache = predict_output[0]

            # make decision
            decision = makeDecision(state, dp1)
            state += decision
            assert state <= 1 or state >= -1, 'something wrong!!!'
            print('decision : {}'.format(decision))

            output_file.write('{}\n'.format(decision))
```
- 股票曲線
![](https://i.imgur.com/IPpPzDB.png)
- profit 使用 助教提供的 testing.csv 測試，取其中表現較好的結果
```shell=
01 : 8.79000000000002 // epoch = 200 (169), mean blur, ksize = 5
02 : 10.139999999999986 // epoch = 120, mean blur, ksize = 5
03 : 7.010000000000048 // epoch = 200, Gaussian blur, ksize = 5
04 : 5.670000000000016 // epoch = 200, Gaussian blur, ksize = 5
```

### 心得
- 使用 Gaussian filter 做 convolution 結果較為穩定，猜測是保留較多原始資訊並且也較為平滑，使用平均平滑化浮動較大，有時表現極好、有時表現較差
- 預測結果曲線大致都會呈現平行狀態，且只有第一天的走勢會比較精確，第一天以後的走勢都會合第一天差不多
- 使用較違反直覺的 逢漲就賣、逢跌就買 不知道為何表現反而較好
- policy 是一門學問阿....
- 越複雜的 policy 風險越高（？


