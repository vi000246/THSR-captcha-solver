# THSR-captcha-solver
###### tags: `pytorch` `captcha-solving` `cnn-pytorch` `pytorch` `thsr`

**僅供學術研究用途，請勿使用於其他用途。**

## Introduction

本專案用Convolutional Neural Network來辨識高鐵訂票網站的驗證碼，並使用pytorch來實作。\
目前模型整張圖片(4個字元全對)辨識率達到98%以上(測試集為2000張)。 \
\
This project uses the convolutional neural network(CNN) to solve the captcha in Taiwan High Speed Rail booking website, and uses pytorch to implement it.
The recognition rate of a whole captcha (4 characters correct) is over 98% (the size of test set is 2000).

## Requirements
* python==3.6
* matplotlib==3.1.1
* numpy==1.17.3
* opencv-python==4.1.1.26
* Pillow==5.3.0
* scikit-learn==0.21.3
* scipy==1.1.0
* torch==1.3.0+cu92
* torchvision==0.4.1+cu92

> torch及torchvision安裝方法可以參考[官方網站](https://pytorch.org/)

## 1. Dataset
資料集原本打算寫一個自動產生模仿資料集的腳本，但考量到對這方面不熟且開發時間有限，因此後來決定全部先手動標記。
之後在用前面手動標的資料訓練出一個堪用的model來做半自動的標記。(此repository僅提供100張驗證碼供測試用)
### 1.1 Preprocess。
前處理主要是參考[[1][2]](#reference)，主要分成兩個部分:
1. 去雜訊
2. 去除曲線

不過實測後發現有無前處理的結果都差不多，可能是驗證碼不夠複雜基本的CNN就可以應付了。\
另外不論是否有做前處裡，進CNN之前一率會將圖片resize至128x128的大小。
### 注意
> 記得將dataset目錄傳給dataset.py中Data class的dir參數

## 2. CNN Model
下面為此專案的CNN架構，主要參考[[1]](#reference)

``` python
CNN(
  (hidden1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Dropout2d(p=0.3, inplace=False)
  )
  (hidden2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Dropout2d(p=0.3, inplace=False)
  )
  (hidden3): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Dropout2d(p=0.3, inplace=False)
  )
  (hidden4): Sequential(
    (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Flatten()
    (5): Dropout(p=0.3, inplace=False)
  )
  (digit1): Linear(in_features=9216, out_features=36, bias=True)
  (digit2): Linear(in_features=9216, out_features=36, bias=True)
  (digit3): Linear(in_features=9216, out_features=36, bias=True)
  (digit4): Linear(in_features=9216, out_features=36, bias=True)
)
```

## Training
main.py的train funcion中有一些hyperparameters可以做調整，預設值也是參考[[1]](#reference)\
train時只要直接執行main.py:
``` python
python main.py
```
預設每個epoch會存一個checkpoint

## Testing
可以使用main.py裡的test function做單張圖的inference。


## Reference
[1] [simple-railway-captcha-solver](https://github.com/JasonLiTW/simple-railway-captcha-solver)\
[2] [[爬蟲實戰] 如何破解高鐵驗證碼 (1) - 去除圖片噪音點?](https://youtu.be/6HGbKdB4kVY)\
[3] [[爬蟲實戰] 如何破解高鐵驗證碼 (2) - 使用迴歸方法去除多餘弧線?](https://youtu.be/4DHcOPSfC4c)\
[4] [pytorch-book
](https://github.com/chenyuntc/pytorch-book)
