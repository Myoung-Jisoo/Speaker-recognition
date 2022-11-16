# Speaker-recognition
여러 명의 화자를 인식하는 것을 목표로 하는 프로젝트입니다. 
(딥러닝 수업 개인 프로젝트로 진행했습니다.)
Class는 8개로 목소리X(none), 연예인 3명, 일반인 4명의 목소리를 분류했습니다.

tensorflow-2.3.0

pip install tensorflow==2.3.0

pip install opencv-python

pip install opencv-contrib-python

librosa


### Dataset
|Speaker|Gender|Number of frames (Train / Test)|
|:---:|:---:|:---:|
|None| - |166 (140 / 26)|
|A|Female|240 (140 / 100)|
|KLN|Female|161 (140 / 21)|
|B|Female|207 (140 / 67)|
|C|Male|180 (140 / 40)|
|SJH|Male|167 (140 / 27)|
|IU|Female|168 (140 / 38)|
|D|Male|175 (140 / 35)|





### Results
#### 1. Accuracy
|Speaker|Accuracy(%)|Precision|Recall|
|:---:|:---:|:---:|:---:|
|None|100|0.93|1.00|
|A|100|0.99|1.00|
|KLN|100|0.95|1.00|
|B|98.51|0.99|0.99|
|C|92.50|1.00|0.93|
|SJH|96.30|0.93|0.96|
|IU|94.74|1.00|0.95|
|D|97.14|0.97|0.97|



#### 2. Confusion Matrix
![cm](https://user-images.githubusercontent.com/76679855/202119162-3947adc4-75e7-4ae5-9874-dea46a63d12a.png)
