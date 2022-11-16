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
|Speaker|Gender|Number of frames\n(Train / Test)|
|------|---|---|
|None| - |166(140/26)|


### Results
![cm](https://user-images.githubusercontent.com/76679855/202119162-3947adc4-75e7-4ae5-9874-dea46a63d12a.png)
