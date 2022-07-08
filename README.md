# YOLOv5-object-detection

앱을 실행하면 카메라가 켜지고, 과자/음료/컵라면의 상품을 비추면 앞, 뒷면에 관계 없이 상품명이 음성으로 출력된다.
현재 1,000개의 상품이 학습되어 있다.

YOLOv5 모델로 상품 이미지를 학습시켜 tflite로 변환하였다.

https://github.com/AarohiSingla/TFLite-Object-Detection-Android-App-Tutorial-Using-YOLOv5.git 프로젝트를 인용하여
가장 크게 보이는 상품 한 개만 검출하도록 코드를 수정하고, 음성 기술을 추가하였다.
