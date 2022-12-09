#  :oncoming_automobile: Jeju Road Traffic Prediction :sunrise_over_mountains:
[제주 도로 교통량 예측 경진대회](https://dacon.io/competitions/official/235985/overview/description)

![image](https://user-images.githubusercontent.com/84713532/206641256-137974fc-f097-4c6f-843c-d23976861409.png)

## :bookmark_tabs: Mini Project 1 (2022/12/06 ~ 2022/12/09) :date:

> :family: 팀명: 차탄갑서
- [고석주](https://github.com/SeokJuGo)
- [지우근](https://github.com/UGeunJi)
- [이재영](https://github.com/JAYJAY1005)
- [양효준](https://github.com/Raphael)

---

## :scroll: 프로젝트에 대한 전반적인 설명

### 주제 : <머신러능 성능 올리기>

#### 1. 문제 정의
```
- 문제 유형 (수치 예측, 분류)
- 성능 평가 지표 (수치 예측  : MSE, RMSE, MAE, RMSLE,  분류 : 정확도, f1 score, roc_auc )
```
#### 2. 탐색적 데이터 분석
```
2. 1 데이터 훑어보기
  - 데이터 양 (샘플수, 특성 수...)
  - 특성 이해
  - 타깃값 이해 (<--- 예측해야 하는 값)
  - 데이터세트(훈련/테스트) 분리
    (1) 합쳐서 전체를 전처리 한 후에 데이터 세트 분리 (bike sharing)
    (2) 훈련데이터 전처리한 후에, 동일한 과정을 테스트 세트에 적용 (california housing)
```
```
2.2 데이터 시각화
    (1) 수치형 데이터
       - 히스토그램 : 빠르게 데이터의 분포를 파악
                    : 상한~하한, 많이 분포한 데이터의 위치
                         : 스케일 여부
                         : 왜곡도 여부(꼬리가 긴 분포는 이후에 정규분포 형태로 로그 변환)

    (2) 범주형 데이터 : 범주형 데이터에 따른 수치값을 확인
       - 바플롯 : 타깃값과의 관계 확인
          ( 예 : 성별이 생존에 어떤 영향을 미치는지, 성별이 중요한특성인지, 예측력이 있는지..)
          ( 예 : Pclass가 생존에 어떤 영향을 미치는지)
          ( 참고 : Age와 같이 수치형 데이터의 예측력을 알고 싶다면 범주화(pd.cut)해서 시각화 해볼수 있음)
       - 박스 플롯 : 데이터의 사분위 분포, 이상치 검출
        
    (3) 상관관계 파악
       - 산점도
       - 상관계수
```

#### 3. 전처리 

```
   : 탐색적 데이터 분석을 통해 전처리 전략 적용
   : pandas 사용, 또는 scikit-learn pipeline 사용 가능
```
```
3.1 특성 선택 / 특성 삭제 / 특성 조합
```
```
3.2 인코딩
    (1) 레이블 인코딩
    (2) 원핫인코딩
```
```
3.3 스케일링
    (1) MinMax 스케일링(정규화) : 0~1사이로
    (2) Standardized 스케일링(표준화) : 평균 0, 표준편차 1
    (3) 로그 스케일링 : 왜곡된 데이터를 정규 분포 형태로 (특성, 타깃 모두 적용 가능)
```
#### 4. 베이스라인 모델
```
4. 1 교차검증을 통한 모델 간 성능 비교
    : 교차 검증 = 훈련 + 검증을 교번하며 진행
    : 성능 측정 지표 설정해줘야 함

   (1) 수치 예측
       선형 회귀 (회귀 계수 : 특성의 영향력)
       트리 모델 (특성의 중요도)
       앙상블 모델
       서포트 벡터 머신
       kNN

   (2) 분류
      로지스틱 회귀 : 이진분류 (회귀 계수 : 특성의 영향력)
      소프트맥스 회귀 : 다중 분류 (회귀 계수 : 특성의 영향력)
      트리 모델 (특성의 중요도)
      앙상블 모델
      서포트 벡터 머신
      kNN
```
#### 5. 성능 올리기
```
5.1 데이터 측면
    탐색적 데이터 분석을 통해 나온 결과를 다양하게 실험(적용)
     : 결측치처리, 이상치 제거
     : 특성 추가, 삭제, 파생 특성 추가
     : 특성 스케일링, 특성 인코딩
```
```
5.2 모델 측면
     : 앙상블 모델 강력
     : 하이퍼파라미터 튜닝 : 그리드 탐색, 랜덤 탐색
```
#### 6. 최종 예측과 성능 평가
```
   테스트 데이터로 예측
   : 테스트 데이터는 훈련데이터의 전처리된 형식과 동일해야함
   (1) 테스트 데이터에 대한 정답이 있으면 바로 평가 가능
   (2) 경진대회용 테스트 데이터의 경우 예측 결과를 제출함으로 평가를 받을 수 있음
```

---

## 실행 코드
