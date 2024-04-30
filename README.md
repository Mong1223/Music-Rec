# Music-Rec
VK-Music-Rec
# Задание
Дан датасет прослушиваний музыки, который нужно скачать [по ссылке](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data).
Задача: 
На основе этих данных построить рекомендательную систему релевантных треков для пользователей и оценить ее качество.

В качестве метрики качества используйте NDCG@20.

Что будет оцениваться:
1.Наличие и содержательность exploratory data analysis.
2.Пайплайн обучения модели: подготовка данных, feature engineering, выбор модели, выбор loss функции, корректная оценка метрик качества.
3.Наличие и содержательность комментариев к происходящему.
4.Финальный скор.


# Данные состоят из 6 таблиц
## members csv

msno - идентификатор пользователя

city - город

bd: возраст

gender - пол

registered_via - способ регистрации

registration_init_time - начало регистрации

expiration_date - дата окончания

```
members_csv.info()
```

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 34403 entries, 0 to 34402
Data columns (total 7 columns):
   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   msno                    34403 non-null  object
 1   city                    34403 non-null  int64 
 2   bd                      34403 non-null  int64 
 3   gender                  14501 non-null  object
 4   registered_via          34403 non-null  int64 
 5   registration_init_time  34403 non-null  int64 
 6   expiration_date         34403 non-null  int64 
dtypes: int64(5), object(2)
memory usage: 1.8+ MB