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
## members 

msno - идентификатор пользователя

city - город

bd: возраст

gender - пол

registered_via - способ регистрации

registration_init_time - начало регистрации

expiration_date - дата окончания

столбец gender имеет пропущенные значения, а также столбцы с датами (registration_init_time и expiration_date) представлены в формате целых чисел.

## sample submission

Пример результирующего файла

id - такой же как id в test.csv

target - целевая переменная

## song extra info 

song_id - идентификатор песни

name - название

iscr - Международный стандартный код записи, в теории может использоваться как идентификатор песни

## song

song_id - идентификатор песни

song_length - длительность в миллисекундах

genre_ids - категория жанра

artist_name - имя артиста

composer - композитор

lyricist - автор слов

language - язык

## Train

msno - идентификатор пользователя 

song_id - идентификатор песни

source_system_tab - название вкладки, где произошло событие

source_screen_name - название макета, который видит пользователь

source_type - точка входа, с которой пользователь впервые воспроизводит музыку в мобильных приложениях

target - целевая переменная

## Test

id - идентификатор строки

msno - идентификатор пользователя 

song_id - идентификатор песни

source_system_tab - название вкладки, где произошло событие

source_screen_name - название макета, который видит пользователь

source_type - точка входа, с которой пользователь впервые воспроизводит музыку в мобильных приложениях

# New Data

## Добавим в train данные из таблиц songs, song_extra_info, members
объединение будем производить по ключевым столбцам:

+ train -- songs --> 'song_id'

+ train -- song_extra_info_csv --> 'song_id'

+ train -- members --> 'msno'

```
train_songs = pd.merge(train_csv, songs_csv, on='song_id', how='left')
train_songs_se = pd.merge(train_songs, song_extra_info_csv, on='song_id', how='left')
songs = pd.merge(train_songs_se, members_csv, on='msno', how='left')
# del songs_df, songs_extra_df, members_df, train_df, train_songs, train_songs_se
songs.head()
```

## Заполним пустые строки 'unknown' для object, 0 для числовых

```
for i in songs.select_dtypes(include=['object']).columns:
    songs.loc[songs[i].isnull(), i] = 'unknown'
songs = songs.fillna(value=0)
```

## Распишем даты в отдельные столбцы day, month, year

```
# registration_init_time
songs.registration_init_time = pd.to_datetime(songs.registration_init_time, format='%Y%m%d', errors='ignore')
songs['registration_init_time_year'] = songs['registration_init_time'].dt.year
songs['registration_init_time_month'] = songs['registration_init_time'].dt.month
songs['registration_init_time_day'] = songs['registration_init_time'].dt.day

# expiration_date
songs.expiration_date = pd.to_datetime(songs.expiration_date,  format='%Y%m%d', errors='ignore')
songs['expiration_date_year'] = songs['expiration_date'].dt.year
songs['expiration_date_month'] = songs['expiration_date'].dt.month
songs['expiration_date_day'] = songs['expiration_date'].dt.day
```

## Сформиурем итоговый датасет

```
X = songs.drop('target', axis = 1)
y = songs.target
```

## Закодируем данные

```
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
label_encoder = LabelEncoder()
#one_hot = OneHotEncoder()

for i in X.columns :
    X[i] = label_encoder.fit_transform(X[i])
```

С тестовыми данными проделываем тоже самое, но при этом изначально записываем столбец id, удаляем id

## Разделим данные на тренировочные и валидационные

```from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 0)
```
# Модель Случайны лес

Модель случайного леса хорошо справляется с большим количеством признаков, что делает ее подходящей для задач с большим числом признаков.

```
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
        n_estimators = 100,
        
)

rf.fit(X_train, y_train)
```

```
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
val_pred = rf.predict(X_val)
print("Accuracy :", accuracy_score(y_val, val_pred))
```

| Accuracy|: 0.7277530627238249      |

## Вероятности предсказаний

```
import numpy as np
# Получить вероятности для каждого класса
predictions_proba = np.zeros((len(X_val), 2))  # Создать массив для хранения вероятностей
for tree in rf.estimators_:
    proba = tree.predict_proba(X_val)
    predictions_proba += proba  # Суммировать вероятности отдельных деревьев

# Получить средние вероятности от всех деревьев
predictions_proba /= len(rf.estimators_)

# Вероятность принадлежности к классу 1
probability_class_1 = predictions_proba[:, 1]

```

## NDCG@20

```
import numpy as np
from sklearn.metrics import ndcg_score

# Задать y_true, содержащий только фактические метки класса
y_true = np.array(y)

#Предсказанные вероятности
predictions_proba = probability_class_1

# Отсортировать вероятности предсказаний для каждого примера по убыванию вероятности класса
sorted_indices = np.argsort(predictions_proba)
sorted_predictions_proba = predictions_proba[sorted_indices]

# Отсортировать y_true соответственно
sorted_y_true = y_true[sorted_indices]

# Вычислить NDCG@20 для отсортированных вероятностей
ndcg_score_val = ndcg_score([sorted_y_true], [sorted_predictions_proba])

print("NDCG@20:", ndcg_score_val)
```
| NDCG@20 |: 0.967556191017638 |

## Предсказание для тестовых данных и схоранение

```
# Получить прогнозы модели
predictions = rf.predict(T_songs)

# Создать DataFrame с прогнозами
predictions_df = pd.DataFrame(T_id, columns=['id'])
# Добавить колонку с идентификаторами объектов
predictions_df['target'] = predictions
# predictions_df = pd.DataFrame(predictions, columns=['target'])
# Сохранить DataFrame в CSV файл
predictions_df.to_csv('predictions.csv', index=False)
```

**predictions.csv** - результирующее предсказание
