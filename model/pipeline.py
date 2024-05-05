import datetime

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def main():
    print('Start')
    # прочитаем два датасета и запишем их в перемеyные df_hits и df_sessions
    df_hits = pd.read_csv("C:\\Users\\mitsa\\PycharmProjects\\python_JN\\data\\ga_hits.csv")
    df_sessions = pd.read_csv("C:\\Users\\mitsa\\PycharmProjects\\python_JN\\data\\ga_sessions.csv")

    # ОБРАБОТАЕМ df_hits
    # удалим ненужные столбцы из df_hits
    df_hits.drop(
        columns=['hit_date', 'hit_time', 'hit_number', 'hit_type', 'hit_referer', 'hit_page_path', 'event_category',
                 'event_label', 'event_value'], axis=1, inplace=True)

    # создадим список целевых действий в колонке 'event_action' согласно описанию в задании
    target = ['sub_car_claim_click', 'sub_car_claim_submit_click',
              'sub_open_dialog_click', 'sub_custom_question_submit_click',
              'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
              'sub_car_request_submit_click']

    # создадим новую колонку 'target' в df_clean_hits и проставим в ней 1/0 в зависимости от соответсвия целевого действия в колонке 'event_action'
    for i in target:
        df_hits['target'] = df_hits['event_action'].apply(lambda x: 1 if x == i else 0)

    # понизим разрядность в столбце target в датасете df_hits
    df_hits['target'] = df_hits['target'].astype(np.int16)

    # ОБРАБОТАЕМ df_sessions
    # удалим столбцы с пропущенными значениями более 50% и ненужные столбцы из df_sessions
    df_sessions.drop(columns=['client_id', 'visit_date', 'visit_time', 'utm_keyword', 'device_os', 'device_model'],
                     axis=1, inplace=True)

    # объеденим df_hits и df_sessions в один финальный
    df = df_sessions.merge(df_hits, on='session_id')

    # удаляем колонку 'session_id' и 'event_action'
    df.drop(columns=['session_id', 'event_action'], axis=1, inplace=True)

    # сбалансируем датасет
    # отношение количества строк  0 к 1 в колонке target
    rat = len(df.loc[df['target'] == 0]) // len(df.loc[df['target'] == 1])

    df_1 = df.loc[df['target'] == 1]
    df_1 = df_1.loc[df_1.index.repeat(rat)]
    df = pd.concat([df.loc[df['target'] == 0], df_1]).sample(frac=1)

    # возьмём 1% от данных
    df_1 = df.sample(frac=0.01)

    # выделим целевой признак
    x = df_1.drop('target', axis=1)
    y = df_1['target']

    # создаем списки количественных
    # и категориальных столбцов
    numerical_features = x.select_dtypes(include=['int64', 'int16', 'float64', 'float16']).columns
    categorical_features = x.select_dtypes(include='object').columns

    # создаем конвейер для категориальных переменных
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')), # пропуски заполняем модой
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore')) # кодируем категориальные фичи с помощью OneHotEncoder
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)
    ])

    # определим список моделей
    models = (
        LogisticRegression(solver='liblinear'),
        #MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64))
    )

    best_score = .0
    best_model = None

    # пройдём циклом по моделям и пременим пайплайн первым шагом будет препроцессор, а вторым сам классификатор
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # посчитаем метрику. Применим кросвалидацию на 4 фолда, выведем среднее метрики ROC-AUC
        score = cross_val_score(pipe, x, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model),__name__}, roc_auc_mean: {score.mean():.4f}, roc_auc_std: {score.std():.4f}')

        # выберем лучшую модель по результатам скора
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(x, y) # обучим модель на всём датасете
    print(f'best_model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.4f}')
    joblib.dump({
        'model': best_pipe,
        'metadata': {
            'name': 'client predict model',
            'author': 'Valeriy Oleynikov',
            'version': 1,
            'date': datetime.datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'ROC-AUC': best_score
        }
    },'final_ds.pkl') # сохраним модель в пикл файл


if __name__ == '__main__':
    main()
