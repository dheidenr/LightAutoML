# Стандартные python библиотеки

# Важные для "Науки о Данных" библиотеки
import numpy as np
import pandas as pd
import torch
# LightAutoML предустановки, класса "Задача" и генерация отчета.
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.report.report_deco import ReportDeco
from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split

# Константы
# Тут мы установим константы для использования в ядре kaggle
# N_THREADS - количество виртуальных ядер CPU для создания модели в LightAutoML
# N_FOLDS - число слоея в LightAutoML внутри CV
# RANDOM_STATE - зерно генератора случайных чисел для лучшей воспроизводимости
# TEST_SIZE - размер тестовой части данных
# TIMEOUT - предел в секундах для тренеровки модели.
# TARGET_NAME - имя целевого столбца в наборе данных.

N_THREADS = 4
N_FOLDS = 3
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 2 * 3600
TARGET_NAME = 'target'

# Для лучшей воспроизводимости устанавливаем зерно для numpy
# и  число потоков для Torch (который любит
#  использовать все доступные потоки сервера)
np.random.seed(RANDOM_STATE)
torch.set_num_threads(N_THREADS)

# Проверим данные которые у нас есть:
train_data = pd.read_csv('input/lightautoml-course-finals/train.csv')
print(train_data.head())
print(train_data.shape)

test_data = pd.read_csv('input/lightautoml-course-finals/test.csv')
print(test_data.head())
print(test_data.shape)

sample_sub = pd.read_csv('input/lightautoml-course-finals/sample_submission.csv')
print(sample_sub.head())
print(sample_sub.shape)

# Поскольку у нас есть только один файл с целевыми значениями
# мы можем его разбить 80% на 20%:

tr_data, te_data = train_test_split(train_data,
                                    test_size=TEST_SIZE,
                                    stratify=train_data[TARGET_NAME],
                                    random_state=RANDOM_STATE)
print(f'Данные разбиты. '
      f'Доли: tr_data = {tr_data.shape}, te_data = {te_data.shape}')
print(tr_data.head())


# Построение модели используя LightAutoML
# Ниже мы создадим объект класса Task.
# Task - класс устанавливающий какую задачу должна решать модель LightAutoML
# с определенными потерями и метрикой, если необходимо.

task = Task('binary', )

# Установим роли для характеристик.
# Для решения задачи нам необходимо настроить роли для столбцав.
# При этом обязательной ролью евляется только 'target' (целевая).
# Все остальные (drop, numeric, categorical, group, wights и т.д.) по желанию.

roles = {'target': TARGET_NAME,
         'drop': ['text_0', 'text_1']}

# Создание модели через предустановку TabulaAutoML.
# ...

automl = TabularAutoML(task=task,
                       timeout=TIMEOUT,
                       reader_params=
                       {'n_jobs': N_THREADS,
                        'cv': N_FOLDS,
                        'random_state': RANDOM_STATE},
                       general_params={'use_algos': [['linear_l2', 'lgb']]}
                       )
RD = ReportDeco(output_path='tabularAutoMl_model_report')
automl_rd = RD(automl)

oof_pred = automl_rd.fit_predict(tr_data, roles=roles)
print(f'oof_pred:\n{oof_pred}\nShape = {oof_pred.shape}')

# Создание новыех характеристик...
