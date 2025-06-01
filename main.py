import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_pacf

# Загрузка данных
df = pd.read_csv('tovar_moving.csv', parse_dates=['date'], index_col='date')

# 1. Отложить последнее значение в тестовую выборку
test = df.iloc[[-1]]
train = df.iloc[:-1]

# 2. Анализ тренда и сезонности
plt.figure(figsize=(12, 6))
plt.plot(train)
plt.title('Товарооборот книжного магазина (2009-2013)')
plt.xlabel('Дата')
plt.ylabel('Количество заказов')
plt.grid(True)
plt.show()

# 3. Экспоненциальное сглаживание (α=0.7)
# Используем только обучающую выборку
model_ses = SimpleExpSmoothing(train['qty']).fit(smoothing_level=0.7, optimized=False)
forecast_ses = model_ses.forecast(1)

# Сравнение с фактическим значением
actual = test['qty'].values[0]
ses_pred = forecast_ses.values[0]
print(f'Прогноз SES: {ses_pred:.0f}, Фактическое: {actual:.0f}, Ошибка: {abs(ses_pred-actual):.0f}')

# 4. Проверка на стационарность
def check_stationarity(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Критические значения:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    return result[1] > 0.05

# Проверка исходного ряда
print("\nПроверка исходного ряда на стационарность:")
is_non_stationary = check_stationarity(train['qty'])

# Определение порядка интегрирования
d = 0
current_series = train['qty'].copy()

while is_non_stationary and d < 2:
    d += 1
    current_series = current_series.diff().dropna()
    print(f"\nРяд после {d}-го дифференцирования:")
    is_non_stationary = check_stationarity(current_series)

print(f'\nПорядок интегрирования d = {d}')

# 5. Определение порядка AR через PACF
# Используем исходный ряд для определения порядка AR
plt.figure(figsize=(12, 6))
plot_pacf(train['qty'], lags=40, method='ols')
plt.title('График частичной автокорреляции (PACF)')
plt.xlabel('Лаг')
plt.ylabel('Значение PACF')
plt.grid(True)
plt.show()

# 6. Построение модели AR (на исходном ряде)
# Выберем лаги 1, 7, 14, 21, 28 как значимые (визуально по PACF)
lags = [1, 7, 14, 21, 28]
model_ar = AutoReg(train['qty'], lags=lags).fit()
print(model_ar.summary())

# Прогноз последнего значения
forecast_ar = model_ar.predict(start=len(train), end=len(train))
ar_pred = forecast_ar.values[0]
print(f'\nПрогноз AR: {ar_pred:.0f}, Фактическое: {actual:.0f}, Ошибка: {abs(ar_pred-actual):.0f}')

# 7. Сравнение результатов
results = pd.DataFrame({
    'Метод': ['Экспоненциальное сглаживание', 'Авторегрессия (AR)'],
    'Прогноз': [ses_pred, ar_pred],
    'Фактическое': [actual, actual],
    'Ошибка': [abs(ses_pred-actual), abs(ar_pred-actual)]
})

print("\nСравнение результатов:")
print(results)