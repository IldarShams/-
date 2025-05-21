from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Загрузка датасета
data = pd.read_csv("ai4i2020.csv")
# Удаление ненужных столбцов
data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
# Преобразование категориальной переменной Type в числовую
data['Type'] = LabelEncoder().fit_transform(data['Type'])
# Проверка на пропущенные значения
# print(data.isnull().sum())

# Признаки (X) и целевая переменная (y)
X = data.drop(columns=['Machine failure'])
y = data['Machine failure']
pd.set_option('display.max_columns', 500)
# print(X.take([2167]))
# print(y.take([2167]))
# exit()
# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Создание и обучение модели
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# productID = 2
# air_temp = 299.2
# process_temp = 308.9
# rotational_speed = 1447
# torque = 39.6
# tool_wear = 22
# 302.2	311.3	1530	37.3	207

# input = [[productID, air_temp, process_temp, rotational_speed, torque, tool_wear]]
# pred = log_reg.predict(input)
# print(pred)
# exit()


# Функция для оценки модели
def evaluate_model(model, X_test, y_test):
    # Предсказания
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Вероятности для ROC-AUC
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    # Вывод результатов
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)
    print("ROC-AUC:", roc_auc)
    # Построение ROC-кривой
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{model.__class__.__name__} (AUC ={roc_auc:.2f})")
    # Оценка Logistic Regression
    print("Logistic Regression:")
    evaluate_model(log_reg, X_test, y_test)
    # Визуализация ROC-кривых
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые')
    plt.legend()
    plt.show()
