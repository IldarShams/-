import streamlit as st
import reveal_slides as rs


def presentation_page():
    st.title("Презентация проекта")
    # Содержание презентации в формате Markdown
    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Задача: сразработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования или нет.
    - Используемой датасет: Используется датасет "AI4I 2020 Predictive Maintenance Dataset", содержащий 10 000 записей с 14 признаками.
    - Ссылка на датасет: https://archive.ics.uci.edu/dataset/601/predictive+maintenance+data
    - Цель: предсказать отказ оборудования (Target = 1) или его отсутствие
    (Target = 0).
    ---
    ## Этапы работы
    1. Загрузка данных.
    2. Предобработка данных.
    3. Обучение модели.
    4. Оценка модели.
    5. Визуализация результатов.
    ---
    ### 1. Загрузка данных.
    Для загрузки использовалась библиотека ucimlrepo
    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601)
    ---
    ### 2. Предобработка данных.
    Для обучения необходимо предобработать данные, убрав лишние столбцы, которые не будут учавствовать в обучении
    data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    Также надо разделить данные на тренировочные и тестовые множества
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    --- 
    ### 3. Обучение модели.
    После предобработки запускаем обучение модели
    model = LogisticRegression()
    model.fit(X_train, y_train)
    ---
    ### 4. Оценка модели.
    Оценка модели на тестовых данных
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    ---
    ### 5. Визуализация результатов.
    Визуализация результатов подготовлена на основе streamlit и matplotlib
    st.write(f"Accuracy: {accuracy:.2f}")
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    st.subheader("Classification Report")
    st.text(classification_rep)
    ---
    ## Streamlit-приложение
    Графическая часть состоит из
    - Основная страница: анализ данных и предсказания.
    - Страница с презентацией: описание проекта.
    ---
    ## Заключение
    - Загрузили и предобработали данные 
    - Обучили регрессионную модель;
    - Оценили регрессионную модель. Точность составила 0.99, F1-score = 0.99
    """
    # Настройки презентации
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige",
                                      "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота слайдов", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave",
                                              "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex",
                                             "mathjax2", "mathjax3", "notes", "search", "zoom"], [])
    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )

presentation_page()