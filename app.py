import streamlit as st

# Настройка навигации
pages = [
    st.Page("analysis_and_model.py", title="Анализ и модель"),
    st.Page("presentation.py", title="Презентация"),
]
# Отображение навигации
current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()

# import streamlit as st
#
# st.set_page_config(page_title="Прогнозирование отказов", layout="wide")
# st.title("Главная страница")
# st.write("Выберите страницу в меню слева.")