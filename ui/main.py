import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Настройка соединения
API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="Оценка стоимости недвижимости",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Опции выбора
FURNISHED_OPTIONS = {
    "Eşyalı": "Меблировано", "Eşyasız": "Без мебели",
    "Sadece Beyaz Eşya": "Только бытовая техника", "Sadece Mutfak": "Только кухня",
}
HEATING_OPTIONS = {
    "Kalorifer (Doğalgaz)": "Центральное (газ)", "Kombi (Doğalgaz)": "Газовый котёл",
    "Klima": "Кондиционер", "Merkezi Sistem": "Центральное отопление",
    "Yerden Isıtma": "Тёплый пол", "Soba (Kömür)": "Печь (уголь)",
    "Güneş Enerjisi": "Солнечная энергия", "Yok": "Отсутствует",
}
LISTING_TYPE_OPTIONS = {"Satılık": "Продажа", "Kiralık": "Аренда"}

# Стилизация
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {background: #fafafa;}
    .block-container {padding: 3rem; max-width: 1180px;}
    div[data-testid="stForm"] {border: 1px solid #ececec; border-radius: 16px; background: #ffffff; padding: 1.6rem;}
    div[data-testid="stFormSubmitButton"] button {background: #1a1a1a; color: #ffffff; border-radius: 10px; border: none;}
    div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #ececec; border-radius: 14px; padding: 0.9rem;}
    </style>
""", unsafe_allow_html=True)

# Функции API
def call_predict(payload: dict) -> dict:
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": "Ошибка подключения", "details": str(exc)}

@st.cache_data(ttl=300)
def get_models_list() -> pd.DataFrame:
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard/models", timeout=10)
        return pd.DataFrame(response.json())
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_dataset_sample() -> pd.DataFrame:
    try:
        response = requests.get(f"{API_BASE_URL}/dashboard/dataset/sample", timeout=30)
        # Если статус не 200, это выбросит исключение
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        # Теперь мы увидим реальную причину ошибки на экране
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

def get_api_info() -> dict:
    try: return requests.get(f"{API_BASE_URL}/info", timeout=5).json()
    except: return {}

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df

# Навигация
if "page" not in st.session_state: st.session_state.page = "Прогноз"
nav_col, main_col = st.columns([1, 6], gap="large")

with nav_col:
    st.caption("МЕНЮ")
    for p in ["Прогноз", "Дашборд", "Справка"]:
        if st.button(p, key=f"nav_{p}", use_container_width=True, type="primary" if st.session_state.page == p else "secondary"):
            st.session_state.page = p
            st.rerun()

# Основной контент
with main_col:
    if st.session_state.page == "Прогноз":
        st.title("Оценка стоимости")
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                size = st.number_input("Площадь, м²", 10.0, 2000.0, 100.0)
                room_count = st.selectbox("Комнаты", ["1+0", "1+1", "2+1", "3+1", "4+1", "5+1"])
                building_age = st.number_input("Возраст здания, лет", 0, 150, 5)
            with c2:
                total_floor_count = st.number_input("Этажей в здании", 1, 100, 5)
                floor_no = st.number_input("Номер этажа", -3, 100, 2)
                listing_type = st.selectbox("Тип объявления", list(LISTING_TYPE_OPTIONS.values()))
            with c3:
                furnished = st.selectbox("Меблировка", list(FURNISHED_OPTIONS.values()))
                heating_type = st.selectbox("Отопление", list(HEATING_OPTIONS.values()))
                address = st.text_input("Район / адрес", placeholder="например, Kadıköy")

            submitted = st.form_submit_button("Рассчитать стоимость", use_container_width=True)

        if submitted:
            if not address.strip(): st.error("Укажите район/адрес.")
            elif floor_no > total_floor_count: st.error("Этаж > этажности здания.")
            else:
                payload = {"size": size, "room_count": room_count, "building_age": building_age, "total_floor_count": total_floor_count, "floor_no": floor_no, "listing_type": listing_type, "furnished": furnished, "heating_type": heating_type, "address": address.strip()}
                with st.spinner("Расчет..."):
                    res = call_predict(payload)
                if "error" in res: st.error(res["error"])
                else:
                    st.success("Прогноз получен")
                    m1, m2 = st.columns(2)
                    m1.metric("Цена", f"{float(res['predicted_price']):,.0f} TRY")
                    m2.metric("Диапазон", f"{float(res['price_range_low']):,.0f} - {float(res['price_range_high']):,.0f} TRY")
                    st.caption(f"Модель: {res.get('model_name')}")

    elif st.session_state.page == "Дашборд":
            st.title("Дашборд")
            models_df, df = clean_dataframe(get_models_list()), clean_dataframe(get_dataset_sample())

            if df.empty:
                st.error("Данные не загружены.")
            else:
                # --- БОКОВАЯ ПАНЕЛЬ С ФИЛЬТРАМИ ---
                st.sidebar.header("Параметры фильтрации")

                # Фильтр типов
                types = st.sidebar.multiselect("Тип объявления", options=df["listing_type"].unique(), default=df["listing_type"].unique())

                # Фильтр цен (динамический)
                min_p, max_p = float(df["price"].min()), float(df["price"].quantile(0.99))
                price_range = st.sidebar.slider("Диапазон цены", min_p, max_p, (min_p, max_p))

                # Фильтр комнат
                rooms = st.sidebar.multiselect("Количество комнат", options=sorted(df["room_count"].unique()), default=sorted(df["room_count"].unique()))

                # Применение всех фильтров сразу
                mask = (df["listing_type"].isin(types)) & (df["price"].between(*price_range)) & (df["room_count"].isin(rooms))
                filtered_df = df[mask]

                # --- ОБЩИЕ МЕТРИКИ ---
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Объектов", f"{len(filtered_df):,}")
                k2.metric("Средняя цена", f"{filtered_df['price'].mean():,.0f} ₺" if not filtered_df.empty else "0")
                k3.metric("Средняя площадь", f"{filtered_df['size'].mean():.1f} м²" if not filtered_df.empty else "0")
                k4.metric("Моделей в анализе", str(len(models_df)))

                st.divider()

                # --- ИНТЕРАКТИВНЫЕ ГРАФИКИ ---
                # 1. Сравнение цен (Histogram)
                st.subheader("Распределение стоимости")
                fig1 = px.histogram(filtered_df, x="price", color="listing_type", log_x=True, template="plotly_white", barmode="overlay")
                st.plotly_chart(fig1, use_container_width=True)

                # 2. Корреляция и тренд (Scatter)
                st.subheader("Зависимость цены от площади")
                fig2 = px.scatter(
                    filtered_df,
                    x="size",
                    y="price",
                    color="room_count",
                    opacity=0.6,
                    template="plotly_white"
                )
                st.plotly_chart(fig2, use_container_width=True)

                # 3. Категориальный анализ (Violin + Bar)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("Отопление vs Цена")
                    fig3 = px.violin(filtered_df, x="heating_type", y="price", box=True, template="plotly_white")
                    fig3.update_layout(xaxis={'tickangle': -45})
                    st.plotly_chart(fig3, use_container_width=True)
                with c2:
                    st.subheader("Цена по типу комнат")
                    fig4 = px.bar(filtered_df.groupby("room_count")["price"].mean().reset_index(), x="room_count", y="price", color="price", template="plotly_white")
                    st.plotly_chart(fig4, use_container_width=True)

                # 4. Сравнение ML моделей
                if not models_df.empty and "model" in models_df.columns:
                    st.subheader("Сравнение качества моделей")
                    # Для работы этого графика убедитесь, что в моделях есть нужные колонки
                    metrics_to_plot = [m for m in ["MAE", "RMSE", "R²"] if m in models_df.columns]
                    if metrics_to_plot:
                        fig5 = px.bar(models_df, x="model", y=metrics_to_plot, barmode="group", template="plotly_white")
                        st.plotly_chart(fig5, use_container_width=True)

    else:
        st.title("Справка")
        st.markdown("### Описание проекта\nПриложение для прогнозирования цен на недвижимость с помощью ML моделей.")
        st.table(pd.DataFrame([
            ["Площадь", "Площадь в м²"], ["Комнаты", "Формат N+1"],
            ["Возраст здания", "Лет"], ["Этажность", "Всего этажей"],
            ["Тип", "Продажа/Аренда"], ["Адрес", "Район"]
        ], columns=["Поле", "Описание"]))

        st.markdown("### Информация о системе")
        api_info = get_api_info()
        if api_info: st.table(pd.DataFrame(list(api_info.items()), columns=["Параметр", "Значение"]))
        else: st.info("API Info недоступно.")
