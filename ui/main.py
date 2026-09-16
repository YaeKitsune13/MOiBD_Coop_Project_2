import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import random

API_BASE_URL = "http://localhost:8000/api"

st.set_page_config(
    page_title="Оценка стоимости недвижимости",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FURNISHED_OPTIONS = {
    "Eşyalı": "Меблировано",
    "Eşyasız": "Без мебели",
    "Sadece Beyaz Eşya": "Только бытовая техника",
    "Sadece Mutfak": "Только кухня",
}

HEATING_OPTIONS = {
    "Kalorifer (Doğalgaz)": "Центральное (газ)",
    "Kombi (Doğalgaz)": "Газовый котёл",
    "Klima": "Кондиционер",
    "Merkezi Sistem": "Центральное отопление",
    "Yerden Isıtma": "Тёплый пол",
    "Soba (Kömür)": "Печь (уголь)",
    "Güneş Enerjisi": "Солнечная энергия",
    "Yok": "Отсутствует",
}

LISTING_TYPE_OPTIONS = {
    "Satılık": "Продажа",
    "Kiralık": "Аренда"
}

RANDOM_ADDRESSES = [
    "İstanbul/Kadıköy",
    "İstanbul/Kartal",
    "İstanbul/Beşiktaş",
    "İstanbul/Üsküdar",
    "İstanbul/Bakırköy",
    "İstanbul/Maltepe",
    "İstanbul/Pendik",
    "İstanbul/Şişli",
    "İstanbul/Avcılar",
    "İstanbul/Beylikdüzü",
]

ROOM_OPTIONS = [
    "1+0",
    "1+1",
    "2+1",
    "3+1",
    "4+1",
    "5+1"
]

st.markdown("""
<style>
#MainMenu, footer, header {
    visibility: hidden;
}

.stApp {
    background: #fafafa;
}

.block-container {
    padding: 3rem;
    max-width: 1180px;
}

div[data-testid="stForm"] {
    border: 1px solid #ececec;
    border-radius: 16px;
    background: #ffffff;
    padding: 1.6rem;
}

div[data-testid="stFormSubmitButton"] button {
    background: #1a1a1a;
    color: #ffffff;
    border-radius: 10px;
    border: none;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #ececec;
    border-radius: 14px;
    padding: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


def call_predict(payload: dict) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {
            "error": "Ошибка подключения",
            "details": str(exc)
        }


@st.cache_data(ttl=300)
def get_dataset_sample() -> pd.DataFrame:
    try:
        response = requests.get(
            f"{API_BASE_URL}/dashboard/dataset/sample",
            timeout=30
        )
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()


def get_api_info() -> dict:
    try:
        return requests.get(
            f"{API_BASE_URL}/info",
            timeout=5
        ).json()
    except Exception:
        return {}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    num_cols = df.select_dtypes(
        include="number"
    ).columns

    df[num_cols] = df[num_cols].replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


def generate_random_values():
    rooms = random.choice(ROOM_OPTIONS)

    if rooms == "1+0":
        size = random.randint(30, 55)
    elif rooms == "1+1":
        size = random.randint(45, 80)
    elif rooms == "2+1":
        size = random.randint(70, 125)
    elif rooms == "3+1":
        size = random.randint(100, 180)
    elif rooms == "4+1":
        size = random.randint(140, 250)
    else:
        size = random.randint(180, 400)

    total_floors = random.randint(3, 15)
    floor = random.randint(1, total_floors)
    building_age = random.randint(0, 30)

    return {
        "size": float(size),
        "room_count": rooms,
        "building_age": building_age,
        "total_floor_count": total_floors,
        "floor_no": floor,
        "listing_type": random.choice(
            list(LISTING_TYPE_OPTIONS.values())
        ),
        "furnished": random.choice(
            list(FURNISHED_OPTIONS.values())
        ),
        "heating_type": random.choice(
            list(HEATING_OPTIONS.values())
        ),
        "address": random.choice(RANDOM_ADDRESSES)
    }


if "page" not in st.session_state:
    st.session_state.page = "Прогноз"

if "random_values" not in st.session_state:
    st.session_state.random_values = None


nav_col, main_col = st.columns(
    [1, 6],
    gap="large"
)

with nav_col:
    st.caption("МЕНЮ")

    for p in [
        "Прогноз",
        "Дашборд",
        "Справка"
    ]:
        if st.button(
            p,
            key=f"nav_{p}",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.page == p
                else "secondary"
            )
        ):
            st.session_state.page = p
            st.rerun()


with main_col:

    if st.session_state.page == "Прогноз":

        st.title("Оценка стоимости")

        if st.button(
            "🎲 Случайные параметры",
            use_container_width=True
        ):
            st.session_state.random_values = (
                generate_random_values()
            )
            st.rerun()

        random_values = (
            st.session_state.random_values or {}
        )

        with st.form("predict_form"):

            c1, c2, c3 = st.columns(3)

            with c1:

                size = st.number_input(
                    "Площадь, м²",
                    10.0,
                    2000.0,
                    float(
                        random_values.get(
                            "size",
                            100.0
                        )
                    )
                )

                room_count = st.selectbox(
                    "Комнаты",
                    ROOM_OPTIONS,
                    index=(
                        ROOM_OPTIONS.index(
                            random_values["room_count"]
                        )
                        if random_values.get(
                            "room_count"
                        ) in ROOM_OPTIONS
                        else 2
                    )
                )

                building_age = st.number_input(
                    "Возраст здания, лет",
                    0,
                    150,
                    int(
                        random_values.get(
                            "building_age",
                            5
                        )
                    )
                )

            with c2:

                total_floor_count = st.number_input(
                    "Этажей в здании",
                    1,
                    100,
                    int(
                        random_values.get(
                            "total_floor_count",
                            5
                        )
                    )
                )

                floor_no = st.number_input(
                    "Номер этажа",
                    -3,
                    100,
                    int(
                        random_values.get(
                            "floor_no",
                            2
                        )
                    )
                )

                listing_type = st.selectbox(
                    "Тип объявления",
                    list(
                        LISTING_TYPE_OPTIONS.values()
                    ),
                    index=(
                        list(
                            LISTING_TYPE_OPTIONS.values()
                        ).index(
                            random_values[
                                "listing_type"
                            ]
                        )
                        if random_values.get(
                            "listing_type"
                        ) in LISTING_TYPE_OPTIONS.values()
                        else 0
                    )
                )

            with c3:

                furnished = st.selectbox(
                    "Меблировка",
                    list(
                        FURNISHED_OPTIONS.values()
                    ),
                    index=(
                        list(
                            FURNISHED_OPTIONS.values()
                        ).index(
                            random_values["furnished"]
                        )
                        if random_values.get(
                            "furnished"
                        ) in FURNISHED_OPTIONS.values()
                        else 1
                    )
                )

                heating_type = st.selectbox(
                    "Отопление",
                    list(
                        HEATING_OPTIONS.values()
                    ),
                    index=(
                        list(
                            HEATING_OPTIONS.values()
                        ).index(
                            random_values["heating_type"]
                        )
                        if random_values.get(
                            "heating_type"
                        ) in HEATING_OPTIONS.values()
                        else 1
                    )
                )

                address = st.text_input(
                    "Район / адрес",
                    value=random_values.get(
                        "address",
                        ""
                    ),
                    placeholder="например, Kadıköy"
                )

            submitted = st.form_submit_button(
                "Рассчитать стоимость",
                use_container_width=True
            )

        if submitted:

            if not address.strip():
                st.error("Укажите район/адрес.")

            elif floor_no > total_floor_count:
                st.error(
                    "Этаж > этажности здания."
                )

            else:

                payload = {
                    "size": size,
                    "room_count": room_count,
                    "building_age": building_age,
                    "total_floor_count": total_floor_count,
                    "floor_no": floor_no,
                    "listing_type": listing_type,
                    "furnished": furnished,
                    "heating_type": heating_type,
                    "address": address.strip()
                }

                with st.spinner("Расчет..."):
                    res = call_predict(payload)

                if "error" in res:
                    st.error(res["error"])

                else:

                    st.success(
                        "Прогноз получен"
                    )

                    m1, m2 = st.columns(2)

                    m1.metric(
                        "Цена",
                        f"{float(res['predicted_price']):,.0f} TRY"
                    )

                    m2.metric(
                        "Диапазон",
                        f"{float(res['price_range_low']):,.0f} - "
                        f"{float(res['price_range_high']):,.0f} TRY"
                    )

                    st.caption(
                        f"Модель: {res.get('model_name')}"
                    )

    elif st.session_state.page == "Дашборд":

        st.title("Дашборд")

        df = clean_dataframe(
            get_dataset_sample()
        )

        if df.empty:

            st.error(
                "Данные не загружены."
            )

        else:

            st.sidebar.header(
                "Общие фильтры"
            )

            types = st.sidebar.multiselect(
                "Тип объявления",
                options=df[
                    "listing_type"
                ].unique(),
                default=df[
                    "listing_type"
                ].unique()
            )

            min_price = float(
                df["price"].min()
            )

            max_price = float(
                df["price"].quantile(0.99)
            )

            global_price_range = st.sidebar.slider(
                "Общий диапазон цены, TRY",
                min_value=min_price,
                max_value=max_price,
                value=(
                    min_price,
                    max_price
                ),
                step=10000.0
            )

            rooms = st.sidebar.multiselect(
                "Количество комнат",
                options=sorted(
                    df["room_count"].dropna().unique()
                ),
                default=sorted(
                    df["room_count"].dropna().unique()
                )
            )

            mask = (
                df["listing_type"].isin(types)
                & df["price"].between(
                    *global_price_range
                )
                & df["room_count"].isin(rooms)
            )

            filtered_df = df[mask]

            k1, k2, k3 = st.columns(3)

            k1.metric(
                "Объектов",
                f"{len(filtered_df):,}"
            )

            k2.metric(
                "Средняя цена",
                (
                    f"{filtered_df['price'].mean():,.0f} ₺"
                    if not filtered_df.empty
                    else "0"
                )
            )

            k3.metric(
                "Средняя площадь",
                (
                    f"{filtered_df['size'].mean():.1f} м²"
                    if not filtered_df.empty
                    else "0"
                )
            )

            st.divider()

            st.subheader(
                "Распределение цен"
            )

            st.write(
                "Показывает, в каких ценовых диапазонах "
                "сосредоточено большинство объявлений."
            )

            histogram_price = st.slider(
                "Диапазон цены для графика",
                min_value=min_price,
                max_value=max_price,
                value=(
                    min_price,
                    max_price
                ),
                step=10000.0,
                key="histogram_price"
            )

            histogram_df = filtered_df[
                filtered_df["price"].between(
                    histogram_price[0],
                    histogram_price[1]
                )
            ]

            fig1 = px.histogram(
                histogram_df,
                x="price",
                color="listing_type",
                nbins=40,
                barmode="overlay",
                template="plotly_white"
            )

            fig1.update_layout(
                xaxis_title="Цена, TRY",
                yaxis_title="Количество объектов",
                legend_title="Тип объявления"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            st.subheader(
                "Цена в зависимости от площади"
            )

            st.write(
                "Показывает, как изменяется стоимость "
                "недвижимости при увеличении площади."
            )

            size_min = float(
                filtered_df["size"].min()
            )

            size_max = float(
                filtered_df["size"].max()
            )

            selected_size = st.slider(
                "Диапазон площади, м²",
                min_value=size_min,
                max_value=size_max,
                value=(
                    size_min,
                    size_max
                ),
                step=1.0,
                key="size_range"
            )

            size_df = filtered_df[
                filtered_df["size"].between(
                    selected_size[0],
                    selected_size[1]
                )
            ]

            fig2 = px.scatter(
                size_df,
                x="size",
                y="price",
                color="room_count",
                opacity=0.55,
                template="plotly_white"
            )

            fig2.update_layout(
                xaxis_title="Площадь, м²",
                yaxis_title="Цена, TRY",
                legend_title="Комнаты"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            c1, c2 = st.columns(2)

            with c1:

                st.subheader(
                    "Цена по типу отопления"
                )

                st.write(
                    "Позволяет сравнить типичную стоимость "
                    "объектов с разными системами отопления."
                )

                heating_options = sorted(
                    filtered_df[
                        "heating_type"
                    ].dropna().unique()
                )

                selected_heating = st.multiselect(
                    "Типы отопления",
                    options=heating_options,
                    default=heating_options,
                    key="heating_filter"
                )

                heating_df = filtered_df[
                    filtered_df[
                        "heating_type"
                    ].isin(selected_heating)
                ]

                heating_price = (
                    heating_df
                    .groupby(
                        "heating_type"
                    )["price"]
                    .median()
                    .sort_values()
                    .reset_index()
                )

                fig3 = px.bar(
                    heating_price,
                    x="price",
                    y="heating_type",
                    orientation="h",
                    template="plotly_white"
                )

                fig3.update_layout(
                    xaxis_title="Медианная цена, TRY",
                    yaxis_title="Тип отопления"
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

            with c2:

                st.subheader(
                    "Цена по количеству комнат"
                )

                st.write(
                    "Показывает, как отличается типичная "
                    "стоимость объектов с разным количеством комнат."
                )

                room_options = sorted(
                    filtered_df[
                        "room_count"
                    ].dropna().unique()
                )

                selected_room_chart = st.multiselect(
                    "Комнаты для графика",
                    options=room_options,
                    default=room_options,
                    key="room_chart_filter"
                )

                room_df = filtered_df[
                    filtered_df[
                        "room_count"
                    ].isin(selected_room_chart)
                ]

                room_price = (
                    room_df
                    .groupby(
                        "room_count"
                    )["price"]
                    .median()
                    .reset_index()
                )

                fig4 = px.bar(
                    room_price,
                    x="room_count",
                    y="price",
                    template="plotly_white"
                )

                fig4.update_layout(
                    xaxis_title="Количество комнат",
                    yaxis_title="Медианная цена, TRY"
                )

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )

            st.subheader(
                "Цена квадратного метра"
            )

            st.write(
                "Показывает стоимость одного квадратного метра "
                "для объектов разной площади."
            )

            sqm_min = float(
                filtered_df["size"].min()
            )

            sqm_max = float(
                filtered_df["size"].max()
            )

            selected_sqm_size = st.slider(
                "Диапазон площади для цены за м²",
                min_value=sqm_min,
                max_value=sqm_max,
                value=(
                    sqm_min,
                    sqm_max
                ),
                step=1.0,
                key="sqm_size_range"
            )

            sqm_df = filtered_df[
                filtered_df["size"].between(
                    selected_sqm_size[0],
                    selected_sqm_size[1]
                )
            ].copy()

            sqm_df["price_per_m2"] = (
                sqm_df["price"]
                / sqm_df["size"]
            )

            sqm_df = sqm_df.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna(
                subset=["price_per_m2"]
            )

            fig5 = px.scatter(
                sqm_df,
                x="size",
                y="price_per_m2",
                color="room_count",
                opacity=0.55,
                template="plotly_white"
            )

            fig5.update_layout(
                xaxis_title="Площадь, м²",
                yaxis_title="Цена за м², TRY",
                legend_title="Комнаты"
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )

    else:

        st.title("Справка")

        st.markdown(
            "### Описание проекта\n"
            "Приложение для прогнозирования цен на недвижимость "
            "с помощью ML моделей."
        )

        st.table(
            pd.DataFrame(
                [
                    ["Площадь", "Площадь в м²"],
                    ["Комнаты", "Формат N+1"],
                    ["Возраст здания", "Лет"],
                    ["Этажность", "Всего этажей"],
                    ["Тип", "Продажа/Аренда"],
                    ["Адрес", "Район"]
                ],
                columns=[
                    "Поле",
                    "Описание"
                ]
            )
        )

        st.markdown(
            "### Информация о системе"
        )

        api_info = get_api_info()

        if api_info:
            st.table(
                pd.DataFrame(
                    list(api_info.items()),
                    columns=[
                        "Параметр",
                        "Значение"
                    ]
                )
            )
        else:
            st.info(
                "API Info недоступно."
            )
