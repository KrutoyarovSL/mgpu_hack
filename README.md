# Customer Experience Platform

> Комплексное аналитическое решение для прогнозирования оттока клиентов, персонализированных рекомендаций и бизнес-аналитики розничной торговли.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-FFCC00?logo=catboost&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## О проекте

Проект разработан в рамках **Московский студенческий DATA_Хакатона задача: Ростелеком** командой **«мисис сладкий подарок»**.

Решение реализует **замкнутый цикл управления клиентским опытом** — от диагностики проблем через бизнес-аналитику и прогнозирование оттока, до проактивного воздействия через персонализированные рекомендации.

### Что внутри

- **Разведочный анализ и feature engineering** — RFM-сегментация, ABC/XYZ-анализ, поведенческие и NLP-признаки
- **Прогнозирование оттока** — CatBoost с временными снэпшотами и sentiment-фичами (ROC-AUC = 0.81)
- **Hybrid ALS рекомендательная система** — Collective Matrix Factorization с side features (HitRate@10 = 0.45)
- **Прогнозирование продаж** — Ridge / Decision Tree / Random Forest с временными лагами
- **Production-ready инфраструктура** — FastAPI + Streamlit + Docker Compose

---

## Задача

> Разработать комплексное аналитическое приложение, объединяющее прогнозную модель оттока клиентов, рекомендательную систему для формирования персонализированных предложений и систему интерактивных дашбордов.

**Бизнес-контекст:** борьба с оттоком и удержание клиентов — ключевые задачи любого бизнеса. Решение должно обеспечить переход от реактивного управления к проактивному удержанию аудитории.

**Исходные данные:** синтетический e-commerce датасет
- `data.csv` — **545 778** строк, **49** полей (заказы, товары, логистика, отзывы)
- `events.csv` — **4 865 324** строк, **13** полей (поведенческие события)

---

## Архитектура решения

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│   data.csv (545K orders)    events.csv (4.8M events)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                            │
│   • RFM / ABC / XYZ сегментация                                 │
│   • Поведенческие признаки (30/90/180-дневные окна)             │
│   • NLP-фичи из отзывов (sentiment)                             │
│   • Product quality risk score                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────────┐
        │  Churn   │   │  Hybrid  │   │    Sales     │
        │ CatBoost │   │   ALS    │   │  Forecast    │
        │          │   │ (cmfrec) │   │  (Ridge/RF)  │
        └──────────┘   └──────────┘   └──────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING LAYER                                │
│   FastAPI REST API  ◀─────▶  Streamlit Demo UI                  │
│   • /predict_churn/{id}      • Churn инференс                   │
│   • /recommend/{id}          • Рекомендации                     │
│   • /forecast_sales          • Прогноз продаж                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ключевые результаты

### Churn Prediction (CatBoost)

Задача сформулирована как **prediction of future inactivity** — предсказание того, что клиент не сделает заказ в горизонте 120 дней. Модель обучалась на временных снэпшотах с историческими окнами поведения.

| Метрика | Значение |
|---|---|
| **ROC-AUC** | **0.8109** |
| **PR-AUC** | **0.9238** |
| **Top 10% risk precision** | **0.995** |

**Ключевые факторы оттока** (по feature importance):
1. `is_loyal` — 30.09
2. `favorite_brand` — 8.26
3. `state` — 7.71
4. `favorite_category` — 5.15
5. `tenure_days_in_window` — 2.54

> **Бизнес-вывод:** риск оттока определяется не только историей покупок, но и сочетанием лояльности, товарных предпочтений, вовлечённости, логистического опыта и клиентского sentiment.

### Hybrid ALS Recommender (cmfrec)

Использована **Collective Matrix Factorization** на implicit feedback с гибридными признаками пользователей и товаров. Учитывает quality risk товара — клиентам с высоким риском оттока рекомендуются товары с низкой долей возвратов.

Изначально было создано бейзлайн-решение: при оттоке каждому пользователю предлагали самые популярные товары в их любимых категориях (любимые категории высчитывались явно). 

Финальное решение: были собраны описания товаров, пользователей и матрица взаимодействий, где взаимодействие считалось как взвешенный составной признак.

`user_item` - матрица взаимодействий:

- UserId
- ItemId
- Value (times_bought, total_spend, last_seen


`item_info` — матрица товаров:

- количество продаж товара;
число уникальных покупателей;
- retail price;
- return rate;
- cancel rate;
- quality risk;
- категория;
- бренд
- и другие.

`user_info` - матрица пользователей:

- количество купленных товаров;
- общие траты;
- средняя цена покупки;
- return rate;
- cancel rate;
- completion rate;
- возраст;
- количество заказов;
- средний чек;
- частота покупок;
- пол;
- страна;
- и другие.


| Метрика @10 | Baseline | **Final (Hybrid ALS)** | Прирост |
|---|---|---|---|
| **HitRate** | 0.094 | **0.4465** | **×4.75** |
| **Recall** | 0.189 | **0.3640** | ×1.93 |
| **MRR** | 0.094 | **0.1747** | ×1.86 |

**Конфигурация модели:**
- `k=48` латентных факторов;
- `lambda=3.0`, `alpha=8.0`, 15 итераций;
- 415K train / 130K test взаимодействий;
- 80K пользователей × 28K товаров с side features (41 признак для пользователей, 66 признак для айтемов);
- Production: 50K рекомендаций для 5K юзеров с высоким риском оттока.

### Sales Forecasting

Прогнозирование объёмов продаж на основе временных лагов и rolling-статистик. Сравнивались Linear Regression / Decision Tree / Random Forest на 5-fold time-series CV.

| Fold | Linear MAE |
|---|---|
| 1 | 111.88 |
| 2 | 72.65 |
| 3 | 78.64 |
| 4 | 80.30 |

Линейная регрессия с лагами и `rolling_mean_6` показала лучший результат — модель аккуратно следует за восходящим трендом продаж.

### Кластеризация клиентов

Выделено **5 поведенческих групп** клиентов по покупательской активности, выручке, возвратам, логистике и review-признакам:
- **VIP / High Value** — удерживать любой ценой
- **Core Customers** — стандартная коммуникация
- **Light Customers** — стимулировать частоту
- **At Risk Dormant** — реактивировать
- **Return Sensitive** — снижать вероятность повторного негативного опыта

Дополнительно построены 6 **RFM-сегментов**:

| Сегмент | Пользователей |
|---|---|
| At Risk | 16 812 |
| Hibernating | 15 197 |
| Regular | 14 509 |
| Loyal Customers | 12 499 |
| Potential Loyalists | 10 625 |
| Champions | 10 379 |

---

## Технологический стек

| Слой | Технологии |
|---|---|
| **Языки** | Python 3.12 |
| **Анализ данных** | pandas, numpy, scipy |
| **ML** | scikit-learn, CatBoost, cmfrec |
| **Визуализация** | matplotlib, seaborn |
| **API** | FastAPI, uvicorn, pydantic |
| **UI** | Streamlit |
| **Инфраструктура** | Docker, docker-compose |
| **Landing** | HTML, CSS (vanilla) |

---

## Структура проекта

```
.
├── analytics_solution.py          # Core: EDA, фичи, baseline churn, рекомендации
├── hybrid_als_recommender.py      # Hybrid ALS на cmfrec.CMF_implicit
├── evaluate_recommendations.py    # Оценка рекомендаций (HitRate, MRR, Recall)
│
├── final_lending/                 # Production-обвязка
│   ├── api.py                     # FastAPI приложение
│   ├── service.py                 # Сервисный слой + кеширование артефактов
│   ├── churn_model.py             # CatBoost churn модель (temporal snapshots)
│   ├── streamlit_app.py           # Demo UI
│   ├── Dockerfile.api
│   ├── Dockerfile.streamlit
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── run.sh                     # Запуск всего стека одной командой
│   └── landing/                   # Презентационная страница
│       ├── index.html
│       └── styles.css
│
├── notebooks/
│   ├── gaz.ipynb                  # Основной EDA (12 секций по task.md)
│   ├── data_preprocess.ipynb      # Очистка данных
│   ├── events_preprocess.ipynb    # Очистка событий
│   └── mgpu-67.ipynb              # Discovery / model development
│
├── artifacts/
│   └── hybrid_als/                # Артефакты обученной ALS-модели
│       ├── cmf_implicit_model.pkl
│       ├── hybrid_als_metrics.json
│       ├── hybrid_als_recommendations.csv
│       ├── user_info.csv
│       ├── item_info.csv
│       └── user_item.csv
│
├── docs/
│   └── images/                    # Графики и скриншоты
│
├── task.md                        # Условие хакатона
└── README.md
```

---

## Быстрый старт

### Вариант 1: Docker Compose (рекомендуется)

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Положите данные в m_pred/archive/
#   m_pred/archive/data.csv
#   m_pred/archive/events.csv

bash final_lending/run.sh
```

После запуска будут доступны:
- **FastAPI docs:** http://127.0.0.1:8000/docs
- **Streamlit demo:** http://127.0.0.1:8501

### Вариант 2: Ручной запуск

```bash
pip install -r final_lending/requirements.txt

# Терминал 1: API
uvicorn final_lending.api:app --reload

# Терминал 2: UI
streamlit run final_lending/streamlit_app.py
```

### Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `FINAL_LENDING_DATA_PATH` | `m_pred/archive/data.csv` | Путь к датасету заказов |
| `FINAL_LENDING_EVENTS_PATH` | `m_pred/archive/events.csv` | Путь к датасету событий |
| `FINAL_LENDING_API_URL` | `http://127.0.0.1:8000` | URL API для Streamlit |

---

## API эндпоинты

| Метод | Эндпоинт | Описание |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/ready` | Готовность (загружены ли артефакты) |
| `GET` | `/summary` | Сводная статистика по платформе |
| `GET` | `/users?limit=N` | Список доступных user_id |
| `GET` | `/predict_churn/{user_id}` | Churn probability + risk group + features |
| `GET` | `/recommend/{user_id}?top_n=5` | Топ-N персональных рекомендаций |
| `GET` | `/forecast_sales` | Прогноз выручки на 3 месяца |

### Пример ответа `/predict_churn/{user_id}`

```json
{
  "user_id": 12345,
  "churn_probability": 0.873,
  "risk_group": "high",
  "customer_features": {
    "orders": 4,
    "total_spend": 287.50,
    "return_rate": 0.25,
    "recency_days": 142,
    "customer_segment": "service_risk_543",
    "abc_segment": "B",
    "xyz_segment": "Z"
  }
}
```

### Пример ответа `/recommend/{user_id}`

```json
{
  "user_id": 12345,
  "recommendations": [
    {
      "rank": 1,
      "recommended_product_id": 8721,
      "product_name": "The North Face Denali Down Mens Jacket",
      "category": "Outerwear & Coats",
      "brand": "The North Face",
      "score": 4.21,
      "item_quality_risk": 0.08,
      "source": "hybrid_als"
    }
  ]
}
```

---

## Детали реализации

### Churn Model

Реализована в `final_lending/churn_model.py`:

- **Temporal snapshots**: 12 месячных срезов, для каждого вычисляются признаки в окне 180 дней назад и таргет — отсутствие заказов в окне 120 дней вперёд
- **Multi-window features**: агрегаты на 30/90/180 дней (orders, revenue, return rate)
- **NLP-фичи из отзывов**: positive/negative word count, sentiment score
- **Логистические фичи**: shipping_delay, delivery_days, completion_rate
- **Event-based**: cnt по типам событий, дни с последнего события, сессии
- **Модель**: `CatBoostClassifier(iterations=300, depth=6, lr=0.05)` с категориальными фичами

### Hybrid ALS Recommender

Реализован в `hybrid_als_recommender.py` на библиотеке `cmfrec`:

- **Interaction value**: взвешенная комбинация `times_bought`, `total_spend`, `completed` минус `quality_penalty` за возвраты и отмены
- **User side features (41)**: спенд, частота, recency, RFM, traffic, demographics, one-hot топ-25 локаций
- **Item side features (66)**: цена, маржа, return rate, quality risk, one-hot топ-25 категорий/брендов
- **Temporal split**: последний заказ каждого пользователя — в test
- **Exclude seen items** при генерации рекомендаций
- **Fallback strategy**: prebuilt CSV → ALS inference → business rules

### Service Layer

`final_lending/service.py` реализует production-патерны:

- **Кеширование артефактов** через `@lru_cache` и файловые snapshots
- **Multi-tier fallback** для рекомендаций (prebuilt → model → rules)
- **JSON-сериализация** numpy/pandas типов
- **Environment-based configuration**

---

## Заключение

### Чем проект ценен

Customer Experience Platform — это не просто набор ML-моделей, а **полноценный замкнутый цикл управления клиентским опытом**: от диагностики через сегментацию и quality-анализ, через прогнозирование оттока и продаж, до проактивного воздействия через персонализированные рекомендации с учётом риска и качества товаров.

Решение покрывает все четыре уровня задачи и связывает их в единый pipeline: high-risk клиент → персональные рекомендации → метрики эффективности. В отличие от изолированных подходов (только churn или только рекомендер), интеграция churn-score в логику рекомендаций даёт бизнесу инструмент **точечного удержания** вместо массовых рассылок.

С инженерной стороны проект готов к production: упакован в Docker, имеет REST API с документацией, multi-tier fallback для отказоустойчивости и кеширование тяжёлых артефактов.

### Что можно улучшить

- **A/B тестирование** рекомендаций на реальных пользователях для подтверждения uplift в удержании
- **Online learning** для churn-модели — переобучение на свежих снэпшотах без полного перезапуска
- **MLOps-обвязка**: model registry (MLflow), мониторинг drift'а, алерты на деградацию метрик
- **Real-time inference** через очереди (Kafka/RabbitMQ) для интеграции с реальными e-commerce событиями
- **BI-дашборды** в Superset / Apache Superset / DataLens поверх артефактов для бизнес-пользователей
- **Глубокие модели** для рекомендера — SASRec, BERT4Rec — там, где важна последовательность действий
- **Расширение фичей**: текстовый анализ отзывов через трансформеры вместо bag-of-words sentiment

---

## Команда «мисис сладкий подарок»

| Имя | Telegram | Роль |
|---|---|---|
| Крутояров Вячеслав | [@krutoyarovsl](https://t.me/krutoyarovsl) | ML |
| Усенко Тимофей | [@lmetacortexl](https://t.me/lmetacortexl) | ML |
| Пикель Герман | [@germanpikel](https://t.me/germanpikel) | ML |
| Шубин Вадим | [@cvbnqq](https://t.me/cvbnqq) | ML |

---

[Демонстрация работы приложения](https://drive.google.com/file/d/1u3fp18TXXNMdekKCvwvFeEHrWIQW0_Vv/view?usp=sharing)
[Дашборды, метрики, инсайды их данных, ключевые графики](https://drive.google.com/file/d/1v2_On4m2mDGGCtug-5Ka8JrguhlhIirG/view?usp=sharing)

## 📜 Лицензия

MIT 

---

