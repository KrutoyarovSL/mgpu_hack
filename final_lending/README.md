# final_lending

Отдельный финальный контур проекта:

- `api.py` — FastAPI-слой над churn / recommendations / forecast.
- `streamlit_app.py` — demo UI.
- `landing/` — отдельная презентационная страница.
- `artifacts/` — место для generated outputs.

## Быстрый запуск одной командой

```bash
bash final_lending/run.sh
```

После старта будут доступны:

- FastAPI docs: `http://127.0.0.1:8000/docs`
- Streamlit demo: `http://127.0.0.1:8501`

## Ручной запуск

```bash
uvicorn final_lending.api:app --reload
```

Во втором терминале:

```bash
streamlit run final_lending/streamlit_app.py
```

## Переменные окружения

- `FINAL_LENDING_DATA_PATH`
- `FINAL_LENDING_EVENTS_PATH`
- `FINAL_LENDING_API_URL`

Если переменные не заданы, используются:

- `m_pred/archive/data.csv`
- `m_pred/archive/events.csv`
