import pandas as pd
from agent import run_agent

questions = [
    "Сколько дней отпуска положено сотруднику?",
    "Какой размер компенсации обучения в компании?",
    "Посчитай 1500 * 4",
    "Сколько суточных получит сотрудник за командировку длительностью 4 дня?",
    "Если сотрудник пошел на курс за 25000 сом, сколько ему компенсирует компания?"
]

print("=== Запуск тестов ===")
for q in questions:
    print(f"\nВопрос: {q}")
    ans = run_agent(q)
    print(f"Ответ: {ans}")

print("\n=== Метрики из agent_log.jsonl ===")
df = pd.read_json("agent_log.jsonl", lines=True)

steps_df = df[df.event == "step"]
final_df = df[df.event == "final"]

print("Всего запусков:", df[df.event == "start"]["run_id"].nunique())
print("Среднее число шагов:", steps_df.groupby("run_id").size().mean())
print("Средняя задержка LLM (сек):", round(df["latency_s"].mean(), 2))
print("Доля успешных вызовов инструментов (tool_ok):", f"{steps_df['tool_ok'].mean() * 100:.1f}%")
