import json
import os
import re
import time
from datetime import datetime, timezone
from openai import OpenAI
from tools import calculator, text_search

SYSTEM_PROMPT = """Ты ассистент, который решает задачи пошагово в формате ReAct.
Доступные инструменты:
{tools}

Строго соблюдай формат. На каждом шаге пиши:
Thought: рассуждение о том, что делать дальше
Action: имя_инструмента
Action Input: входные данные инструмента одной строкой

После этого ОСТАНОВИСЬ и жди строку Observation c результатом.
Когда информации достаточно, напиши:
Thought: итоговое рассуждение
Final Answer: ответ пользователю

Правила:
1. Если вопрос содержит ТОЛЬКО математическое выражение, сразу вызывай calculator.
2. Факты о компании бери из text_search.
3. ВНИМАНИЕ К ЛИМИТАМ: Если в правилах есть ограничение (например, "не более 20000 сом"), а полученная через калькулятор сумма превышает этот лимит, итоговым ответом должен быть именно лимит (20000 сом).
4. Все арифметические вычисления делай через calculator.
5. Не придумывай Observation сам."""

ACTION_RE = re.compile(r"Action:\s*([\w\-]+)\s*\n\s*Action Input:\s*(.+)")
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.S)

def parse(text: str):
    text = text.split("Observation:")[0]
    m_act, m_fin = ACTION_RE.search(text), FINAL_RE.search(text)
    if m_fin and (not m_act or m_fin.start() < m_act.start()):
        return ("final", m_fin.group(1).strip())
    if m_act:
        return ("action", m_act.group(1).strip(), m_act.group(2).strip())
    return ("error", "Нет 'Action:' + 'Action Input:' и нет 'Final Answer:'. Соблюдай формат.")

client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    api_key=os.getenv("GROQ_API_KEY", "")
)
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
TOOLS = {"calculator": calculator, "text_search": text_search}

def log_event(run_id, step, event, llm_output="", action="", action_input="", observation="", tool_ok=True, latency=0.0, usage=None):
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "step": step,
        "event": event,
        "llm_output": llm_output,
        "action": action,
        "action_input": action_input,
        "observation": observation,
        "tool_ok": tool_ok,
        "latency_s": round(latency, 2),
        "usage": usage or {}
    }
    with open("agent_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def run_agent(question: str, max_steps: int = 8) -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    tools_desc = "- calculator: вычисляет арифметическое выражение\n- text_search: ищет фрагменты во внутренних документах"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_desc)},
        {"role": "user", "content": f"Question: {question}"}
    ]

    log_event(run_id, 0, "start", observation=f"Question: {question}")

    for step in range(1, max_steps + 1):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=None
        )
        latency = time.perf_counter() - t0
        text = (resp.choices[0].message.content or "").strip()
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            "total_tokens": resp.usage.total_tokens if resp.usage else 0,
        }

        parsed = parse(text)

        if parsed[0] == "final":
            log_event(run_id, step, "final", llm_output=text, observation=parsed[1], latency=latency, usage=usage)
            return parsed[1]

        elif parsed[0] == "action":
            _, name, arg = parsed
            tool_fn = TOOLS.get(name)
            if tool_fn:
                obs = tool_fn(arg)
                tool_ok = not obs.startswith("Ошибка:")
            else:
                obs = f"Ошибка: неизвестный инструмент '{name}'. Доступны: {list(TOOLS.keys())}"
                tool_ok = False

            log_event(run_id, step, "step", llm_output=text, action=name, action_input=arg, observation=obs, tool_ok=tool_ok, latency=latency, usage=usage)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

        elif parsed[0] == "error":
            obs = f"Ошибка формата: {parsed[1]}"
            log_event(run_id, step, "step", llm_output=text, observation=obs, tool_ok=False, latency=latency, usage=usage)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})

    log_event(run_id, max_steps, "max_steps_reached")
    return "Не удалось получить ответ за отведённое число шагов."
