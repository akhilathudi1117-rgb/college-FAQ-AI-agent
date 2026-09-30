"""Test the agent across categories.  Run:  python test_agent.py"""
from rag_agent import CollegeFAQAgent, NOT_FOUND

agent = CollegeFAQAgent()

# (category, question, text that MUST appear, or None if refusal expected)
TESTS = [
    ("Normal",        "What time does the college start?",      "8:00"),
    ("Normal",        "When does the library close?",           "6:00"),
    ("Knowledge",     "How many books can I borrow?",           "3"),
    ("Knowledge",     "What is the minimum attendance needed?", "75"),
    ("Unknown",       "Who is the prime minister of Canada?",   None),
    ("Hallucination", "Who won the 2030 World Cup?",            None),
    ("Hallucination", "What is the salary of the principal?",   None),
]

passed = 0
for cat, q, must in TESTS:
    answer, _, trace = agent.ask(q)
    refused = "don't have that information" in answer.lower()
    ok = refused if must is None else (must in answer and not refused)
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {cat:13} | {q}\n         -> {answer}\n")
print(f"{passed}/{len(TESTS)} tests passed")
