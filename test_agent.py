"""Offline test: runs the full agent graph with a fake LLM and fake search tool.
Run with:  python test_agent.py"""
from agent import ResearchAgent, build_final_report
from export import report_to_pdf


class FakeLLM:
    def __init__(self):
        self.review_calls = 0

    def invoke(self, prompt: str):
        if "research planner" in prompt:
            out = '["What are EV adoption trends?", "How do EVs affect NO2 levels?"]'
        elif "Extract the key facts" in prompt:
            out = "- EV sales grew 35% in 2023\n- Cities saw lower NO2 near roads"
        elif "research writer" in prompt:
            out = ("# EVs and Urban Air\n## Introduction\nEVs are growing [1].\n"
                   "## Key Findings\n### Adoption\nSales rose 35% [1][2].\n"
                   "## Analysis\nNO2 fell [3].\n## Conclusion\nPositive impact [2].")
        elif "strict research reviewer" in prompt:
            self.review_calls += 1
            out = ('{"approved": false, "feedback": "Add data on PM2.5", '
                   '"missing_queries": ["EV PM2.5 tyre wear"]}'
                   if self.review_calls == 1 else '{"approved": true, "feedback": "", "missing_queries": []}')
        else:
            out = ""
        return type("R", (), {"content": out})()


def fake_search(q):
    slug = q.replace(" ", "-")[:20]
    return [{"title": f"Article on {q}", "url": f"https://example.com/{slug}/{i}",
             "content": f"Some content about {q}. " * 5} for i in range(2)]


agent = ResearchAgent(llm=FakeLLM(), search_fn=fake_search, fetch_fn=lambda u: "")
state = {}
for node, update in agent.stream("Impact of electric vehicles on urban air quality"):
    print(f"--- {node}")
    for line in update.get("log", []):
        print(line)
    state.update({k: v for k, v in update.items() if k != "log"})

report = build_final_report(state["draft"], state["sources"])
print("\n" + report)
assert "## References" in report and len(state["sources"]) == 6
pdf = report_to_pdf(report)
assert pdf[:4] == b"%PDF"
print(f"PDF OK ({len(pdf)} bytes). All tests passed.")
