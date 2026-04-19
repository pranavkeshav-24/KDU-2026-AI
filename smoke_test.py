import requests
import json

queries = [
    ("What are your operating hours?", "sess-001"),
    ("I need to reschedule my appointment to next week", "sess-002"),
    ("My plumber never showed up and I want a refund NOW", "sess-003"),
    ("My AC has error code E5, what should I do?", "sess-004"),
]

print("=== QUERY ROUTING TEST ===\n")
for q, sid in queries:
    r = requests.post("http://localhost:8000/query", json={"text": q, "session_id": sid})
    d = r.json()
    cat = d["category"].upper().ljust(10)
    comp = d["complexity"].upper().ljust(6)
    tier = d["tier_used"].ljust(12)
    print(f"[{cat}] [{comp}] -> {tier} | {q[:55]}")
    print(f"  Response: {d['response_text'][:100]}")
    print()

print("\n=== BUDGET STATUS ===")
r2 = requests.get("http://localhost:8000/budget/status")
print(json.dumps(r2.json(), indent=2))

print("\n=== PROMPTS REGISTRY ===")
r3 = requests.get("http://localhost:8000/admin/prompts")
d3 = r3.json()
for cat, versions in d3["categories"].items():
    for v in versions:
        print(f"  {cat}/{v['version']} [{v['status']}] eval={v['eval_score']}")

print("\n=== CONFIG ACTIVE ===")
r4 = requests.get("http://localhost:8000/admin/config")
d4 = r4.json()
print(f"  version={d4['version']} env={d4['environment']} dry_run={d4['features']['dry_run_mode']}")
for tier, m in d4["models"].items():
    print(f"  {tier}: {m['model_id']} (provider={m['provider']})")
