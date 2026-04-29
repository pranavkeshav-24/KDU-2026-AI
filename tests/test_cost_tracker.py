from src.cost.cost_tracker import CostTracker


def test_gpt_4o_mini_cost_formula():
    cost = CostTracker().calculate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 0.75


def test_embedding_cost_formula():
    cost = CostTracker().calculate_cost("text-embedding-3-small", input_tokens=1_000_000)
    assert cost == 0.02

