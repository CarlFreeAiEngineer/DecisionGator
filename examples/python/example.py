#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""DecisionGator from Python. Run from the repository: uv run examples/python/example.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # this checkout's decisiongator package
from decisiongator import choose_p, is_yes, is_yes_p

ticket = "Our whole warehouse can't print shipping labels and trucks leave in an hour."
question = "Is the customer describing an urgent problem?"

# A yes/no decision at the usual 0.5 cutoff.
print("urgent:", str(is_yes(ticket, question)).lower())

# The probability, so you can keep an uncertain range for a person.
print(f"p_yes = {is_yes_p(ticket, question):.3f}")

# Criteria and a stricter threshold.
criteria = {"yes": "Work is blocked and there is a deadline within hours.",
            "no": "The problem is an inconvenience with no near deadline."}
print("confident urgent:", str(is_yes(ticket, question, criteria, threshold=0.90)).lower())

# Several options instead of yes or no.
teams = ["billing", "technical support", "sales"]
for index, p in choose_p("My card was charged twice for last month's invoice.",
                         "Which team should handle this message?", teams):
    print(f"{teams[index]:<18} {p:.3f}")
