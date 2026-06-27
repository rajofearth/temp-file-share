import json
with open("/tmp/modal_billing.json") as f:
    data = json.load(f)
curr = [x for x in data if x["object_id"] == "ap-4FUgbdLe9uast2t4qe0MWi"]
total = sum(float(x["cost"]) for x in curr)
print(f"Current run total: ${total:.3f}")
hours = [x["interval_start"] for x in curr]
print(f"Intervals: {hours}")
all_total = sum(float(x["cost"]) for x in data)
print(f"All nanochat-sft today: ${all_total:.3f}")
