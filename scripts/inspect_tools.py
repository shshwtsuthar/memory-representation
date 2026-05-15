import json
import sys
from collections import Counter

tools = Counter()

for line in sys.stdin:
    if not line.strip():
        continue

    row = json.loads(line)

    for msg in row["trajectory"]:
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tools[fn.get("name", "unknown")] += 1

        elif msg["role"] == "tool":
            tools["tool_msg:" + str(msg.get("name", "unknown"))] += 1

print(tools.most_common(50))
