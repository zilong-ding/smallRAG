import json

# 假设数据在文件 data.jsonl 中
counts = {0: 0, 1: 0, 2: 0}
with open("./Gift_Cards.jsonl") as f:
    for line in f:
        item = json.loads(line)
        r = item["rating"]
        if r <= 2.0:
            counts[0] += 1
        elif r == 3.0:
            counts[1] += 1
        else:
            counts[2] += 1

total = sum(counts.values())
print("Class distribution:")
for cls, cnt in counts.items():
    print(f"  Class {cls}: {cnt} ({cnt/total:.1%})")