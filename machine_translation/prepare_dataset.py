import os
import json


base_dir = "./data"

for split in ["train", "valid", "test"]:
    en = open(os.path.join(base_dir, split + ".en"), "r", encoding="utf-8").readlines()
    zh = open(os.path.join(base_dir, split + ".zh"), "r", encoding="utf-8").readlines()
    assert len(en) == len(
        zh
    ), "The number of English sentences and Chinese sentences are not equal."
    data = []
    for e, z in zip(en, zh):
        data.append({"translation": {"en": e.strip(), "zh": z.strip()}})
    json.dump(
        data,
        open(os.path.join(base_dir, split + ".json"), "w", encoding="utf-8"),
        ensure_ascii=False,
    )
