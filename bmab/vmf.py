"""Hammer map sources (Source .vmf, GoldSrc .map): the labels designers left behind.

Visgroup names, func_instance files and the names of named entities, with counts. The keyvalue
format is read as a flat stream: a block name followed by `{`, or a `"key" "value"` pair.
"""
import re
from collections import Counter

_TOKEN = re.compile(r'"([^"]*)"\s*"([^"]*)"|([A-Za-z_]\w*)\s*\{|(\})')


def labels(text):
    visgroups, instances, names, comments = Counter(), Counter(), Counter(), []
    stack, current = [], {}
    for key, value, block, close in _TOKEN.findall(text):
        if block:
            stack.append((block, current))
            current = {}
        elif close:
            if not stack:
                continue
            block_name, parent = stack.pop()
            if block_name == "visgroup" and current.get("name"):
                visgroups[current["name"]] += 1
            elif block_name == "entity":
                cls = current.get("classname", "")
                if cls == "func_instance" and current.get("file"):
                    instances[current["file"].replace("\\", "/").lower()] += 1
                if current.get("targetname"):
                    names[current["targetname"]] += 1
                if cls == "info_overlay" and current.get("material"):
                    names["overlay:" + current["material"].lower()] += 1
            current = parent
        else:
            current[key] = value
            if key == "comments" and value:
                comments.append(value)
    return {"visgroups": visgroups, "instances": instances, "names": names, "comments": comments}
