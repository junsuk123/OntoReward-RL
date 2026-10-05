"""Strict JSON for research artifacts: an unavailable metric is null, not NaN."""
from __future__ import annotations

import json
import math


def json_text(value, *, indent=None):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key,value in item.items()}
        if isinstance(item, (list,tuple)):
            return [clean(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    return json.dumps(clean(value), indent=indent, allow_nan=False)
