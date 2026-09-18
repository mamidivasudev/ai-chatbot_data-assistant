"""
Summaries of dashboard charts, whole dashboards and documents.

Charts and dashboards
    Every number is computed here: totals, shares, condition groups, trends,
    two-series gaps and cross-chart ratios. The model only turns those facts
    into sentences, and each number it writes is checked against them. A reply
    with numbers the facts don't contain is retried once; sentences that still
    carry one are dropped. When the model is unreachable or its reply unusable,
    the summary is built from the facts by rule. Either way a chart summary
    never states a figure the data doesn't support.

Documents
    The whole text is read, not just the top RAG chunks /ask-your-query sees:
    in one call when it fits the model's window, otherwise map-reduce (notes per
    part, merged until they fit, then summarized). Numbers in the summary that
    don't occur in the source are reported back as unverified_numbers.
"""

import bisect
import json
import logging
import math
import re
from typing import Any, AsyncIterator, Optional

from llm_provider import LLMError

logger = logging.getLogger("summarizer")

MAX_CHART_ROWS = 5000
MAX_DASHBOARD_CHARTS = 30
MAX_LISTED_ITEMS = 15        # categories listed one by one; the rest are summed as "others"
MAX_LISTED_PERIODS = 36      # trend points listed one by one; longer series keep only the stats
MAX_DOCUMENT_CHARS = 400_000  # about 150 dense pages; longer documents are summarized up to here
MAX_MERGE_ROUNDS = 3


# ─────────────────────────────────────────────
# Numbers
# ─────────────────────────────────────────────
_NUMBER = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(\s*(?:thousand|lakhs?|crores?|million|billion|mn|bn|k)\b)?",
    re.IGNORECASE,
)
_SCALE = {
    "thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7,
    "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9,
}


def _numbers_in(text: str) -> list[tuple[str, float, float]]:
    """(as written, value, tolerance) for every number in text.

    The tolerance allows honest rounding: half a unit of the last digit written
    (22,325 for 22,324.72; 50.4 for 50.39), and for round figures like 22,000
    half the rounded place, capped at 5% of the value.
    """
    found = []
    for match in _NUMBER.finditer(text):
        written = match.group(1).rstrip(",")
        digits = written.replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        if "." in digits:
            tolerance = 0.5 * 10 ** -len(digits.split(".")[1])
        else:
            trailing_zeros = len(digits) - len(digits.rstrip("0"))
            tolerance = 0.5
            if value >= 1000 and trailing_zeros:
                tolerance = min(0.5 * 10 ** trailing_zeros, 0.05 * value)
        scale = _SCALE.get((match.group(2) or "").strip().lower(), 1.0)
        found.append((match.group(0).strip(), value * scale, tolerance * scale))
    return found


def _allowed_values(text: str) -> list[float]:
    return sorted({value for _, value, _ in _numbers_in(text)})


def _unsupported_numbers(text: str, allowed: list[float]) -> list[str]:
    """Numbers written in text that no value in `allowed` (sorted) rounds to."""
    unsupported = []
    for written, value, tolerance in _numbers_in(text):
        # Small whole numbers are counts and ranks ("top 3", "2 categories").
        if value <= 12 and value.is_integer():
            continue
        i = bisect.bisect_left(allowed, value - tolerance - 1e-9)
        if i < len(allowed) and allowed[i] <= value + tolerance + 1e-9:
            continue
        if written not in unsupported:
            unsupported.append(written)
    return unsupported


def _to_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.replace(",", "").strip().rstrip("%"))
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _fmt(value: float, signed: bool = False) -> str:
    text = f"{value:+,.2f}" if signed else f"{value:,.2f}"
    return text.rstrip("0").rstrip(".")


# ─────────────────────────────────────────────
# Chart facts
# ─────────────────────────────────────────────
# amCharts rows often carry styling and ids next to the values.
_NOT_DATA_FIELD = re.compile(r"^_|^id$|_id$|colou?r|fill|stroke|settings|icon", re.IGNORECASE)
_PERIOD_LABEL = re.compile(
    r"^((19|20)\d{2}\b|(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|q[1-4]\b|w(ee)?k)",
    re.IGNORECASE,
)
_TREND_CHART_TYPES = {"line", "area", "spline", "trend", "timeseries", "time-series"}
_GOOD_LABEL = re.compile(r"\b(good|excellent)\b", re.IGNORECASE)
_POOR_LABEL = re.compile(r"\b(poor|bad|failed|critical)\b", re.IGNORECASE)


def _resolve_fields(rows: list[dict], category_field: Optional[str],
                    value_fields: Optional[list[str]]) -> tuple[str, list[str]]:
    """The label field and the numeric fields, inferred when the caller doesn't name them."""
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in keys and not isinstance(value, (dict, list)) and not _NOT_DATA_FIELD.search(str(key)):
                keys.append(key)

    def is_numeric(key: str) -> bool:
        present = [row[key] for row in rows if row.get(key) not in (None, "")]
        return bool(present) and all(_to_number(v) is not None for v in present)

    if not category_field:
        category_field = next((k for k in keys if not is_numeric(k)), keys[0] if keys else None)
    if not value_fields:
        value_fields = [k for k in keys if k != category_field and is_numeric(k)]
    if not category_field or not value_fields:
        raise ValueError(
            "Chart data needs a label field and at least one numeric field; "
            "pass category_field and value_fields."
        )
    return category_field, value_fields


def _condition_groups(labels: list[str]) -> dict[str, list[str]]:
    """'Good or better' and 'Poor or worse' for condition scales (roughness, PCI, bridge condition)."""
    groups = {}
    good = [label for label in labels if _GOOD_LABEL.search(label)]
    poor = [label for label in labels if _POOR_LABEL.search(label)]
    if len(good) >= 2:
        groups["Good or better"] = good
    if len(poor) >= 2:
        groups["Poor or worse"] = poor
    return groups


def _category_series(name: str, points: list[tuple[str, float]],
                     groups: Optional[dict[str, list[str]]]) -> dict:
    total = sum(value for _, value in points)
    # Shares only mean something when the values are parts of one positive whole.
    has_shares = total > 0 and all(value >= 0 for _, value in points)

    def item(label: str, value: float) -> dict:
        entry = {"label": label, "value": round(value, 2)}
        if has_shares:
            entry["share_pct"] = round(value / total * 100, 1)
        return entry

    ranked = sorted(points, key=lambda p: p[1], reverse=True)
    series = {
        "name": name,
        "count": len(points),
        "total": round(total, 2),
        "items": [item(*p) for p in ranked[:MAX_LISTED_ITEMS]],
        "largest": item(*ranked[0]),
        "smallest": item(*ranked[-1]),
    }
    rest = ranked[MAX_LISTED_ITEMS:]
    if rest:
        series["others"] = {"count": len(rest), **item("others", sum(v for _, v in rest))}

    values = dict(points)
    for group_name, members in (groups or _condition_groups([label for label, _ in points])).items():
        present = [m for m in members if m in values]
        if present:
            series.setdefault("groups", []).append(
                {**item(group_name, sum(values[m] for m in present)), "members": present}
            )
    return series


def _trend_series(name: str, points: list[tuple[str, float]]) -> dict:
    (first_label, first), (last_label, last) = points[0], points[-1]
    high = max(points, key=lambda p: p[1])
    low = min(points, key=lambda p: p[1])
    series = {
        "name": name,
        "count": len(points),
        "first": {"label": first_label, "value": round(first, 2)},
        "last": {"label": last_label, "value": round(last, 2)},
        "change": round(last - first, 2),
        "highest": {"label": high[0], "value": round(high[1], 2)},
        "lowest": {"label": low[0], "value": round(low[1], 2)},
        "average": round(sum(v for _, v in points) / len(points), 2),
    }
    if first:
        series["change_pct"] = round((last - first) / abs(first) * 100, 1)
    if len(points) <= MAX_LISTED_PERIODS:
        series["items"] = [{"label": label, "value": round(value, 2)} for label, value in points]
    return series


def _compare(first_name: str, first: dict[str, float], second_name: str, second: dict[str, float]) -> Optional[dict]:
    """Gap between two series per label, second minus first (e.g. achieved minus target)."""
    common = [label for label in first if label in second]
    if not common:
        return None

    def gap(label: str) -> dict:
        base, difference = first[label], second[label] - first[label]
        entry = {"label": label, "difference": round(difference, 2)}
        if base:
            entry["difference_pct"] = round(difference / abs(base) * 100, 1)
        return entry

    gaps = sorted((gap(label) for label in common), key=lambda g: abs(g["difference"]), reverse=True)
    first_sum = sum(first[label] for label in common)
    second_sum = sum(second[label] for label in common)
    comparison = {
        "series": [first_name, second_name],
        "first_sum": round(first_sum, 2),
        "second_sum": round(second_sum, 2),
        "difference": round(second_sum - first_sum, 2),
        "largest_gap": gaps[0],
        "items": gaps[:MAX_LISTED_ITEMS],
    }
    if first_sum:
        comparison["difference_pct"] = round((second_sum - first_sum) / abs(first_sum) * 100, 1)
    return comparison


def chart_facts(chart: dict) -> dict:
    """Everything a summary may say about one chart, computed from its data rows."""
    title = (chart.get("title") or "").strip() or "Untitled chart"
    rows = chart.get("data") or []
    if not rows:
        raise ValueError(f"Chart '{title}' has no data.")
    if len(rows) > MAX_CHART_ROWS:
        raise ValueError(f"Chart '{title}' has {len(rows)} rows; the limit is {MAX_CHART_ROWS}.")

    category_field, value_fields = _resolve_fields(rows, chart.get("category_field"), chart.get("value_fields"))
    labels = [str(row.get(category_field, "")).strip() or "(blank)" for row in rows]
    chart_type = (chart.get("chart_type") or "").strip().lower()
    is_trend = chart_type in _TREND_CHART_TYPES or (
        len(labels) >= 2 and all(_PERIOD_LABEL.match(label) for label in labels)
    )
    names = chart.get("series_names") or {}

    series, raw = [], []
    for field in value_fields:
        # Rows repeating a label (one per district, say) add up under that label.
        values: dict[str, float] = {}
        for label, row in zip(labels, rows):
            number = _to_number(row.get(field))
            if number is not None:
                values[label] = values.get(label, 0.0) + number
        if not values:
            continue
        name = names.get(field) or field
        points = list(values.items())
        series.append(_trend_series(name, points) if is_trend else _category_series(name, points, chart.get("groups")))
        raw.append((name, values))
    if not series:
        raise ValueError(f"Chart '{title}' has no numeric values in {', '.join(value_fields)}.")

    facts = {
        "title": title,
        "chart_type": chart_type or None,
        "unit": (chart.get("unit") or "").strip() or None,
        "filters": chart.get("filters") or {},
        "kind": "trend" if is_trend else "categories",
        "series": series,
    }
    if len(raw) == 2:
        comparison = _compare(raw[0][0], raw[0][1], raw[1][0], raw[1][1])
        if comparison:
            facts["comparison"] = comparison
    return facts


# ─────────────────────────────────────────────
# Facts as prompt text
# ─────────────────────────────────────────────
def _amount(value: float, unit: str, signed: bool = False) -> str:
    return f"{_fmt(value, signed)} {unit}" if unit else _fmt(value, signed)


def _share(entry: dict) -> str:
    if "share_pct" not in entry:
        return ""
    # A real but tiny share rounds to 0.0, which reads as "none".
    pct = "under 0.1%" if entry["share_pct"] == 0 and entry.get("value") else f"{entry['share_pct']}%"
    return f" ({pct} of total)"


def render_chart_facts(facts: dict) -> str:
    unit = facts.get("unit") or ""
    lines = [f"Chart: {facts['title']}" + (f" ({facts['chart_type']} chart)" if facts.get("chart_type") else "")]
    if facts.get("filters"):
        lines.append("Filters: " + ", ".join(f"{k} = {v}" for k, v in facts["filters"].items()))
    for s in facts["series"]:
        lines.append(f"Series: {s['name']}")
        if facts["kind"] == "trend":
            change = _amount(s["change"], unit, signed=True)
            if "change_pct" in s:
                change += f" ({s['change_pct']:+.1f}%)"
            lines += [
                f"  First: {s['first']['label']}, {_amount(s['first']['value'], unit)}",
                f"  Last: {s['last']['label']}, {_amount(s['last']['value'], unit)}",
                f"  Change from first to last: {change}",
                f"  Highest: {s['highest']['label']}, {_amount(s['highest']['value'], unit)}",
                f"  Lowest: {s['lowest']['label']}, {_amount(s['lowest']['value'], unit)}",
                f"  Average per period: {_amount(s['average'], unit)}",
            ]
            if "items" in s:
                lines.append("  Values: " + " | ".join(f"{i['label']} {_fmt(i['value'])}" for i in s["items"]))
        else:
            lines.append(f"  Total: {_amount(s['total'], unit)} across {s['count']} categories")
            lines += [f"  {i['label']}: {_amount(i['value'], unit)}{_share(i)}" for i in s["items"]]
            if "others" in s:
                o = s["others"]
                lines.append(f"  Other {o['count']} categories combined: {_amount(o['value'], unit)}{_share(o)}")
            lines.append(f"  Largest: {s['largest']['label']}, {_amount(s['largest']['value'], unit)}{_share(s['largest'])}")
            lines.append(f"  Smallest: {s['smallest']['label']}, {_amount(s['smallest']['value'], unit)}{_share(s['smallest'])}")
            for g in s.get("groups", []):
                lines.append(
                    f"  Group \"{g['label']}\" ({', '.join(g['members'])}): {_amount(g['value'], unit)}{_share(g)}"
                )
    c = facts.get("comparison")
    if c:
        first, second = c["series"]
        total = _amount(c["difference"], unit, signed=True)
        if "difference_pct" in c:
            total += f" ({c['difference_pct']:+.1f}%)"

        def gap(g: dict) -> str:
            pct = f" ({g['difference_pct']:+.1f}%)" if "difference_pct" in g else ""
            return f"{g['label']} {_amount(g['difference'], unit, signed=True)}{pct}"

        lines += [
            f"Comparison, {second} minus {first}:",
            f"  Sum of {first}: {_amount(c['first_sum'], unit)}; sum of {second}: {_amount(c['second_sum'], unit)}; difference {total}",
            f"  Largest gap: {gap(c['largest_gap'])}",
            "  By label: " + " | ".join(gap(g) for g in c["items"]),
        ]
    return "\n".join(lines)


def _cross_chart_ratios(facts_list: list[dict]) -> list[dict]:
    """Each chart's total as a share of the largest total in the same unit.

    Dashboards often cut one network several ways (by road type, by condition).
    When those totals differ, the gap is usually coverage, such as roads not yet
    surveyed, and the model can only point that out if the ratio is given.
    """
    by_unit: dict[str, list[dict]] = {}
    for facts in facts_list:
        if facts["kind"] == "categories" and len(facts["series"]) == 1 and facts.get("unit"):
            by_unit.setdefault(facts["unit"].lower(), []).append(facts)
    ratios = []
    for charts in by_unit.values():
        if len(charts) < 2:
            continue
        reference = max(charts, key=lambda f: f["series"][0]["total"])
        reference_total = reference["series"][0]["total"]
        if reference_total <= 0:
            continue
        for facts in charts:
            if facts is reference:
                continue
            total = facts["series"][0]["total"]
            ratios.append({
                "title": facts["title"], "total": total,
                "reference_title": reference["title"], "reference_total": reference_total,
                "pct": round(total / reference_total * 100, 1), "difference": round(reference_total - total, 2),
                "unit": reference["unit"],
            })
    return ratios


def _cross_chart_line(r: dict) -> str:
    return (f"\"{r['title']}\" total ({_amount(r['total'], r['unit'])}) is {r['pct']:.1f}% of "
            f"\"{r['reference_title']}\" total ({_amount(r['reference_total'], r['unit'])}); "
            f"difference {_amount(r['difference'], r['unit'])}")


def _cross_chart_sentence(r: dict) -> str:
    return (f"{r['title']} covers {_readable_pct(r['pct'])} of the {r['reference_title']} total "
            f"({_readable(r['total'], r['unit'])} of {_readable(r['reference_total'], r['unit'])}).")


# ─────────────────────────────────────────────
# Rule-based summary (no model)
# ─────────────────────────────────────────────
# Written for someone glancing at a dashboard: whole numbers for large figures,
# "more than half" where it fits, and the smallest categories folded into one line.
def _readable(value: float, unit: str) -> str:
    """15,521.67 km → 15,522 km; small values keep their decimals (3.22 km)."""
    return _amount(round(value) if abs(value) >= 100 else value, unit)


def _readable_pct(pct: float) -> str:
    """19.0 → 19%, 56.5 → 56.5%, and a share that rounds to zero → under 0.1%."""
    return "under 0.1%" if 0 < abs(pct) < 0.1 else f"{_fmt(round(pct, 1))}%"


def _with_pct(entry: dict, unit: str) -> str:
    """"15,522 km (56.5%)", or just the amount when the chart has no shares."""
    text = _readable(entry["value"], unit)
    if "share_pct" in entry:
        text += f" ({_readable_pct(entry['share_pct'])})"
    return text


def _join(labels: list[str]) -> str:
    return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"


def _plain_name(name: str, unit: str) -> str:
    """"Length (km)" → "Length" when the unit is shown next to every figure anyway."""
    return re.sub(rf"\s*\({re.escape(unit)}\)\s*$", "", name, flags=re.IGNORECASE) if unit else name


def _rule_categories(s: dict, unit: str) -> tuple[str, list[str]]:
    largest, items = s["largest"], s["items"]
    share = largest.get("share_pct")
    if share is not None and share > 50:
        lead = f"{largest['label']} makes up more than half of the total"
    elif share is not None and share >= 45:
        lead = f"{largest['label']} makes up almost half of the total"
    else:
        lead = f"{largest['label']} is the largest"
    if share is not None:
        headline = (f"{lead}: {_readable(largest['value'], unit)} out of "
                    f"{_readable(s['total'], unit)} ({_readable_pct(share)}).")
    else:
        headline = f"{lead} at {_readable(largest['value'], unit)}."

    points = [f"{_join(g['members'])} together: {_with_pct(g, unit)}." for g in s.get("groups", [])]
    if len(items) >= 3:
        points.append(f"{items[1]['label']} is next with {_with_pct(items[1], unit)}, "
                      f"followed by {items[2]['label']} with {_with_pct(items[2], unit)}.")
    elif len(items) == 2:
        points.append(f"{items[1]['label']} has {_with_pct(items[1], unit)}.")

    # Everything after the top three, in one sentence rather than one line each.
    rest = items[3:]
    rest_count = len(rest) + s.get("others", {}).get("count", 0)
    if rest_count:
        rest_value = sum(i["value"] for i in rest) + s.get("others", {}).get("value", 0)
        names = [i["label"] for i in rest[:4]] + (["others"] if rest_count > 4 else [])
        amount = _readable(rest_value, unit)
        if share is not None and s["total"] > 0:
            rest_pct = rest_value / s["total"] * 100
            amount += f" ({_readable_pct(rest_pct)})"
            only = "only " if rest_pct < 10 else ""
        else:
            only = ""
        if rest_count == 1:
            points.append(f"{rest[0]['label']} has {only}{amount}.")
        else:
            points.append(f"The other {rest_count} categories ({_join(names)}) add up to {only}{amount}.")
    return headline, points


def _rule_trend(s: dict, unit: str) -> tuple[str, list[str]]:
    name = _plain_name(s["name"], unit)
    first, last = s["first"], s["last"]
    if s["change"] == 0:
        headline = f"{name} stayed at {_readable(first['value'], unit)} from {first['label']} to {last['label']}."
    else:
        rose = s["change"] > 0
        headline = (f"{name} went {'up' if rose else 'down'} from {_readable(first['value'], unit)} in {first['label']} "
                    f"to {_readable(last['value'], unit)} in {last['label']}")
        if "change_pct" in s:
            headline += f", a {'rise' if rose else 'drop'} of {_readable_pct(abs(s['change_pct']))}"
        headline += "."
    points = [f"It was highest in {s['highest']['label']} ({_readable(s['highest']['value'], unit)}) "
              f"and lowest in {s['lowest']['label']} ({_readable(s['lowest']['value'], unit)})."]
    return headline, points


def rule_based_chart_summary(facts: dict) -> dict:
    unit = facts.get("unit") or ""
    is_trend = facts["kind"] == "trend"
    headline, points = (_rule_trend if is_trend else _rule_categories)(facts["series"][0], unit)

    for other in facts["series"][1:]:
        name = _plain_name(other["name"], unit)
        if is_trend:
            points.append(f"{name} went from {_readable(other['first']['value'], unit)} "
                          f"to {_readable(other['last']['value'], unit)}.")
        else:
            points.append(f"{name} adds up to {_readable(other['total'], unit)}.")

    c = facts.get("comparison")
    if c and c["difference"]:
        first, second = (_plain_name(n, unit) for n in c["series"])
        gap = _readable(abs(c["difference"]), unit)
        if "difference_pct" in c:
            gap += f" ({_readable_pct(abs(c['difference_pct']))})"
        points.append(f"Overall, {second} is {gap} {'above' if c['difference'] > 0 else 'below'} {first}.")
    return {"headline": headline, "points": points[:5], "watch": []}


# ─────────────────────────────────────────────
# Model replies
# ─────────────────────────────────────────────
CHART_SYSTEM = """You write short summaries of dashboard charts for road and bridge asset managers.

Rules:
- Use only the facts in the user message. Do not use outside knowledge.
- Copy every number from the facts. You may round it, but never add, subtract, divide or estimate a new number. If a comparison has no number in the facts, describe it in words.
- Start with the most important finding. Name categories exactly as the facts name them.
- Write for a busy manager: short sentences in plain, everyday words. Round long figures to whole numbers and percentages to at most one decimal, and say "more than half" or "about a quarter" where it reads better.
- Refer to the measure in plain words, never as a field name or with a unit in brackets.
- Do not mention the facts, JSON, these rules or how the data was prepared.
- Write in {language}, keeping numbers as digits 0-9.
- Use an empty list for "watch" when nothing needs attention.

Reply with a JSON object only:
{"headline": "one sentence", "points": ["2 to 4 short sentences"], "watch": ["0 to 2 short sentences on what may need attention"]}"""

DASHBOARD_SYSTEM = """You write the summary of a dashboard with several charts for road and bridge asset managers.

Rules:
- Use only the facts in the user message. Do not use outside knowledge.
- Copy every number from the facts. You may round it, but never add, subtract, divide or estimate a new number. If a comparison has no number in the facts, describe it in words.
- The headline covers the dashboard as a whole. Each point may draw on one chart or relate several; relate charts only through numbers listed under "Across charts".
- Write for a busy manager: short sentences in plain, everyday words. Round long figures to whole numbers and percentages to at most one decimal, and say "more than half" or "about a quarter" where it reads better.
- Refer to the measure in plain words, never as a field name or with a unit in brackets.
- Do not mention the facts, JSON, these rules, chart numbers like [1], or how the data was prepared.
- Write in {language}, keeping numbers as digits 0-9.
- Use an empty list for "watch" when nothing needs attention.

Reply with a JSON object only:
{"headline": "one sentence", "points": ["3 to 6 short sentences, most important first"], "watch": ["0 to 3 short sentences on what may need attention"]}"""


def _parse_json_object(raw: str) -> Optional[dict]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _as_summary(obj: Optional[dict]) -> Optional[dict]:
    if not obj or not isinstance(obj.get("headline"), str) or not obj["headline"].strip():
        return None
    return {
        "headline": obj["headline"].strip(),
        "points": _str_list(obj.get("points"))[:8],
        "watch": _str_list(obj.get("watch"))[:4],
    }


def _keep_supported(summary: dict, allowed: list[float], fallback: dict) -> dict:
    """Drop sentences whose numbers aren't in the facts; refill from the rule-based summary."""
    def supported(sentence: str) -> bool:
        return not _unsupported_numbers(sentence, allowed)

    return {
        "headline": summary["headline"] if supported(summary["headline"]) else fallback["headline"],
        "points": [p for p in summary["points"] if supported(p)] or fallback["points"],
        "watch": [w for w in summary["watch"] if supported(w)],
    }


async def _write_from_facts(system: str, facts_text: str, provider, language: str, fallback: dict) -> dict:
    allowed = _allowed_values(facts_text)
    system = system.replace("{language}", language)
    prompt = facts_text
    summary = None
    try:
        for _ in range(2):
            summary = _as_summary(_parse_json_object(await provider.complete(system, prompt, json_mode=True)))
            if summary is None:
                prompt = f"{facts_text}\n\nReply with the JSON object only, with keys headline, points and watch."
                continue
            unsupported = _unsupported_numbers(" ".join([summary["headline"], *summary["points"], *summary["watch"]]), allowed)
            if not unsupported:
                return {**summary, "source": "llm", "warnings": []}
            logger.info("Summary used numbers not in the facts, retrying: %s", unsupported)
            prompt = (f"{facts_text}\n\nYour previous answer used numbers that are not in these facts: "
                      f"{', '.join(unsupported)}. Write it again using only numbers that appear above.")
    except LLMError as exc:
        logger.warning("Summary model unavailable, using the rule-based summary: %s", exc)
        return {**fallback, "source": "rules",
                "warnings": ["The AI model could not be reached, so this summary was built directly from the data."]}
    if summary is None:
        return {**fallback, "source": "rules",
                "warnings": ["The AI model's reply could not be used, so this summary was built directly from the data."]}
    return {**_keep_supported(summary, allowed, fallback), "source": "llm+rules",
            "warnings": ["Sentences with figures that could not be verified against the data were removed."]}


# ─────────────────────────────────────────────
# Charts and dashboards
# ─────────────────────────────────────────────
async def summarize_chart(chart: dict, provider, language: str = "English") -> dict:
    facts = chart_facts(chart)
    result = await _write_from_facts(CHART_SYSTEM, render_chart_facts(facts), provider, language,
                                     rule_based_chart_summary(facts))
    return {**result, "facts": facts}


async def summarize_dashboard(dashboard: dict, provider, language: str = "English") -> dict:
    charts = dashboard.get("charts") or []
    if not charts:
        raise ValueError("The dashboard has no charts.")
    if len(charts) > MAX_DASHBOARD_CHARTS:
        raise ValueError(f"The dashboard has {len(charts)} charts; the limit is {MAX_DASHBOARD_CHARTS}.")

    facts_list, skipped = [], []
    for chart in charts:
        try:
            facts_list.append(chart_facts(chart))
        except ValueError as exc:
            skipped.append(str(exc))
    if not facts_list:
        raise ValueError("None of the charts has usable data: " + " ".join(skipped))

    header = [f"Dashboard: {dashboard.get('title') or 'Dashboard'}"]
    if dashboard.get("filters"):
        header.append("Filters: " + ", ".join(f"{k} = {v}" for k, v in dashboard["filters"].items()))
    blocks = [f"[{i}] {render_chart_facts(f)}" for i, f in enumerate(facts_list, 1)]
    ratios = _cross_chart_ratios(facts_list)
    cross = [_cross_chart_line(r) for r in ratios]
    text = "\n".join(header) + "\n\n" + "\n\n".join(blocks)
    if cross:
        text += "\n\nAcross charts:\n" + "\n".join(f"  {line}" for line in cross)

    per_chart = [rule_based_chart_summary(f) for f in facts_list]
    fallback = {
        "headline": per_chart[0]["headline"],
        "points": ([p["headline"] for p in per_chart[1:]] + [_cross_chart_sentence(r) for r in ratios])[:6]
        or per_chart[0]["points"],
        "watch": [],
    }
    result = await _write_from_facts(DASHBOARD_SYSTEM, text, provider, language, fallback)
    result["warnings"] = result["warnings"] + [f"Skipped: {reason}" for reason in skipped]
    return {**result, "facts": facts_list, "cross_chart": cross}


# ─────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────
DOC_NOTES_SYSTEM = """You are reading one part of a longer document. Write concise bullet notes of what this part contains: main topics, decisions, requirements, figures, dates, road/bridge/place names, responsibilities and actions. Copy numbers exactly as written. Output the notes only, with no preamble."""

DOC_MERGE_SYSTEM = """Below are notes taken from consecutive parts of one document. Merge them into one set of concise bullet notes: remove repetition, keep every distinct figure, date, decision and action. Copy numbers exactly as written. Output the notes only, with no preamble."""

DOC_SUMMARY_SYSTEM = """You summarize documents for road and bridge asset managers.

Rules:
- Use only the text in the user message. Do not use outside knowledge.
- Copy numbers exactly as written there; never compute new ones.
- Do not mention these rules, notes, or "the text provided".
- Write in {language}, keeping numbers as digits 0-9.
- Use an empty list for "action_items" when the document asks for nothing.

Reply with a JSON object only:
{"title": "short title for the document", "overview": "2 to 4 sentences", "key_points": ["4 to 8 short points"], "action_items": ["deadlines, required actions or pending decisions"]}"""


def split_text(text: str, max_chars: int) -> list[str]:
    """Split on paragraph breaks into parts of at most max_chars."""
    parts: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        while len(paragraph) > max_chars:
            cut = paragraph.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            if current:
                parts.append(current)
                current = ""
            parts.append(paragraph[:cut].strip())
            paragraph = paragraph[cut:].strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > max_chars:
            parts.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        parts.append(current)
    return parts


def _group_to_fit(notes: list[str], max_chars: int) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for note in notes:
        if current and size + len(note) + 2 > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append(note)
        size += len(note) + 2
    if current:
        groups.append(current)
    return groups


def _as_document_summary(obj: Optional[dict]) -> Optional[dict]:
    if not obj or not isinstance(obj.get("overview"), str) or not obj["overview"].strip():
        return None
    return {
        "title": str(obj.get("title") or "").strip(),
        "overview": obj["overview"].strip(),
        "key_points": _str_list(obj.get("key_points"))[:10],
        "action_items": _str_list(obj.get("action_items"))[:10],
    }


async def summarize_document_events(text: str, provider, language: str = "English") -> AsyncIterator[tuple[str, dict]]:
    """Yield ("progress", {...}) while working and ("result", summary) at the end.

    Raises LLMError if the model fails or never returns a usable summary.
    """
    text = text.strip()
    if not text:
        raise ValueError("The document has no readable text.")
    warnings = []
    truncated = len(text) > MAX_DOCUMENT_CHARS
    if truncated:
        text = text[:MAX_DOCUMENT_CHARS]
        warnings.append(f"The document is long; this summary covers roughly its first {MAX_DOCUMENT_CHARS:,} characters.")

    budget = provider.max_input_chars
    parts = split_text(text, budget)
    if len(parts) == 1:
        source = parts[0]
    else:
        notes = []
        for i, part in enumerate(parts, 1):
            yield "progress", {"stage": "reading", "part": i, "parts": len(parts)}
            notes.append(await provider.complete(DOC_NOTES_SYSTEM, part, temperature=0.1))
        for _ in range(MAX_MERGE_ROUNDS):
            if len("\n\n".join(notes)) <= budget:
                break
            groups = _group_to_fit(notes, budget)
            yield "progress", {"stage": "merging", "groups": len(groups)}
            notes = [await provider.complete(DOC_MERGE_SYSTEM, "\n\n".join(g), temperature=0.1) for g in groups]
        source = "\n\n".join(notes)
        if len(source) > budget:
            source = source[:budget]
            warnings.append("Some notes were cut to fit the model's context window.")

    yield "progress", {"stage": "writing"}
    system = DOC_SUMMARY_SYSTEM.replace("{language}", language)
    prompt = source
    summary = None
    for _ in range(2):
        summary = _as_document_summary(_parse_json_object(await provider.complete(system, prompt, json_mode=True)))
        if summary:
            break
        prompt = f"{source}\n\nReply with the JSON object only, with keys title, overview, key_points and action_items."
    if summary is None:
        raise LLMError("The model did not return a usable summary.")

    written = " ".join([summary["title"], summary["overview"], *summary["key_points"], *summary["action_items"]])
    yield "result", {
        **summary,
        "unverified_numbers": _unsupported_numbers(written, _allowed_values(text)),
        "parts": len(parts),
        "truncated": truncated,
        "warnings": warnings,
    }
