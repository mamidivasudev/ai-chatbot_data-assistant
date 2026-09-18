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


# Averages, rates and indices do not add up: a total or share of "AxelAvg" or
# "IRI" per zone is meaningless. Field names are split on camelCase and
# underscores first, so "AxelAvg" reads as "Axel Avg".
_NON_ADDITIVE_WORD = re.compile(r"\b(avg|average|mean|median|rate|ratio|percent|pct|index|score|iri|pci|bci)\b",
                                re.IGNORECASE)


def is_additive(*names: str) -> bool:
    for name in names:
        words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name or "").replace("_", " ")
        if "%" in words or _NON_ADDITIVE_WORD.search(words):
            return False
    return True


def _category_series(name: str, points: list[tuple[str, float]],
                     groups: Optional[dict[str, list[str]]], additive: bool = True) -> dict:
    total = sum(value for _, value in points)
    # Shares only mean something when the values are parts of one positive whole.
    has_shares = additive and total > 0 and all(value >= 0 for _, value in points)

    def item(label: str, value: float) -> dict:
        entry = {"label": label, "value": round(value, 2)}
        if has_shares:
            entry["share_pct"] = round(value / total * 100, 1)
        return entry

    ranked = sorted(points, key=lambda p: p[1], reverse=True)
    series = {
        "name": name,
        "count": len(points),
        "additive": additive,
        "items": [item(*p) for p in ranked[:MAX_LISTED_ITEMS]],
        "largest": item(*ranked[0]),
        "smallest": item(*ranked[-1]),
    }
    if additive:
        series["total"] = round(total, 2)
    else:
        series["average"] = round(total / len(points), 2)
    rest = ranked[MAX_LISTED_ITEMS:]
    if rest:
        series["others"] = ({"count": len(rest), **item("others", sum(v for _, v in rest))} if additive
                            else {"count": len(rest), "label": "others"})

    if not additive:
        return series  # groups are sums, which averages do not support
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


def _compare(first_name: str, first: dict[str, float], second_name: str, second: dict[str, float],
             additive: bool = True) -> Optional[dict]:
    """Gap between two series per label, second minus first (e.g. achieved minus target).

    Sums across labels are included only for additive series.
    """
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
    comparison = {"series": [first_name, second_name], "largest_gap": gaps[0], "items": gaps[:MAX_LISTED_ITEMS]}
    if additive:
        first_sum = sum(first[label] for label in common)
        second_sum = sum(second[label] for label in common)
        comparison.update(first_sum=round(first_sum, 2), second_sum=round(second_sum, 2),
                          difference=round(second_sum - first_sum, 2))
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
        name = names.get(field) or field
        additive = is_additive(field, name)
        # Rows repeating a label (one per district, say) add up under that label;
        # averages and rates are averaged instead, since they do not add up.
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for label, row in zip(labels, rows):
            number = _to_number(row.get(field))
            if number is not None:
                sums[label] = sums.get(label, 0.0) + number
                counts[label] = counts.get(label, 0) + 1
        if not sums:
            continue
        values = sums if additive else {label: sums[label] / counts[label] for label in sums}
        points = list(values.items())
        series.append(_trend_series(name, points) if is_trend
                      else _category_series(name, points, chart.get("groups"), additive))
        raw.append((name, values, additive))
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
        (first_name, first, first_additive), (second_name, second, second_additive) = raw
        comparison = _compare(first_name, first, second_name, second, first_additive and second_additive)
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
            if "total" in s:
                lines.append(f"  Total: {_amount(s['total'], unit)} across {s['count']} categories")
            else:
                lines.append(f"  These are averages or rates, which do not add up; no total or shares. "
                             f"Mean across {s['count']} categories: {_amount(s['average'], unit)}")
            lines += [f"  {i['label']}: {_amount(i['value'], unit)}{_share(i)}" for i in s["items"]]
            if "others" in s:
                o = s["others"]
                combined = f": {_amount(o['value'], unit)}{_share(o)}" if "value" in o else " not listed"
                lines.append(f"  Other {o['count']} categories{' combined' if 'value' in o else ''}{combined}")
            lines.append(f"  Largest: {s['largest']['label']}, {_amount(s['largest']['value'], unit)}{_share(s['largest'])}")
            lines.append(f"  Smallest: {s['smallest']['label']}, {_amount(s['smallest']['value'], unit)}{_share(s['smallest'])}")
            for g in s.get("groups", []):
                lines.append(
                    f"  Group \"{g['label']}\" ({', '.join(g['members'])}): {_amount(g['value'], unit)}{_share(g)}"
                )
    c = facts.get("comparison")
    if c:
        first, second = c["series"]

        def gap(g: dict) -> str:
            pct = f" ({g['difference_pct']:+.1f}%)" if "difference_pct" in g else ""
            return f"{g['label']} {_amount(g['difference'], unit, signed=True)}{pct}"

        lines.append(f"Comparison, {second} minus {first}:")
        if "difference" in c:
            total = _amount(c["difference"], unit, signed=True)
            if "difference_pct" in c:
                total += f" ({c['difference_pct']:+.1f}%)"
            lines.append(f"  Sum of {first}: {_amount(c['first_sum'], unit)}; sum of {second}: "
                         f"{_amount(c['second_sum'], unit)}; difference {total}")
        lines += [f"  Largest gap: {gap(c['largest_gap'])}", "  By label: " + " | ".join(gap(g) for g in c["items"])]
    return "\n".join(lines)


def _cross_chart_ratios(facts_list: list[dict]) -> list[dict]:
    """Relations between chart totals that the model cannot work out on its own.

    same_total — two charts adding up to the same figure usually count the same
    items two ways (stations by traffic band and by zone), even without a unit.
    share — each chart's total as a share of the largest total in the same unit.
    When a dashboard cuts one network several ways (by road type, by condition),
    the gap is usually coverage, such as roads not yet surveyed.
    """
    totals = [f for f in facts_list
              if f["kind"] == "categories" and len(f["series"]) == 1 and f["series"][0].get("total", 0) > 0]
    relations, matched = [], set()
    for i, first in enumerate(totals):
        for second in totals[i + 1:]:
            a, b = first["series"][0]["total"], second["series"][0]["total"]
            if abs(a - b) <= 0.005 * max(a, b):
                relations.append({"kind": "same_total", "title": first["title"], "reference_title": second["title"],
                                  "total": a, "unit": first.get("unit") or second.get("unit") or ""})
                matched.add(frozenset((id(first), id(second))))

    by_unit: dict[str, list[dict]] = {}
    for facts in totals:
        if facts.get("unit"):
            by_unit.setdefault(facts["unit"].lower(), []).append(facts)
    for charts in by_unit.values():
        if len(charts) < 2:
            continue
        reference = max(charts, key=lambda f: f["series"][0]["total"])
        reference_total = reference["series"][0]["total"]
        for facts in charts:
            if facts is reference or frozenset((id(facts), id(reference))) in matched:
                continue
            total = facts["series"][0]["total"]
            relations.append({
                "kind": "share", "title": facts["title"], "total": total,
                "reference_title": reference["title"], "reference_total": reference_total,
                "pct": round(total / reference_total * 100, 1), "difference": round(reference_total - total, 2),
                "unit": reference["unit"],
            })
    return relations


def _cross_chart_line(r: dict) -> str:
    if r["kind"] == "same_total":
        return (f"\"{r['title']}\" and \"{r['reference_title']}\" both total {_amount(r['total'], r['unit'])}: "
                f"they likely count the same items, broken down two ways")
    return (f"\"{r['title']}\" total ({_amount(r['total'], r['unit'])}) is {r['pct']:.1f}% of "
            f"\"{r['reference_title']}\" total ({_amount(r['reference_total'], r['unit'])}); "
            f"difference {_amount(r['difference'], r['unit'])}")


def _cross_chart_sentence(r: dict) -> str:
    if r["kind"] == "same_total":
        return (f"{r['title']} and {r['reference_title']} both add up to {_readable(r['total'], r['unit'])}, "
                f"so they likely count the same items.")
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
    if "total" not in s:
        # Averages and rates: highest and lowest, never totals or shares.
        smallest = s["smallest"]
        headline = (f"{largest['label']} is the highest at {_readable(largest['value'], unit)}, "
                    f"and {smallest['label']} the lowest at {_readable(smallest['value'], unit)}.")
        points = []
        if len(items) >= 3:
            points.append(f"{items[1]['label']} ({_readable(items[1]['value'], unit)}) and {items[2]['label']} "
                          f"({_readable(items[2]['value'], unit)}) come next.")
        points.append(f"The average across {s['count']} categories is {_readable(s['average'], unit)}.")
        return headline, points
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
        elif "total" in other:
            points.append(f"{name} adds up to {_readable(other['total'], unit)}.")
        else:
            points.append(f"{name} averages {_readable(other['average'], unit)}.")

    c = facts.get("comparison")
    if c and c.get("difference"):
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
- Say what the chart counts in plain words taken from its title (such as roads or stations), never a field name or a unit in brackets.
- Do not mention the facts, JSON, these rules or how the data was prepared.
- Write in {language}, keeping numbers as digits 0-9.
- Use an empty list for "watch" when nothing needs attention.

Reply with a JSON object only:
{"headline": "one sentence", "points": ["2 to 4 short sentences"], "watch": ["0 to 2 short sentences on what may need attention"]}"""

# One sentence per chart, in chart order: a small model asked for free-form points
# tends to pour everything into one paragraph, and fixed slots can be checked and
# repaired one by one. "..." placeholders, so the example is not copied as text.
DASHBOARD_SYSTEM = """You brief a busy road and bridge asset manager on a dashboard of several charts.

Rules:
- Use only the facts in the user message. Do not use outside knowledge.
- Copy every number from the facts. You may round it, but never add, subtract, divide or estimate a new number. If a comparison has no number in the facts, describe it in words.
- Say what each chart counts in plain words taken from its title (such as roads or stations), never a field name or a unit in brackets.
- Write for a busy manager: short sentences in plain, everyday words. Round long figures to whole numbers and percentages to at most one decimal.
- Do not mention the facts, JSON, these rules, chart numbers like [1], or how the data was prepared.
- Write in {language}, keeping numbers as digits 0-9.

What to write:
- "headline": the single most important finding on the whole dashboard, as one sentence with its key number. Not a title.
- "charts": exactly one entry per chart, in the order given ([1], [2], ...): one sentence of at most 25 words with that chart's main finding.
- "points": 0 to 2 sentences relating charts to each other, using only what is listed under "Across charts"; an empty list if nothing is listed.
- "watch": 0 to 2 sentences on what may need attention; an empty list if nothing stands out.

Reply with a JSON object only:
{"headline": "...", "charts": ["...", "..."], "points": [], "watch": []}"""


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

    result = await _write_dashboard(text, facts_list, [_cross_chart_sentence(r) for r in ratios], provider, language)
    result["warnings"] = result["warnings"] + [f"Skipped: {reason}" for reason in skipped]
    return {**result, "facts": facts_list, "cross_chart": cross}


def _as_dashboard(obj: Optional[dict], chart_count: int) -> Optional[dict]:
    if not obj or not isinstance(obj.get("headline"), str) or not obj["headline"].strip():
        return None
    return {
        "headline": obj["headline"].strip(),
        "charts": _str_list(obj.get("charts"))[:chart_count],
        "points": _str_list(obj.get("points"))[:3],
        "watch": _str_list(obj.get("watch"))[:3],
    }


async def _write_dashboard(text: str, facts_list: list[dict], cross_sentences: list[str],
                           provider, language: str) -> dict:
    """Headline, one sentence per chart, cross-chart points and watch items.

    Every sentence is checked against the facts. After one retry, a chart sentence
    with a number the facts don't hold is replaced by that chart's rule-based
    headline, so each chart always gets a correct line.
    """
    allowed = _allowed_values(text)
    titles = [f["title"] for f in facts_list]
    rule_lines = [rule_based_chart_summary(f)["headline"] for f in facts_list]

    def supported(sentence: str) -> bool:
        return not _unsupported_numbers(sentence, allowed)

    def rules_only(warning: str) -> dict:
        return {"headline": rule_lines[0], "charts": [{"title": t, "summary": s} for t, s in zip(titles, rule_lines)],
                "points": cross_sentences[:2], "watch": [], "source": "rules", "warnings": [warning]}

    system = DASHBOARD_SYSTEM.replace("{language}", language)
    prompt, reply = text, None
    try:
        for attempt in range(2):
            reply = _as_dashboard(_parse_json_object(await provider.complete(system, prompt, json_mode=True)), len(titles))
            if reply is None:
                prompt = f"{text}\n\nReply with the JSON object only, with keys headline, charts, points and watch."
                continue
            unsupported = _unsupported_numbers(
                " ".join([reply["headline"], *reply["charts"], *reply["points"], *reply["watch"]]), allowed)
            if not unsupported and len(reply["charts"]) == len(titles):
                break
            problems = []
            if unsupported:
                problems.append(f"used numbers that are not in these facts: {', '.join(unsupported)}")
            if len(reply["charts"]) != len(titles):
                problems.append(f"gave {len(reply['charts'])} chart sentences for {len(titles)} charts")
            logger.info("Dashboard summary retried: %s", "; ".join(problems))
            prompt = (f"{text}\n\nYour previous answer {' and '.join(problems)}. Write it again with exactly one "
                      f"sentence per chart, using only numbers that appear above.")
    except LLMError as exc:
        logger.warning("Summary model unavailable, using the rule-based summary: %s", exc)
        return rules_only("The AI model could not be reached, so this summary was built directly from the data.")
    if reply is None:
        return rules_only("The AI model's reply could not be used, so this summary was built directly from the data.")

    repaired = not supported(reply["headline"])
    charts = []
    for i, title in enumerate(titles):
        sentence = reply["charts"][i] if i < len(reply["charts"]) else ""
        if not sentence or not supported(sentence):
            sentence, repaired = rule_lines[i], True
        charts.append({"title": title, "summary": sentence})
    points = [p for p in reply["points"] if supported(p)]
    watch = [w for w in reply["watch"] if supported(w)]
    repaired = repaired or len(points) < len(reply["points"]) or len(watch) < len(reply["watch"])
    return {
        "headline": reply["headline"] if supported(reply["headline"]) else rule_lines[0],
        "charts": charts,
        "points": points,
        "watch": watch,
        "source": "llm+rules" if repaired else "llm",
        "warnings": ["Sentences with figures that could not be verified against the data were replaced."] if repaired else [],
    }


# ─────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────
DOC_NOTES_SYSTEM = """You are reading one part of a longer document. Write concise bullet notes of what this part contains: main topics, decisions, requirements, figures, dates, road/bridge/place names, and for each item its status (done, pending, in progress), owner and due date when given. Copy numbers and dates exactly as written. Output the notes only, with no preamble."""

DOC_MERGE_SYSTEM = """Below are notes taken from consecutive parts of one document. Merge them into one set of concise bullet notes: remove repetition, keep every distinct figure, date, decision, status and action. Copy numbers and dates exactly as written. Output the notes only, with no preamble."""

# The JSON example uses "..." on purpose: descriptive placeholder text gets copied
# into the answer as labels ("Deadlines: ...").
DOC_SUMMARY_SYSTEM = """You brief a busy road and bridge asset manager on a document, the way a good analyst would.

Rules:
- Use only the text in the user message. Do not use outside knowledge.
- Copy numbers, dates and names exactly as written; never compute new numbers. Counts listed under "Counted from the document's tables" are exact and may be quoted.
- Write in {language}, in short, plain sentences, keeping numbers as digits 0-9.
- Do not add purposes, benefits or opinions the document does not state (for example, that the changes improve the user experience).
- Do not mention these rules, notes, tables, or "the text provided".

What to write:
- "title": a short, specific title for the document.
- "overview": 2 to 3 sentences: what the document is, what it covers, and its overall status. Quote the counted statuses when they are given; otherwise describe status only as the document states it.
- "key_points": 3 to 6 points. Group related items into one point per area, screen, section or theme instead of listing every line. Each point is one complete sentence.
- "action_items": what is still to be done or decided, each a complete sentence saying what must happen, and by whom and by when if the document says so. Never output bare item or section numbers, and never start an item with a label such as "Deadlines:". Use an empty list when nothing is pending.

Reply with a JSON object only:
{"title": "...", "overview": "...", "key_points": ["...", "..."], "action_items": ["..."]}"""

# A leading label the model sometimes adds ("Action items: …", "Deadlines: …").
_LABEL_PREFIX = re.compile(
    r"^\s*(deadlines?|required actions?(?: or pending decisions?)?|pending decisions?|action items?|key points?|actions?)\s*:\s*",
    re.IGNORECASE,
)


def _as_sentences(items: list[str]) -> list[str]:
    """Drop labels and fragments that carry no words, like "2.{6,7,8}, 3 {1,2,3}"."""
    kept = []
    for item in items:
        item = _LABEL_PREFIX.sub("", item).strip()
        if len(re.findall(r"[^\W\d_]{2,}", item)) >= 2:
            kept.append(item)
    return kept


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
        "key_points": _as_sentences(_str_list(obj.get("key_points")))[:8],
        "action_items": _as_sentences(_str_list(obj.get("action_items")))[:8],
    }


async def summarize_document_events(text: str, provider, language: str = "English",
                                    status: Optional[list[dict]] = None) -> AsyncIterator[tuple[str, dict]]:
    """Yield ("progress", {...}) while working and ("result", summary) at the end.

    `status` is document_tables.status_counts() output: exact counts from status
    columns, given to the model as facts and returned with the result for display.
    Raises LLMError if the model fails or never returns a usable summary.
    """
    from document_tables import render_status_facts

    status = status or []
    facts = render_status_facts(status) if status else ""
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
    # The counts go last, right before the answer, where a small model weighs them most.
    source = f"{source}\n\n{facts}" if facts else source
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
        "status": status,
        "unverified_numbers": _unsupported_numbers(written, _allowed_values(f"{text}\n{facts}")),
        "parts": len(parts),
        "truncated": truncated,
        "warnings": warnings,
    }
