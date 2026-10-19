#!/usr/bin/env python3
"""quakewatch: plain-English earthquake briefings and Q&A.

Pulls the free USGS earthquake GeoJSON feed (public data, no API key), computes
statistics locally, and asks a local Ollama model to explain them. Standard
library only. The tool works on earthquakes, never on people or photos.

Usage:
  python3 quakewatch.py summary [--feed day] [--min-mag 2.5] [--model llama3.2]
  python3 quakewatch.py ask "Where was the deepest quake?" [--feed week]
  python3 quakewatch.py regions [--feed week] [--top 15]
  python3 quakewatch.py strongest [--feed month] [--top 5]
  python3 quakewatch.py selftest
"""

import argparse
import datetime
import json
import sys
import threading
import unittest.mock as mock
import urllib.error
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer

OLLAMA_URL = "http://localhost:11434/api/generate"
USGS_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"
FEEDS = {
    "hour": "all_hour.geojson",
    "day": "all_day.geojson",
    "week": "all_week.geojson",
    "month": "all_month.geojson",
}
DEFAULT_MODEL = "llama3.2"

SAMPLE_FEED = {
    "type": "FeatureCollection",
    "features": [
        {"properties": {"mag": 5.1, "place": "75 km SW of Hilo, Hawaii",
                        "time": 1760900000000, "title": "M 5.1 - Hawaii"},
         "geometry": {"coordinates": [-155.5, 19.2, 12.0]}},
        {"properties": {"mag": 2.4, "place": "10 km N of Volcano, Hawaii",
                        "time": 1760900500000, "title": "M 2.4 - Hawaii"},
         "geometry": {"coordinates": [-155.3, 19.5, 8.0]}},
        {"properties": {"mag": 1.0, "place": "5 km E of Pahala, Hawaii",
                        "time": 1760901000000, "title": "M 1.0 - Hawaii"},
         "geometry": {"coordinates": [-155.4, 19.2, 4.0]}},
        {"properties": {"mag": 3.2, "place": "8 km W of Ridgecrest, CA",
                        "time": 1760902000000, "title": "M 3.2 - CA"},
         "geometry": {"coordinates": [-117.8, 35.6, 9.0]}},
        {"properties": {"mag": 3.0, "place": None,
                        "time": 1760903000000, "title": "M 3.0 - unknown"},
         "geometry": {"coordinates": [10.0, 10.0, 7.0]}},
        {"properties": {"mag": None, "place": "bad record", "time": None},
         "geometry": {"coordinates": [0, 0]}},
    ],
}


class QuakeError(Exception):
    """Raised for feed or Ollama problems that the user can act on."""


def valid_number(value):
    """Return True for a finite int or float (bool and NaN are rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float("inf"), float("-inf"))


def fetch_json(url, timeout=20):
    """Download and decode JSON from a URL."""
    request = urllib.request.Request(url, headers={"User-Agent": "quakewatch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise QuakeError("Could not load the earthquake feed: %s" % exc)
    except ValueError as exc:
        raise QuakeError("Feed returned invalid JSON: %s" % exc)


def feed_url(feed):
    if feed not in FEEDS:
        raise QuakeError("Unknown feed %r; choose from %s" % (feed, ", ".join(FEEDS)))
    return USGS_BASE + FEEDS[feed]


def region_of(place):
    """Reduce a USGS place string like '10 km N of Volcano, Hawaii' to a region."""
    if not place or not isinstance(place, str):
        return "Unknown region"
    if "," in place:
        return place.rsplit(",", 1)[1].strip() or "Unknown region"
    if " of " in place:
        return place.split(" of ", 1)[1].strip() or "Unknown region"
    return place.strip()


def normalize_events(data):
    """Turn a GeoJSON feed into a list of clean event dicts, skipping bad rows."""
    events = []
    features = data.get("features") if isinstance(data, dict) else None
    if not isinstance(features, list):
        raise QuakeError("Feed has no 'features' list")
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        mag = props.get("mag")
        if not valid_number(mag):
            continue
        depth = coords[2] if len(coords) > 2 and valid_number(coords[2]) else None
        when = props.get("time")
        if valid_number(when):
            when = datetime.datetime.fromtimestamp(
                when / 1000.0, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        else:
            when = "unknown time"
        events.append({
            "mag": float(mag),
            "depth_km": float(depth) if depth is not None else None,
            "place": props.get("place") or "Unknown place",
            "region": region_of(props.get("place")),
            "when": when,
        })
    return events


def filter_events(events, min_mag=None):
    if min_mag is None:
        return list(events)
    return [e for e in events if e["mag"] >= min_mag]


def compute_stats(events):
    """Compute count, magnitude, depth and region statistics."""
    mags = [e["mag"] for e in events]
    depths = [e["depth_km"] for e in events if e["depth_km"] is not None]
    regions = Counter(e["region"] for e in events)
    stats = {
        "count": len(events),
        "max_mag": max(mags) if mags else None,
        "min_mag": min(mags) if mags else None,
        "mean_mag": sum(mags) / len(mags) if mags else None,
        "min_depth_km": min(depths) if depths else None,
        "max_depth_km": max(depths) if depths else None,
        "mean_depth_km": sum(depths) / len(depths) if depths else None,
        "top_regions": regions.most_common(10),
    }
    if events:
        stats["strongest"] = max(events, key=lambda e: e["mag"])
        with_depth = [e for e in events if e["depth_km"] is not None]
        stats["deepest"] = max(with_depth, key=lambda e: e["depth_km"]) if with_depth else None
    else:
        stats["strongest"] = None
        stats["deepest"] = None
    return stats


def band_counts(events):
    """Count events by magnitude band."""
    bands = [("<2", 0, 2), ("2-3", 2, 3), ("3-4", 3, 4), ("4-5", 4, 5), ("5+", 5, 99)]
    return {name: sum(1 for e in events if lo <= e["mag"] < hi) for name, lo, hi in bands}


def _fmt(value, suffix=""):
    return "unavailable" if value is None else "%.2f%s" % (value, suffix)


def format_stats(stats):
    lines = ["Earthquakes: %d" % stats["count"]]
    if stats["count"] == 0:
        lines.append("No events matched.")
        return "\n".join(lines)
    lines.append("Magnitude max/mean/min: %s / %s / %s" % (
        _fmt(stats["max_mag"]), _fmt(stats["mean_mag"]), _fmt(stats["min_mag"])))
    lines.append("Depth km max/mean/min: %s / %s / %s" % (
        _fmt(stats["max_depth_km"]), _fmt(stats["mean_depth_km"]), _fmt(stats["min_depth_km"])))
    lines.append("Top regions:")
    for name, count in stats["top_regions"][:5]:
        lines.append("  %s: %d" % (name, count))
    return "\n".join(lines)


def build_context(events, stats, limit=15):
    """Build the data block that is given to the model."""
    lines = [format_stats(stats), "Magnitude bands: " + json.dumps(band_counts(events))]
    ranked = sorted(events, key=lambda e: e["mag"], reverse=True)[:limit]
    lines.append("Strongest events:")
    for e in ranked:
        depth = "n/a" if e["depth_km"] is None else "%.1f km" % e["depth_km"]
        lines.append("- M%.1f, %s, depth %s, %s" % (e["mag"], e["place"], depth, e["when"]))
    return "\n".join(lines)


def ollama_generate(prompt, model=DEFAULT_MODEL, timeout=120, url=None):
    """Call the local Ollama HTTP API and return the response text."""
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        url or OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise QuakeError(
            "Could not reach Ollama at %s (%s). Start it with 'ollama serve' and run "
            "'ollama pull %s'." % (url or OLLAMA_URL, exc, model))
    except ValueError as exc:
        raise QuakeError("Ollama returned invalid JSON: %s" % exc)
    text = body.get("response") if isinstance(body, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise QuakeError("Ollama returned no text: %s" % str(body)[:200])
    return text.strip()


def summary_prompt(context):
    return ("You are a calm science communicator. Using only the earthquake data below, "
            "write a short plain-English briefing (under 150 words). Mention the "
            "strongest event, where activity is concentrated, and whether anything "
            "looks unusual. Do not invent numbers.\n\nDATA:\n" + context)


def question_prompt(context, question):
    return ("Answer the question using only the earthquake data below. If the data "
            "cannot answer it, say so. Keep it under 100 words.\n\nDATA:\n%s\n\n"
            "QUESTION: %s" % (context, question))


def load_events(args):
    data = fetch_json(feed_url(args.feed))
    return filter_events(normalize_events(data), args.min_mag)


def cmd_summary(args, generate=ollama_generate):
    events = load_events(args)
    stats = compute_stats(events)
    print(format_stats(stats))
    if not events:
        return 0
    print()
    print(generate(summary_prompt(build_context(events, stats)), model=args.model))
    return 0


def cmd_ask(args, generate=ollama_generate):
    events = load_events(args)
    if not events:
        print("No events matched, nothing to ask about.")
        return 0
    stats = compute_stats(events)
    print(generate(question_prompt(build_context(events, stats), args.question), model=args.model))
    return 0


def cmd_regions(args):
    events = load_events(args)
    counts = Counter(e["region"] for e in events)
    for name, count in counts.most_common(args.top):
        print("%5d  %s" % (count, name))
    return 0


def cmd_strongest(args):
    events = load_events(args)
    for e in sorted(events, key=lambda e: e["mag"], reverse=True)[:args.top]:
        print("M%.1f  %s  (%s)" % (e["mag"], e["place"], e["when"]))
    return 0


class _FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        json.loads(self.rfile.read(length).decode("utf-8"))
        body = json.dumps({"response": " stub briefing "}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def selftest():
    """Offline checks using a sample feed and a stubbed Ollama."""
    checks = 0

    def check(condition, label):
        nonlocal checks
        if not condition:
            raise AssertionError("selftest failed: " + label)
        checks += 1

    check(valid_number(1) and valid_number(2.5), "numbers accepted")
    check(not valid_number(True) and not valid_number(None) and not valid_number(float("nan")),
          "bad numbers rejected")
    check(region_of("10 km N of Volcano, Hawaii") == "Hawaii", "region from comma")
    check(region_of("Gulf of Alaska") == "Alaska", "region from 'of'")
    check(region_of(None) == "Unknown region", "region of None")

    events = normalize_events(SAMPLE_FEED)
    check(len(events) == 5, "bad record skipped")
    stats = compute_stats(events)
    check(stats["count"] == 5, "count")
    check(stats["max_mag"] == 5.1 and stats["min_mag"] == 1.0, "magnitude range")
    check(abs(stats["mean_mag"] - 2.94) < 1e-9, "mean magnitude")
    check(stats["max_depth_km"] == 12.0 and stats["min_depth_km"] == 4.0, "depth range")
    check(stats["top_regions"][0] == ("Hawaii", 3), "top region")
    check(stats["strongest"]["place"].startswith("75 km SW"), "strongest event")
    check(stats["deepest"]["mag"] == 5.1, "deepest event")
    check(band_counts(events) == {"<2": 1, "2-3": 1, "3-4": 2, "4-5": 0, "5+": 1}, "bands")
    check(len(filter_events(events, 3.0)) == 3, "min magnitude filter")

    empty = compute_stats([])
    check(empty["count"] == 0 and empty["max_mag"] is None, "empty stats")
    check("No events" in format_stats(empty), "empty format")

    try:
        normalize_events({"nope": 1})
        check(False, "missing features should raise")
    except QuakeError:
        check(True, "missing features raises")
    try:
        feed_url("decade")
        check(False, "unknown feed should raise")
    except QuakeError:
        check(True, "unknown feed raises")

    context = build_context(events, stats)
    check("M5.1" in context and "Hawaii" in context, "context content")
    check("DATA:" in summary_prompt(context), "summary prompt")
    check("QUESTION: why" in question_prompt(context, "why"), "question prompt")

    server = HTTPServer(("127.0.0.1", 0), _FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:%d/api/generate" % server.server_port
        check(ollama_generate("hi", url=url) == "stub briefing", "ollama stub over HTTP")
    finally:
        server.shutdown()
        server.server_close()
    try:
        ollama_generate("hi", url="http://127.0.0.1:1/api/generate", timeout=2)
        check(False, "dead Ollama should raise")
    except QuakeError as exc:
        check("ollama serve" in str(exc), "dead Ollama gives a helpful error")

    args = argparse.Namespace(feed="day", min_mag=None, model="x", question="q", top=3)
    with mock.patch(__name__ + ".fetch_json", return_value=SAMPLE_FEED):
        check(cmd_summary(args, generate=lambda p, model: "fake briefing") == 0, "summary command")
        check(cmd_ask(args, generate=lambda p, model: "fake answer") == 0, "ask command")
        check(cmd_regions(args) == 0, "regions command")
        check(cmd_strongest(args) == 0, "strongest command")

    print("selftest passed (%d checks)" % checks)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="quakewatch", description="Earthquake briefings from USGS data and local Ollama.")
    sub = parser.add_subparsers(dest="command")

    def common(p):
        p.add_argument("--feed", choices=sorted(FEEDS), default="day", help="time window")
        p.add_argument("--min-mag", type=float, default=None, help="minimum magnitude")
        p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")

    p = sub.add_parser("summary", help="stats plus an Ollama briefing")
    common(p)
    p.set_defaults(func=cmd_summary)
    p = sub.add_parser("ask", help="ask a question about the feed")
    p.add_argument("question")
    common(p)
    p.set_defaults(func=cmd_ask)
    p = sub.add_parser("regions", help="regions ranked by quake count")
    common(p)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_regions)
    p = sub.add_parser("strongest", help="strongest events")
    common(p)
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(func=cmd_strongest)
    p = sub.add_parser("selftest", help="offline self-test")
    p.set_defaults(func=lambda args: selftest())
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except QuakeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
