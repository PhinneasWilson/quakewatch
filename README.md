# quakewatch

Plain-English earthquake briefings and Q&A, built on free public data and a local Ollama model.

quakewatch downloads the USGS earthquake GeoJSON feed (no API key), computes the statistics itself (count, magnitude, depth, busiest regions), then asks a local Ollama model to explain what the numbers mean or answer your question about them. It only handles earthquake data. No people, no photos.

## Requirements

- Python 3.8+ (standard library only, no pip packages)
- Ollama running locally for `summary` and `ask`: `ollama serve` and `ollama pull llama3.2`

## Usage

```
python3 quakewatch.py summary --feed day --min-mag 2.5
python3 quakewatch.py ask "Where was the deepest quake?" --feed week
python3 quakewatch.py regions --feed week --top 15
python3 quakewatch.py strongest --feed month --top 5
python3 quakewatch.py selftest
```

Feeds: hour, day, week, month. Use `--model` to pick another Ollama model.

## Self-test

`python3 quakewatch.py selftest` runs offline with a small sample feed and a stub Ollama server on localhost. It checks parsing, stats, region grouping, prompts, error handling and every command.

Data: USGS Earthquake Hazards Program, https://earthquake.usgs.gov/earthquakes/feed/
