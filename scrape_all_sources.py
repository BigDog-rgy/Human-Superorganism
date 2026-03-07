"""
Standalone scraper for all Prime Mover Tracker sources.

This script fetches:
1. Forbes 400
2. Forbes Billionaires
3. Current U.S. Congress legislators
4. Current world heads of state via Wikidata SPARQL

Each source is written to its own JSON output file.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request


BASE_DIR = Path(__file__).resolve().parent

FORBES_SOURCES = [
    {
        "label": "Forbes 400",
        "url": "https://www.forbes.com/forbesapi/person/forbes-400/2024/position/true.json?fields=personName,rank&limit=400",
        "output_file": "forbes_400_names.json",
    },
    {
        "label": "Forbes Billionaires",
        "url": "https://www.forbes.com/forbesapi/person/billionaires/2024/position/true.json?fields=personName,rank&limit=2500",
        "output_file": "forbes_billionaires_names.json",
    },
]

CONGRESS_URL = "https://unitedstates.github.io/congress-legislators/legislators-current.json"

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WORLD_LEADERS_QUERY = """
SELECT DISTINCT ?country ?countryLabel ?countryISO ?person ?personLabel ?personDesc ?wikidata_id
WHERE {
  ?country wdt:P31 wd:Q3624078 .
  ?country wdt:P35 ?person .
  OPTIONAL { ?country wdt:P297 ?countryISO . }
  ?person wdt:P31 wd:Q5 .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
  BIND(STRAFTER(STR(?person), "http://www.wikidata.org/entity/") AS ?wikidata_id)
}
ORDER BY ?countryLabel
"""


def output_path(filename: str) -> Path:
    return BASE_DIR / filename


def fetch_json(url: str, timeout: int = 30, headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def save_json(filename: str, payload) -> None:
    path = output_path(filename)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def fetch_forbes_names(api_url: str) -> list[str]:
    payload = fetch_json(api_url, timeout=20)
    people = payload.get("personList", {}).get("personsLists", [])
    names = [p.get("personName", "").strip() for p in people if isinstance(p, dict)]
    return [name for name in names if name]


def scrape_forbes() -> None:
    for source in FORBES_SOURCES:
        print(f"Fetching {source['label']}...")
        names = fetch_forbes_names(source["url"])
        save_json(source["output_file"], names)
        print(f"  Saved {len(names)} names to {source['output_file']}")


def fetch_legislators() -> list[dict]:
    print(f"Fetching Congress data from {CONGRESS_URL}...")
    data = fetch_json(CONGRESS_URL, timeout=30)
    print(f"  Retrieved {len(data)} legislators")
    return data


def get_current_term(terms: list[dict]) -> dict:
    return terms[-1] if terms else {}


def normalize_legislator(legislator: dict) -> dict:
    name = legislator.get("name", {})
    bio = legislator.get("bio", {})
    term = get_current_term(legislator.get("terms", []))
    ids = legislator.get("id", {})

    full_name = name.get("official_full") or f"{name.get('first', '')} {name.get('last', '')}".strip()
    chamber = term.get("type", "")
    start = term.get("start", "")

    try:
        years_in_term = (datetime.now() - datetime.strptime(start, "%Y-%m-%d")).days / 365
    except ValueError:
        years_in_term = None

    return {
        "name": full_name,
        "source": "us_congress",
        "scope": "us",
        "category": "political",
        "subcategory": "senator" if chamber == "sen" else "representative",
        "chamber": "Senate" if chamber == "sen" else "House",
        "state": term.get("state", ""),
        "party": term.get("party", ""),
        "term_start": start,
        "years_in_current_term": round(years_in_term, 1) if years_in_term is not None else None,
        "bioguide_id": ids.get("bioguide"),
        "wikipedia": ids.get("wikipedia"),
        "wikidata": ids.get("wikidata"),
        "elo_score": 1000,
        "elo_matches": 0,
        "metadata": {
            "gender": bio.get("gender"),
            "born": bio.get("birthday"),
            "official_full": name.get("official_full"),
        },
    }


def scrape_congress() -> None:
    legislators = fetch_legislators()
    candidates = [normalize_legislator(legislator) for legislator in legislators]
    parties = Counter(candidate["party"] for candidate in candidates)

    payload = {
        "generated": datetime.now().isoformat(),
        "source_url": CONGRESS_URL,
        "total": len(candidates),
        "candidates": candidates,
    }

    save_json("congress_candidates.json", payload)
    print(f"  Party breakdown: {dict(parties)}")
    print("  Saved Congress candidates to congress_candidates.json")


def run_sparql(query: str, label: str, retries: int = 5) -> list[dict]:
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    url = f"{SPARQL_ENDPOINT}?{params}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PrimeMoverTracker/1.0 (academic research)",
            "Accept": "application/sparql-results+json",
        },
    )

    print(f"Querying {label}...")
    for attempt in range(1, retries + 1):
        try:
            timeout = 90 + (attempt - 1) * 30
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            results = data["results"]["bindings"]
            print(f"  Retrieved {len(results)} rows")
            return results
        except Exception as exc:
            wait = 30 * (2 ** (attempt - 1))
            print(f"  Attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}")
            if attempt == retries:
                raise RuntimeError(f"Query '{label}' failed after {retries} attempts") from exc
            print(f"  Waiting {wait}s before retry...")
            time.sleep(wait)


def extract_value(row: dict, key: str) -> str:
    return row.get(key, {}).get("value", "")


def scrape_world_leaders() -> None:
    rows = run_sparql(WORLD_LEADERS_QUERY, "world heads of state")
    country_map: dict[str, dict] = {}

    for row in rows:
        iso = extract_value(row, "countryISO")
        country = extract_value(row, "countryLabel")
        country_key = iso or country

        if country_key not in country_map:
            country_map[country_key] = {
                "country": country,
                "country_iso2": iso,
                "country_qid": extract_value(row, "country").replace("http://www.wikidata.org/entity/", ""),
                "head_of_state": [],
            }

        person = {
            "name": extract_value(row, "personLabel"),
            "wikidata": extract_value(row, "wikidata_id"),
            "description": extract_value(row, "personDesc"),
        }

        people = country_map[country_key]["head_of_state"]
        if not any(existing["wikidata"] == person["wikidata"] for existing in people):
            people.append(person)

    seen: set[str] = set()
    candidates = []

    for entry in country_map.values():
        for person in entry["head_of_state"]:
            wikidata_id = person["wikidata"]
            if not wikidata_id or wikidata_id in seen:
                continue
            seen.add(wikidata_id)
            candidates.append(
                {
                    "name": person["name"],
                    "source": "wikidata_world_leaders",
                    "scope": "global",
                    "category": "political",
                    "subcategory": "head_of_state",
                    "roles": ["head_of_state"],
                    "country": entry["country"],
                    "country_iso2": entry["country_iso2"],
                    "country_qid": entry["country_qid"],
                    "description": person["description"],
                    "wikidata": wikidata_id,
                    "elo_score": 1000,
                    "elo_matches": 0,
                    "metadata": {},
                }
            )

    payload = {
        "generated": datetime.now().isoformat(),
        "source": "Wikidata SPARQL (P35=head of state)",
        "total_countries": len(country_map),
        "total_candidates": len(candidates),
        "candidates": candidates,
        "country_detail": list(country_map.values()),
    }

    save_json("world_leaders_candidates.json", payload)
    print(f"  Saved {len(candidates)} world leaders to world_leaders_candidates.json")


def main() -> None:
    started_at = time.time()
    print(f"Running standalone scraper from {BASE_DIR}...")

    print("\n=== Forbes ===")
    scrape_forbes()

    print("\n=== Congress ===")
    scrape_congress()

    print("\n=== World Leaders ===")
    scrape_world_leaders()

    print(f"\nAll sources scraped in {time.time() - started_at:.1f}s")


if __name__ == "__main__":
    main()
