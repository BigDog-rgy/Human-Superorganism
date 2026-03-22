"""
World Leaders Scraper — via Wikidata SPARQL
Fetches current heads of state for all sovereign states.

Outputs: world_leaders_candidates.json

Wikidata properties:
  P35       = head of state
  P31       = instance of
  P297      = ISO 3166-1 alpha-2 country code
  Q3624078  = sovereign state
  Q5        = human
"""

import json
import time
import urllib.request
import urllib.parse
from datetime import datetime

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

COMBINED_QUERY = """
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


def run_sparql(query, label, retries=5):
    """Run a SPARQL query with exponential backoff."""
    params = urllib.parse.urlencode({"query": query, "format": "json"})
    url = f"{SPARQL_ENDPOINT}?{params}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "PrimeMoverTracker/1.0 (academic research)",
        "Accept": "application/sparql-results+json"
    })

    print(f"  Querying: {label}...")
    for attempt in range(1, retries + 1):
        try:
            timeout = 90 + (attempt - 1) * 30   # 90s, 120s, 150s, 180s, 210s
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            results = data["results"]["bindings"]
            print(f"    -> {len(results)} results")
            return results
        except Exception as e:
            wait = 30 * (2 ** (attempt - 1))  # 30s, 60s, 120s, 240s, 480s
            print(f"    x Attempt {attempt}/{retries} failed: {type(e).__name__}: {e}")
            if attempt < retries:
                print(f"      Waiting {wait}s before retry (exponential backoff)...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Query '{label}' failed after {retries} attempts") from e


def extract(row, key):
    return row.get(key, {}).get("value", "")


def main():
    # --- Single query for heads of state ---
    rows = run_sparql(COMBINED_QUERY, "heads of state")

    # --- Build per-country map ---
    country_map = {}

    for row in rows:
        iso = extract(row, "countryISO")
        country = extract(row, "countryLabel")
        country_key = iso or country

        if country_key not in country_map:
            country_map[country_key] = {
                "country": country,
                "country_iso2": iso,
                "country_qid": extract(row, "country").replace("http://www.wikidata.org/entity/", ""),
                "head_of_state": []
            }

        person = {
            "name": extract(row, "personLabel"),
            "wikidata": extract(row, "wikidata_id"),
            "description": extract(row, "personDesc")
        }

        role_list = country_map[country_key]["head_of_state"]
        if not any(p["wikidata"] == person["wikidata"] for p in role_list):
            role_list.append(person)

    # --- Flatten to candidate pool, dedup by wikidata ID ---
    seen = set()
    candidates = []

    for entry in country_map.values():
        for person in entry["head_of_state"]:
            wid = person["wikidata"]
            if not wid or wid in seen:
                continue
            seen.add(wid)

            candidates.append({
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
                "wikidata": wid,
                "elo_score": 1000,
                "elo_matches": 0,
                "metadata": {}
            })

    # --- Stats ---
    by_subcat = {}
    for c in candidates:
        by_subcat[c["subcategory"]] = by_subcat.get(c["subcategory"], 0) + 1

    print(f"\n  Countries mapped: {len(country_map)}")
    print(f"  Unique candidates: {len(candidates)}")
    for k, v in sorted(by_subcat.items()):
        print(f"    {k}: {v}")

    # --- Save ---
    out = {
        "generated": datetime.now().isoformat(),
        "source": "Wikidata SPARQL (P35=head of state)",
        "total_countries": len(country_map),
        "total_candidates": len(candidates),
        "candidates": candidates,
        "country_detail": list(country_map.values())
    }

    outfile = "world_leaders_candidates.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(candidates)} world leaders to {outfile}")


if __name__ == "__main__":
    main()