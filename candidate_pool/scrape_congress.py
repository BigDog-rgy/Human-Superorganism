"""
Congress Legislators Scraper
Source: https://unitedstates.github.io/congress-legislators/legislators-current.json
Outputs: congress_candidates.json — normalized candidate pool entries
"""

import json
import urllib.request
from datetime import datetime

URL = "https://unitedstates.github.io/congress-legislators/legislators-current.json"

def fetch_legislators():
    print(f"Fetching from {URL}...")
    with urllib.request.urlopen(URL) as response:
        data = json.loads(response.read().decode())
    print(f"  → {len(data)} legislators fetched")
    return data

def get_current_term(terms):
    """Return the most recent term."""
    return terms[-1] if terms else {}

def normalize(legislator):
    name = legislator.get("name", {})
    bio = legislator.get("bio", {})
    term = get_current_term(legislator.get("terms", []))
    ids = legislator.get("id", {})

    full_name = name.get("official_full") or f"{name.get('first', '')} {name.get('last', '')}".strip()
    chamber = term.get("type", "")  # "sen" or "rep"
    state = term.get("state", "")
    party = term.get("party", "")
    start = term.get("start", "")

    # Seniority proxy: years in current term
    try:
        years_in_term = (datetime.now() - datetime.strptime(start, "%Y-%m-%d")).days / 365
    except:
        years_in_term = None

    return {
        "name": full_name,
        "source": "us_congress",
        "scope": "us",
        "category": "political",
        "subcategory": "senator" if chamber == "sen" else "representative",
        "chamber": "Senate" if chamber == "sen" else "House",
        "state": state,
        "party": party,
        "term_start": start,
        "years_in_current_term": round(years_in_term, 1) if years_in_term else None,
        "bioguide_id": ids.get("bioguide"),
        "wikipedia": ids.get("wikipedia"),
        "wikidata": ids.get("wikidata"),
        "elo_score": 1000,  # default starting ELO
        "elo_matches": 0,
        "metadata": {
            "gender": bio.get("gender"),
            "born": bio.get("birthday"),
            "official_full": name.get("official_full"),
        }
    }

def main():
    legislators = fetch_legislators()
    candidates = [normalize(l) for l in legislators]

    # Summary stats
    senators = [c for c in candidates if c["chamber"] == "Senate"]
    reps = [c for c in candidates if c["chamber"] == "House"]
    print(f"  → {len(senators)} senators, {len(reps)} representatives")

    # Party breakdown
    from collections import Counter
    parties = Counter(c["party"] for c in candidates)
    print(f"  → Party breakdown: {dict(parties)}")

    # Save
    out = {
        "generated": datetime.now().isoformat(),
        "source_url": URL,
        "total": len(candidates),
        "candidates": candidates
    }

    outfile = "congress_candidates.json"
    with open(outfile, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved {len(candidates)} candidates to {outfile}")

if __name__ == "__main__":
    main()