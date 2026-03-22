import json
from urllib.request import urlopen


SOURCES = [
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


def fetch_forbes_names(api_url: str) -> list[str]:
    with urlopen(api_url, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    people = payload.get("personList", {}).get("personsLists", [])
    names = [p.get("personName", "").strip() for p in people if isinstance(p, dict)]
    return [n for n in names if n]


def main():
    for source in SOURCES:
        print(f"Fetching {source['label']} from API...")
        names = fetch_forbes_names(source["url"])

        with open(source["output_file"], "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2, ensure_ascii=False)

        print(f"Found {len(names)} names.")
        print(f"Saved to {source['output_file']}")
        print("Preview:")
        print(json.dumps(names[:10], indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
