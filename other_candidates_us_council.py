"""
Other Candidates US Council - Multi-Model Consensus System
Queries Claude, Grok, ChatGPT, and Gemini to construct consensus lists
of the top 20 US prime movers across 5 specialized categories.

Designed to fill gaps not covered by existing billionaire and congressional
tracking lists. Each category explicitly excludes billionaires and elected
officials, focusing on the influential figures who move systems without
appearing on those lists.

Categories:
  academics_intellectuals — policy-shaping intellectuals, economists,
      scientists, and scholars whose ideas actively drive US outcomes
  religious_leaders       — US-based religious figures with demonstrable
      influence on American political behavior and legislation
  media_cultural          — operational media figures (editors, hosts,
      anchors) with genuine agenda-setting power; not owner-billionaires
  ngo_foundation          — heads of foundations, think tanks, and legal
      movement builders who translate funding into US policy outcomes
  political_operators     — lobbyists, dark money network operators,
      K Street fixers, and behind-the-scenes power brokers
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
from google import genai

# Load environment variables
load_dotenv()

MODEL_NAMES = ['claude', 'chatgpt', 'grok', 'gemini']

CATEGORIES = {
    'academics_intellectuals': {
        'label': 'Academics & Public Intellectuals',
        'description': (
            'US-based economists, scientists, legal scholars, historians, and '
            'public intellectuals whose ideas actively shape domestic policy, '
            'legislation, or mainstream public debate right now — prioritize '
            'real-world impact over academic prestige. Think: economic advisers '
            'whose frameworks move policy, scientists whose work drives '
            'regulation, public philosophers who shift how Americans think. '
            'Exclude sitting government officials and billionaires '
            '(tracked separately).'
        ),
    },
    'religious_leaders': {
        'label': 'Religious Leaders',
        'description': (
            'US-based religious figures with demonstrable influence over '
            'American political behavior, legislation, or social norms — '
            'not just congregation size or denominational prestige. '
            'Prioritize Evangelical pastors and political organizers who '
            'mobilize voters or lobby legislation, prominent US Catholic '
            'bishops and cardinals shaping domestic policy, influential '
            'rabbis, imams, and leaders of other traditions who actively '
            'shape American civic life. All figures must be US-based and '
            'primarily operate within the US — exclude global-only figures '
            'such as the Pope or Dalai Lama.'
        ),
    },
    'media_cultural': {
        'label': 'Media & Cultural Figures',
        'description': (
            'US-based journalists, editors, talk radio hosts, podcast hosts, '
            'cable news anchors, and cultural figures who genuinely set the '
            'national agenda. Focus on operational figures whose editorial '
            'judgment, on-air reach, or platform access shapes what Americans '
            'think and discuss — not entertainment celebrities and not '
            'media-owner billionaires (tracked separately). Prioritize '
            'those who drive political narratives, control major editorial '
            'decisions, or have demonstrated measurable influence on public '
            'opinion and policy debate.'
        ),
    },
    'ngo_foundation': {
        'label': 'NGOs, Foundations, Think Tanks & Legal Movement Builders',
        'description': (
            'US-based heads and key principals of major foundations, think '
            'tanks, advocacy organizations, and legal movement builders who '
            'translate funding and ideas into real domestic policy outcomes. '
            'Include figures running institutions like the Federalist Society, '
            'ACLU, Heritage Foundation, Brookings, major private foundations, '
            'and influential cross-sector advocacy groups. Exclude the '
            'billionaire donors themselves (tracked separately); focus on '
            'the institutional operators and strategists who direct these '
            'organizations and their policy impact.'
        ),
    },
    'political_operators': {
        'label': 'Political Operators & Power Brokers',
        'description': (
            'US-based fixers, lobbyists, dark money network operators, '
            'major political strategists, super PAC controllers, and '
            'behind-the-scenes power brokers who demonstrably move systems '
            'without holding elected office. Includes K Street lobbying '
            'principals, major political fundraisers and bundlers, '
            'influential party strategists, dark money conduit operators, '
            'and those who connect money to policy outcomes in ways that '
            'are real but rarely headline. Exclude elected and appointed '
            'officials (tracked separately) and billionaires '
            '(tracked separately); focus on the intermediary operators '
            'who make the system function behind the scenes.'
        ),
    },
    'officials_judiciary': {
        'label': 'Appointed Officials, Federal Judiciary & Governors',
        'description': (
            'US formal-authority holders whose power derives from office or '
            'appointment rather than wealth or a congressional seat — the '
            'tier of institutional power not captured by the billionaire '
            'or Congress trackers. Include: sitting and recently departed '
            'Cabinet secretaries and major agency heads whose regulatory or '
            'enforcement decisions shape US domestic life (Attorney General, '
            'Treasury Secretary, EPA Administrator, FTC/FCC/SEC chairs, FBI '
            'Director, Federal Reserve leadership); federal judges with '
            'demonstrably outsized national impact, prioritizing sitting '
            'Supreme Court justices and circuit judges whose rulings are '
            'actively reshaping law; sitting US governors with genuine '
            'national policy reach; and influential state Attorneys General '
            'coordinating multi-state legal action with national consequence. '
            'Exclude sitting members of Congress (tracked separately), '
            'billionaires (tracked separately), and behind-the-scenes '
            'operators without formal office (tracked in political_operators). '
            'Rank by current real-world impact of their formal authority '
            'as of February 2026.'
        ),
    },
}


def build_category_prompt(category_key: str, category_info: dict) -> str:
    return f"""Today is February 2026.

You are tasked with identifying the 20 most influential living figures in the \
United States in the category of **{category_info['label']}** AS OF TODAY.

Category definition: {category_info['description']}

IMPORTANT CONSTRAINTS:
- US focus: individuals must primarily operate within the US or have their \
greatest influence on US domestic outcomes
- Must be currently alive and active as of February 2026
- Must have demonstrable, current, real-world influence — not just historical \
reputation or institutional prestige
- Do not include figures primarily known for top-tier political or tech-CEO \
roles (e.g. sitting Cabinet members, Fortune 10 CEOs) unless they genuinely \
and distinctly fit this category's definition
- Use your most up-to-date knowledge

Please provide exactly 20 names, ranked by current influence (most influential \
first). For each person provide:
1. Full name
2. Brief role or affiliation
3. One-sentence justification for their inclusion

Format your response as a JSON array with no extra text:
[
  {{"rank": 1, "name": "Full Name", "role": "Brief role", "justification": "One sentence."}}
]"""


class OtherCandidatesUSCouncil:
    """Orchestrates multiple LLM APIs across 5 specialized US prime-mover categories."""

    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.xai_client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )
        self.gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    # ------------------------------------------------------------------ #
    # Low-level query helpers                                              #
    # ------------------------------------------------------------------ #

    def _extract_json_array(self, text: str) -> list:
        start = text.find('[')
        end = text.rfind(']') + 1
        return json.loads(text[start:end])

    def _extract_json_object(self, text: str) -> dict:
        start = text.find('{')
        end = text.rfind('}') + 1
        return json.loads(text[start:end])

    def _query_model(self, model_name: str, prompt: str,
                     max_tokens: int = 4096, temperature: float = 0.7) -> str:
        if model_name == 'claude':
            message = self.anthropic_client.messages.create(
                model="claude-opus-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        elif model_name == 'chatgpt':
            resp = self.openai_client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": "You are an expert analyst of US influence and power dynamics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return resp.choices[0].message.content

        elif model_name == 'grok':
            resp = self.xai_client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[
                    {"role": "system", "content": "You are an expert analyst of US influence and power dynamics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            return resp.choices[0].message.content

        elif model_name == 'gemini':
            resp = self.gemini_client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=prompt
            )
            return resp.text

        raise ValueError(f"Unknown model: {model_name}")

    # ------------------------------------------------------------------ #
    # Stage 1: Independent responses                                       #
    # ------------------------------------------------------------------ #

    def stage_1_query_all(self, category_key: str,
                          category_info: dict) -> Dict[str, List[Dict[str, Any]]]:
        prompt = build_category_prompt(category_key, category_info)
        results = {}

        for name in MODEL_NAMES:
            print(f"    Querying {name}...")
            try:
                text = self._query_model(name, prompt, max_tokens=4096, temperature=0.7)
                results[name] = self._extract_json_array(text)
                print(f"      → {len(results[name])} entries")
            except Exception as e:
                print(f"      ✗ {name} failed: {e}")
                results[name] = []

        return results

    # ------------------------------------------------------------------ #
    # Stage 2: Peer review                                                 #
    # ------------------------------------------------------------------ #

    def stage_2_peer_review(self, all_responses: Dict[str, List[Dict[str, Any]]],
                             category_label: str) -> Dict[str, Any]:
        peer_reviews = {}

        for reviewer in MODEL_NAMES:
            if not all_responses.get(reviewer):
                continue

            print(f"    Peer review from {reviewer}...")

            # Assign anonymous labels to the other three models
            other_responses: Dict[str, List] = {}
            label_map: Dict[str, str] = {}
            labels = ['Model A', 'Model B', 'Model C']
            idx = 0
            for name in MODEL_NAMES:
                if name != reviewer and all_responses.get(name):
                    other_responses[labels[idx]] = all_responses[name]
                    label_map[labels[idx]] = name
                    idx += 1

            review_prompt = (
                f"You previously provided a list of the 20 most influential US figures "
                f"in the category '{category_label}'.\n\n"
                "Now review these three independent responses to the same question "
                "(labeled Model A, Model B, Model C):\n\n"
            )
            for label, response in other_responses.items():
                review_prompt += f"### {label}'s Response:\n"
                for person in response[:5]:
                    review_prompt += f"{person['rank']}. {person['name']} - {person['justification']}\n"
                review_prompt += f"...and {len(response) - 5} more\n\n"

            review_prompt += (
                "Rank these three responses from best to worst based on:\n"
                "1. Accuracy of current information (February 2026)\n"
                "2. Quality of reasoning and justifications\n"
                f"3. Adherence to the '{category_label}' category definition\n"
                "4. Comprehensiveness and insight\n\n"
                "Return only a JSON object with no extra text:\n"
                '{"rankings": ['
                '{"model": "Model A", "rank": 1, "reasoning": "brief explanation"}, '
                '{"model": "Model B", "rank": 2, "reasoning": "brief explanation"}, '
                '{"model": "Model C", "rank": 3, "reasoning": "brief explanation"}'
                ']}'
            )

            try:
                text = self._query_model(reviewer, review_prompt,
                                         max_tokens=2048, temperature=0.3)
                review_data = self._extract_json_object(text)
                for ranking in review_data['rankings']:
                    ranking['actual_model'] = label_map.get(ranking['model'], 'unknown')
                peer_reviews[reviewer] = review_data
                print(f"      ✓ done")
            except Exception as e:
                print(f"      ✗ parse error: {e}")
                peer_reviews[reviewer] = None

        return peer_reviews

    # ------------------------------------------------------------------ #
    # Stage 3: Chairman synthesis                                          #
    # ------------------------------------------------------------------ #

    def stage_3_chairman_synthesis(self,
                                    all_responses: Dict[str, List[Dict[str, Any]]],
                                    peer_reviews: Dict[str, Any],
                                    category_info: dict) -> List[Dict[str, Any]]:
        synthesis_prompt = (
            f"You are the Chairman of an LLM Council tasked with identifying the "
            f"20 most influential living US figures in the category "
            f"**{category_info['label']}** as of February 2026.\n\n"
            f"Category definition: {category_info['description']}\n\n"
            "You have received independent responses from four leading AI models "
            "and peer reviews where each model ranked the others.\n\n"
            "## Original Responses:\n\n"
        )

        for model_name, response in all_responses.items():
            if not response:
                continue
            synthesis_prompt += f"### {model_name.upper()}'s List:\n"
            for person in response:
                synthesis_prompt += (
                    f"{person['rank']}. {person['name']} "
                    f"({person.get('role', 'N/A')}) — "
                    f"{person.get('justification', 'N/A')}\n"
                )
            synthesis_prompt += "\n"

        synthesis_prompt += "\n## Peer Review Rankings:\n\n"
        for reviewer, review in peer_reviews.items():
            if not review:
                continue
            synthesis_prompt += f"### {reviewer.upper()}'s rankings:\n"
            for ranking in review['rankings']:
                synthesis_prompt += (
                    f"{ranking['rank']}. {ranking['actual_model']} — "
                    f"{ranking.get('reasoning', '')}\n"
                )
            synthesis_prompt += "\n"

        synthesis_prompt += (
            "\n## Your Task:\n\n"
            "Synthesize these inputs into a single, definitive list of the "
            "20 most influential living US figures in this category.\n\n"
            "Consider:\n"
            "1. Consensus across models (use judgment, not just averaging)\n"
            "2. Peer review feedback on quality and accuracy\n"
            "3. Current events as of February 2026\n"
            f"4. Strict adherence to the '{category_info['label']}' definition\n"
            "5. No duplicates — consolidate name variations into single entries\n"
            "6. US focus: domestic influence over American life\n\n"
            "Provide exactly 20 names ranked by current influence. "
            "Return only a JSON array with no extra text:\n"
            "[\n"
            "  {\n"
            '    "rank": 1,\n'
            '    "name": "Full Name",\n'
            '    "role": "Brief role or affiliation",\n'
            '    "justification": "One clear sentence on current influence",\n'
            '    "consensus_notes": "Brief note on model agreement/disagreement"\n'
            "  }\n"
            "]\n"
        )

        try:
            message = self.anthropic_client.messages.create(
                model="claude-opus-4-6",
                max_tokens=8192,
                messages=[{"role": "user", "content": synthesis_prompt}]
            )
            text = message.content[0].text
            final_list = self._extract_json_array(text)
            print(f"    ✓ Chairman synthesis complete ({len(final_list)} entries)")
            return final_list
        except Exception as e:
            print(f"    ✗ Chairman synthesis failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Per-category orchestration                                           #
    # ------------------------------------------------------------------ #

    def run_category(self, category_key: str, category_info: dict) -> dict:
        label = category_info['label']
        print(f"\n{'=' * 60}")
        print(f"CATEGORY: {label.upper()}")
        print(f"{'=' * 60}")

        print("\n  -- Stage 1: Independent Responses --")
        all_responses = self.stage_1_query_all(category_key, category_info)

        valid = sum(1 for r in all_responses.values() if r)
        print(f"\n  Stage 1 complete: {valid}/4 models responded")

        print("\n  -- Stage 2: Peer Review --")
        peer_reviews = self.stage_2_peer_review(all_responses, label)

        print("\n  -- Stage 3: Chairman Synthesis --")
        final_list = self.stage_3_chairman_synthesis(
            all_responses, peer_reviews, category_info
        )

        return {
            'stage_1_individual_responses': all_responses,
            'stage_2_peer_reviews': peer_reviews,
            'stage_3_final_list': final_list,
        }

    # ------------------------------------------------------------------ #
    # Main run                                                             #
    # ------------------------------------------------------------------ #

    def run(self):
        print("=" * 60)
        print("OTHER CANDIDATES US COUNCIL")
        print("Five-Category Prime Mover Identification (US Focus)")
        print("Three-Stage Process (Karpathy Method)")
        print("=" * 60)

        all_category_results = {}
        for category_key, category_info in CATEGORIES.items():
            all_category_results[category_key] = self.run_category(
                category_key, category_info
            )

        self.save_results(all_category_results)

        print("\n" + "=" * 60)
        print("Council session complete!")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #

    def save_results(self, all_category_results: dict):
        output = {
            'metadata': {
                'date': '2026-02-17',
                'focus': 'US prime movers — specialized gap-filling categories (billionaires and Congress tracked separately)',
                'categories': list(CATEGORIES.keys()),
                'models_queried': MODEL_NAMES,
                'methodology': (
                    'Three-stage LLM Council (Karpathy method): '
                    'Stage 1 - Independent responses, '
                    'Stage 2 - Peer review, '
                    'Stage 3 - Chairman synthesis by Claude Opus 4.6'
                ),
            }
        }
        for category_key, result in all_category_results.items():
            output[category_key] = result

        output_file = 'other_candidates_us.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {output_file}")

        self.create_markdown_summary(all_category_results)

    def create_markdown_summary(self, all_category_results: dict):
        md = "# LLM Council: US Prime Movers — Gap-Filling Categories (2026)\n\n"
        md += "*Generated: February 2026*\n\n"
        md += (
            "> These lists are designed to surface influential Americans **not** "
            "captured by existing billionaire or congressional tracking lists. "
            "Each category explicitly excludes billionaires and elected/appointed "
            "officials.\n\n"
        )
        md += "## Methodology: Three-Stage Council Process (Karpathy Method)\n\n"
        md += (
            "**Stage 1 — Independent Responses**: Four leading LLMs (Claude Opus 4.6, "
            "GPT-5.2, Grok 4.1, Gemini 3 Pro) independently generated top-20 lists "
            "per category.\n\n"
            "**Stage 2 — Peer Review**: Each model anonymously reviewed and ranked "
            "the other models' responses for accuracy, reasoning quality, and "
            "category adherence.\n\n"
            "**Stage 3 — Chairman Synthesis**: Claude Opus 4.6, serving as Chairman, "
            "synthesized all inputs into the final authoritative lists.\n\n"
            "---\n\n"
        )

        for category_key, result in all_category_results.items():
            info = CATEGORIES[category_key]
            final_list = result.get('stage_3_final_list', [])
            md += f"# {info['label']}\n\n"
            md += f"*{info['description']}*\n\n"
            for person in final_list:
                md += f"## {person['rank']}. {person['name']}\n\n"
                md += f"**Role**: {person.get('role', 'N/A')}\n\n"
                md += f"**Justification**: {person.get('justification', 'N/A')}\n\n"
                if person.get('consensus_notes'):
                    md += f"**Consensus Notes**: {person['consensus_notes']}\n\n"
                md += "---\n\n"

        output_file = 'other_candidates_us_summary.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✓ Summary saved to {output_file}")


if __name__ == "__main__":
    council = OtherCandidatesUSCouncil()
    council.run()
