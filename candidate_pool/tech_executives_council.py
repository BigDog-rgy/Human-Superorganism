"""
Tech Executives Council — Gap-Filling Category
Three-Stage LLM Council (Karpathy Method)

Surfaces AI/tech company founders and C-suite operators who are genuinely
among the world's most influential people but fall through the cracks of
all automated sources:

  - Not on Forbes lists: privately-held companies (OpenAI, Anthropic, SpaceX
    pre-IPO equity) meant people like Sam Altman and Dario Amodei were not
    captured, even though they are almost certainly billionaires in practice.
    Forbes net worth estimates lag private valuations significantly.
  - Not in Congress or world leaders: they hold no formal government office.
  - Not in existing council categories: the original categories (academics,
    religious, media, NGO, political operators, officials) had no tech bucket.

Produces two lists — US-focused and Global — in a single output file.
Output: tech_executives_results.json

Usage:
  python tech_executives_council.py
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
from google import genai

load_dotenv()

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_NAMES = ['claude', 'chatgpt', 'grok', 'gemini']

CATEGORIES = {
    'us_tech_executives': {
        'label':   'US Tech & AI Executives',
        'scope':   'us',
        'description': (
            'Founders, CEOs, and C-suite decision-makers at US-headquartered AI and '
            'technology companies whose operational decisions today are actively '
            'reshaping economic, geopolitical, or social power structures at scale. '
            'Focus on people whose *current* control over platforms, models, capital '
            'allocation, or standards-setting has outsized real-world consequences — '
            'not historical founders who have stepped back from operations. '
            '\n\n'
            'CRITICAL — these figures are NOT captured by other lists in this system:\n'
            '  • Forbes billionaires list: people like Sam Altman (OpenAI) and '
            'Dario Amodei (Anthropic) almost certainly have net worth above the '
            'Forbes threshold given their equity stakes in companies valued at '
            '$150B+ and $60B+ respectively, but Forbes net worth estimates for '
            'private-company equity routinely lag actual valuations by years. '
            'Include figures in this situation — likely billionaires by any real '
            'measure, simply not yet captured by the Forbes methodology.\n'
            '  • Congress/elected officials: these are private-sector figures.\n'
            '  • Existing council categories: academics, media, NGO, religious, '
            'and political operator categories were explicitly defined to exclude '
            'tech-CEO roles.\n\n'
            'DO exclude: figures already confirmed on the Forbes 400 or Forbes '
            'Global Billionaires list (e.g. Elon Musk, Jeff Bezos, Mark Zuckerberg, '
            'Larry Page, Sergey Brin, Michael Dell — already tracked). Focus on '
            'the operational layer just below or adjacent to the mega-billionaires: '
            'the CEOs running the AI labs, the chip architects, the platform '
            'operators, the VC gatekeepers who decide which technologies get '
            'built. Rank by current real-world impact as of March 2026.'
        ),
    },
    'global_tech_executives': {
        'label':   'Global Tech & AI Executives',
        'scope':   'global',
        'description': (
            'Founders, CEOs, and C-suite decision-makers at AI and technology '
            'companies worldwide whose operational decisions are actively '
            'reshaping economic, geopolitical, or social power at a global scale. '
            'Include non-US figures (e.g. Chinese tech leaders, European platform '
            'regulators with executive-equivalent power, Asian semiconductor '
            'executives) alongside US figures not yet on Forbes lists. '
            '\n\n'
            'CRITICAL — these figures are NOT captured by other lists in this system:\n'
            '  • Forbes Global Billionaires list: people like Sam Altman (OpenAI) '
            'and Dario Amodei (Anthropic) almost certainly have net worth above the '
            'Forbes threshold — OpenAI is valued at ~$150B, Anthropic at ~$60B, '
            'and their equity stakes imply billionaire-level wealth — but Forbes '
            'net worth estimates for private-company equity routinely lag actual '
            'valuations significantly. Similarly, some Chinese tech executives '
            'have obscured wealth due to regulatory pressure. Include anyone who '
            'is a de facto billionaire by real valuation even if not yet listed.\n'
            '  • World leaders list: these are private-sector figures.\n'
            '  • Existing global council categories: global_operators and '
            'global_officials categories did not cover the private-sector '
            'technology executive tier.\n\n'
            'DO exclude: figures already confirmed on the Forbes Global Billionaires '
            'list (Musk, Bezos, Zuckerberg, Jensen Huang if listed, etc. — '
            'already tracked). Focus on the C-suite AI lab operators, major chip '
            'and hardware decision-makers, platform executives with genuine '
            'geopolitical reach, and the capital allocators shaping which '
            'technologies and which nations win the AI race. '
            'Rank by current real-world global impact as of March 2026.'
        ),
    },
}


def build_category_prompt(category_info: dict) -> str:
    scope_phrase = (
        'United States' if category_info['scope'] == 'us'
        else 'world'
    )
    return (
        f"Today is March 2026.\n\n"
        f"You are tasked with identifying the 20 most influential living figures "
        f"in the {scope_phrase} in the category of "
        f"**{category_info['label']}** AS OF TODAY.\n\n"
        f"Category definition: {category_info['description']}\n\n"
        f"IMPORTANT CONSTRAINTS:\n"
        f"- Must be currently alive and operationally active as of March 2026\n"
        f"- Must have demonstrable, current, real-world influence through their "
        f"technology decisions — not just wealth or historical reputation\n"
        f"- Use your most up-to-date knowledge of company valuations, leadership "
        f"roles, and geopolitical tech dynamics\n\n"
        f"Please provide exactly 20 names, ranked by current influence (most "
        f"influential first). For each person provide:\n"
        f"1. Full name\n"
        f"2. Brief role or affiliation\n"
        f"3. One-sentence justification for their inclusion\n\n"
        f"Format your response as a JSON array with no extra text:\n"
        f"[\n"
        f'  {{"rank": 1, "name": "Full Name", "role": "Brief role", '
        f'"justification": "One sentence."}}\n'
        f"]"
    )


class TechExecutivesCouncil:
    """Single-category council covering US and Global tech executive lists."""

    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        self.openai_client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self.xai_client = OpenAI(
            api_key=os.getenv('XAI_API_KEY'),
            base_url='https://api.x.ai/v1'
        )
        self.gemini_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

    # ------------------------------------------------------------------ #
    # Low-level query helpers (identical pattern to existing councils)    #
    # ------------------------------------------------------------------ #

    def _extract_json_array(self, text: str) -> list:
        start = text.find('[')
        end   = text.rfind(']') + 1
        return json.loads(text[start:end])

    def _extract_json_object(self, text: str) -> dict:
        start = text.find('{')
        end   = text.rfind('}') + 1
        return json.loads(text[start:end])

    def _query_model(self, model_name: str, prompt: str,
                     max_tokens: int = 4096, temperature: float = 0.7,
                     scope: str = 'global') -> str:
        sys_msg = (
            f'You are an expert analyst of {"US" if scope == "us" else "global"} '
            f'technology industry influence and power dynamics.'
        )
        if model_name == 'claude':
            msg = self.anthropic_client.messages.create(
                model='claude-opus-4-6',
                max_tokens=max_tokens,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return msg.content[0].text

        elif model_name == 'chatgpt':
            resp = self.openai_client.chat.completions.create(
                model='gpt-5.2',
                messages=[
                    {'role': 'system', 'content': sys_msg},
                    {'role': 'user',   'content': prompt},
                ],
                temperature=temperature
            )
            return resp.choices[0].message.content

        elif model_name == 'grok':
            resp = self.xai_client.chat.completions.create(
                model='grok-4-1-fast-reasoning',
                messages=[
                    {'role': 'system', 'content': sys_msg},
                    {'role': 'user',   'content': prompt},
                ],
                temperature=temperature
            )
            return resp.choices[0].message.content

        elif model_name == 'gemini':
            resp = self.gemini_client.models.generate_content(
                model='gemini-3-pro-preview',
                contents=prompt
            )
            return resp.text

        raise ValueError(f'Unknown model: {model_name}')

    # ------------------------------------------------------------------ #
    # Stage 1                                                             #
    # ------------------------------------------------------------------ #

    def stage_1_query_all(self, category_info: dict) -> Dict[str, List[Dict[str, Any]]]:
        prompt  = build_category_prompt(category_info)
        results = {}
        for name in MODEL_NAMES:
            print(f'    Querying {name}...')
            try:
                text          = self._query_model(name, prompt, max_tokens=4096,
                                                   temperature=0.7,
                                                   scope=category_info['scope'])
                results[name] = self._extract_json_array(text)
                print(f'      → {len(results[name])} entries')
            except Exception as e:
                print(f'      ✗ {name} failed: {e}')
                results[name] = []
        return results

    # ------------------------------------------------------------------ #
    # Stage 2                                                             #
    # ------------------------------------------------------------------ #

    def stage_2_peer_review(self, all_responses: Dict[str, List[Dict[str, Any]]],
                             category_info: dict) -> Dict[str, Any]:
        peer_reviews = {}
        for reviewer in MODEL_NAMES:
            if not all_responses.get(reviewer):
                continue
            print(f'    Peer review from {reviewer}...')

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
                f"You previously provided a list of the 20 most influential figures "
                f"in the category '{category_info['label']}'.\n\n"
                "Now review these three independent responses to the same question "
                "(labeled Model A, Model B, Model C):\n\n"
            )
            for label, response in other_responses.items():
                review_prompt += f'### {label}\'s Response:\n'
                for person in response[:5]:
                    review_prompt += (
                        f"{person['rank']}. {person['name']} "
                        f"- {person.get('justification', '')}\n"
                    )
                review_prompt += f'...and {len(response) - 5} more\n\n'

            review_prompt += (
                'Rank these three responses from best to worst based on:\n'
                '1. Accuracy of current information (March 2026)\n'
                '2. Quality of reasoning and justifications\n'
                f"3. Adherence to the '{category_info['label']}' category definition\n"
                '4. Correct identification of figures missing from Forbes lists '
                'despite likely billionaire-level wealth (e.g. private AI lab CEOs)\n\n'
                'Return only a JSON object with no extra text:\n'
                '{"rankings": ['
                '{"model": "Model A", "rank": 1, "reasoning": "brief explanation"}, '
                '{"model": "Model B", "rank": 2, "reasoning": "brief explanation"}, '
                '{"model": "Model C", "rank": 3, "reasoning": "brief explanation"}'
                ']}'
            )

            try:
                text        = self._query_model(reviewer, review_prompt,
                                                max_tokens=2048, temperature=0.3,
                                                scope=category_info['scope'])
                review_data = self._extract_json_object(text)
                for ranking in review_data['rankings']:
                    ranking['actual_model'] = label_map.get(ranking['model'], 'unknown')
                peer_reviews[reviewer] = review_data
                print(f'      ✓ done')
            except Exception as e:
                print(f'      ✗ parse error: {e}')
                peer_reviews[reviewer] = None

        return peer_reviews

    # ------------------------------------------------------------------ #
    # Stage 3                                                             #
    # ------------------------------------------------------------------ #

    def stage_3_chairman_synthesis(self,
                                   all_responses: Dict[str, List[Dict[str, Any]]],
                                   peer_reviews: Dict[str, Any],
                                   category_info: dict) -> List[Dict[str, Any]]:
        synthesis_prompt = (
            f"You are the Chairman of an LLM Council tasked with identifying the "
            f"20 most influential living figures in the category "
            f"**{category_info['label']}** as of March 2026.\n\n"
            f"Category definition: {category_info['description']}\n\n"
            "You have received independent responses from four leading AI models "
            "and peer reviews where each model ranked the others.\n\n"
            "## Original Responses:\n\n"
        )
        for model_name, response in all_responses.items():
            if not response:
                continue
            synthesis_prompt += f'### {model_name.upper()}\'s List:\n'
            for person in response:
                synthesis_prompt += (
                    f"{person['rank']}. {person['name']} "
                    f"({person.get('role', 'N/A')}) — "
                    f"{person.get('justification', 'N/A')}\n"
                )
            synthesis_prompt += '\n'

        synthesis_prompt += '\n## Peer Review Rankings:\n\n'
        for reviewer, review in peer_reviews.items():
            if not review:
                continue
            synthesis_prompt += f'### {reviewer.upper()}\'s rankings:\n'
            for ranking in review['rankings']:
                synthesis_prompt += (
                    f"{ranking['rank']}. {ranking['actual_model']} — "
                    f"{ranking.get('reasoning', '')}\n"
                )
            synthesis_prompt += '\n'

        synthesis_prompt += (
            '\n## Your Task:\n\n'
            'Synthesize these inputs into a single, definitive list of the '
            '20 most influential living figures in this category.\n\n'
            'Consider:\n'
            '1. Consensus across models (use judgment, not just averaging)\n'
            '2. Peer review feedback on quality and accuracy\n'
            '3. Current events as of March 2026\n'
            f"4. Strict adherence to the '{category_info['label']}' definition\n"
            '5. Explicitly include figures like Sam Altman (OpenAI) and '
            'Dario Amodei (Anthropic) if the model responses support it — '
            'their absence from Forbes lists is a data collection artifact, '
            'not a reflection of their actual wealth or influence\n'
            '6. No duplicates — consolidate name variations into single entries\n\n'
            'Provide exactly 20 names ranked by current influence. '
            'Return only a JSON array with no extra text:\n'
            '[\n'
            '  {\n'
            '    "rank": 1,\n'
            '    "name": "Full Name",\n'
            '    "role": "Brief role or affiliation",\n'
            '    "justification": "One clear sentence on current influence",\n'
            '    "consensus_notes": "Brief note on model agreement/disagreement"\n'
            '  }\n'
            ']\n'
        )

        try:
            msg = self.anthropic_client.messages.create(
                model='claude-opus-4-6',
                max_tokens=8192,
                messages=[{'role': 'user', 'content': synthesis_prompt}]
            )
            final_list = self._extract_json_array(msg.content[0].text)
            print(f'    ✓ Chairman synthesis complete ({len(final_list)} entries)')
            return final_list
        except Exception as e:
            print(f'    ✗ Chairman synthesis failed: {e}')
            return []

    # ------------------------------------------------------------------ #
    # Per-category orchestration                                          #
    # ------------------------------------------------------------------ #

    def run_category(self, category_key: str, category_info: dict) -> dict:
        print(f"\n{'=' * 60}")
        print(f"CATEGORY: {category_info['label'].upper()}")
        print(f"{'=' * 60}")

        print('\n  -- Stage 1: Independent Responses --')
        all_responses = self.stage_1_query_all(category_info)
        valid = sum(1 for r in all_responses.values() if r)
        print(f'\n  Stage 1 complete: {valid}/4 models responded')

        print('\n  -- Stage 2: Peer Review --')
        peer_reviews = self.stage_2_peer_review(all_responses, category_info)

        print('\n  -- Stage 3: Chairman Synthesis --')
        final_list = self.stage_3_chairman_synthesis(
            all_responses, peer_reviews, category_info
        )

        return {
            'stage_1_individual_responses': all_responses,
            'stage_2_peer_reviews':         peer_reviews,
            'stage_3_final_list':           final_list,
        }

    # ------------------------------------------------------------------ #
    # Main run                                                            #
    # ------------------------------------------------------------------ #

    def run(self):
        print('=' * 60)
        print('TECH EXECUTIVES COUNCIL')
        print('Gap-Filling Category — US & Global')
        print('Three-Stage Process (Karpathy Method)')
        print('=' * 60)

        results = {}
        for category_key, category_info in CATEGORIES.items():
            results[category_key] = self.run_category(category_key, category_info)

        self._save(results)

        print('\n' + '=' * 60)
        print('Council session complete!')
        print('=' * 60)

    def _save(self, results: dict):
        output = {
            'metadata': {
                'date':       '2026-03-03',
                'focus':      'Tech & AI executives missing from Forbes lists and existing council categories',
                'categories': list(CATEGORIES.keys()),
                'models_queried': MODEL_NAMES,
                'methodology': (
                    'Three-stage LLM Council (Karpathy method): '
                    'Stage 1 - Independent responses, '
                    'Stage 2 - Peer review, '
                    'Stage 3 - Chairman synthesis by Claude Opus 4.6'
                ),
                'note': (
                    'Figures like Sam Altman (OpenAI) and Dario Amodei (Anthropic) '
                    'are almost certainly billionaires by real valuation but do not '
                    'appear on Forbes lists due to the lag in private-equity '
                    'net worth estimation. This category was created specifically '
                    'to capture such figures.'
                ),
            }
        }
        for key, result in results.items():
            output[key] = result

        out_path = os.path.join(BASE_DIR, 'tech_executives_results.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f'\n✓ Results saved to tech_executives_results.json')


if __name__ == '__main__':
    council = TechExecutivesCouncil()
    council.run()
