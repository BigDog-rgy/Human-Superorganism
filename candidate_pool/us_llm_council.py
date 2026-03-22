"""
US LLM Council - Multi-Model Consensus System (Domestic Focus)
Queries Claude, Grok, ChatGPT, and Gemini to construct a consensus list
of the 12 most influential living Americans (domestic influence focus).
"""

import os
import json
import asyncio
from typing import List, Dict, Any
from collections import Counter
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
from google import genai

# Load environment variables
load_dotenv()


class USLLMCouncil:
    """Orchestrates multiple LLM APIs to reach consensus on US domestic prime movers."""

    def __init__(self):
        # Initialize API clients
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

        self.prompt = """Today is February 17, 2026.

You are tasked with identifying the 12 most influential living Americans AS OF TODAY — individuals who are currently shaping major US domestic outcomes across politics, business, technology, media, finance, and other critical domestic domains.

IMPORTANT CONSTRAINTS:
- Focus is DOMESTIC influence within the United States, not global geopolitical reach
- All individuals must be US citizens currently residing and operating primarily in the US
- Must be currently alive and active as of February 2026
- Must have demonstrable influence over US markets, policy, technology, or public behavior RIGHT NOW
- Must have current power to materially shape US domestic direction
- Consider influence across: US federal policy, US technology sector, US financial markets, US media/narrative, US defense and industry
- Use your most up-to-date knowledge about current US leadership and events

Please provide exactly 12 names, ranked by current domestic influence (most influential first). For each person, provide:
1. Full name
2. Primary domain of influence
3. Brief one-sentence justification for their inclusion

Format your response as a JSON array of objects with fields: "rank", "name", "domain", "justification"
"""

    def query_claude(self) -> List[Dict[str, Any]]:
        """Query Claude (Anthropic) for US prime movers list."""
        print("Querying Claude...")
        try:
            message = self.anthropic_client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": self.prompt}
                ]
            )

            response_text = message.content[0].text
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]

            return json.loads(json_str)
        except Exception as e:
            print(f"Error querying Claude: {e}")
            return []

    def query_gpt(self) -> List[Dict[str, Any]]:
        """Query ChatGPT (OpenAI) for US prime movers list."""
        print("Querying ChatGPT...")
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system", "content": "You are an expert analyst of US domestic influence and power dynamics."},
                    {"role": "user", "content": self.prompt}
                ],
                temperature=0.7
            )

            response_text = response.choices[0].message.content
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]

            return json.loads(json_str)
        except Exception as e:
            print(f"Error querying ChatGPT: {e}")
            return []

    def query_grok(self) -> List[Dict[str, Any]]:
        """Query Grok (xAI) for US prime movers list."""
        print("Querying Grok...")
        try:
            response = self.xai_client.chat.completions.create(
                model="grok-4-1-fast-reasoning",
                messages=[
                    {"role": "system", "content": "You are an expert analyst of US domestic influence and power dynamics."},
                    {"role": "user", "content": self.prompt}
                ],
                temperature=0.7
            )

            response_text = response.choices[0].message.content
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]

            return json.loads(json_str)
        except Exception as e:
            print(f"Error querying Grok: {e}")
            return []

    def query_gemini(self) -> List[Dict[str, Any]]:
        """Query Gemini (Google) for US prime movers list."""
        print("Querying Gemini...")
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=self.prompt
            )
            response_text = response.text

            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]

            return json.loads(json_str)
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return []

    def query_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Query all LLM models and return their responses."""
        results = {}

        results['claude'] = self.query_claude()
        results['chatgpt'] = self.query_gpt()
        results['grok'] = self.query_grok()
        results['gemini'] = self.query_gemini()

        return results

    def stage_2_peer_review(self, all_responses: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Stage 2: Each model reviews the other models' responses anonymously.
        Returns peer rankings from each model.
        """
        print("\n" + "=" * 60)
        print("STAGE 2: PEER REVIEW")
        print("=" * 60)
        print("\nEach model will review and rank the other responses...\n")

        peer_reviews = {}
        model_names = ['claude', 'chatgpt', 'grok', 'gemini']

        for reviewer in model_names:
            if not all_responses.get(reviewer):
                continue

            print(f"Getting peer review from {reviewer}...")

            other_responses = {}
            label_map = {}
            labels = ['Model A', 'Model B', 'Model C']
            idx = 0

            for model_name in model_names:
                if model_name != reviewer and all_responses.get(model_name):
                    other_responses[labels[idx]] = all_responses[model_name]
                    label_map[labels[idx]] = model_name
                    idx += 1

            review_prompt = f"""You previously provided a list of the 12 most influential living Americans (domestic focus).

Now, please review these three other independent responses to the same question (labeled Model A, Model B, and Model C):

"""
            for label, response in other_responses.items():
                review_prompt += f"\n### {label}'s Response:\n"
                for person in response[:5]:
                    review_prompt += f"{person['rank']}. {person['name']} - {person['justification']}\n"
                review_prompt += f"...and {len(response) - 5} more\n"

            review_prompt += """
Please rank these three responses from best to worst based on:
1. Accuracy of current information (February 2026)
2. Quality of reasoning and justifications
3. Focus on domestic US influence (not global reach)
4. Comprehensiveness and insight

Provide your ranking as a JSON object with this format:
{
  "rankings": [
    {"model": "Model A", "rank": 1, "reasoning": "brief explanation"},
    {"model": "Model B", "rank": 2, "reasoning": "brief explanation"},
    {"model": "Model C", "rank": 3, "reasoning": "brief explanation"}
  ]
}
"""

            try:
                if reviewer == 'claude':
                    message = self.anthropic_client.messages.create(
                        model="claude-opus-4-6",
                        max_tokens=2048,
                        messages=[{"role": "user", "content": review_prompt}]
                    )
                    response_text = message.content[0].text
                elif reviewer == 'chatgpt':
                    response = self.openai_client.chat.completions.create(
                        model="gpt-5.2",
                        messages=[{"role": "user", "content": review_prompt}],
                        temperature=0.3
                    )
                    response_text = response.choices[0].message.content
                elif reviewer == 'grok':
                    response = self.xai_client.chat.completions.create(
                        model="grok-4-1-fast-reasoning",
                        messages=[{"role": "user", "content": review_prompt}],
                        temperature=0.3
                    )
                    response_text = response.choices[0].message.content
                elif reviewer == 'gemini':
                    response = self.gemini_client.models.generate_content(
                        model="gemini-3-pro-preview",
                        contents=review_prompt
                    )
                    response_text = response.text

                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                review_data = json.loads(response_text[json_start:json_end])

                for ranking in review_data['rankings']:
                    ranking['actual_model'] = label_map.get(ranking['model'], 'unknown')

                peer_reviews[reviewer] = review_data
                print(f"  ✓ {reviewer} completed peer review")

            except Exception as e:
                print(f"  ✗ Error getting peer review from {reviewer}: {e}")
                peer_reviews[reviewer] = None

        return peer_reviews

    def stage_3_chairman_synthesis(self, all_responses: Dict[str, List[Dict[str, Any]]],
                                   peer_reviews: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Stage 3: Claude (Chairman) synthesizes final authoritative response
        based on original answers and peer rankings.
        """
        print("\n" + "=" * 60)
        print("STAGE 3: CHAIRMAN SYNTHESIS")
        print("=" * 60)
        print("\nClaude (Chairman) is synthesizing the final US list...\n")

        synthesis_prompt = """You are the Chairman of an LLM Council tasked with identifying the 12 most influential living Americans (domestic focus) as of February 17, 2026.

You have received:
1. Independent responses from four leading AI models
2. Peer reviews where each model ranked the others' responses

## Original Responses:

"""
        for model_name, response in all_responses.items():
            if not response:
                continue
            synthesis_prompt += f"\n### {model_name.upper()}'s List:\n"
            for person in response:
                synthesis_prompt += f"{person['rank']}. {person['name']} ({person.get('domain', 'N/A')}) - {person.get('justification', 'N/A')}\n"

        synthesis_prompt += "\n\n## Peer Review Rankings:\n\n"
        for reviewer, review in peer_reviews.items():
            if not review:
                continue
            synthesis_prompt += f"\n### {reviewer.upper()}'s rankings:\n"
            for ranking in review['rankings']:
                synthesis_prompt += f"{ranking['rank']}. {ranking['actual_model']} - {ranking.get('reasoning', '')}\n"

        synthesis_prompt += """

## Your Task:

As Chairman, synthesize these inputs into a single, definitive list of the 12 most influential living Americans as of February 2026.

Consider:
1. Consensus across models (but don't just average - use judgment)
2. Peer review feedback on quality and accuracy
3. Current events and up-to-date knowledge (Donald Trump is the current U.S. President as of 2026)
4. Balance across domains (politics, technology, finance, media, defense)
5. Focus on DOMESTIC US influence, not global geopolitical reach
6. Actual current influence, not historical reputation

**IMPORTANT**: Ensure there are NO DUPLICATES. Some models may list the same person with slight name variations. Consolidate these into single entries.

Provide exactly 12 names ranked by current domestic influence. Format as JSON:
[
  {
    "rank": 1,
    "name": "Full Name",
    "domain": "Primary domain",
    "justification": "One clear sentence explaining their current domestic influence and why they made the final list",
    "consensus_notes": "Brief note on how models agreed/disagreed"
  },
  ...
]
"""

        try:
            message = self.anthropic_client.messages.create(
                model="claude-opus-4-6",
                max_tokens=8192,
                messages=[{"role": "user", "content": synthesis_prompt}]
            )

            response_text = message.content[0].text

            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]

            final_list = json.loads(json_str)
            print("✓ Chairman synthesis complete!")
            return final_list

        except Exception as e:
            print(f"✗ Error in chairman synthesis: {e}")
            return []

    def build_consensus(self, all_responses: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Build consensus from all model responses.

        Strategy:
        1. Count frequency of each name across all models
        2. Weight by inverse rank (rank 1 = 12 points, rank 12 = 1 point)
        3. Combine frequency and weighted score for final ranking
        """
        print("\nBuilding consensus...")

        name_data = {}

        for model_name, responses in all_responses.items():
            if not responses:
                continue

            print(f"\n{model_name.upper()} responses:")
            for item in responses:
                name = item['name']
                rank = item.get('rank', 12)
                domain = item.get('domain', 'Unknown')
                justification = item.get('justification', '')

                print(f"  {rank}. {name} ({domain})")

                # Weighted score: rank 1 = 12 points, rank 12 = 1 point
                weighted_score = 13 - rank

                if name not in name_data:
                    name_data[name] = {
                        'name': name,
                        'mentions': 0,
                        'total_score': 0,
                        'domains': [],
                        'justifications': [],
                        'models': []
                    }

                name_data[name]['mentions'] += 1
                name_data[name]['total_score'] += weighted_score
                name_data[name]['domains'].append(domain)
                name_data[name]['justifications'].append(f"[{model_name}] {justification}")
                name_data[name]['models'].append(model_name)

        for name, data in name_data.items():
            data['final_score'] = (data['mentions'] * 100) + data['total_score']
            data['primary_domain'] = max(set(data['domains']), key=data['domains'].count)

        ranked_names = sorted(name_data.values(), key=lambda x: x['final_score'], reverse=True)

        consensus_list = []
        for i, person in enumerate(ranked_names[:12], 1):
            consensus_list.append({
                'rank': i,
                'name': person['name'],
                'domain': person['primary_domain'],
                'mentions': person['mentions'],
                'models_mentioned_by': person['models'],
                'consensus_score': person['final_score'],
                'justifications': person['justifications']
            })

        return consensus_list

    def save_results(self, all_responses: Dict[str, List], peer_reviews: Dict[str, Any],
                    final_list: List[Dict[str, Any]]):
        """Save all results to JSON file."""
        output = {
            'metadata': {
                'date': '2026-02-17',
                'focus': 'US domestic prime movers',
                'models_queried': ['claude', 'chatgpt', 'grok', 'gemini'],
                'methodology': 'Three-stage LLM Council (Karpathy method): Stage 1 - Independent responses, Stage 2 - Peer review, Stage 3 - Chairman synthesis by Claude Opus 4.6'
            },
            'stage_1_individual_responses': all_responses,
            'stage_2_peer_reviews': peer_reviews,
            'stage_3_final_list': final_list
        }

        output_file = 'us_llm_council_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to {output_file}")

        self.create_markdown_summary(final_list)

    def create_markdown_summary(self, final_list: List[Dict[str, Any]]):
        """Create a readable markdown summary of the final US list."""
        md_content = "# LLM Council: Top 12 Living US Prime Movers (2026)\n\n"
        md_content += "*Generated: February 17, 2026*\n\n"
        md_content += "## Methodology: Three-Stage Council Process (Karpathy Method)\n\n"
        md_content += "**Stage 1 - Independent Responses**: Four leading LLMs (Claude Opus 4.6, GPT-5.2, Grok 4.1, Gemini 3 Pro) "
        md_content += "independently generated their top 12 lists (domestic US focus).\n\n"
        md_content += "**Stage 2 - Peer Review**: Each model anonymously reviewed and ranked the other models' responses "
        md_content += "for accuracy, reasoning quality, and comprehensiveness.\n\n"
        md_content += "**Stage 3 - Chairman Synthesis**: Claude Opus 4.6, serving as Chairman, synthesized all responses "
        md_content += "and peer reviews into this final authoritative list.\n\n"
        md_content += "---\n\n"

        for person in final_list:
            md_content += f"## {person['rank']}. {person['name']}\n\n"
            md_content += f"**Domain**: {person['domain']}\n\n"
            md_content += f"**Justification**: {person.get('justification', 'N/A')}\n\n"

            if person.get('consensus_notes'):
                md_content += f"**Consensus Notes**: {person['consensus_notes']}\n\n"

            md_content += "---\n\n"

        output_file = 'us_llm_council_summary.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"✓ Summary saved to {output_file}")

    def run(self):
        """Execute the full three-stage US LLM council process."""
        print("=" * 60)
        print("US LLM COUNCIL - Domestic Prime Mover Identification")
        print("Three-Stage Process (Karpathy Method)")
        print("=" * 60)

        # STAGE 1: Independent Responses
        print("\n" + "=" * 60)
        print("STAGE 1: INDEPENDENT RESPONSES")
        print("=" * 60)
        print("\nQuerying 4 leading LLMs for their top 12 US prime movers...\n")

        all_responses = self.query_all_models()

        print("\n--- Stage 1 Summary ---")
        for model_name, response in all_responses.items():
            if response:
                print(f"✓ {model_name}: {len(response)} responses")
            else:
                print(f"✗ {model_name}: No response")

        # STAGE 2: Peer Review
        peer_reviews = self.stage_2_peer_review(all_responses)

        # STAGE 3: Chairman Synthesis
        final_list = self.stage_3_chairman_synthesis(all_responses, peer_reviews)

        # Display final results
        print("\n" + "=" * 60)
        print("FINAL RESULTS - Top 12 US Prime Movers")
        print("=" * 60 + "\n")

        for person in final_list:
            print(f"{person['rank']:2d}. {person['name']}")
            print(f"    Domain: {person['domain']}")
            print(f"    {person.get('justification', '')}")
            if person.get('consensus_notes'):
                print(f"    Notes: {person['consensus_notes']}")
            print()

        # Save results
        self.save_results(all_responses, peer_reviews, final_list)

        print("\n" + "=" * 60)
        print("US Council session complete!")
        print("=" * 60)


if __name__ == "__main__":
    council = USLLMCouncil()
    council.run()
