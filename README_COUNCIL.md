# LLM Council - Multi-Model Prime Mover Identification

## Overview

The LLM Council is a consensus-based system that queries four leading language models (Claude Opus 4.6, ChatGPT-4, Grok-2, and Gemini 2.0) to identify and rank the world's 17 most influential living "prime movers."

## How It Works

1. **Query Phase**: Each LLM is independently asked to identify the 17 most influential living individuals who are currently shaping major global outcomes
2. **Consensus Building**: Responses are aggregated using:
   - Frequency scoring (how many models mentioned each person)
   - Position-weighted scoring (higher weight for higher rankings)
   - Final score = (mentions × 100) + weighted position scores
3. **Output Generation**: Creates both JSON and Markdown reports with detailed justifications

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

- **Anthropic**: Get from https://console.anthropic.com/
- **OpenAI**: Get from https://platform.openai.com/api-keys
- **xAI (Grok)**: Get from https://x.ai/api
- **Google (Gemini)**: Get from https://aistudio.google.com/apikey

### 3. Run the Council

```bash
python llm_council.py
```

## Output Files

The council generates two output files:

1. **llm_council_results.json**: Complete data including all individual model responses and consensus rankings
2. **llm_council_summary.md**: Human-readable summary with justifications

## Consensus Methodology

Each person's final score is calculated as:
- **Base score**: mentions × 100 (rewards consensus across models)
- **Position bonus**: Sum of weighted positions where rank 1 = 17 points, rank 17 = 1 point
- **Final ranking**: Sorted by total score, top 17 selected

This approach balances:
- Cross-model agreement (frequency)
- Relative importance (ranking position)
- Diversity of perspectives (multiple models)

## Customization

You can modify the prompt in `llm_council.py` to:
- Change selection criteria
- Adjust the number of individuals (default: 17)
- Focus on specific domains or regions
- Add temporal constraints

## Models Used

- **Claude Opus 4.6** (Anthropic): Latest frontier model
- **GPT-4o** (OpenAI): Advanced reasoning and analysis
- **Grok-2** (xAI): Real-time data and unique perspective
- **Gemini 2.0 Flash** (Google): Multimodal analysis capability

## Cost Estimate

Approximate costs per run (as of Feb 2026):
- Claude Opus 4.6: ~$0.30-0.50
- GPT-4o: ~$0.10-0.20
- Grok-2: ~$0.15-0.25
- Gemini 2.0: ~$0.05-0.10

**Total: ~$0.60-1.05 per council session**

## Notes

- API keys are never logged or stored outside your local `.env` file
- Each model query is independent to avoid bias
- Responses are parsed for structured JSON output
- Error handling ensures partial results even if one API fails
