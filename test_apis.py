"""
Test each LLM API individually to diagnose connection issues
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("LLM API Diagnostic Test")
print("=" * 60)

# Test 1: Anthropic (Claude)
print("\n1. Testing Anthropic (Claude)...")
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say 'Claude is working'"}]
    )
    print(f"   [OK] SUCCESS: {message.content[0].text}")
except Exception as e:
    print(f"   [FAIL] FAILED: {type(e).__name__}: {str(e)}")

# Test 2: OpenAI (ChatGPT)
print("\n2. Testing OpenAI (ChatGPT)...")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say 'ChatGPT is working'"}],
        max_tokens=50
    )
    print(f"   [OK] SUCCESS: {response.choices[0].message.content}")
except Exception as e:
    print(f"   [FAIL] FAILED: {type(e).__name__}: {str(e)}")

# Test 3: xAI (Grok)
print("\n3. Testing xAI (Grok)...")
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    print("   Attempting to connect to xAI API...")
    response = client.chat.completions.create(
        model="grok-2-latest",
        messages=[{"role": "user", "content": "Say 'Grok is working'"}],
        max_tokens=50
    )
    print(f"   [OK] SUCCESS: {response.choices[0].message.content}")
except Exception as e:
    print(f"   [FAIL] FAILED: {type(e).__name__}: {str(e)}")
    print(f"   Note: Check that XAI_API_KEY is set correctly and has access")
    print(f"   API Key present: {bool(os.getenv('XAI_API_KEY'))}")
    if os.getenv("XAI_API_KEY"):
        print(f"   API Key starts with: {os.getenv('XAI_API_KEY')[:10]}...")

# Test 4: Google (Gemini)
print("\n4. Testing Google (Gemini)...")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    # Try different model names
    model_names = [
        "gemini-2.0-flash-exp",
        "gemini-pro",
        "gemini-1.5-pro",
        "gemini-1.5-flash"
    ]

    success = False
    for model_name in model_names:
        try:
            print(f"   Trying model: {model_name}...")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Say 'Gemini is working'")
            print(f"   [OK] SUCCESS with {model_name}: {response.text}")
            success = True
            break
        except Exception as e:
            print(f"   - {model_name} failed: {type(e).__name__}")

    if not success:
        print(f"   [FAIL] FAILED: All model names failed")
        print(f"   API Key present: {bool(os.getenv('GOOGLE_API_KEY'))}")

except Exception as e:
    print(f"   [FAIL] FAILED: {type(e).__name__}: {str(e)}")
    print(f"   API Key present: {bool(os.getenv('GOOGLE_API_KEY'))}")

print("\n" + "=" * 60)
print("Diagnostic Complete")
print("=" * 60)
