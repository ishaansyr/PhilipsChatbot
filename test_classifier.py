from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """
You are an intent classifier for an enterprise knowledge assistant.
Classify the user query into exactly one of the following categories:

SYSTEM_LOOKUP — The user is asking where something is stored, accessed, downloaded, logged, or found in a system.

TEAM_LOOKUP — The user is asking which team or department is responsible for something.

PROJECT_CONTEXT — The user is asking about project history, lessons learned, prior failures, approaches tried, or contextual analysis.

Return ONLY one of the three labels. No explanation.
"""

def classify(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # lightweight and sufficient
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"}
        ],
        max_tokens=5
    )

    return response.choices[0].message.content.strip()

# Test queries
queries = [
    "I need help with CAPA",
    "I’m working on packaging and need context.",
    "Where is verification owned?",
    "Who do I talk to about verification reports?",
    "I need details on past firmware issues.",
    "How do we handle complaint trending?",
    "Who handles complaint trending?",
    "What approaches were tried for ageing reduction?"
]

for q in queries:
    label = classify(q)
    print(f"Query: {q}")
    print(f"Classified as: {label}")
    print("-" * 50)
