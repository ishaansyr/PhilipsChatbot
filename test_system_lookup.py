import sqlite3
from openai import OpenAI
import re

client = OpenAI()

DB_PATH = "philips_knowledge.db"

SYSTEM_PROMPT = """
You are an intent classifier for an enterprise knowledge assistant.

Classify the user query into exactly one of the following categories:

SYSTEM_LOOKUP — The user is asking where something is stored, accessed, downloaded, logged, or found in an internal system.

TEAM_LOOKUP — The user is asking which internal team or department handles, owns, manages, oversees, or is responsible for a specific operational or project-related function.

PROJECT_CONTEXT — The user is asking about project history, lessons learned, prior failures, approaches tried, analysis, or contextual project information. 
Also classify here if the query is general company information, external information, or outside the scope of internal systems and team responsibilities.

Return ONLY one of:
SYSTEM_LOOKUP
TEAM_LOOKUP
PROJECT_CONTEXT

Do not explain.

"""

def classify(query):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"}
        ],
        max_tokens=5
    )
    return response.choices[0].message.content.strip()


def system_lookup(query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query_lower = query.lower()
    words = re.findall(r'\b\w+\b', query_lower)

    cursor.execute("""
        SELECT task, system, link, user_intent
        FROM systems_navigator
    """)

    rows = cursor.fetchall()
    conn.close()

    best_match = None
    best_score = -1

    for row in rows:
        task, system, link, user_intent = row
        text_blob = f"{task} {user_intent}".lower()

        score = 0
        match_count = 0

        for word in words:
            if word in text_blob:
                score += len(word)  # longer words weigh more
                match_count += 1

        # Prefer rows with more distinct word matches
        score += match_count * 2

        if score > best_score:
            best_score = score
            best_match = (task, system, link)

    if best_score <= 0:
        return None

    return best_match

def team_lookup(query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query_lower = query.lower()
    words = re.findall(r'\b\w+\b', query_lower)

    cursor.execute("""
        SELECT team, responsibility, question
        FROM team_responsibility
    """)

    rows = cursor.fetchall()
    conn.close()

    best_match = None
    best_score = -1

    for row in rows:
        team, responsibility, question = row
        text_blob = f"{team} {responsibility} {question}".lower()

        score = 0
        match_count = 0

        for word in words:
            if word in text_blob:
                score += len(word)
                match_count += 1

        score += match_count * 2

        if score > best_score:
            best_score = score
            best_match = (team, responsibility)

    if best_score <= 0:
        return None

    return best_match

def handle_query(query):
    label = classify(query)

    print(f"\nQuery: {query}")
    print(f"Intent: {label}")

    if label == "SYSTEM_LOOKUP":
        result = system_lookup(query)
        if result:
            task, system, link = result
            print("Structured Result:")
            print(f"Task: {task}")
            print(f"System: {system}")
            print(f"Link: {link}")
        else:
            print("No structured match found. (Would fallback to RAG)")

    elif label == "TEAM_LOOKUP":
        result = team_lookup(query)
        if result:
            team, responsibility = result
            print("Structured Result:")
            print(f"Team: {team}")
            print(f"Responsibility: {responsibility}")
        else:
            print("No structured match found. (Would fallback to RAG)")

    else:
        print("PROJECT_CONTEXT (Would route to RAG)")


# Test cases
queries = [
    "Where do I log my working hours?",
    "How do I access the DHF?",
    "Where can I download approved software?",
    "Who handles complaint trending?",
    "Which team owns CAPA investigations?",
    "Who handles complaint trending?",
    "Who manages product roadmaps?",
    "Who is responsible for risk documentation?",
    "What went wrong in packaging?"
]

for q in queries:
    handle_query(q)
