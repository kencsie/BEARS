"""
Agentic agent prompt templates.

Ported from archive/AgenticFlow/prompts/ and run_agent.py defaults.
"""

STEP_REASONER_PROMPT = """You are a "next-step planner" for a retrieval-based QA system.
Goal: Iteratively find information to answer the user's question. Output one instruction at a time:

1) If current content is sufficient to answer: output DONE
2) If not sufficient: output the next search query (one sentence, be specific)

Constraints:
- No explanations, no multi-line, no JSON
- Only output DONE or a single query

Important: If the question contains constraints (ranking, second largest, specific year, etc.),
you must verify the constraint is satisfied before outputting DONE.

[Original Question]
{question}

[Accumulated Retrieval Snippets (may contain noise)]
{contexts}

[Previous Steps (step -> query)]
{history}
"""

LLM_RERANK_PROMPT = """You are a professional document re-ranker.
The user's query is: "{query}"

Below are {num_docs} candidate documents. Analyze their relevance to the query:

{docs_list}

[Task]
Select the most relevant {top_k} documents, ranked by relevance (high to low).
Return format must be a pure JSON array of document index numbers.
Example: [2, 0, 5, 1, 8]

Notes:
1. If a document is completely irrelevant, do not include it.
2. Even if fewer than {top_k} are relevant, only return relevant ones.
3. Output only JSON, no explanations.
"""

LLM_GRADE_PROMPT = """You are a retrieval result relevance scorer.

Judge whether the "Document Content" helps answer the "Question".
Output only a single integer score, no explanations.

Scoring criteria:
3 = Directly contains answerable facts (names/titles/companies/dates/numbers)
2 = Contains key bridging information (necessary for multi-hop)
1 = Background relevant but insufficient as evidence
0 = Irrelevant or noise

[Question]
{question}

[Document Content]
{content}
"""

GENERATE_SYSTEM_PROMPT = """You are a professional assistant. Answer the [Question] based on the [Reference Documents].
If documents mention role changes (e.g., former vs. current), be sure to distinguish clearly.
If documents are insufficient, say "Insufficient data to answer" and indicate what is missing.
"""
