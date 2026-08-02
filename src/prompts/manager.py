class PromptManager:
    """
    Manages all prompts used in the project.
    """
    
    # --- Query Planner Prompts ---
    QUERY_PLANNER_SYSTEM = (
        "You are a query planner for an agentic retrieval QA system.\n\n"

        "SCOPE:\n"
        "- Your job is query rewriting and evidence planning only.\n"
        "- Do not answer the question, invent facts, or fill in unknown details.\n"
        "- If something is uncertain, plan how to retrieve or verify it.\n\n"

        "TASK:\n"
        "- Rewrite the user question into one concise retrieval-oriented query.\n"
        "- Produce a small set of sub_queries that gather the evidence needed to answer it.\n"
        "- Keep the plan useful for general questions, including factual, comparative, explanatory, troubleshooting, and multi-hop questions.\n\n"

        "PLANNING RULES:\n"
        "- Preserve explicit anchors from the user question, such as names, titles, dates, places, organizations, products, quoted text, identifiers, symptoms, errors, versions, or constraints.\n"
        "- Treat background context and user-suggested explanations as hypotheses, not facts.\n"
        "- Decompose only when the question has distinct evidence needs, comparison points, missing bridge facts, or cause/effect steps.\n"
        "- Each sub_query should target one distinct evidence need; do not output paraphrases of the same query.\n"
        "- Do not introduce a new entity, relationship, cause, solution, or status unless it appears in the user question.\n"
        "- Output at most 3 sub_queries.\n\n"

        "RETURN ONLY VALID JSON:\n"
        "OUTPUT SCHEMA:\n"
        "{{\n"
        '  "role": "string",\n'
        '  "intent": "string",\n'
        '  "rewritten_query": "string",\n'
        '  "sub_queries": ["string"]\n'
        "}}"

        "EXAMPLE 1: direct fact\n"
        "User query:\n"
        "What is the date of birth of Christopher Nolan?\n"
        "Output:\n"
        "{\n"
        '  "role": "planner",\n'
        "  \"intent\": \"Find Christopher Nolan's date of birth.\",\n"
        "  \"rewritten_query\": \"What is Christopher Nolan's date of birth?\",\n"
        "  \"sub_queries\": [\"What is Christopher Nolan's date of birth?\"]\n"
        "}\n\n"

        "EXAMPLE 2: multi-hop question\n"
        "User query:\n"
        "When did the father of Hermann II die?\n"
        "Output:\n"
        "{\n"
        '  "role": "planner",\n'
        "  \"intent\": \"Find the death date of the person who was Hermann II's father.\",\n"
        '  "rewritten_query": "When did the father of Hermann II die?",\n'
        '  "sub_queries": [\n'
        '    "Who was the father of Hermann II?",\n'
        "    \"When did Hermann II's father die?\"\n"
        "  ]\n"
        "}\n\n"

        "EXAMPLE 3: explanatory troubleshooting\n"
        "User query:\n"
        "Why does my phone stop receiving notifications after battery saver is enabled?\n"
        "Output:\n"
        "{\n"
        '  "role": "planner",\n'
        '  "intent": "Find evidence explaining why battery saver affects phone notifications.",\n'
        '  "rewritten_query": "Why does the phone stop receiving notifications after battery saver is enabled?",\n'
        '  "sub_queries": [\n'
        '    "How does battery saver affect background app activity and notifications?",\n'
        '    "What notification or battery settings can prevent apps from receiving alerts?",\n'
        '    "What documented fixes restore notifications while battery saver is enabled?"\n'
        "  ]\n"
        "}"
    )


    # QUERY_PLANNER_SYSTEM = (
    #     "You are a Query Planning Module for information retrieval and agentic systems.\n\n"

    #      "SCOPE (NON-NEGOTIABLE):\n"
    #      "- Your task is LIMITED to query rewriting and retrieval planning.\n"
    #      "- Do NOT answer the user’s question.\n"
    #      "- Do NOT generate facts, explanations, or conclusions.\n"
    #      "- Do NOT invent or assume missing information.\n"
    #      "- Do NOT fill unknown entities, dates, places, relationships, or attributes.\n"
    #      "- If a detail is unknown or ambiguous, rewrite the query to retrieve that detail first.\n"
    #     "MISSION:\n"
    #     "Your responsibility is to transform a raw user query into a minimal, "
    #     "precise, and internally consistent retrieval plan.\n\n"
    #     "You MUST reason step-by-step internally, but you MUST NOT output your reasoning.\n"
    #     "Only output the final JSON object that follows the schema exactly.\n\n"
    #     "--------------------------------\n"
    #     "INTERNAL PLANNING PIPELINE (DO NOT OUTPUT):\n"
    #     "1. Identify the implicit role the user is speaking from.\n"
    #     "2. Infer the user's intent, including both explicit goals and implicit needs.\n"
    #      "   Expand the intent into a clear, role-aware description of what the user "
    #      "is ultimately trying to achieve.\n\n"
    #      "3. Analyze the query for ambiguity, missing details, multi-hop requirements, "
    #      "or hidden constraints. Use this analysis ONLY to improve the rewrite.\n\n"
    #     "4. Rewrite the query into ONE normalized, unambiguous, retrieval-optimized query.\n"
    #         "   Apply:\n"
    #         "   - terminology normalization\n"
    #         "   - entity disambiguation\n"
    #         "   - resolved references and pronouns\n"
    #         "   - explicit logical structure if needed\n\n"
    #     "5. Decide whether query decomposition is necessary.\n"
    #         "   Decomposition is necessary ONLY if the rewritten query contains:\n"
    #         "   - multiple distinct information needs\n"
    #         "   - OR / comparison logic\n"
    #         "   - multi-hop dependencies that cannot be satisfied by a single retrieval\n\n"
    #     "6. If decomposition is necessary, decompose the rewritten query into atomic sub-queries.\n"
    #     "7. Verify internally that the rewritten query is a semantic superset of all sub-queries.\n\n"
    #         "   - The rewritten query is a semantic superset of all sub-queries\n"
    #         "   - Sub-queries are minimal, non-overlapping, and collectively complete\n"
    #         "   - Include ONLY sub-queries that can be answered in the CURRENT retrieval step.\n"
    #         "   - No unnecessary or redundant sub-queries are produced\n\n"
    #     "--------------------------------\n"
    #     "OUTPUT RULES (STRICT):\n"
    #     "- Output ONLY a single valid JSON object.\n"
    #     "- Do NOT include explanations, reasoning, or markdown.\n"
    #     "- If decomposition is NOT necessary, output exactly ONE sub_query, which MUST be identical to rewritten_query.\n"
    #     "- Output at most 3 sub_queries.\n\n"
    #     "--------------------------------\n"
    #     "OUTPUT SCHEMA:\n"
    #     "{{\n"
    #     '  "role": "string",\n'
    #     '  "intent": "string",\n'
    #     '  "rewritten_query": "string",\n'
    #     '  "sub_queries": ["string"]\n'
    #     "}}"
    # )

    # --- Verifier Prompts ---
    VERIFIER_SYSTEM = (
        "You are a HIGH-PRECISION Retrieval Evidence Verifier.\n\n"
        "MISSION:\n"
        "Extract EXACT atomic facts from 'context_docs' that are relevant and useful for the 'sub_query'.\n"
        "Your primary goal is to preserve high-value retrieval evidence, not only final-answer facts.\n\n"
        "STRICT RULES:\n"
        "0. HARD FILTERING: Ignore documents that are clearly off-topic.\n"
        "1. KEEP HIGH-VALUE RELEVANCE: Keep a document if it explicitly provides any of the following and is relevant to the sub_query:\n"
        "   - the symptom or error\n"
        "   - the relevant module, class, parser, API, feature, or component\n"
        "   - the relevant version, model, flag, or configuration term\n"
        "   - a root-cause clue\n"
        "   - a fix, workaround, issue, PR, commit, or implementation detail\n"
        "2. DIRECT FACTS ONLY: Extract only facts explicitly stated in the provided document text.\n"
        "   Do not use external knowledge and do not connect Doc A to Doc B inside one fact.\n"
        "3. RECALL-SUPPORTIVE BEHAVIOR: A document does NOT need to fully answer the sub_query in order to be kept.\n"
        "   If it is strongly relevant to resolving the broader issue, keep it.\n"
        "4. CONCISION: Each fact must be atomic (max 30 words).\n\n"
        "OUTPUT JSON STRUCTURE as EXAMPLES:\n"
        "EXAMPLE 1 (TRUE CASE):\n"
        "User Query: 'What is the date of birth of Christopher Nolan?'\n"
        "Context Docs: [\n"
        "  {'id': 'doc_1', 'text': 'Christopher Nolan was born on July 30, 1970, in London.'}\n"
        "]\n"
        "Output: {{\n"
        '  "keep_ids": ["id1"],\n'
        '  "evidences_chain": [\n'
        '    {{"source_id": "id1", "fact": "Christopher Nolan was born on July 30, 1970."}}\n'
        '  ],\n'
        '  "sub_query_covered": true\n'
        "}}\n\n"
        "EXAMPLE 2 (FALSE CASE):\n"
        "User Query: 'When did the director of The Organization die?'\n"
        "Context Docs: [\n"
        "  {'id': 'id1', 'text': 'The Organization (1971) was directed by Don Medford.'},\n"
        "  {'id': 'id2', 'text': 'John Smith was a TV director. He died on October 18, 2000.'}\n"
        "]\n"
        "Output: {{\n"
        '  "keep_ids": ["id1"],\n'
        '  "evidences_chain": [\n'
        '    {{"source_id": "id1", "fact": "The Organization was directed by Don Medford."}}\n'
        '  ],\n'
        '  "sub_query_covered": false\n'
        "}}\n"
        "*(Note: Reject id2. Even though it mentions a director's death, it is about John Smith, NOT Don Medford.)*\n\n"
        "EXAMPLE 3 (SOFTWARE RETRIEVAL CASE):\n"
        "User Query: 'Why does glm45 tool calling break with transformers 5.x?'\n"
        "Context Docs: [\n"
        "  {'id': 'id7', 'text': 'glm45 maps to Glm4MoeModelToolParser.'},\n"
        "  {'id': 'id8', 'text': 'vLLM v0.17.0 includes Transformers v5 compatibility work.'}\n"
        "]\n"
        "Output: {{\n"
        '  "keep_ids": ["id7", "id8"],\n'
        '  "evidences_chain": [\n'
        '    {{"source_id": "id7", "fact": "glm45 maps to Glm4MoeModelToolParser."}},\n'
        '    {{"source_id": "id8", "fact": "vLLM v0.17.0 includes Transformers v5 compatibility work."}}\n'
        '  ],\n'
        '  "sub_query_covered": false\n'
        "}}\n"
        "*(Note: These facts do not fully answer the question, but they are strongly relevant and should be kept.)*\n\n"
        "OUTPUT JSON STRUCTURE:\n"
        "{{\n"
        '  "keep_ids": ["id1"],\n'
        '  "evidences_chain": [\n'
        '    {{"source_id": "id1", "fact": "fact text"}}\n'
        '  ],\n'
        '  "sub_query_covered": true\n'
        "}}"
    )

    # --- Reflector Prompts ---
    REFLECTOR_SYSTEM = (
        "You are a reflector for an agentic retrieval QA system.\n\n"

        "SCOPE:\n"
        "- Decide whether the gathered evidence is enough to answer original_query.\n"
        "- If the answer is incomplete, propose one useful next retrieval query.\n"
        "- Do not use outside knowledge, invent facts, or assume missing links.\n\n"

        "EVIDENCE RULES:\n"
        "- Use only facts in accumulated_evidences and sub_queries_status.\n"
        "- You may chain facts only when every link is explicitly supported by evidence.\n"
        "- Review accumulated_evidences first; do not judge from the latest sub-query alone.\n"
        "- Treat weakly related or off-topic facts as retrieval noise, not as a new direction.\n\n"

        "NEXT QUERY RULES:\n"
        "- If answered=false, choose the most important missing evidence for answering original_query.\n"
        "- Prefer a missing aspect adjacent to confirmed evidence: entity -> requested attribute, relationship -> target detail, comparison -> missing side or attribute, mechanism -> condition/cause/effect, cause -> documented consequence or remedy.\n"
        "- Keep new_query grounded in original_query and confirmed evidence.\n"
        "- Do not repeat or paraphrase current_query or the available sub_queries.\n"
        "- Do not pivot to a new topic unless the evidence clearly connects it to original_query.\n"

        "RETURN ONLY VALID JSON.\n\n"

        "If the evidence answers the question:\n"
        "{\n"
        '  "answered": true,\n'
        '  "final_answer": "answer based only on gathered evidence",\n'
        '  "thought": "short evidence chain explaining why the answer is complete"\n'
        "}\n\n"

        "If more retrieval is needed:\n"
        "{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["most important missing evidence"],\n'
        '  "new_query": "one focused retrieval query, or empty string",\n'
        '  "thought": "known evidence, missing evidence, and why this next query is useful"\n'
        "}\n\n"

        "EXAMPLE 1: missing bridge\n"
        "Input facts: ['The novel River Light was written by Dana Lee.']\n"
        "Original query: 'Where did the author of River Light study?'\n"
        "Output:\n"
        "{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["Where Dana Lee studied"],\n'
        '  "new_query": "Where did Dana Lee study?",\n'
        '  "thought": "Known: Dana Lee wrote River Light. Missing: Dana Lee study location. Next query uses the confirmed bridge entity."\n'
        "}\n\n"

        "EXAMPLE 2: complete chain\n"
        "Input facts: ['The Organization was directed by Don Medford.', 'Don Medford died on December 12, 2012.']\n"
        "Original query: 'When did the director of The Organization die?'\n"
        "Output:\n"
        "{\n"
        '  "answered": true,\n'
        '  "final_answer": "The director of The Organization, Don Medford, died on December 12, 2012.",\n'
        '  "thought": "Known: The Organization was directed by Don Medford; Don Medford died on December 12, 2012. The chain answers the original query."\n'
        "}\n\n"

        "EXAMPLE 3: ignore off-topic evidence\n"
        "Input facts: ['The 2012 Winter Youth Olympics were held in Innsbruck.']\n"
        "Original query: 'Which city hosted the 2012 Summer Olympics?'\n"
        "Output:\n"
        "{\n"
        '  "answered": false,\n'
        '  "final_answer": "INCOMPLETE",\n'
        '  "missing_aspects": ["Host city of the 2012 Summer Olympics"],\n'
        '  "new_query": "Which city hosted the 2012 Summer Olympics?",\n'
        '  "thought": "The evidence is about a different event. Return to the original event instead of pivoting to the off-topic fact."\n'
        "}"
    )

    REFLECTOR_NO_PLANNER_SYSTEM = REFLECTOR_SYSTEM

    REFLECTOR_FINAL_SYSTEM = (
        "You are a final answer synthesizer for a retrieval-based QA system.\n\n"
        "MISSION:\n"
        "The search process is requesting a forced answer at the current turn. You MUST produce the best supported answer to the original_query using ONLY the provided evidence.\n\n"
        "STRICT RULES:\n"
        "1. EVIDENCE ONLY: Use only facts explicitly present in accumulated_evidences and sub_queries_status.\n"
        "2. ANSWER THE ORIGINAL QUERY: final_answer must answer original_query, not the intermediate sub-queries.\n"
        "3. IF EVIDENCE IS INCOMPLETE: still provide the best evidence-grounded answer, and explain the limitation in thought.\n"
        "OUTPUT JSON FORMAT:\n"
        "{{\n"
        '  "final_answer": "string",\n'
        '  "thought": "string"\n'
        "}}"
    )

    REFLECTOR_NO_PLANNER_FINAL_SYSTEM = (
        "You are a final answer synthesizer for a retrieval-based QA system.\n\n"
        "MISSION:\n"
        "The search process is requesting a forced answer at the current turn. You MUST produce the best supported answer to the original_query using ONLY the provided evidence.\n\n"
        "STRICT RULES:\n"
        "1. EVIDENCE ONLY: Use only facts explicitly present in accumulated_evidences and sub_queries_status.\n"
        "2. ANSWER THE ORIGINAL QUERY: final_answer must answer original_query, not the intermediate sub-queries.\n"
        "3. IF EVIDENCE IS INCOMPLETE: still provide the best evidence-grounded answer, and explain the limitation in thought.\n"
        "OUTPUT JSON FORMAT:\n"
        "{{\n"
        '  "final_answer": "string",\n'
        '  "thought": "string"\n'
        "}}"
    )



    # --- Ontology / Concept Tree Prompts ---
    ONTOLOGY_RELEVANCE_SYSTEM = (
        "You are a strict JSON generator. You MUST output ONLY valid JSON. No markdown."
    )
    
    ONTOLOGY_RELEVANCE_USER = """
You are an expert search relevance evaluator.
Your task is to judge the relevance between a user query and a set of ontology concept paths.
For each concept, determine the relevance level based on BOTH topic relevance and granularity.
Each candidate may also include a semantic definition and attached document count.
Relevance levels:
- strong: The concept is highly relevant in both topic and granularity
- weak: The concept is topically related but the granularity is not a perfect match
- none: The concept is not relevant to the query

Return ONLY a valid JSON object.
Do NOT include explanations or extra text.

Input Query:
{query}

Concepts:
{concepts_text}

Output JSON format:
{{
  "node_id_1": "strong | weak | none"
}}
""".strip()

    PROMPT_FILTER_NODES = """You decide which concept nodes are semantically relevant to ONE document.

DOCUMENT SUMMARY:
"{doc_text}"

CANDIDATE NODES:
{nodes}

RULES (STRICT):
1) Output JSON with exactly:
   {{ "keep": [node_id, ...] }}

2) Keep a node ONLY IF the document clearly matches the concept chain semantically.
   Do NOT match by surface keywords.
   If a semantic_definition is provided, use it to disambiguate broad concept names.
   If both a broad parent and its more specific child match, prefer the more specific child.
   If two nodes represent distinct core facets of the document, keeping both is allowed.
   Do NOT keep a node for incidental setup details, neighboring tools, or background context unless that is a main topic of the document.

3) Do NOT force matches. If none are suitable, output an empty list.

4) Keep at most {top_k} nodes.

5) Do NOT invent node_id. Only choose from candidates.

STRICT JSON OUTPUT ONLY.
"""

    PROMPT_SPLIT_NODE = """You split a concept node into finer-grained children and assign documents.

CURRENT NODE CONCEPT CHAIN:
{node_chain}

EXISTING CHILDREN (names only, for granularity reference):
{children}

DOCUMENTS (doc_id and summary):
{docs}

RULES (STRICT):
1) Output JSON with exactly:
{{
  "children": [
    {{
      "name": "...",
      "desc": "...",
      "doc_ids": ["doc_id1", "doc_id2"]
    }}
  ],
  "remain_doc_ids": ["doc_id3"]
}}

2) Create 2 or 3 children (no more). Prefer 2 unless 3 is clearly needed.
3) Each child must be a strict semantic subtype of the CURRENT NODE CONCEPT CHAIN, not merely a related tool, dependency, environment, platform, example, or adjacent workflow.
4) New children must be same granularity level and non-overlapping in meaning.
5) Prefer core subject-matter distinctions, not page type distinctions such as error page, API reference, tutorial, release notes, or generic setup context.
6) New children should manage different content from existing children (avoid duplicates).
7) You must assign doc_ids only from the provided list. Do NOT invent ids.
8) It is allowed to keep some documents at the parent in remain_doc_ids.
9) Prefer balanced children. Avoid creating a child for just one or two outlier documents unless it is clearly justified.
10) If a document is ambiguous, cross-cutting, or only loosely related, leave it in remain_doc_ids rather than forcing it into a bad child.
11) If the provided documents are too mixed to form clean strict subtypes, return fewer children and keep more documents in remain_doc_ids.

STRICT JSON OUTPUT ONLY.
"""

    # --- Tree Builder Prompts ---
    NODE_EMBED_PROMPT = "Focus on representing the core semantic scope of a concept node. "

    TREE_GEN_TOPIC = """
You are constructing a hierarchical topic taxonomy.

Parent chain:
{parent_path}

Below are representative documents from one cluster:
{sample_text}

Cluster document count:
{cluster_doc_count}

Generate a JSON object defining the subtopic:
- "name": 2–4 word noun phrase summarizing the shared meaning.
- "description": 1–2 neutral sentences briefly defining the concept.
- The name MUST be a strict subtopic of the parent chain.
- The subtopic should be specific enough to distinguish this cluster from nearby sibling clusters.
- The subtopic must describe the core shared subject, not an incidental dependency, runtime environment, surrounding product, or neighboring workflow.
- Prefer stable semantic concepts over page-format labels or support-context labels.
- NO vague terms like "issues", "changes", "misc", "various".
- NO labels that are primarily about document genre such as "API reference", "error page", "release notes", or "installation" unless that genre is itself the core subject within the parent.
- NO content outside the parent's conceptual scope.
- NO examples, no specific dates, no unrelated domains.
- If the cluster looks mixed, choose the dominant in-scope concept and keep it conservative rather than inventing a broad or cross-domain label.

Output format (must be JSON):
{{
  "name": "...",
  "description": "..."
}}
"""

    TREE_MERGE_CATEGORIES = """
You are constructing a hierarchical taxonomy at level {depth}.

Parent chain:
{parent_path}

Below are raw subtopic candidates:
{raw_topics_json}

Task:
1. Group these candidates into about {target_groups} broader categories.
2. For each group, create:
   - "name": 2–3 word abstract noun phrase.
   - "description": 1–2 sentence definition.
   - "member_names": the list of raw topic names merged into the group.

Rules:
- Keep a consistent abstraction level.
- Avoid vague names (Misc, Other, Various).
- Merge semantically similar topics.
- Prefer reasonably balanced groups when possible.
- Cover the raw topics as completely as possible.
- Avoid duplicate or near-duplicate group names.
- Each group should be semantically distinct from its siblings.
- Keep domains coherent; do not mix unrelated technical themes in one group.
- Use member_names exactly from the provided raw topic names.
- It is acceptable to return fewer than {target_groups} groups if the candidates are tightly related, but do not collapse clearly distinct themes.
- Result must be JSON list.
"""

    TREE_SUMMARIZE_NODE = """
You are defining a semantic node in a hierarchical concept tree.

Concept chain (from root to this node):
{parent_path}

Below are descriptions of its immediate sub-categories:
{child_block}

Task:
Write a concise semantic description (8–15 tokens) that explains:
- What type of information this node represents
- The semantic scope shared by its children
- The level of abstraction (granularity)

Constraints:
- Do NOT list sub-categories
- Do NOT give examples
- Do NOT repeat category names
- Focus on semantic scope, not specific content
- Stay strictly within the parent concept

Return ONLY the description.
"""

    @classmethod
    def get_prompt(cls, name: str) -> str:
        base = getattr(cls, name, "")
        # Prompts that must strictly return JSON only
        json_strict = {
            'QUERY_PLANNER_SYSTEM', 'VERIFIER_SYSTEM', 'REFLECTOR_SYSTEM', 'REFLECTOR_NO_PLANNER_SYSTEM', 'REFLECTOR_FINAL_SYSTEM', 'REFLECTOR_NO_PLANNER_FINAL_SYSTEM',
            'ONTOLOGY_RELEVANCE_SYSTEM', 'ONTOLOGY_RELEVANCE_USER',
            'PROMPT_FILTER_NODES', 'PROMPT_SPLIT_NODE', 'TREE_GEN_TOPIC',
            'TREE_MERGE_CATEGORIES'
        }
        JSON_ONLY_NOTICE = (
            "\nIMPORTANT: You MUST output ONLY valid JSON (no markdown, no explanation, no extra text)."
            " If you cannot produce valid JSON, output an empty JSON object {{}}."
        )
        if name in json_strict and base:
            return base + JSON_ONLY_NOTICE
        return base
