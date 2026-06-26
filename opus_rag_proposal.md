# AWS-Based RAG System for Financial Institution

Here is a practical, cost-conscious architecture and implementation plan for a customer-facing, slightly agentic RAG chatbot built on AWS.

## High-Level Architecture

The system splits into five logical layers: **ingestion**, **retrieval/storage**, **orchestration (agent + MCP)**, **serving/API**, and **observability/governance**. Keeping these decoupled lets you swap components (e.g., change the LLM or vector store) without rewriting the system.

```
Website Chat Widget
        │  (WSS/HTTPS)
        ▼
API Gateway (WebSocket) ──► Lambda (orchestrator) ──► Bedrock (LLM + Guardrails)
        │                          │
        │                          ├──► Retrieval (Knowledge Base / OpenSearch)
        │                          └──► MCP Tool Router ──► Lambda tools (interest calc, etc.)
        ▼
   Cognito (auth)                  │
                                   ▼
                          DynamoDB (chat history/session)

Ingestion (async, scheduled):
S3 (docs) + Web Scraper (Lambda/Fargate) ──► Chunking & Embedding ──► Vector Store
```

## Component Choices (Cost-Effective Defaults)

**LLM & embeddings: Amazon Bedrock.** Use Bedrock so you pay per token with no idle GPU cost. Pick a small/cheap model (e.g., Claude Haiku tier or Amazon Nova Lite/Micro) for routine navigation Q&A, and route only complex queries to a larger model. Bedrock also gives you **Guardrails** (PII redaction, denied topics, hallucination/grounding checks), which is essential for a financial institution.

**Knowledge base / retrieval.** Two options depending on scale:
- **Amazon Bedrock Knowledge Bases** (managed RAG): handles chunking, embedding, vector storage, and retrieval with minimal code. Cheapest to operate and fastest to ship. Best starting point.
- **Self-managed: OpenSearch Serverless or pgvector on Aurora Serverless v2.** More control over advanced retrieval (hybrid search, custom rerankers) but more ops cost. Use pgvector on Aurora Serverless if your corpus is small/medium to minimize cost (scales to low capacity when idle).

**Compute: Lambda-first, serverless.** A traffic pattern driven by a website chat is bursty and unpredictable, so serverless avoids paying for idle servers. Use Fargate only for long-running scrape jobs that exceed Lambda's 15-minute limit.

**Session/chat memory: DynamoDB** (on-demand capacity) keyed by session ID, with a TTL to auto-expire old conversations and control storage cost.

**Frontend connection: API Gateway WebSocket API** for streaming token responses to the chat widget; **Cognito** for any authenticated context.

## State-of-the-Art Retrieval Pipeline

To meet the "most state-of-the-art techniques" requirement while staying affordable:

1. **Smart chunking**: semantic/recursive chunking rather than fixed-size, preserving document structure (headings, tables for loan terms).
2. **Hybrid search**: combine dense vector search with keyword/BM25 (OpenSearch supports both natively). This significantly improves recall for exact terms like product names, rates, and fees.
3. **Reranking**: apply a reranker (e.g., Cohere Rerank via Bedrock) on the top-k candidates before passing to the LLM. Big quality gain at low marginal cost since it only runs on a handful of results.
4. **Query rewriting / expansion**: have a cheap model rewrite the user's query using conversation context before retrieval.
5. **Grounded generation with citations**: instruct the model to answer only from retrieved context and return source references. Use Bedrock Guardrails' contextual grounding check to reject ungrounded answers, critical to prevent the bot from inventing loan terms.

## Agentic Layer with MCP Tools

The orchestrator acts as the agent. Keep tools pluggable via **MCP**:

- Each tool (interest calculator, loan eligibility estimator, etc.) is an independent **Lambda function** exposed through a lightweight **MCP server**. Adding/removing a tool becomes a config + deploy change, not a rewrite.
- Maintain a **tool registry** (in DynamoDB or a config file in S3) listing available MCP tools and schemas, loaded at runtime so creators can add/remove tools dynamically.
- The agent loop: receive query → retrieve context → decide if a tool is needed → call MCP tool → synthesize final grounded answer. **Bedrock Agents** can manage this loop natively, or you can run a lightweight custom loop in Lambda for more control and lower cost.
- Keep tools **deterministic and validated** (e.g., interest calc returns exact numbers), so the LLM presents computed results rather than estimating them.

## Ingestion Pipeline

- **Documents**: upload to **S3**; an S3 event triggers an embedding/indexing Lambda. Bedrock Knowledge Bases can sync directly from an S3 bucket.
- **Website scraping**: scheduled **EventBridge** rule triggers a scraper (Lambda for small sites, Fargate for large) that writes cleaned text to S3, reusing the same indexing path.
- **Incremental updates**: track content hashes to re-embed only changed pages, saving embedding API costs.

## Security & Governance (Non-Negotiable for Finance)

- Restrict the corpus to **public information only**; isolate the knowledge base account/bucket and never connect it to internal customer data stores.
- **Bedrock Guardrails** for PII filtering, denied topics (e.g., refuse personalized financial advice), and grounding.
- **WAF** on API Gateway, **Cognito** auth, encryption at rest (KMS) and in transit, and full **CloudTrail/CloudWatch** logging for auditability and compliance.
- Add a disclaimer and a confidence/fallback path ("I can't answer that, here's how to reach support") to avoid liability from wrong answers.

## Cost-Optimization Summary

- **Serverless everywhere** (Lambda, Aurora/OpenSearch Serverless, DynamoDB on-demand) to pay only for usage.
- **Model tiering**: cheap model for most queries, expensive model only when needed.
- **Caching**: cache embeddings and cache answers to frequent FAQ-style questions (e.g., in DynamoDB/ElastiCache) to cut LLM calls.
- **Start managed (Bedrock Knowledge Bases)**, then migrate to self-managed OpenSearch only if scale or retrieval-quality needs justify it.
- **Incremental re-indexing** to minimize embedding costs.

## Suggested Rollout Phases

1. **MVP**: Bedrock Knowledge Base + Bedrock Agent + S3 docs + simple chat widget. Validate answer quality fast and cheap.
2. **Enhance retrieval**: add hybrid search + reranking + query rewriting once you have real query logs.
3. **Add agentic tools**: introduce the MCP tool layer and registry for interest/loan calculations.
4. **Harden**: guardrails, WAF, audit logging, evaluation pipeline, and cost monitoring/alerts.