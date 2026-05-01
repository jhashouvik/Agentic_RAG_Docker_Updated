# Use Case Guide — Agentic RAG PDF Q&A System

## What Problem Does This Solve?

Reading through a 100-page research paper, legal contract, financial report, or technical manual to find one specific answer is slow and error-prone. Traditional keyword search (`Ctrl+F`) matches text literally — it cannot understand meaning or answer questions in natural language.

This system lets you **have a conversation with any PDF document**. You upload the document once, and then ask questions in plain English. The system finds the most relevant sections, verifies their relevance using an LLM judge, and generates a grounded answer — always citing the source pages.

Unlike simple chatbots that hallucinate, this system:
- Only answers from content that exists in the document
- Tells you which page(s) the answer came from
- Maintains conversation context so you can ask follow-up questions
- Remembers everything across sessions (SQLite-persisted)

---

## Who Is This For?

| Role | Example Use |
|------|-------------|
| **Researchers / Academics** | Query papers, literature reviews, survey documents |
| **Legal / Compliance Teams** | Search contracts, regulations, policy documents |
| **Financial Analysts** | Extract data from annual reports, earnings calls, audits |
| **Engineers / Developers** | Navigate technical manuals, API docs, architecture specs |
| **Business Analysts** | Summarize and question market research, RFPs, proposals |
| **Students** | Study from textbooks, lecture notes, past papers |

---

## Real-World Use Cases

### 1. Legal Document Review

**Scenario:** A lawyer receives a 200-page merger agreement and needs to quickly find all clauses related to indemnification, termination conditions, and penalty clauses.

**Without this system:** Hours of manual search or expensive legal software.

**With this system:**
```
Upload: merger_agreement.pdf

Q: "What are the indemnification obligations for each party?"
A: "According to Section 9.2 (Page 45), Party A shall indemnify... [Page 45, 46]"

Q: "Under what conditions can the agreement be terminated?"
A: "Section 12 (Page 87) lists five termination triggers... [Page 87, 88]"

Q: "What are the financial penalties for early termination?"
A: "Building on the termination conditions I just described, Page 91 specifies..."
```

**Why it works:** The grade node filters out generic boilerplate clauses and only surfaces sections directly relevant to the question. Conversation memory means follow-up questions build on previous answers.

---

### 2. Financial Report Analysis

**Scenario:** A fund manager needs to analyze Q3 earnings reports from 5 different companies to compare revenue growth, margins, and forward guidance.

**With this system:**
```
Upload: company_q3_report.pdf

Q: "What was total revenue in Q3 and how does it compare to Q3 last year?"
Q: "What drove gross margin improvement?"
Q: "What is management's guidance for Q4?"
Q: "Are there any risks mentioned that could affect the guidance?"
```

Each answer cites the exact page, making it auditable. Analysts can export the conversation history from SQLite for their research notes.

---

### 3. Technical Manual Navigation

**Scenario:** A DevOps engineer is troubleshooting a networking issue and has a 400-page vendor manual. They need to find configuration steps for a specific error code.

**With this system:**
```
Upload: network_device_manual.pdf

Q: "What does error code ERR_LINK_FLAP mean and how do I fix it?"
Q: "What are the recommended MTU settings for this device?"
Q: "How do I enable OSPF on interface GigabitEthernet0/1?"
```

The **token-aware chunking strategy** is ideal here — it handles dense technical content with code snippets and commands without splitting mid-instruction.

---

### 4. Academic Research

**Scenario:** A PhD student is doing a literature review on transformer attention mechanisms and has 30 papers to read.

**With this system:**
```
Upload: attention_is_all_you_need.pdf

Q: "What is the main contribution of this paper?"
Q: "How does multi-head attention differ from single-head attention?"
Q: "What datasets were used for evaluation and what were the BLEU scores?"
Q: "What limitations do the authors acknowledge?"
```

The **document summary** (auto-generated on upload) gives an instant overview. The conversation memory lets students build understanding progressively, just like a tutorial session.

---

### 5. Compliance and Policy Review

**Scenario:** An HR team needs to check whether their company policies are aligned with a newly released regulatory framework (e.g., GDPR, HIPAA, ISO 27001).

**With this system:**
```
Upload: iso_27001_2022.pdf

Q: "What are the mandatory controls in Annex A?"
Q: "What does the standard require for access control policies?"
Q: "How often must internal audits be conducted?"
Q: "What evidence is required for certification?"
```

**Comparison Mode** is especially useful here — you can run the same question through two chunking strategies (e.g., recursive vs. semantic) and see which one gives a more complete answer, then choose the best approach for your document type.

---

### 6. RFP / Proposal Response

**Scenario:** A pre-sales team receives a 60-page RFP and needs to quickly understand requirements, evaluation criteria, and deadlines before deciding whether to bid.

**With this system:**
```
Upload: rfp_enterprise_software.pdf

Q: "What are the mandatory technical requirements?"
Q: "How will proposals be scored and evaluated?"
Q: "What is the submission deadline and format?"
Q: "Are there any incumbent vendor restrictions?"
Q: "What is the estimated contract value and duration?"
```

Within 10 minutes the team has a clear picture of the opportunity — what used to take a full day of reading.

---

## Feature-to-Use-Case Mapping

| Feature | When to Use It |
|---------|---------------|
| **Recursive chunking** | Most documents — best all-round default |
| **Token chunking** | Technical docs with code, tables, equations |
| **Character chunking** | Well-structured reports with clear section headings |
| **Semantic chunking** | Mixed-topic documents, interviews, meeting transcripts |
| **Comparison Mode** | When you want to evaluate two chunking strategies on the same question |
| **GPT-4o** | Complex reasoning, legal analysis, nuanced interpretation |
| **GPT-4o-mini** | Fast lookups, factual extraction, cost-sensitive workflows |
| **Higher Top-K (6-10)** | Broad questions covering multiple document sections |
| **Lower Top-K (2-3)** | Specific factual questions with a single correct answer |
| **Higher Temperature (0.7+)** | Summaries, synthesis, narrative explanations |
| **Lower Temperature (0.0-0.2)** | Factual extraction, compliance checks, structured data |

---

## What Makes This Different from Other PDF Chat Tools

### vs. ChatGPT / Claude with file upload
- **This system** cites exact page numbers for every answer
- **This system** persists conversation history across browser sessions (SQLite)
- **This system** lets you choose and compare chunking strategies
- **This system** self-grades retrieved chunks before generating (reduces hallucination)
- **This system** is fully self-hosted — your documents never leave your infrastructure

### vs. Simple keyword search
- Understands natural language questions, not just keywords
- Answers follow-up questions with awareness of conversation history
- Surfaces semantically related content even if exact words don't match

### vs. Basic RAG implementations
- Self-reflection (grade node) filters irrelevant retrieved chunks before generation
- Dual memory layer: sliding window for LLM context + full SQLite history
- Four pluggable chunking strategies matched to document type
- A/B comparison mode to evaluate chunking quality empirically

---

## End-to-End Workflow

```
1. Start the app
   docker compose up --build
   → Open http://localhost:8501

2. Upload your PDF
   → System extracts text, chunks it, embeds chunks, builds FAISS index
   → Auto-generates a document summary
   → Takes ~10–30 seconds depending on document size

3. Ask your first question
   → System retrieves top-K relevant chunks from FAISS
   → LLM grades each chunk for relevance (self-reflection)
   → LLM generates answer from graded chunks + conversation history
   → Answer displayed with source pages and chunk metadata

4. Ask follow-up questions
   → Previous conversation is automatically injected into context
   → Questions like "Can you expand on that?" work naturally

5. Try Comparison Mode (optional)
   → Enable in sidebar
   → Same question answered by two different chunking strategies
   → See which gives better results for your document type

6. Come back later
   → Conversation history is persisted in SQLite
   → FAISS index is saved to disk
   → Upload the same PDF → session continues where you left off
```

---

## Limitations and Honest Caveats

| Limitation | Explanation |
|------------|-------------|
| **Image-only PDFs** | PDFs that are scans without embedded text cannot be read. Use a text-based PDF or enable OCR (pytesseract). |
| **Very large documents** | PDFs over 200 pages may take 1–2 minutes to index on first upload. |
| **Semantic chunking is slow** | Calls the embedding API per sentence. Use recursive chunking for speed. |
| **Answers are only as good as the document** | If the document doesn't contain the answer, the system will say so rather than hallucinate (grader + RAG prompt discourage speculation). |
| **Single-user by default** | The Docker setup runs one instance. For team use, add authentication or deploy to a cloud platform with multiple replicas. |
| **OpenAI dependency** | Requires an OpenAI API key. Running costs apply per question (~$0.002–$0.02 depending on model). |

---

## Estimated Costs (OpenAI Pricing, 2025)

| Operation | Cost |
|-----------|------|
| Upload & index a 30-page PDF | ~$0.0005–$0.001 |
| One Q&A turn (GPT-4o-mini) | ~$0.002 |
| One Q&A turn (GPT-4o) | ~$0.02 |
| Full session: 1 doc + 20 questions (GPT-4o-mini) | ~$0.05 |
| Full session: 1 doc + 20 questions (GPT-4o) | ~$0.45 |

For high-volume use, GPT-4o-mini gives strong results at 10x lower cost.

---

## Getting Started

See the main [README.md](README.md) for installation, Docker setup, and configuration details.

```bash
# Fastest path to running the system
git clone <repo-url>
cd pdf-qa-v2
echo "OPENAI_API_KEY=sk-your-key-here" > .env
docker compose up --build
# Open http://localhost:8501
```
