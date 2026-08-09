# AI Research Copilot

![CI](https://github.com/UtkarshOver9000/ai-research-copilot/actions/workflows/ci.yml/badge.svg)

Upload documents (or paste text), ask a question, get ranked passages with real citations
back to the source — not a black box. Retrieval runs entirely server-side with zero
external API calls; generative answers are available as an optional local-only feature.

**Live demo: https://ai-research-copilot-3jwt.vercel.app** — no signup, no API key, click
"Load example" and ask a question.

Backend API: https://ai-research-copilot-delta.vercel.app/docs

## How it works

1. **Chunking**: documents are split into overlapping word-window passages.
2. **Indexing**: TF-IDF + truncated SVD (classic Latent Semantic Analysis) builds a
   lightweight semantic representation of every chunk.
3. **Retrieval**: your question goes through the same pipeline, ranked by cosine
   similarity against every chunk, returned with citations.
4. **Generation (optional, local only)**: if you set your own `OPENAI_API_KEY`, the top
   passages get synthesized into a cited natural-language answer. The public live demo
   runs retrieval-only — no external API cost or abuse surface on a public endpoint.

**Why TF-IDF+SVD instead of a transformer embedding model** (e.g. sentence-transformers)?
That gives better retrieval quality but pulls in `torch` and a multi-hundred-MB model,
which doesn't fit Vercel's serverless size limits. TF-IDF+SVD is scikit-learn only,
trains in milliseconds per request, and — per the benchmark below — is still genuinely
useful. This is a documented tradeoff, not a hidden one.

## Retrieval benchmark

No real labeled Q&A dataset exists for this, so `backend/src/copilot/synthetic_eval.py`
generates one: 40 synthetic documents sharing prefixes and metric types with each other
(e.g. multiple "Team X" documents, multiple documents reporting "uptime" in the same
quarter) so queries have genuine, confusable distractors — not just irrelevant filler.

```bash
cd backend && PYTHONPATH=src python -m copilot.synthetic_eval
```

| Metric | Score |
|---|---|
| Recall@1 | 95.0% |
| Recall@3 | 100.0% |
| Recall@5 | 100.0% |
| Recall@10 | 100.0% |
| MRR | 0.975 |

An earlier, easier version of this benchmark (12 non-confusable documents) scored a
suspicious 1.0/1.0 — not a real result, just an under-powered test. Scaling to 40
documents with deliberate distractors is what makes 95% at strict top-1 credible.

## Real bugs found while building this

- **`TruncatedSVD` crashed on small corpora.** `n_components` was bounded by chunk count
  but not by the actual TF-IDF vocabulary size, so `n_components(150) > n_features(149)`
  on a modest document set. Fixed in `index.py`.
- **A misconfigured API key crashed retrieval.** This machine had a literal
  `OPENAI_API_KEY=your_api_key_here` set system-wide (leftover from a different project's
  README example). `/v1/query` now catches generation failures and falls back to
  retrieval-only instead of failing the whole request — verified against that exact key.
- **The deployed frontend crashed on load** with `Cannot read properties of undefined
  (reading 'toFixed')`. Root cause: `NEXT_PUBLIC_API_BASE_URL` wasn't set in the Vercel
  project, so the build fell back to a `localhost:8000` default that's unreachable in
  production, and the resulting bad response wasn't shape-checked before being rendered.
  Fixed by validating the response shape before use, regardless of what the deployment
  config is set to.

## Run it locally

**Backend:**
```bash
cd backend
python -m venv .venv && .venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m uvicorn copilot.api.app:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local
npm run dev
```

## Tests

```bash
cd backend
pytest --cov=src --cov-report=term-missing
```

25 tests, 95% coverage: chunking, the retrieval index, PDF/text ingestion, the synthetic
benchmark, and the API (including the graceful-degradation-on-bad-key regression test).
`npm run lint && npm run build` for the frontend. CI runs both on every push/PR across
Python 3.10–3.12.

## Project layout

```
backend/
  src/copilot/
    chunking.py       word-window document splitting
    index.py           TF-IDF + SVD retrieval index
    generate.py         optional local LLM answer synthesis
    synthetic_eval.py   labeled benchmark (Recall@k, MRR)
    ingest.py            PDF/text extraction
    api/app.py            FastAPI app (stateless: rebuilds the index per request)
  tests/
frontend/
  src/app/page.tsx        document upload/paste, question box, ranked citations
```

## Limitations & honest notes

- **Stateless by design.** Each query rebuilds the index from the documents sent in that
  same request rather than keeping server-side session state, since serverless
  deployments can't guarantee two requests land on the same warm instance. Fine for
  demo-sized documents; not optimized for large corpora queried repeatedly.
- **TF-IDF+SVD, not a transformer embedding model** — see "How it works" above. Good
  enough to beat 95% Recall@1 on a deliberately hard synthetic benchmark, not
  state-of-the-art on nuanced paraphrased queries.
- **The public live demo has no labeled real-world benchmark**, only the synthetic one
  above — same caveat as this author's other repos: real-world validation is still
  the natural next step, not something to claim without evidence.
- **Generation is local-only** by design (see architecture above), not a limitation to
  fix — a public endpoint calling a paid LLM API is a cost/abuse surface this project
  deliberately avoids.

## License

MIT
