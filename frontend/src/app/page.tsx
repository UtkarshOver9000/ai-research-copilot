"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

type Doc = { name: string; text: string };

type Citation = {
  doc_name: string;
  position: number;
  text: string;
  score: number;
};

type QueryResponse = {
  question: string;
  citations: Citation[];
  answer: string | null;
  generation_available: boolean;
  timestamp: string;
};

type Benchmark = {
  num_documents: number;
  num_chunks: number;
  num_queries: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  recall_at_10: number;
  mrr: number;
};

const EXAMPLE_DOCS: Doc[] = [
  {
    name: "rockets.txt",
    text:
      "Rockets use controlled combustion of propellant to generate thrust. Multi-stage " +
      "rockets discard empty fuel stages to reduce mass during ascent, which is why most " +
      "orbital launch vehicles are built in two or three stages rather than one. Orbital " +
      "launches require reaching a horizontal velocity of roughly 7.8 km/s so the vehicle " +
      "falls around the Earth rather than back onto it -- this is what distinguishes an " +
      "orbital launch from a simple suborbital hop.",
  },
  {
    name: "sourdough.txt",
    text:
      "Sourdough bread relies on a fermented starter culture containing wild yeast and " +
      "lactobacilli bacteria living in symbiosis. The long, slow fermentation develops " +
      "flavor and a chewy crumb structure distinct from commercially yeasted bread, and " +
      "also produces lactic and acetic acid, which is what gives sourdough its " +
      "characteristic tang.",
  },
];

const EXAMPLE_QUESTION = "What velocity do orbital launches need to reach?";

export default function Home() {
  const [documents, setDocuments] = useState<Doc[]>([]);
  const [docName, setDocName] = useState("");
  const [docText, setDocText] = useState("");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/v1/benchmark`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`status ${r.status}`))))
      .then((data) => {
        // Guard against a malformed/unexpected response shape (e.g. hitting
        // the wrong host) silently crashing the page on a missing field.
        if (data && typeof data.recall_at_1 === "number" && typeof data.recall_at_5 === "number") {
          setBenchmark(data);
        } else {
          setBenchmark(null);
        }
      })
      .catch(() => setBenchmark(null));
  }, []);

  function addTextDocument() {
    if (!docName.trim() || !docText.trim()) return;
    setDocuments((prev) => [...prev, { name: docName.trim(), text: docText.trim() }]);
    setDocName("");
    setDocText("");
  }

  function removeDocument(index: number) {
    setDocuments((prev) => prev.filter((_, i) => i !== index));
  }

  function loadExample() {
    setDocuments(EXAMPLE_DOCS);
    setQuestion(EXAMPLE_QUESTION);
    setResult(null);
    setError(null);
  }

  async function handleFileUpload(file: File) {
    setUploading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/v1/extract`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Extraction failed (${res.status})`);
      const data = await res.json();
      setDocuments((prev) => [...prev, { name: data.name || file.name, text: data.text }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function askQuestion() {
    if (!question.trim() || documents.length === 0) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents, question, top_k: 5 }),
      });
      if (!res.ok) throw new Error(`Query failed (${res.status})`);
      const data: QueryResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="flex-1 max-w-5xl mx-auto w-full px-6 py-10">
      <header className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-lg">
            📚
          </div>
          <h1 className="text-xl font-bold tracking-tight">AI Research Copilot</h1>
        </div>
        <p className="text-sm text-slate-400 max-w-2xl">
          Upload documents or paste text, ask a question, and get ranked passages with real
          citations back to the source. Retrieval runs entirely server-side (TF-IDF + LSA), no
          external API calls required.
        </p>
      </header>

      {benchmark && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-8">
          {[
            ["Recall@1", `${(benchmark.recall_at_1 * 100).toFixed(0)}%`],
            ["Recall@5", `${(benchmark.recall_at_5 * 100).toFixed(0)}%`],
            ["MRR", benchmark.mrr.toFixed(3)],
            ["Docs benchmarked", benchmark.num_documents],
            ["Chunks", benchmark.num_chunks],
          ].map(([label, value]) => (
            <div key={label} className="bg-slate-900 border border-slate-800 rounded-lg p-3">
              <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">
                {label}
              </div>
              <div className="text-lg font-bold">{value}</div>
            </div>
          ))}
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-6">
        <section className="bg-slate-900 border border-slate-800 rounded-lg p-5">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-bold">Documents</h2>
            <button
              onClick={loadExample}
              className="text-xs px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 transition-colors"
            >
              ✨ Load example
            </button>
          </div>

          <div className="space-y-2 mb-4">
            {documents.map((doc, i) => (
              <div
                key={i}
                className="flex items-center justify-between bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-xs"
              >
                <span className="truncate">{doc.name}</span>
                <button
                  onClick={() => removeDocument(i)}
                  className="text-slate-500 hover:text-red-400 ml-2"
                >
                  ✕
                </button>
              </div>
            ))}
            {documents.length === 0 && (
              <p className="text-xs text-slate-500">No documents yet.</p>
            )}
          </div>

          <div className="space-y-2 mb-3">
            <input
              value={docName}
              onChange={(e) => setDocName(e.target.value)}
              placeholder="Document name"
              className="w-full bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-xs focus:outline-none focus:border-blue-600"
            />
            <textarea
              value={docText}
              onChange={(e) => setDocText(e.target.value)}
              placeholder="Paste document text here..."
              rows={4}
              className="w-full bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-xs focus:outline-none focus:border-blue-600"
            />
            <button
              onClick={addTextDocument}
              className="w-full text-xs py-2 rounded-md bg-slate-800 hover:bg-slate-700 transition-colors"
            >
              + Add pasted text
            </button>
          </div>

          <label className="block text-xs text-center py-2 rounded-md border border-dashed border-slate-700 hover:border-blue-600 cursor-pointer transition-colors">
            {uploading ? "Extracting..." : "📄 Upload a .pdf or .txt file"}
            <input
              type="file"
              accept=".pdf,.txt"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
            />
          </label>
        </section>

        <section className="bg-slate-900 border border-slate-800 rounded-lg p-5 flex flex-col">
          <h2 className="text-sm font-bold mb-3">Ask a question</h2>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="What do you want to know?"
            rows={2}
            className="w-full bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-xs mb-3 focus:outline-none focus:border-blue-600"
          />
          <button
            onClick={askQuestion}
            disabled={loading || documents.length === 0 || !question.trim()}
            className="w-full py-2 rounded-md bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-600 text-sm font-semibold transition-colors mb-4"
          >
            {loading ? "Searching..." : "▶ Ask"}
          </button>

          {error && (
            <div className="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 mb-3">
              {error}
            </div>
          )}

          {result && (
            <div className="flex-1 overflow-y-auto space-y-3">
              {result.answer && (
                <div className="bg-blue-950/30 border border-blue-900 rounded-md px-3 py-2 text-xs">
                  <div className="text-[10px] uppercase text-blue-400 font-bold mb-1">
                    Generated answer
                  </div>
                  {result.answer}
                </div>
              )}
              {!result.generation_available && (
                <p className="text-[11px] text-slate-500 italic">
                  Generative answers are off in this public demo (no API key configured).
                  Retrieval below is real and unmodified.
                </p>
              )}
              {result.citations.map((c, i) => (
                <div
                  key={i}
                  className="bg-slate-950 border border-slate-800 rounded-md px-3 py-2 text-xs"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-slate-300">
                      {c.doc_name} · passage {c.position}
                    </span>
                    <span className="text-blue-400 font-mono">{c.score.toFixed(3)}</span>
                  </div>
                  <p className="text-slate-400">{c.text}</p>
                </div>
              ))}
              {result.citations.length === 0 && (
                <p className="text-xs text-slate-500">No results.</p>
              )}
            </div>
          )}
        </section>
      </div>

      <footer className="mt-10 text-xs text-slate-500">
        <a href="/docs" target="_blank" className="text-blue-400 hover:underline">
          Interactive API docs
        </a>{" "}
        ·{" "}
        <a
          href="https://github.com/UtkarshOver9000/ai-research-copilot"
          target="_blank"
          className="text-blue-400 hover:underline"
        >
          Source on GitHub
        </a>
      </footer>
    </main>
  );
}
