import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Loader2, CheckCircle2, XCircle, Clock, ArrowRight } from "lucide-react";
import { getSubmission, type Submission } from "../services/api";

const STAGES = [
  "Input validation",
  "Text preprocessing",
  "Claim extraction",
  "Query generation",
  "Web search",
  "Document retrieval",
  "TF-IDF / BM25 ranking",
  "Evidence extraction",
  "Contradiction detection",
  "Source quality scoring",
  "Image forensics",
  "OCR analysis",
  "Perceptual hashing",
  "Multimodal consistency",
  "Verdict engine",
  "Generating report",
];

function StatusIcon({ status }: { status: Submission["status"] }) {
  if (status === "complete") return <CheckCircle2 size={18} className="text-verdict-supported" />;
  if (status === "error") return <XCircle size={18} className="text-verdict-contradicted" />;
  return <Loader2 size={18} className="text-brand-400 animate-spin" />;
}

export default function Progress() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [submission, setSubmission] = useState<Submission | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  // Fake stage progress for UI (real stage tracking added in Phase 2)
  const fakeStage = Math.min(Math.floor(elapsed / 2), STAGES.length - 1);

  useEffect(() => {
    if (!id) return;

    const poll = async () => {
      try {
        const data = await getSubmission(id);
        setSubmission(data);
        if (data.status === "complete") {
          navigate(`/submissions/${id}/results`);
        } else if (data.status === "error") {
          setError(data.error_message || "Analysis failed.");
        }
      } catch {
        setError("Could not reach the verification server.");
      }
    };

    poll();
    const interval = setInterval(poll, 3000);
    const timer = setInterval(() => setElapsed(e => e + 1), 1000);

    return () => { clearInterval(interval); clearInterval(timer); };
  }, [id, navigate]);

  return (
    <div className="max-w-2xl mx-auto px-6 py-20 space-y-12 animate-fade-in">
      {/* Header */}
      <div className="text-center space-y-3">
        <div className="w-16 h-16 rounded-2xl bg-brand-500/10 flex items-center justify-center mx-auto">
          <StatusIcon status={submission?.status ?? "processing"} />
        </div>
        <h1 className="text-2xl font-bold text-white">
          {submission?.status === "processing" ? "Analyzing…" : "Queued"}
        </h1>
        <p className="text-slate-400 text-sm">
          {submission?.status === "processing"
            ? "The pipeline is running. This typically takes 20–60 seconds."
            : "Your submission is queued and will begin shortly."}
        </p>
        {elapsed > 0 && (
          <p className="text-xs text-slate-600 font-mono">
            <Clock size={11} className="inline mr-1" />{elapsed}s elapsed
          </p>
        )}
      </div>

      {/* Stage list */}
      <div className="glass p-6 space-y-3">
        <p className="section-label">Pipeline stages</p>
        <div className="space-y-2 mt-3">
          {STAGES.map((stage, i) => {
            const done = i < fakeStage;
            const active = i === fakeStage && submission?.status === "processing";
            return (
              <div key={stage}
                className={`flex items-center gap-3 text-sm transition-all duration-300 ${
                  done ? "text-slate-300" : active ? "text-brand-300" : "text-slate-600"
                }`}>
                <div className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 ${
                  done ? "bg-verdict-supported/20" : active ? "bg-brand-500/20" : "bg-surface-600"
                }`}>
                  {done ? <CheckCircle2 size={10} className="text-verdict-supported" />
                        : active ? <Loader2 size={10} className="animate-spin text-brand-400" />
                        : <span className="w-1.5 h-1.5 rounded-full bg-slate-600 block" />}
                </div>
                <span className={active ? "font-medium" : ""}>{stage}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="glass p-6 space-y-4 border-red-500/20">
          <div className="flex items-center gap-2 text-red-400 font-semibold">
            <XCircle size={18} /> Analysis Failed
          </div>
          <p className="text-sm text-slate-400">{error}</p>
          <button onClick={() => navigate("/upload")} className="btn-secondary flex items-center gap-2">
            Try Again <ArrowRight size={14} />
          </button>
        </div>
      )}

      {/* Submission ID */}
      <p className="text-center text-xs text-slate-700 font-mono">
        Submission ID: {id}
      </p>
    </div>
  );
}
