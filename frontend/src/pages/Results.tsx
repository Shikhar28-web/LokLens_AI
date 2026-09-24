import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ShieldCheck, ShieldAlert, ShieldQuestion, AlertTriangle,
  CheckCircle2, XCircle, MinusCircle, ChevronRight, ExternalLink
} from "lucide-react";
import { getReport, type FullReport } from "../services/api";

// ── Helpers ────────────────────────────────────────────────────────────────────

type VerdictLevel =
  | "SUPPORTED" | "LIKELY_SUPPORTED" | "PARTIALLY_SUPPORTED"
  | "MISLEADING_CONTEXT" | "UNSUPPORTED" | "CONTRADICTED"
  | "INSUFFICIENT_EVIDENCE"
  | "LIKELY_AUTHENTIC" | "POSSIBLY_AUTHENTIC"
  | "POSSIBLY_MANIPULATED" | "LIKELY_MANIPULATED"
  | "POSSIBLY_AI_GENERATED" | "LIKELY_AI_GENERATED"
  | "INCONCLUSIVE";

const VERDICT_META: Record<VerdictLevel, { color: string; bg: string; Icon: React.ElementType }> = {
  SUPPORTED:            { color: "text-verdict-supported", bg: "bg-verdict-supported/15 border-verdict-supported/30", Icon: CheckCircle2 },
  LIKELY_SUPPORTED:     { color: "text-verdict-likely", bg: "bg-verdict-likely/15 border-verdict-likely/30", Icon: CheckCircle2 },
  PARTIALLY_SUPPORTED:  { color: "text-verdict-partial", bg: "bg-verdict-partial/15 border-verdict-partial/30", Icon: ShieldQuestion },
  MISLEADING_CONTEXT:   { color: "text-verdict-misleading", bg: "bg-verdict-misleading/15 border-verdict-misleading/30", Icon: AlertTriangle },
  UNSUPPORTED:          { color: "text-verdict-unsupported", bg: "bg-verdict-unsupported/15 border-verdict-unsupported/30", Icon: ShieldAlert },
  CONTRADICTED:         { color: "text-verdict-contradicted", bg: "bg-verdict-contradicted/15 border-verdict-contradicted/30", Icon: XCircle },
  INSUFFICIENT_EVIDENCE:{ color: "text-slate-400", bg: "bg-surface-700 border-surface-500", Icon: MinusCircle },
  LIKELY_AUTHENTIC:     { color: "text-verdict-authentic", bg: "bg-verdict-authentic/15 border-verdict-authentic/30", Icon: CheckCircle2 },
  POSSIBLY_AUTHENTIC:   { color: "text-verdict-likely", bg: "bg-verdict-likely/15 border-verdict-likely/30", Icon: ShieldCheck },
  POSSIBLY_MANIPULATED: { color: "text-verdict-misleading", bg: "bg-verdict-misleading/15 border-verdict-misleading/30", Icon: ShieldAlert },
  LIKELY_MANIPULATED:   { color: "text-verdict-manipulated", bg: "bg-verdict-manipulated/15 border-verdict-manipulated/30", Icon: ShieldAlert },
  POSSIBLY_AI_GENERATED:{ color: "text-verdict-ai", bg: "bg-verdict-ai/15 border-verdict-ai/30", Icon: ShieldQuestion },
  LIKELY_AI_GENERATED:  { color: "text-verdict-ai", bg: "bg-verdict-ai/15 border-verdict-ai/30", Icon: XCircle },
  INCONCLUSIVE:         { color: "text-slate-400", bg: "bg-surface-700 border-surface-500", Icon: MinusCircle },
};

function VerdictBadge({ verdict, label }: { verdict: string; label?: string }) {
  const meta = VERDICT_META[verdict as VerdictLevel] ?? VERDICT_META.INCONCLUSIVE;
  const { Icon, color, bg } = meta;
  return (
    <div className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${bg} ${color} font-semibold text-sm`}>
      <Icon size={16} />
      {label && <span className="text-slate-500 text-xs font-normal">{label}:</span>}
      {verdict.replace(/_/g, " ")}
    </div>
  );
}

function ConfidenceBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="score-bar-track">
      <div className={`score-bar-fill ${color}`} style={{ width: `${value * 100}%` }} />
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function Results() {
  const { id } = useParams<{ id: string }>();
  const [report, setReport] = useState<FullReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) return;
    getReport(id)
      .then(setReport)
      .catch(() => setError("Failed to load report. The analysis may still be running."))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-8 h-8" />
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-xl mx-auto px-6 py-20 text-center space-y-4">
        <XCircle size={40} className="text-red-400 mx-auto" />
        <p className="text-slate-300">{error}</p>
        <Link to="/upload" className="btn-primary inline-flex items-center gap-2">
          New Verification <ChevronRight size={14} />
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-16 space-y-10 animate-fade-in">
      {/* ── Header ── */}
      <div className="space-y-2">
        <p className="section-label">Verification report</p>
        <h1 className="text-3xl font-bold text-white">Analysis Results</h1>
        <p className="text-xs text-slate-600 font-mono">Submission: {id}</p>
      </div>

      {/* ── Verdict summary ── */}
      <div className="glass p-6 sm:p-8 space-y-6">
        <div className="flex flex-wrap gap-3">
          {report.claim_verdict && (
            <VerdictBadge verdict={report.claim_verdict} label="Claim" />
          )}
          {report.image_verdict && (
            <VerdictBadge verdict={report.image_verdict} label="Image" />
          )}
        </div>

        {/* Confidence bars */}
        {(report.claim_confidence != null || report.image_confidence != null) && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {report.claim_confidence != null && (
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Claim confidence</span>
                  <span className="font-mono">{(report.claim_confidence * 100).toFixed(0)}%</span>
                </div>
                <ConfidenceBar value={report.claim_confidence} color="bg-brand-500" />
              </div>
            )}
            {report.image_confidence != null && (
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Image confidence</span>
                  <span className="font-mono">{(report.image_confidence * 100).toFixed(0)}%</span>
                </div>
                <ConfidenceBar value={report.image_confidence} color="bg-purple-500" />
              </div>
            )}
          </div>
        )}

        {/* Explanation */}
        {report.summary && (
          <div className="p-4 rounded-xl bg-surface-700/60 border border-white/5">
            <p className="section-label mb-2">Explanation</p>
            <p className="text-sm text-slate-300 leading-relaxed">{report.summary}</p>
          </div>
        )}

        {/* Source stats */}
        <div className="grid grid-cols-3 gap-4 text-center">
          {[
            { label: "Independent sources", value: report.independent_source_count ?? "—" },
            { label: "Supporting evidence", value: report.supporting_evidence.length },
            { label: "Contradicting evidence", value: report.contradicting_evidence.length },
          ].map(({ label, value }) => (
            <div key={label} className="glass p-4 space-y-1">
              <p className="text-2xl font-bold text-white">{value}</p>
              <p className="text-xs text-slate-500">{label}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Claims ── */}
      {report.claims.length > 0 && (
        <section className="space-y-4">
          <p className="section-label">Extracted claims ({report.claims.length})</p>
          {report.claims.map((claim) => (
            <div key={claim.id} className="glass p-5 space-y-3">
              <p className="text-sm text-slate-200 leading-relaxed">{claim.claim_text}</p>
              {claim.entities_json && (
                <div className="flex flex-wrap gap-2">
                  {Object.entries(claim.entities_json).flatMap(([type, vals]) =>
                    (vals as string[]).map(v => (
                      <span key={`${type}-${v}`}
                        className="text-xs px-2 py-0.5 rounded-full bg-brand-500/10 text-brand-300 border border-brand-500/20">
                        {type}: {v}
                      </span>
                    ))
                  )}
                </div>
              )}
            </div>
          ))}
        </section>
      )}

      {/* ── Evidence sections ── */}
      {[
        { title: "Supporting evidence", items: report.supporting_evidence, accent: "text-verdict-supported" },
        { title: "Contradicting evidence", items: report.contradicting_evidence, accent: "text-verdict-contradicted" },
      ].map(({ title, items, accent }) =>
        items.length > 0 ? (
          <section key={title} className="space-y-4">
            <p className="section-label">{title} ({items.length})</p>
            {items.map((ev) => (
              <div key={ev.id} className="glass p-5 space-y-3">
                <p className="text-sm text-slate-300 leading-relaxed italic">"{ev.evidence_text}"</p>
                <div className="flex items-center gap-4 text-xs text-slate-500">
                  <span>Relevance: <span className={`font-mono ${accent}`}>{(ev.relevance_score * 100).toFixed(0)}%</span></span>
                  <span>Method: <span className="font-mono">{ev.retrieval_method}</span></span>
                </div>
              </div>
            ))}
          </section>
        ) : null
      )}

      {/* ── Sources ── */}
      {report.sources.length > 0 && (
        <section className="space-y-4">
          <p className="section-label">Sources ({report.sources.length})</p>
          {report.sources.map((src) => (
            <div key={src.id} className="glass p-4 flex items-center justify-between gap-4">
              <div className="space-y-1 min-w-0">
                <p className="text-sm text-white font-medium truncate">{src.domain}</p>
                <p className="text-xs text-slate-500 capitalize">{src.source_type.replace("_", " ")} ·
                  Quality: <span className="font-mono">{(src.quality_score * 100).toFixed(0)}%</span>
                </p>
              </div>
              <a href={src.url} target="_blank" rel="noopener noreferrer"
                className="shrink-0 text-slate-500 hover:text-brand-400 transition-colors">
                <ExternalLink size={14} />
              </a>
            </div>
          ))}
        </section>
      )}

      {/* ── Limitations ── */}
      {report.limitations.length > 0 && (
        <section className="glass p-5 space-y-3 border-yellow-500/20">
          <p className="section-label text-yellow-500">Analysis limitations</p>
          <ul className="space-y-1.5">
            {report.limitations.map((l, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-400">
                <AlertTriangle size={13} className="text-yellow-500 shrink-0 mt-0.5" /> {l}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ── New verification CTA ── */}
      <div className="flex justify-center">
        <Link to="/upload" className="btn-secondary flex items-center gap-2">
          Run another verification <ChevronRight size={14} />
        </Link>
      </div>
    </div>
  );
}
