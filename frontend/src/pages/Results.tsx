import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  ShieldCheck, ShieldAlert, ShieldQuestion, AlertTriangle,
  CheckCircle2, XCircle, MinusCircle, ChevronRight, ExternalLink,
  Globe, Share2, Search, Cpu, Sparkles, MessageSquare,
  FileText, Copy, Check
} from "lucide-react";
import { getReport, type FullReport, type SocialMediaEvidence } from "../services/api";

// ── Helpers ────────────────────────────────────────────────────────────────────

type VerdictLevel =
  | "SUPPORTED" | "LIKELY_SUPPORTED" | "PARTIALLY_SUPPORTED"
  | "MISLEADING_CONTEXT" | "MISLEADING" | "LIKELY_MISLEADING"
  | "UNSUPPORTED" | "CONTRADICTED"
  | "INSUFFICIENT_EVIDENCE" | "UNVERIFIED" | "OCR_FAILED" | "NO_TEXT_FOUND"
  | "REAL" | "FAKE"
  | "LIKELY_AUTHENTIC" | "POSSIBLY_AUTHENTIC"
  | "POSSIBLY_MANIPULATED" | "LIKELY_MANIPULATED"
  | "POSSIBLY_AI_GENERATED" | "LIKELY_AI_GENERATED"
  | "INCONCLUSIVE" | "UNKNOWN";

const VERDICT_META: Record<VerdictLevel, { color: string; bg: string; Icon: React.ElementType }> = {
  SUPPORTED:            { color: "text-verdict-supported", bg: "bg-verdict-supported/15 border-verdict-supported/30", Icon: CheckCircle2 },
  REAL:                 { color: "text-verdict-supported", bg: "bg-verdict-supported/15 border-verdict-supported/30", Icon: CheckCircle2 },
  LIKELY_SUPPORTED:     { color: "text-verdict-likely", bg: "bg-verdict-likely/15 border-verdict-likely/30", Icon: CheckCircle2 },
  PARTIALLY_SUPPORTED:  { color: "text-verdict-partial", bg: "bg-verdict-partial/15 border-verdict-partial/30", Icon: ShieldQuestion },
  MISLEADING_CONTEXT:   { color: "text-verdict-misleading", bg: "bg-verdict-misleading/15 border-verdict-misleading/30", Icon: AlertTriangle },
  MISLEADING:           { color: "text-verdict-misleading", bg: "bg-verdict-misleading/15 border-verdict-misleading/30", Icon: AlertTriangle },
  LIKELY_MISLEADING:    { color: "text-verdict-misleading", bg: "bg-verdict-misleading/15 border-verdict-misleading/30", Icon: AlertTriangle },
  UNSUPPORTED:          { color: "text-verdict-unsupported", bg: "bg-verdict-unsupported/15 border-verdict-unsupported/30", Icon: ShieldAlert },
  CONTRADICTED:         { color: "text-verdict-contradicted", bg: "bg-verdict-contradicted/15 border-verdict-contradicted/30", Icon: XCircle },
  FAKE:                 { color: "text-verdict-contradicted", bg: "bg-verdict-contradicted/15 border-verdict-contradicted/30", Icon: XCircle },
  INSUFFICIENT_EVIDENCE:{ color: "text-slate-400", bg: "bg-surface-700 border-surface-500", Icon: MinusCircle },
  UNVERIFIED:           { color: "text-slate-400", bg: "bg-surface-700 border-surface-500", Icon: MinusCircle },
  OCR_FAILED:           { color: "text-verdict-unsupported", bg: "bg-verdict-unsupported/15 border-verdict-unsupported/30", Icon: AlertTriangle },
  NO_TEXT_FOUND:        { color: "text-brand-400", bg: "bg-brand-500/15 border-brand-500/30", Icon: Sparkles },
  LIKELY_AUTHENTIC:     { color: "text-verdict-authentic", bg: "bg-verdict-authentic/15 border-verdict-authentic/30", Icon: CheckCircle2 },
  POSSIBLY_AUTHENTIC:   { color: "text-verdict-likely", bg: "bg-verdict-likely/15 border-verdict-likely/30", Icon: ShieldCheck },
  POSSIBLY_MANIPULATED: { color: "text-verdict-misleading", bg: "bg-verdict-misleading/15 border-verdict-misleading/30", Icon: ShieldAlert },
  LIKELY_MANIPULATED:   { color: "text-verdict-manipulated", bg: "bg-verdict-manipulated/15 border-verdict-manipulated/30", Icon: ShieldAlert },
  POSSIBLY_AI_GENERATED:{ color: "text-verdict-ai", bg: "bg-verdict-ai/15 border-verdict-ai/30", Icon: ShieldQuestion },
  LIKELY_AI_GENERATED:  { color: "text-verdict-ai", bg: "bg-verdict-ai/15 border-verdict-ai/30", Icon: XCircle },
  INCONCLUSIVE:         { color: "text-slate-400", bg: "bg-surface-700 border-surface-500", Icon: MinusCircle },
  UNKNOWN:              { color: "text-slate-400", bg: "bg-surface-700 border-surface-500", Icon: MinusCircle },
};

function VerdictBadge({ verdict, label }: { verdict: string; label?: string }) {
  const meta = VERDICT_META[verdict as VerdictLevel] ?? VERDICT_META.INCONCLUSIVE;
  const { Icon, color, bg } = meta;
  return (
    <div className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${bg} ${color} font-semibold text-sm shadow-sm`}>
      <Icon size={16} />
      {label && <span className="text-slate-400 text-xs font-normal">{label}:</span>}
      {verdict.replace(/_/g, " ")}
    </div>
  );
}

function ConfidenceBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="score-bar-track h-2 bg-surface-700 rounded-full overflow-hidden">
      <div className={`score-bar-fill h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} />
    </div>
  );
}

// ── Social Media Platform Badge Helper ──────────────────────────────────────────

function PlatformIcon({ platform }: { platform: string }) {
  const p = platform.toLowerCase();
  if (p === "x" || p === "twitter") {
    return (
      <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
        <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
      </svg>
    );
  }
  if (p === "instagram") {
    return (
      <svg className="w-4 h-4 fill-current text-pink-500" viewBox="0 0 24 24">
        <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z"/>
      </svg>
    );
  }
  if (p === "reddit") {
    return (
      <svg className="w-4 h-4 fill-current text-orange-500" viewBox="0 0 24 24">
        <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.688-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.197-2.512-.73a.326.326 0 0 0-.232-.095z"/>
      </svg>
    );
  }
  if (p === "youtube") {
    return (
      <svg className="w-4 h-4 fill-current text-red-500" viewBox="0 0 24 24">
        <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z"/>
      </svg>
    );
  }
  return <Share2 size={15} className="text-slate-400" />;
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function Results() {
  const { id } = useParams<{ id: string }>();
  const [report, setReport] = useState<FullReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [ocrTab, setOcrTab] = useState<"raw" | "processed">("raw");
  const [copiedOcr, setCopiedOcr] = useState(false);

  useEffect(() => {
    if (!id) return;
    getReport(id)
      .then(setReport)
      .catch(() => setError("Failed to load verification report. The analysis may still be processing."))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 space-y-4">
        <div className="spinner w-10 h-10 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-slate-400 animate-pulse">Running multimodal deep verification...</p>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-xl mx-auto px-6 py-20 text-center space-y-4">
        <XCircle size={44} className="text-red-400 mx-auto" />
        <h2 className="text-xl font-bold text-white">Verification In Progress or Unavailable</h2>
        <p className="text-slate-300 text-sm">{error}</p>
        <div className="pt-4">
          <Link to="/upload" className="btn-primary inline-flex items-center gap-2">
            New Verification <ChevronRight size={14} />
          </Link>
        </div>
      </div>
    );
  }

  const socialEvidence = report.social_media_evidence || [];
  const coverage = report.search_coverage;

  return (
    <div className="max-w-5xl mx-auto px-6 py-12 space-y-10 animate-fade-in">
      {/* ── Top Header ── */}
      <div className="space-y-2 border-b border-white/5 pb-6">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <p className="section-label uppercase tracking-wider text-xs text-brand-400 font-semibold">Verification Intelligence Report</p>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">Claim & Forensics Analysis</h1>
            <p className="text-xs text-slate-500 font-mono mt-1">ID: {id}</p>
          </div>
          <Link to="/upload" className="btn-secondary text-xs px-3 py-1.5 flex items-center gap-1.5">
            New Scan <ChevronRight size={13} />
          </Link>
        </div>
      </div>

      {/* ── Extracted Image OCR & Layout Intelligence (Rendered First as Requested) ── */}
      {report.ocr && (report.ocr.processed_text || report.ocr.raw_text) && (
        <section className="space-y-4">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <FileText size={18} className="text-brand-400" />
              <h2 className="text-lg font-bold text-white">Visual OCR Extraction & Layout Intelligence</h2>
            </div>
            <div className="flex items-center flex-wrap gap-2.5">
              {report.ocr.engine_used && (
                <span className="text-xs px-2.5 py-1 rounded-md bg-surface-700/80 text-slate-300 border border-white/10 font-mono">
                  Engine: <span className="text-brand-300 font-semibold">{report.ocr.engine_used}</span>
                </span>
              )}
              {report.ocr.overall_confidence != null && (
                <span className="text-xs px-2.5 py-1 rounded-md bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 font-semibold font-mono">
                  {(report.ocr.overall_confidence * 100).toFixed(1)}% Real Confidence
                </span>
              )}
              {report.ocr.word_count != null && report.ocr.word_count > 0 && (
                <span className="text-xs px-2.5 py-1 rounded-md bg-surface-700/80 text-slate-300 border border-white/10 font-mono">
                  {report.ocr.word_count} words ({report.ocr.character_count || 0} chars)
                </span>
              )}
            </div>
          </div>

          <div className="glass rounded-xl border border-white/10 overflow-hidden bg-surface-800/80">
            {/* Tab controls & Copy button */}
            <div className="flex items-center justify-between bg-surface-900/70 px-4 py-2.5 border-b border-white/10">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setOcrTab("raw")}
                  className={`text-xs px-3 py-1.5 rounded-lg font-semibold transition-all ${
                    ocrTab === "raw"
                      ? "bg-brand-500 text-white shadow-sm"
                      : "text-slate-400 hover:text-white hover:bg-surface-800"
                  }`}
                >
                  Raw OCR (Without Preprocessing)
                </button>
                <button
                  type="button"
                  onClick={() => setOcrTab("processed")}
                  className={`text-xs px-3 py-1.5 rounded-lg font-semibold transition-all ${
                    ocrTab === "processed"
                      ? "bg-brand-500 text-white shadow-sm"
                      : "text-slate-400 hover:text-white hover:bg-surface-800"
                  }`}
                >
                  Processed & Structured News Text
                </button>
              </div>

              <button
                type="button"
                onClick={() => {
                  const textToCopy = ocrTab === "processed"
                    ? (report.ocr?.processed_text || report.ocr?.cleaned_text || "")
                    : (report.ocr?.raw_text || "");
                  navigator.clipboard.writeText(textToCopy);
                  setCopiedOcr(true);
                  setTimeout(() => setCopiedOcr(false), 2000);
                }}
                className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-brand-300 transition-colors px-2.5 py-1 rounded-md hover:bg-surface-800 border border-transparent hover:border-white/10"
              >
                {copiedOcr ? <Check size={13} className="text-emerald-400" /> : <Copy size={13} />}
                {copiedOcr ? "Copied" : "Copy Output"}
              </button>
            </div>

            {/* Content Display */}
            <div className="p-5">
              {ocrTab === "raw" ? (
                <pre className="text-xs font-mono text-slate-300 bg-surface-950/90 p-4 rounded-lg overflow-x-auto whitespace-pre-wrap leading-relaxed border border-white/5 max-h-96">
                  {report.ocr.raw_text || "No raw OCR text available."}
                </pre>
              ) : (
                <div className="space-y-3 font-sans">
                  {(report.ocr.processed_text || report.ocr.cleaned_text || "No structured text extracted.")
                    .split("\n\n")
                    .map((para, pIdx) => (
                      <p key={pIdx} className="text-sm text-slate-200 leading-relaxed bg-surface-900/50 p-3.5 rounded-lg border border-white/5">
                        {para}
                      </p>
                    ))}
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* ── Main Verdict Hero Card ── */}
      <div className="glass p-6 sm:p-8 space-y-6 rounded-2xl border border-white/10 shadow-xl bg-surface-800/80">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-3">
            {report.claim_verdict && (
              <VerdictBadge verdict={report.claim_verdict} label="Claim Verdict" />
            )}
            {report.image_verdict && (
              <VerdictBadge verdict={report.image_verdict} label="Image Forensics" />
            )}
          </div>
          {report.image_type_detected && (
            <div className="px-3 py-1 rounded-full bg-surface-700/80 border border-white/10 text-xs text-slate-300 font-medium">
              Format: <span className="text-brand-300 font-mono capitalize">{report.image_type_detected.replace('_', ' ')}</span>
            </div>
          )}
        </div>

        {/* Confidence Bars & Engine Diagnostics */}
        {(report.claim_confidence != null || report.image_confidence != null) && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 pt-2">
            {report.claim_confidence != null && (
              <div className="space-y-2 bg-surface-900/40 p-4 rounded-xl border border-white/5">
                <div className="flex justify-between text-xs text-slate-400">
                  <span className="flex items-center gap-1.5 font-medium"><Globe size={13} className="text-brand-400" /> Evidence Confidence</span>
                  <span className="font-mono text-white font-semibold">{(report.claim_confidence * 100).toFixed(0)}%</span>
                </div>
                <ConfidenceBar value={report.claim_confidence} color="bg-brand-500" />
              </div>
            )}
            {report.image_confidence != null && (
              <div className="space-y-2 bg-surface-900/40 p-4 rounded-xl border border-white/5">
                <div className="flex justify-between text-xs text-slate-400">
                  <span className="flex items-center gap-1.5 font-medium"><Cpu size={13} className="text-purple-400" /> Image Authenticity Score</span>
                  <span className="font-mono text-white font-semibold">{(report.image_confidence * 100).toFixed(0)}%</span>
                </div>
                <ConfidenceBar value={report.image_confidence} color="bg-purple-500" />
              </div>
            )}
          </div>
        )}

        {/* Executive Summary with AI Synthesis */}
        {report.summary && (
          <div className="p-5 rounded-xl bg-surface-900/70 border border-white/10 shadow-inner space-y-2">
            <div className="flex items-center gap-2">
              <Sparkles size={16} className="text-brand-400" />
              <p className="text-xs uppercase tracking-wider font-semibold text-brand-300">Executive Synthesis</p>
            </div>
            <p className="text-sm text-slate-200 leading-relaxed font-sans">{report.summary}</p>
          </div>
        )}

        {/* Multi-Vector Coverage Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center pt-1">
          <div className="glass p-3 rounded-xl border border-white/5">
            <p className="text-xl font-bold text-white">{coverage?.web_sources_checked ?? report.sources.length}</p>
            <p className="text-xs text-slate-400">Web Sources</p>
          </div>
          <div className="glass p-3 rounded-xl border border-white/5">
            <p className="text-xl font-bold text-brand-400">{coverage?.social_sources_checked ?? socialEvidence.length}</p>
            <p className="text-xs text-slate-400">Social Posts (X/Insta)</p>
          </div>
          <div className="glass p-3 rounded-xl border border-white/5">
            <p className="text-xl font-bold text-emerald-400">{report.supporting_evidence.length}</p>
            <p className="text-xs text-slate-400">Supporting Citations</p>
          </div>
          <div className="glass p-3 rounded-xl border border-white/5">
            <p className="text-xl font-bold text-rose-400">{report.contradicting_evidence.length}</p>
            <p className="text-xs text-slate-400">Refutations / Debunks</p>
          </div>
        </div>
      </div>



      {/* ── Social Media Intelligence Section (X, Instagram, Reddit, YouTube) ── */}
      {socialEvidence.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Share2 size={18} className="text-brand-400" />
              <h2 className="text-lg font-bold text-white">Social Media Intelligence ({socialEvidence.length})</h2>
            </div>
            <span className="text-xs text-slate-400">X (Twitter) · Instagram · Reddit · YouTube</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {socialEvidence.map((post, idx) => (
              <div key={idx} className="glass p-5 rounded-xl border border-white/10 space-y-3 flex flex-col justify-between hover:border-brand-500/30 transition-all">
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <PlatformIcon platform={post.platform} />
                      <span className="text-xs font-semibold text-white capitalize">{post.platform_label}</span>
                      {post.author && (
                        <span className="text-xs text-slate-400 font-mono truncate max-w-[120px]">{post.author}</span>
                      )}
                    </div>
                    <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-full border ${
                      post.relevance === "SUPPORTS" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" :
                      post.relevance === "CONTRADICTS" ? "bg-rose-500/10 text-rose-400 border-rose-500/30" :
                      "bg-brand-500/10 text-brand-300 border-brand-500/30"
                    }`}>
                      {post.relevance.replace('_', ' ')}
                    </span>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed italic line-clamp-4">
                    "{post.post_text}"
                  </p>
                </div>

                <div className="pt-2 border-t border-white/5 flex items-center justify-between text-xs text-slate-400">
                  <span className="text-[11px]">{post.reason || "Matched social query"}</span>
                  <a
                    href={post.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1 text-brand-400 hover:text-brand-300 transition-colors shrink-0 font-medium"
                  >
                    View Post <ExternalLink size={12} />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Extracted Claims ── */}
      {report.claims.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center gap-2">
            <MessageSquare size={18} className="text-brand-400" />
            <h2 className="text-lg font-bold text-white">Extracted Claims ({report.claims.length})</h2>
          </div>
          {report.claims.map((claim) => (
            <div key={claim.id} className="glass p-5 rounded-xl border border-white/10 space-y-3">
              <p className="text-sm text-slate-100 font-medium leading-relaxed">{claim.claim_text}</p>
              {claim.entities_json && (
                <div className="flex flex-wrap gap-2 pt-1">
                  {Object.entries(claim.entities_json).flatMap(([type, vals]) =>
                    (vals as string[]).map(v => (
                      <span key={`${type}-${v}`}
                        className="text-xs px-2.5 py-0.5 rounded-full bg-brand-500/10 text-brand-300 border border-brand-500/20 font-mono">
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

      {/* ── Evidence Details (Supporting & Contradicting) ── */}
      {[
        { title: "Authoritative Supporting Citations", items: report.supporting_evidence, accent: "text-emerald-400", border: "border-emerald-500/20" },
        { title: "Refutation & Contradicting Findings", items: report.contradicting_evidence, accent: "text-rose-400", border: "border-rose-500/20" },
      ].map(({ title, items, accent, border }) =>
        items.length > 0 ? (
          <section key={title} className="space-y-4">
            <h2 className="text-lg font-bold text-white">{title} ({items.length})</h2>
            <div className="space-y-3">
              {items.map((ev, i) => (
                <div key={ev.id || i} className={`glass p-5 rounded-xl border ${border} space-y-2`}>
                  <p className="text-sm text-slate-200 leading-relaxed italic">"{ev.evidence_text}"</p>
                  <div className="flex items-center gap-4 text-xs text-slate-400 pt-1">
                    <span>Relevance: <span className={`font-mono font-semibold ${accent}`}>{(ev.relevance_score * 100).toFixed(0)}%</span></span>
                    <span>Method: <span className="font-mono text-slate-300">{ev.retrieval_method}</span></span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        ) : null
      )}

      {/* ── Sources Registry ── */}
      {report.sources.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center gap-2">
            <Search size={18} className="text-brand-400" />
            <h2 className="text-lg font-bold text-white">Retrieved Sources & Authority Scores ({report.sources.length})</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {report.sources.map((src, i) => (
              <div key={src.id || i} className="glass p-4 rounded-xl border border-white/5 flex items-center justify-between gap-4">
                <div className="space-y-1 min-w-0">
                  <p className="text-sm text-white font-medium truncate">{src.domain}</p>
                  <p className="text-xs text-slate-400 capitalize">
                    {src.source_type.replace("_", " ")} · Authority: <span className="font-mono text-brand-300 font-semibold">{(src.quality_score * 100).toFixed(0)}%</span>
                  </p>
                </div>
                <a href={src.url} target="_blank" rel="noopener noreferrer"
                  className="shrink-0 p-2 rounded-lg bg-surface-700/60 text-slate-400 hover:text-brand-300 hover:bg-surface-600 transition-colors">
                  <ExternalLink size={14} />
                </a>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Limitations & Methodology Notice ── */}
      {report.limitations.length > 0 && (
        <section className="glass p-5 rounded-xl border border-yellow-500/20 space-y-2 bg-yellow-500/5">
          <div className="flex items-center gap-2">
            <AlertTriangle size={15} className="text-yellow-400" />
            <p className="text-xs uppercase tracking-wider font-semibold text-yellow-400">Analysis Methodology & Limits</p>
          </div>
          <ul className="space-y-1">
            {report.limitations.map((l, i) => (
              <li key={i} className="text-xs text-slate-400 leading-relaxed">
                • {l}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ── CTA ── */}
      <div className="flex justify-center pt-4">
        <Link to="/upload" className="btn-secondary px-6 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2 border border-white/10 hover:border-brand-500/30 transition-all">
          Verify Another Claim or Image <ChevronRight size={15} />
        </Link>
      </div>
    </div>
  );
}
