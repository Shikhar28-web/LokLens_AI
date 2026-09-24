import { Link } from "react-router-dom";
import { ShieldCheck, Search, Image, FileText, ArrowRight, Zap, Eye, Scale } from "lucide-react";

const features = [
  {
    icon: <FileText size={22} className="text-brand-400" />,
    title: "Claim Extraction",
    desc: "NLP pipeline splits complex inputs into atomic factual claims for targeted verification.",
  },
  {
    icon: <Search size={22} className="text-brand-400" />,
    title: "Evidence Retrieval",
    desc: "TF-IDF + BM25 retrieval ranks real documents against each claim. No fabricated sources.",
  },
  {
    icon: <Image size={22} className="text-brand-400" />,
    title: "Image Forensics",
    desc: "ELA, noise analysis, frequency domain, copy-move detection — every indicator exposed.",
  },
  {
    icon: <Scale size={22} className="text-brand-400" />,
    title: "Independent Verdicts",
    desc: "Image authenticity and claim truth are evaluated separately and never conflated.",
  },
  {
    icon: <Eye size={22} className="text-brand-400" />,
    title: "Full Transparency",
    desc: "Every score has documented components. No opaque black-box decisions.",
  },
  {
    icon: <Zap size={22} className="text-brand-400" />,
    title: "Zero Generative AI",
    desc: "Classical NLP, statistical methods, and deterministic rules — not an LLM guess.",
  },
];

const verdicts = [
  { label: "SUPPORTED", color: "bg-verdict-supported/20 text-verdict-supported border-verdict-supported/30" },
  { label: "PARTIALLY SUPPORTED", color: "bg-verdict-partial/20 text-verdict-partial border-verdict-partial/30" },
  { label: "MISLEADING CONTEXT", color: "bg-verdict-misleading/20 text-verdict-misleading border-verdict-misleading/30" },
  { label: "CONTRADICTED", color: "bg-verdict-contradicted/20 text-verdict-contradicted border-verdict-contradicted/30" },
  { label: "INSUFFICIENT EVIDENCE", color: "bg-verdict-insufficient/20 text-slate-400 border-slate-600/30" },
];

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-20 space-y-32 animate-fade-in">

      {/* ── Hero ── */}
      <section className="text-center space-y-8">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-brand-500/30
                        bg-brand-500/10 text-brand-300 text-xs font-semibold uppercase tracking-widest">
          <ShieldCheck size={13} />
          Evidence-Based · No Generative AI · Fully Explainable
        </div>

        <h1 className="text-5xl sm:text-7xl font-extrabold tracking-tight leading-none text-balance">
          <span className="gradient-text">LokLens AI</span>
          <br />
          <span className="text-slate-200 text-4xl sm:text-5xl font-bold">
            Multimodal News & Image Verification
          </span>
        </h1>

        <p className="max-w-2xl mx-auto text-lg text-slate-400 leading-relaxed text-balance">
          Submit a news claim, an image, or both. LokLens AI retrieves real evidence,
          analyzes forensic indicators, and delivers a transparent verdict — with every
          piece of reasoning shown.
        </p>

        <div className="flex items-center justify-center gap-4 flex-wrap">
          <Link to="/upload" className="btn-primary flex items-center gap-2">
            Start Verification <ArrowRight size={16} />
          </Link>
          <a href="#how-it-works" className="btn-secondary">How it works</a>
        </div>
      </section>

      {/* ── Verdict scale preview ── */}
      <section className="space-y-6" id="how-it-works">
        <p className="section-label text-center">Claim verdict scale</p>
        <div className="flex flex-wrap justify-center gap-3">
          {verdicts.map((v) => (
            <span key={v.label}
              className={`verdict-badge border ${v.color}`}>
              {v.label}
            </span>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section className="space-y-10">
        <div className="text-center space-y-3">
          <p className="section-label">What LokLens AI does</p>
          <h2 className="text-3xl font-bold text-white">Built on evidence, not inference</h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f) => (
            <div key={f.title} className="glass-hover p-6 space-y-3 animate-slide-up">
              <div className="w-10 h-10 rounded-xl bg-brand-500/10 flex items-center justify-center">
                {f.icon}
              </div>
              <h3 className="font-semibold text-white">{f.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Pipeline visual ── */}
      <section className="glass p-8 sm:p-12 space-y-8">
        <div className="text-center space-y-2">
          <p className="section-label">Verification pipeline</p>
          <h2 className="text-2xl font-bold text-white">20-stage analysis — nothing hidden</h2>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center text-sm">
          {[
            ["01", "Input Validation"],
            ["02", "NLP Preprocessing"],
            ["03", "Claim Extraction"],
            ["04", "Query Generation"],
            ["05", "Web Search"],
            ["06", "Document Retrieval"],
            ["07", "TF-IDF + BM25"],
            ["08", "Evidence Extraction"],
            ["09", "Contradiction Detection"],
            ["10", "Source Quality"],
            ["11", "Image Forensics"],
            ["12", "OCR Analysis"],
            ["13", "Perceptual Hashing"],
            ["14", "Multimodal Consistency"],
            ["15", "Verdict Engine"],
            ["16", "Evidence Graph"],
          ].map(([num, label]) => (
            <div key={num} className="flex flex-col items-center gap-2 p-3 rounded-xl bg-surface-700/50">
              <span className="text-xs font-mono text-brand-400 font-semibold">{num}</span>
              <span className="text-xs text-slate-300 leading-tight">{label}</span>
            </div>
          ))}
        </div>

        <div className="text-center">
          <Link to="/upload" className="btn-primary inline-flex items-center gap-2">
            Try it now <ArrowRight size={16} />
          </Link>
        </div>
      </section>
    </div>
  );
}
