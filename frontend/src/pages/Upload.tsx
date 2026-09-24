import { useState, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  Upload as UploadIcon, FileText, Image, X, Loader2,
  AlertCircle, CheckCircle2, ArrowRight
} from "lucide-react";
import { createSubmission, triggerAnalysis } from "../services/api";

type Mode = "text" | "image" | "multimodal";

export default function Upload() {
  const navigate = useNavigate();
  const fileRef = useRef<HTMLInputElement>(null);

  const [mode, setMode] = useState<Mode>("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // ── File handling ──────────────────────────────────────────────────────────
  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Only image files are accepted (JPEG, PNG, WebP, GIF).");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("Image must be under 10 MB.");
      return;
    }
    setError(null);
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFile(dropped);
  }, []);

  const removeImage = () => {
    setFile(null);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  // ── Submit ─────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!text.trim() && !file) {
      setError("Please provide a text claim, an image, or both.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const submission = await createSubmission(
        text.trim() || undefined,
        file || undefined
      );
      await triggerAnalysis(submission.id);
      navigate(`/submissions/${submission.id}/progress`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Submission failed. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const canSubmit = (text.trim().length > 0 || !!file) && !loading;

  return (
    <div className="max-w-3xl mx-auto px-6 py-16 space-y-10 animate-fade-in">
      {/* Header */}
      <div className="space-y-2">
        <p className="section-label">New verification</p>
        <h1 className="text-3xl font-bold text-white">Submit for analysis</h1>
        <p className="text-slate-400">
          Provide a news claim, an image, or both. The pipeline runs independently
          on each — results are never conflated.
        </p>
      </div>

      {/* Mode selector */}
      <div className="flex gap-2 p-1 rounded-xl bg-surface-800 w-fit">
        {([
          { id: "text", label: "Text only", Icon: FileText },
          { id: "image", label: "Image only", Icon: Image },
          { id: "multimodal", label: "Both", Icon: UploadIcon },
        ] as { id: Mode; label: string; Icon: React.ElementType }[]).map(({ id, label, Icon }) => (
          <button key={id}
            onClick={() => setMode(id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
              ${mode === id
                ? "bg-brand-600 text-white shadow-lg shadow-brand-900/50"
                : "text-slate-400 hover:text-white"
              }`}>
            <Icon size={15} />
            {label}
          </button>
        ))}
      </div>

      <div className="space-y-5">
        {/* Text input */}
        {(mode === "text" || mode === "multimodal") && (
          <div className="space-y-2 animate-slide-up">
            <label className="section-label block">News claim or statement</label>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              rows={5}
              maxLength={10000}
              placeholder="Paste the claim or news excerpt you want to verify…"
              className="w-full rounded-xl bg-surface-800 border border-white/10 text-slate-100
                         placeholder-slate-600 px-4 py-3 text-sm resize-none
                         focus:outline-none focus:border-brand-500/60 focus:ring-1 focus:ring-brand-500/40
                         transition-all duration-200"
            />
            <p className="text-xs text-slate-600 text-right">{text.length} / 10,000</p>
          </div>
        )}

        {/* Image upload */}
        {(mode === "image" || mode === "multimodal") && (
          <div className="space-y-2 animate-slide-up">
            <label className="section-label block">Image</label>

            {preview ? (
              <div className="relative rounded-2xl overflow-hidden border border-white/10 bg-surface-800">
                <img src={preview} alt="Preview" className="w-full max-h-72 object-contain" />
                <button onClick={removeImage}
                  className="absolute top-3 right-3 w-8 h-8 rounded-full bg-black/60 flex items-center
                             justify-center text-white hover:bg-red-600 transition-colors">
                  <X size={14} />
                </button>
                <div className="px-4 py-2 border-t border-white/10 flex items-center gap-2">
                  <CheckCircle2 size={14} className="text-verdict-supported" />
                  <span className="text-xs text-slate-400">{file?.name} ({(file!.size / 1024).toFixed(0)} KB)</span>
                </div>
              </div>
            ) : (
              <div
                className={`upload-zone ${dragging ? "drag-over" : ""}`}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileRef.current?.click()}
              >
                <div className="w-14 h-14 rounded-2xl bg-brand-500/10 flex items-center justify-center">
                  <UploadIcon size={26} className="text-brand-400" />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-300">
                    Drag & drop or <span className="text-brand-400">browse</span>
                  </p>
                  <p className="text-xs text-slate-600 mt-1">JPEG, PNG, WebP, GIF · Max 10 MB</p>
                </div>
                <input ref={fileRef} type="file" accept="image/*" className="hidden"
                  onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f); }} />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-sm text-red-300">
          <AlertCircle size={16} className="shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!canSubmit}
        className="btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none">
        {loading ? (
          <><Loader2 size={16} className="animate-spin" /> Submitting…</>
        ) : (
          <>Run Verification <ArrowRight size={16} /></>
        )}
      </button>

      {/* Disclaimer */}
      <p className="text-xs text-slate-600 text-center leading-relaxed">
        LokLens AI uses classical NLP, information retrieval, and image forensics — not a generative AI model.
        Every verdict is derived from structured evidence retrieved from public sources.
      </p>
    </div>
  );
}
