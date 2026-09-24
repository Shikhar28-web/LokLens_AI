import { Outlet, NavLink, Link } from "react-router-dom";
import { ShieldCheck } from "lucide-react";

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Navigation ── */}
      <header className="sticky top-0 z-50 border-b border-white/[0.06]"
        style={{ background: "rgba(13,15,20,0.85)", backdropFilter: "blur(20px)" }}>
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-brand-600 flex items-center justify-center
                            group-hover:bg-brand-500 transition-colors shadow-lg shadow-brand-900/50">
              <ShieldCheck size={18} className="text-white" />
            </div>
            <span className="font-bold text-white tracking-tight">LokLens AI</span>
            <span className="text-xs text-slate-500 font-medium hidden sm:block">Evidence-Based Verification</span>
          </Link>

          <nav className="flex items-center gap-1">
            <NavLink to="/" end className={({ isActive }) =>
              `nav-link ${isActive ? "active" : ""}`
            }>Home</NavLink>
            <NavLink to="/upload" className={({ isActive }) =>
              `nav-link ${isActive ? "active" : ""}`
            }>Verify</NavLink>
          </nav>
        </div>
      </header>

      {/* ── Main content ── */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-white/[0.06] py-8 text-center">
        <p className="text-xs text-slate-600">
          LokLens AI · Evidence-based verification · No generative AI · Every verdict is explainable
        </p>
      </footer>
    </div>
  );
}
