import { useEffect, useState } from "react";
import AnalysisPage from "./AnalysisPage";
import PredictPage from "./PredictPage";

type Page = "estimator" | "analysis";

const TITLES: Record<Page, string> = {
  estimator: "Class Estimator · Loan Default",
  analysis: "ROC-Curve · Loan Default",
};

const currentPage = (): Page =>
  window.location.hash.startsWith("#/analysis") ? "analysis" : "estimator";

/* Self-contained styles: independent of styles.css, with unique class names */
const TOPBAR_CSS = `
.lt-bar{position:sticky;top:0;z-index:50;display:block;width:100%;background:#fff;border-bottom:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(15,23,42,.06)}
.lt-inner{display:flex;flex-direction:column;gap:10px;max-width:1200px;margin:0 auto;padding:14px 24px 12px}
.lt-title{margin:0;color:#0f172a;font-size:1.05rem;font-weight:800;letter-spacing:-.025em;line-height:1.3}
.lt-tabs{display:flex;flex-wrap:wrap;gap:8px}
a.lt-tab{display:inline-flex;align-items:center;justify-content:center;min-height:40px;padding:0 18px;color:#334155;background:#f1f5f9;border:1px solid #cbd5e1;border-radius:10px;font-size:.9rem;font-weight:600;text-decoration:none;white-space:nowrap}
a.lt-tab:hover{background:#e2e8f0;color:#0f172a}
a.lt-tab[aria-current="page"]{color:#fff;background:#1e5fb4;border-color:#1e5fb4}
a.lt-tab:focus-visible{outline:3px solid #1e5fb4;outline-offset:3px}
:root[data-theme="dark"] .lt-bar{background:#0f1a2e;border-bottom-color:#27374f}
:root[data-theme="dark"] .lt-title{color:#fff}
:root[data-theme="dark"] a.lt-tab{color:#c5d1e2;background:#16223a;border-color:#3d516f}
:root[data-theme="dark"] a.lt-tab[aria-current="page"]{color:#04223a;background:#4aa8f0;border-color:#4aa8f0}
@media (max-width:560px){.lt-inner{padding:12px 16px 10px}a.lt-tab{flex:1 1 0;padding:0 10px}}
`;

export default function App() {
  const [page, setPage] = useState<Page>(currentPage);

  useEffect(() => {
    const onHashChange = () => {
      setPage(currentPage());
      window.scrollTo({ top: 0, behavior: "auto" });
    };

    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    document.title = TITLES[page];
  }, [page]);

  const tab = (id: Page, href: string, label: string) => (
    <a
      className="lt-tab"
      href={href}
      aria-current={page === id ? "page" : undefined}
    >
      {label}
    </a>
  );

  return (
    <>
      <style>{TOPBAR_CSS}</style>

      <div className="lt-bar">
        <div className="lt-inner">
          <p className="lt-title">Loan Default Classifier</p>

          <div className="lt-tabs" role="navigation" aria-label="Main navigation">
            {tab("estimator", "#/", "Class Estimator")}
            {tab("analysis", "#/analysis", "ROC-Curve")}
          </div>
        </div>
      </div>

      {page === "analysis" ? <AnalysisPage /> : <PredictPage />}
    </>
  );
}