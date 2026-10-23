import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  ApiError,
  fetchHealth,
  predict,
  type Health,
  type LoanFeatures,
  type LoanResponse,
} from "./api";
import { GROUPS, INITIAL, type Field, type Key } from "./fields";

type Form = Record<Key, string>;
type FieldErrors = Partial<Record<Key, string>>;

const ALL_FIELDS = GROUPS.flatMap((g) => g.fields);

const clamp01 = (n: number): number =>
  Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 0;

/** Validate against the same limits as the API's Pydantic model; returns payload or field errors. */
function buildPayload(form: Form): { payload?: LoanFeatures; errors: FieldErrors } {
  const errors: FieldErrors = {};
  const out: Record<string, string | number | null> = {};

  for (const f of ALL_FIELDS) {
    const raw = form[f.key].trim();

    if (f.kind === "select") {
      out[f.key] = raw;
      continue;
    }

    if (raw === "") {
      if (f.optional) out[f.key] = null;
      else errors[f.key] = "Enter a value.";
      continue;
    }

    const n = Number(raw);
    if (!Number.isFinite(n)) errors[f.key] = "Enter a number.";
    else if (f.min !== undefined && n < f.min) errors[f.key] = `Must be at least ${f.min}.`;
    else if (f.max !== undefined && n > f.max) errors[f.key] = `Must be at most ${f.max}.`;
    else out[f.key] = n;
  }

  return Object.keys(errors).length
    ? { errors }
    : { payload: out as unknown as LoanFeatures, errors };
}

function Gauge({ p, level }: { p: number; level: string }) {
  const value = clamp01(p);
  return (
    <figure className={`gauge ${level.toLowerCase()}`}>
      <svg
        viewBox="0 0 120 70"
        role="img"
        aria-label={`Default probability ${(value * 100).toFixed(1)} percent`}
      >
        <path className="track" d="M10 60 A50 50 0 0 1 110 60" pathLength="100" />
        <path
          className="fill"
          d="M10 60 A50 50 0 0 1 110 60"
          pathLength="100"
          strokeDasharray={`${value * 100} 100`}
        />
      </svg>
      <figcaption>{(value * 100).toFixed(1)}%</figcaption>
    </figure>
  );
}

export default function PredictPage() {
  const [form, setForm] = useState<Form>(INITIAL);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [health, setHealth] = useState<Health | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const [healthAttempt, setHealthAttempt] = useState(0);
  const [loading, setLoading] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [result, setResult] = useState<LoanResponse | null>(null);

  const resultRef = useRef<HTMLElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // API health check. Cancelled requests are not treated as "unreachable".
  useEffect(() => {
    const controller = new AbortController();

    fetchHealth(controller.signal)
      .then((h) => {
        setHealth(h);
        setApiDown(false);
      })
      .catch(() => {
        if (!controller.signal.aborted) setApiDown(true);
      });

    return () => controller.abort();
  }, [healthAttempt]);

  // Cancel any in-flight prediction when leaving the page.
  useEffect(() => () => abortRef.current?.abort(), []);

  function set(key: Key, value: string) {
    setForm((f) => ({ ...f, [key]: value }));
    setErrors((e) => (e[key] ? { ...e, [key]: undefined } : e));
  }

  function resetForm() {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
    setForm(INITIAL);
    setErrors({});
    setFormError(null);
    setResult(null);
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (loading) return;

    const { payload, errors: errs } = buildPayload(form);
    setErrors(errs);
    setFormError(null);

    if (!payload) {
      setFormError("Fix the highlighted fields and try again.");
      // Focus by id: the error state has not rendered yet, so an
      // [aria-invalid] query would find nothing at this point.
      const first = ALL_FIELDS.find((f) => errs[f.key]);
      if (first) document.getElementById(`f-${first.key}`)?.focus();
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);

    try {
      const res = await predict(payload, controller.signal);
      setResult(res);
      setApiDown(false);

      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      resultRef.current?.scrollIntoView({
        block: "nearest",
        behavior: reduceMotion ? "auto" : "smooth",
      });
    } catch (err) {
      if (controller.signal.aborted) return; // cancelled or superseded

      setResult(null);
      if (err instanceof ApiError) {
        setFormError(err.message);
        if (Object.keys(err.fieldErrors).length) {
          setErrors(err.fieldErrors as FieldErrors);
        }
        if (err.status === 0) setApiDown(true);
      } else {
        setFormError("Something went wrong. Try again.");
      }
    } finally {
      // Only the latest request may clear the loading state.
      if (abortRef.current === controller) setLoading(false);
    }
  }

  const status = apiDown ? "down" : health?.model_loaded ? "up" : "wait";
  const statusText =
    status === "up"
      ? `Model ready (${health?.features_expected} inputs)`
      : status === "down"
        ? "API unreachable"
        : "Connecting to API…";

  const renderField = (f: Field) => {
    const err = errors[f.key];
    const id = `f-${f.key}`;
    const messageId = `${id}-h`;
    const message = err ?? f.hint;
    const common = {
      id,
      value: form[f.key],
      "aria-invalid": err ? true : undefined,
      // Only reference the message element when it is actually rendered.
      "aria-describedby": message ? messageId : undefined,
    } as const;

    return (
      <div className="field" key={f.key}>
        <label htmlFor={id}>
          {f.label}
          {f.kind === "number" && f.optional && <em> (optional)</em>}
        </label>

        {f.kind === "select" ? (
          <select {...common} onChange={(e) => set(f.key, e.target.value)}>
            {f.options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        ) : (
          <input
            {...common}
            type="number"
            inputMode="decimal"
            step={f.step}
            min={f.min}
            max={f.max}
            onChange={(e) => set(f.key, e.target.value)}
          />
        )}

        {message && (
          <p id={messageId} className={err ? "msg err" : "msg"}>
            {message}
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="shell">
      <main id="main">
        <header>
          <h1>Loan default risk check</h1>
          <p>Enter the loan application details to estimate the chance the loan defaults.</p>
        </header>

        <form onSubmit={onSubmit} noValidate>
          {GROUPS.map((g) => (
            <fieldset key={g.title}>
              <legend>{g.title}</legend>
              <div className="grid">{g.fields.map(renderField)}</div>
            </fieldset>
          ))}

          {formError && (
            <p className="banner" role="alert">
              {formError}
            </p>
          )}

          <div className="actions">
            <button className="primary" type="submit" disabled={loading} aria-busy={loading}>
              {loading ? "Checking…" : "Check default risk"}
            </button>
            <button className="ghost" type="button" onClick={resetForm}>
              Reset form
            </button>
          </div>
        </form>
      </main>

      <aside className="result" ref={resultRef} aria-live="polite">
        <p className={`status ${status}`}>
          <i aria-hidden="true" />
          {statusText}
        </p>

        {apiDown && (
          <button
            type="button"
            className="ghost"
            onClick={() => {
              setApiDown(false);
              setHealthAttempt((n) => n + 1);
            }}
          >
            Retry connection
          </button>
        )}

        {result ? (
          <>
            <p className="eyebrow">Default probability</p>
            <Gauge p={result.default_probability} level={result.risk_level} />
            <p className={`badge ${result.risk_level.toLowerCase()}`}>{result.risk_level} risk</p>
            <dl>
              <dt>Model classification</dt>
              <dd>{result.prediction === 1 ? "Predicted to default" : "Predicted not to default"}</dd>
            </dl>
            <p className="note">
              Risk levels: low under 30%, medium 30–60%, high 60% or more. Classification uses
              the model's own 50% cut-off, so it can differ from the risk level.
            </p>
          </>
        ) : (
          <p className="empty">
            Your result will appear here. Complete the form and select <b>Check default risk</b>.
          </p>
        )}
      </aside>
    </div>
  );
}