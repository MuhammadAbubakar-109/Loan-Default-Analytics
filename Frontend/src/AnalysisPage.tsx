import { useEffect, useMemo, useState } from "react";

/* ------------------------------ Types ------------------------------ */

interface Confusion {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

interface RocPoint {
  fpr: number;
  tpr: number;
}

interface Analysis {
  total: number;
  confusion: Confusion;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  rocAuc: number | null;
  roc: RocPoint[];
}

/* --------------------------- Parsing helpers --------------------------- */

type RecordValue = Record<string, unknown>;

const isRecord = (value: unknown): value is RecordValue =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const toNum = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  // Also support numeric strings in JSON responses.
  if (
    typeof value === "string" &&
    value.trim() !== "" &&
    Number.isFinite(Number(value))
  ) {
    return Number(value);
  }

  return null;
};

function pick(
  sources: RecordValue[],
  keys: string[],
): number | null {
  for (const source of sources) {
    for (const key of keys) {
      const value = toNum(source[key]);

      if (value !== null) {
        return value;
      }
    }
  }

  return null;
}

function ratio(numerator: number, denominator: number): number | null {
  return denominator > 0 ? numerator / denominator : null;
}

/**
 * Collect common nested metric objects, including model_metrics.
 * The top-level response is checked first.
 */
function collectSources(raw: RecordValue): RecordValue[] {
  const sources: RecordValue[] = [raw];

  const nestedKeys = [
    "model_metrics",
    "metrics",
    "evaluation_metrics",
    "evaluation",
    "results",
    "analysis",
    "classification_report",
  ];

  for (const key of nestedKeys) {
    const nested = raw[key];

    if (isRecord(nested)) {
      sources.push(nested);
    }
  }

  return sources;
}

/* --------------------------- Confusion matrix --------------------------- */

function parseConfusionValue(value: unknown): Confusion | null {
  // Format: [[tn, fp], [fn, tp]]
  if (Array.isArray(value) && value.length === 2) {
    const row0 = value[0];
    const row1 = value[1];

    if (Array.isArray(row0) && Array.isArray(row1)) {
      const tn = toNum(row0[0]);
      const fp = toNum(row0[1]);
      const fn = toNum(row1[0]);
      const tp = toNum(row1[1]);

      if (tn !== null && fp !== null && fn !== null && tp !== null) {
        return { tn, fp, fn, tp };
      }
    }
  }

  if (isRecord(value)) {
    // Support: { matrix: [[tn, fp], [fn, tp]] }
    if (value.matrix !== undefined) {
      const nestedMatrix = parseConfusionValue(value.matrix);

      if (nestedMatrix) {
        return nestedMatrix;
      }
    }

    // Support: { values: [[tn, fp], [fn, tp]] }
    if (value.values !== undefined) {
      const nestedValues = parseConfusionValue(value.values);

      if (nestedValues) {
        return nestedValues;
      }
    }

    // Support: { tn, fp, fn, tp }
    const tn = pick([value], ["tn", "TN", "true_negative"]);
    const fp = pick([value], ["fp", "FP", "false_positive"]);
    const fn = pick([value], ["fn", "FN", "false_negative"]);
    const tp = pick([value], ["tp", "TP", "true_positive"]);

    if (tn !== null && fp !== null && fn !== null && tp !== null) {
      return { tn, fp, fn, tp };
    }
  }

  return null;
}

function findConfusion(sources: RecordValue[]): Confusion | null {
  for (const source of sources) {
    const value =
      source.confusion_matrix ??
      source.confusionMatrix ??
      source.confusion ??
      source.cm;

    const confusion = parseConfusionValue(value);

    if (confusion) {
      return confusion;
    }
  }

  return null;
}

/* ------------------------------ ROC curve ------------------------------ */

function findRoc(sources: RecordValue[]): RocPoint[] {
  for (const source of sources) {
    const value =
      source.roc_curve ??
      source.rocCurve ??
      source.roc_model ??
      source.rocModel ??
      source.roc ??
      source.roc_points;

    const points: RocPoint[] = [];

    // Format: [{ fpr: 0, tpr: 0 }, ...]
    if (Array.isArray(value)) {
      for (const item of value) {
        if (!isRecord(item)) continue;

        const fpr = toNum(item.fpr);
        const tpr = toNum(item.tpr);

        if (fpr !== null && tpr !== null) {
          points.push({ fpr, tpr });
        }
      }
    }

    // Format: { fpr: [...], tpr: [...], roc_auc: 0.86 }
    if (
      isRecord(value) &&
      Array.isArray(value.fpr) &&
      Array.isArray(value.tpr)
    ) {
      const length = Math.min(value.fpr.length, value.tpr.length);

      for (let i = 0; i < length; i += 1) {
        const fpr = toNum(value.fpr[i]);
        const tpr = toNum(value.tpr[i]);

        if (fpr !== null && tpr !== null) {
          points.push({ fpr, tpr });
        }
      }
    }

    if (points.length > 0) {
      return points.sort(
        (a, b) => a.fpr - b.fpr || a.tpr - b.tpr,
      );
    }
  }

  return [];
}

/* --------------------------- Analysis parser --------------------------- */

function parseAnalysis(raw: unknown): Analysis {
  if (!isRecord(raw)) {
    throw new Error("The server returned an unexpected response.");
  }

  const sources = collectSources(raw);
  const confusion = findConfusion(sources);

  if (!confusion) {
    throw new Error(
      "The API responded, but no recognized confusion matrix was found. Check the JSON response from /api/model-analysis.",
    );
  }

  const { tn, fp, fn, tp } = confusion;
  const total = tn + fp + fn + tp;

  // Use saved test-set metrics where available; otherwise calculate them.
  const accuracy =
    pick(sources, ["accuracy", "accuracy_score"]) ??
    ratio(tp + tn, total);

  const precision =
    pick(sources, ["precision", "positive_precision"]) ??
    ratio(tp, tp + fp);

  const recall =
    pick(sources, [
      "recall",
      "sensitivity",
      "true_positive_rate",
    ]) ?? ratio(tp, tp + fn);

  const f1 =
    pick(sources, ["f1", "f1_score", "f1-score", "f1Score"]) ??
    (precision !== null &&
    recall !== null &&
    precision + recall > 0
      ? (2 * precision * recall) / (precision + recall)
      : null);

  const roc = findRoc(sources);

  // ROC-AUC may be stored in model_metrics or inside the ROC object.
  const rocObject = sources
    .map(
      (source) =>
        source.roc_curve ??
        source.rocCurve ??
        source.roc_model ??
        source.rocModel ??
        source.roc,
    )
    .find(isRecord);

  const rocAuc =
    pick(sources, [
      "roc_auc",
      "rocAuc",
      "auc",
      "roc_auc_score",
    ]) ??
    (isRecord(rocObject)
      ? pick([rocObject], ["roc_auc", "rocAuc", "auc"])
      : null);

  return {
    total,
    confusion,
    accuracy,
    precision,
    recall,
    f1,
    rocAuc,
    roc,
  };
}

/* ------------------------------ Formatting ------------------------------ */

const integer = new Intl.NumberFormat("en", {
  maximumFractionDigits: 0,
});

const pct = (value: number | null): string =>
  value === null ? "N/A" : `${(value * 100).toFixed(2)}%`;

const score = (value: number | null): string =>
  value === null ? "N/A" : value.toFixed(4);

/* ------------------------------- ROC chart ------------------------------- */

function RocChart({
  points,
  auc,
}: {
  points: RocPoint[];
  auc: number | null;
}) {
  const W = 480;
  const H = 400;
  const M = { t: 20, r: 20, b: 56, l: 62 };

  const plotWidth = W - M.l - M.r;
  const plotHeight = H - M.t - M.b;

  const x = (value: number) => M.l + value * plotWidth;
  const y = (value: number) => H - M.b - value * plotHeight;

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  const path = useMemo(
    () =>
      points
        .map((point, index) => {
          const fpr = Math.min(1, Math.max(0, point.fpr));
          const tpr = Math.min(1, Math.max(0, point.tpr));

          return `${index === 0 ? "M" : "L"}${x(fpr).toFixed(1)} ${y(
            tpr,
          ).toFixed(1)}`;
        })
        .join(" "),
    [points],
  );

  return (
    <svg
      className="an-chart"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`ROC curve showing true positive rate against false positive rate.${
        auc !== null ? ` Area under the curve ${auc.toFixed(4)}.` : ""
      }`}
    >
      {ticks.map((tick) => (
        <g key={tick}>
          <line
            className="an-grid"
            x1={M.l}
            x2={W - M.r}
            y1={y(tick)}
            y2={y(tick)}
          />

          <line
            className="an-grid"
            x1={x(tick)}
            x2={x(tick)}
            y1={M.t}
            y2={H - M.b}
          />

          <text
            className="an-tick"
            x={M.l - 10}
            y={y(tick) + 4}
            textAnchor="end"
          >
            {tick.toFixed(2)}
          </text>

          <text
            className="an-tick"
            x={x(tick)}
            y={H - M.b + 20}
            textAnchor="middle"
          >
            {tick.toFixed(2)}
          </text>
        </g>
      ))}

      <line
        className="an-axis"
        x1={M.l}
        y1={M.t}
        x2={M.l}
        y2={H - M.b}
      />

      <line
        className="an-axis"
        x1={M.l}
        y1={H - M.b}
        x2={W - M.r}
        y2={H - M.b}
      />

      {/* Random guessing baseline */}
      <line
        className="an-diag"
        x1={x(0)}
        y1={y(0)}
        x2={x(1)}
        y2={y(1)}
      />

      {path && <path className="an-roc" d={path} />}

      <text
        className="an-axis-title"
        x={(M.l + W - M.r) / 2}
        y={H - 14}
        textAnchor="middle"
      >
        False positive rate
      </text>

      <text
        className="an-axis-title"
        transform={`translate(16 ${(M.t + H - M.b) / 2}) rotate(-90)`}
        textAnchor="middle"
      >
        True positive rate
      </text>
    </svg>
  );
}

/* --------------------------------- Page --------------------------------- */

const BASE_URL = (
  (import.meta.env.VITE_API_URL as string | undefined) ?? "/api"
).replace(/\/+$/, "");

export default function AnalysisPage() {
  const [data, setData] = useState<Analysis | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();

    async function loadAnalysis() {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(`${BASE_URL}/model-analysis`, {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
          signal: controller.signal,
        });

        if (!response.ok) {
          const body = await response.text();

          throw new Error(
            `Model analysis request failed (${response.status}).${
              body ? ` ${body.slice(0, 250)}` : ""
            }`,
          );
        }

        const raw: unknown = await response.json();
        const parsed = parseAnalysis(raw);

        if (!controller.signal.aborted) {
          setData(parsed);
          setError(null);
        }
      } catch (err) {
        if (controller.signal.aborted) return;

        setData(null);

        setError(
          err instanceof Error
            ? err.message
            : "Unable to load model analysis.",
        );
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadAnalysis();

    return () => controller.abort();
  }, [attempt]);

  return (
    <main className="an-page" id="main">
      <header className="an-header">
        <h1>Model analysis: loan default classifier</h1>

        <p>
          Performance on held-out test applicants that were not used to train
          the classifier. Positive class: loan default.
        </p>
      </header>

      {loading && !data ? (
        <p className="an-card an-state" role="status">
          Loading model analysis…
        </p>
      ) : error ? (
        <div className="an-card an-state">
          <p className="an-error" role="alert">
            {error}
          </p>

          <button
            type="button"
            className="an-btn"
            onClick={() => setAttempt((current) => current + 1)}
          >
            Try again
          </button>
        </div>
      ) : data ? (
        <>
          <section
            className="an-metrics"
            aria-label="Model performance metrics"
          >
            <dl className="an-metric-grid">
              <div>
                <dt>Accuracy</dt>
                <dd>{pct(data.accuracy)}</dd>
              </div>

              <div>
                <dt>Precision</dt>
                <dd>{pct(data.precision)}</dd>
              </div>

              <div>
                <dt>Recall</dt>
                <dd>{pct(data.recall)}</dd>
              </div>

              <div>
                <dt>F1 score</dt>
                <dd>{pct(data.f1)}</dd>
              </div>

              <div>
                <dt>ROC AUC</dt>
                <dd>{score(data.rocAuc)}</dd>
              </div>

              <div>
                <dt>Test applicants</dt>
                <dd>{integer.format(data.total)}</dd>
              </div>
            </dl>
          </section>

          <div className="an-grid-2">
            <section className="an-card" aria-labelledby="cm-title">
              <h2 id="cm-title">Confusion matrix</h2>

              <table className="an-cm">
                <caption className="sr-only">
                  Actual outcome versus predicted outcome, in applicants
                </caption>

                <thead>
                  <tr>
                    <td />
                    <th scope="col">Predicted: repaid</th>
                    <th scope="col">Predicted: default</th>
                  </tr>
                </thead>

                <tbody>
                  <tr>
                    <th scope="row">Actual: repaid</th>

                    <td className="an-good">
                      <b>{integer.format(data.confusion.tn)}</b>
                      <span>True negatives</span>
                    </td>

                    <td className="an-bad">
                      <b>{integer.format(data.confusion.fp)}</b>
                      <span>False positives</span>
                    </td>
                  </tr>

                  <tr>
                    <th scope="row">Actual: default</th>

                    <td className="an-bad">
                      <b>{integer.format(data.confusion.fn)}</b>
                      <span>False negatives</span>
                    </td>

                    <td className="an-good">
                      <b>{integer.format(data.confusion.tp)}</b>
                      <span>True positives</span>
                    </td>
                  </tr>
                </tbody>
              </table>

              <p className="an-note">
                {integer.format(data.confusion.fn)} actual defaults were
                missed and predicted as repaid.{" "}
                {integer.format(data.confusion.fp)} applicants who repaid were
                incorrectly flagged as likely defaults.
              </p>
            </section>

            <section className="an-card" aria-labelledby="roc-title">
              <h2 id="roc-title">ROC curve</h2>

              {data.roc.length > 1 ? (
                <>
                  <RocChart points={data.roc} auc={data.rocAuc} />

                  <ul className="an-legend">
                    <li>
                      <i className="an-sw an-sw-roc" aria-hidden="true" />
                      Classifier
                    </li>

                    <li>
                      <i className="an-sw an-sw-diag" aria-hidden="true" />
                      Random guessing
                    </li>
                  </ul>
                </>
              ) : (
                <p className="an-note">
                  The API did not return enough ROC curve points to draw the
                  chart. The ROC-AUC score can still be displayed if available.
                  Check that the backend returns numeric arrays named{" "}
                  <code>fpr</code> and <code>tpr</code> inside{" "}
                  <code>roc_curve</code>.
                </p>
              )}
            </section>
          </div>
        </>
      ) : (
        <p className="an-card an-state" role="status">
          No analysis data available.
        </p>
      )}
    </main>
  );
}