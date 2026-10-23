// Types mirror LoanFeatures / LoanResponse in main.py (field names are sent exactly as the API defines them).
export interface LoanFeatures {
  loan_limit: string; Gender: string; approv_in_adv: string; loan_type: string; loan_purpose: string;
  Credit_Worthiness: string; open_credit: string; business_or_commercial: string; Neg_ammortization: string;
  interest_only: string; lump_sum_payment: string; construction_type: string; occupancy_type: string;
  Secured_by: string; total_units: string; credit_type: string; co_applicant_credit_type: string;
  submission_of_application: string; Region: string; Security_Type: string;
  loan_amount: number; term: number; income: number; Credit_Score: number; age: number; LTV: number;
  rate_of_interest: number | null; Interest_rate_spread: number | null; Upfront_charges: number | null;
  property_value: number | null; dtir1: number | null;
}

export interface LoanResponse {
  prediction: 0 | 1;
  default_probability: number;
  risk_level: "Low" | "Medium" | "High";
}

export interface Health { status: string; model_loaded: boolean; features_expected: number }

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public fieldErrors: Record<string, string> = {},
    public retryable = false,
  ) {
    super(message);
  }
}

const BASE: string = import.meta.env.VITE_API_URL ?? "/api";
const TIMEOUT_MS = 15_000;

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;

function isLoanResponse(v: unknown): v is LoanResponse {
  return (
    isObj(v) &&
    (v.prediction === 0 || v.prediction === 1) &&
    typeof v.default_probability === "number" &&
    (v.risk_level === "Low" || v.risk_level === "Medium" || v.risk_level === "High")
  );
}

async function request(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<unknown> {
  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; ctrl.abort(); }, TIMEOUT_MS);
  signal?.addEventListener("abort", () => ctrl.abort());

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init.headers },
      signal: ctrl.signal,
    });
  } catch (e) {
    if (timedOut) throw new ApiError("The server took too long to respond. Try again.", 0, {}, true);
    if (e instanceof DOMException && e.name === "AbortError") throw e; // user-cancelled: not shown
    throw new ApiError("Can't reach the API. Check that the backend is running (uvicorn main:app).", 0, {}, true);
  } finally {
    clearTimeout(timer);
  }

  let body: unknown = undefined;
  if (res.status !== 204) {
    try { body = await res.json(); } catch { body = undefined; }
  }

  if (!res.ok) {
    const detail = isObj(body) ? body.detail : undefined;
    const fieldErrors: Record<string, string> = {};
    let message = `Request failed (${res.status}).`;
    if (Array.isArray(detail)) {
      for (const d of detail) {
        if (isObj(d) && Array.isArray(d.loc) && typeof d.msg === "string") {
          fieldErrors[String(d.loc[d.loc.length - 1])] = d.msg;
        }
      }
      message = "Some values were rejected. Check the highlighted fields.";
    } else if (typeof detail === "string") {
      message = res.status === 503 ? "The prediction model isn't loaded on the server yet." : detail;
    }
    if (res.status === 429) {
      const wait = res.headers.get("Retry-After");
      message = `Too many requests.${wait ? ` Try again in ${wait} seconds.` : " Try again shortly."}`;
    }
    throw new ApiError(message, res.status, fieldErrors, res.status >= 500 || res.status === 429);
  }
  return body;
}

export async function fetchHealth(signal?: AbortSignal): Promise<Health> {
  const b = await request("/health", {}, signal);
  if (!isObj(b) || typeof b.model_loaded !== "boolean") throw new ApiError("Unexpected /health response.", 200);
  return b as unknown as Health;
}

export async function predict(features: LoanFeatures, signal?: AbortSignal): Promise<LoanResponse> {
  // POST is not idempotent-safe to auto-retry by policy; user retries explicitly.
  const b = await request("/predict", { method: "POST", body: JSON.stringify(features) }, signal);
  if (!isLoanResponse(b)) throw new ApiError("The server returned an unexpected prediction format.", 200);
  return b;
}
