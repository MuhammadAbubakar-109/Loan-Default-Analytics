import type { LoanFeatures } from "./api";

export type Key = keyof LoanFeatures;
export interface Opt { value: string; label: string }
export type Field =
  | { kind: "select"; key: Key; label: string; options: Opt[]; hint?: string }
  | { kind: "number"; key: Key; label: string; min?: number; max?: number; step: number; optional?: boolean; hint?: string };

const o = (value: string, label = value): Opt => ({ value, label });
const sel = (key: Key, label: string, options: Opt[], hint?: string): Field => ({ kind: "select", key, label, options, hint });
const num = (key: Key, label: string, step: number, extra: Partial<Extract<Field, { kind: "number" }>> = {}): Field =>
  ({ kind: "number", key, label, step, ...extra });

const UNUSED = "Required by the API, but the current model doesn't use this value.";

// Option values are the exact strings found in Loan_Default.csv (including the dataset's own spelling "Indriect").
export const GROUPS: { title: string; fields: Field[] }[] = [
  {
    title: "Loan",
    fields: [
      num("loan_amount", "Loan amount", 1000, { min: 0 }),
      num("term", "Term (months)", 12, { min: 0 }),
      sel("loan_purpose", "Loan purpose", [o("p1", "Purpose 1"), o("p2", "Purpose 2"), o("p3", "Purpose 3"), o("p4", "Purpose 4")]),
      sel("loan_type", "Loan type", [o("type1", "Type 1"), o("type2", "Type 2"), o("type3", "Type 3")]),
      sel("loan_limit", "Loan limit", [o("cf", "Conforming"), o("ncf", "Non-conforming")]),
      sel("approv_in_adv", "Pre-approval", [o("nopre", "Not pre-approved"), o("pre", "Pre-approved")]),
    ],
  },
  {
    title: "Applicant",
    fields: [
      num("age", "Age", 1, { min: 18, max: 100, hint: UNUSED }),
      sel("Gender", "Gender", [o("Male"), o("Female"), o("Joint"), o("Sex Not Available", "Not available")]),
      num("income", "Income (per month)", 100, { min: 0 }),
      sel("co_applicant_credit_type", "Co-applicant credit bureau", [o("CIB"), o("EXP", "Experian")], UNUSED),
    ],
  },
  {
    title: "Credit",
    fields: [
      num("Credit_Score", "Credit score", 1, { min: 300, max: 850 }),
      sel("credit_type", "Credit bureau", [o("CIB"), o("CRIF"), o("EXP", "Experian"), o("EQUI", "Equifax")]),
      sel("Credit_Worthiness", "Credit worthiness tier", [o("l1", "Tier 1"), o("l2", "Tier 2")]),
      sel("open_credit", "Open credit", [o("nopc", "No open credit"), o("opc", "Has open credit")]),
      num("dtir1", "Debt-to-income ratio (%)", 1, { min: 0, optional: true }),
    ],
  },
  {
    title: "Property",
    fields: [
      num("property_value", "Property value", 1000, { min: 0, optional: true }),
      num("LTV", "Loan-to-value (%)", 0.1, { min: 0 }),
      sel("occupancy_type", "Occupancy", [o("pr", "Primary residence"), o("sr", "Secondary residence"), o("ir", "Investment property")]),
      sel("construction_type", "Construction", [o("sb", "Site-built"), o("mh", "Manufactured home")]),
      sel("total_units", "Units", [o("1U", "1 unit"), o("2U", "2 units"), o("3U", "3 units"), o("4U", "4 units")]),
      sel("Secured_by", "Secured by", [o("home", "Home"), o("land", "Land")]),
    ],
  },
  {
    title: "Pricing and structure",
    fields: [
      num("rate_of_interest", "Interest rate (%)", 0.01, { min: 0, optional: true }),
      num("Interest_rate_spread", "Interest rate spread", 0.01, { optional: true }),
      num("Upfront_charges", "Upfront charges", 100, { min: 0, optional: true }),
      sel("Neg_ammortization", "Negative amortization", [o("not_neg", "No"), o("neg_amm", "Yes")]),
      sel("interest_only", "Interest-only", [o("not_int", "No"), o("int_only", "Yes")]),
      sel("lump_sum_payment", "Lump-sum payment", [o("not_lpsm", "No"), o("lpsm", "Yes")]),
    ],
  },
  {
    title: "Application",
    fields: [
      sel("business_or_commercial", "Purpose of loan", [o("nob/c", "Personal"), o("b/c", "Business / commercial")]),
      sel("submission_of_application", "Submitted to institution", [o("to_inst", "Yes"), o("not_inst", "No")]),
      sel("Region", "Region", [o("North"), o("North-East"), o("central", "Central"), o("south", "South")]),
      sel("Security_Type", "Security type", [o("direct", "Direct"), o("Indriect", "Indirect")]),
    ],
  },
];

export const INITIAL: Record<Key, string> = {
  loan_limit: "cf", Gender: "Male", approv_in_adv: "nopre", loan_type: "type1", loan_purpose: "p3",
  Credit_Worthiness: "l1", open_credit: "nopc", business_or_commercial: "nob/c", Neg_ammortization: "not_neg",
  interest_only: "not_int", lump_sum_payment: "not_lpsm", construction_type: "sb", occupancy_type: "pr",
  Secured_by: "home", total_units: "1U", credit_type: "EXP", co_applicant_credit_type: "CIB",
  submission_of_application: "to_inst", Region: "North", Security_Type: "direct",
  loan_amount: "300000", term: "360", income: "6000", Credit_Score: "700", age: "35", LTV: "75",
  rate_of_interest: "4", Interest_rate_spread: "0.4", Upfront_charges: "3000", property_value: "450000", dtir1: "37",
};
