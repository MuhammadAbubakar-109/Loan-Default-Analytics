"""
test_api.py
───────────
End-to-end smoke tests for the Loan Default Prediction API.

Prerequisites:
    1. python train_model.py
    2. python -m uvicorn main:app --reload
    3. python test_api.py
"""

import sys
import requests

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT  = 15

_passed = 0
_failed = 0


# ═══════════════════════════════════════════════════════════
#  HELPER
# ═══════════════════════════════════════════════════════════

def check(label, method, path, expected_status=200, **kwargs):
    global _passed, _failed
    url = f"{BASE_URL}{path}"

    try:
        r = getattr(requests, method)(url, timeout=TIMEOUT, **kwargs)
        ok = r.status_code == expected_status
        icon = "✅" if ok else "❌"
        _passed += ok
        _failed += not ok
        print(f"  {icon}  {label:<35} {r.status_code}  {r.json()}")
    except requests.ConnectionError:
        print(f"  ❌  {label:<35} ---  SERVER NOT REACHABLE")
        _failed += 1
    except Exception as e:
        print(f"  ❌  {label:<35} ERR  {e}")
        _failed += 1


# ═══════════════════════════════════════════════════════════
#  BASE PAYLOAD  (all 31 features)
# ═══════════════════════════════════════════════════════════

BASE_PAYLOAD = {
    # Categorical
    "loan_limit": "cf",
    "Gender": "Male",
    "approv_in_adv": "pre",
    "loan_type": "conventional",
    "loan_purpose": "p1",
    "Credit_Worthiness": "I1",
    "open_credit": "op",
    "business_or_commercial": "b/c",
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lps",
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Secured_by": "home",
    "total_units": "1U",
    "credit_type": "CIB",
    "co_applicant_credit_type": "CIB",
    "submission_of_application": "to_inst",
    "Region": "south",
    "Security_Type": "direct",
    # Numeric
    "loan_amount": 250000,
    "term": 360,
    "income": 75000,
    "Credit_Score": 720,
    "age": 35,
    "LTV": 80.0,
    # Optional numeric
    "rate_of_interest": 4.5,
    "Interest_rate_spread": 1.5,
    "Upfront_charges": 2500,
    "property_value": 300000,
    "dtir1": 35.0,
}


# ═══════════════════════════════════════════════════════════
#  TEST SUITES
# ═══════════════════════════════════════════════════════════

def test_info():
    print("\n── Info Routes ──────────────────────────────────")
    check("GET  /",       "get", "/")
    check("GET  /health", "get", "/health")


def test_users():
    print("\n── User Routes ──────────────────────────────────")
    check("GET  /users/1",  "get", "/users/1")
    check("GET  /users/99", "get", "/users/99")
    check("GET  /users/-1", "get", "/users/-1", expected_status=400)


def test_predict_valid():
    print("\n── Prediction (valid) ──────────────────────────")

    # Low-risk profile
    low_risk = {**BASE_PAYLOAD, "Credit_Score": 780, "income": 120000,
                "loan_amount": 150000, "LTV": 60.0}
    check("POST /predict  (Low risk)", "post", "/predict", json=low_risk)

    # Medium-risk profile
    check("POST /predict  (Medium risk)", "post", "/predict", json=BASE_PAYLOAD)

    # High-risk profile
    high_risk = {**BASE_PAYLOAD, "Credit_Score": 520, "income": 25000,
                 "loan_amount": 500000, "LTV": 95.0,
                 "rate_of_interest": 9.5, "dtir1": 55.0}
    check("POST /predict  (High risk)", "post", "/predict", json=high_risk)

    # Missing optional fields (imputer handles it)
    minimal = {**BASE_PAYLOAD, "rate_of_interest": None,
               "Interest_rate_spread": None, "Upfront_charges": None,
               "property_value": None, "dtir1": None}
    check("POST /predict  (Missing optional)", "post", "/predict", json=minimal)


def test_predict_invalid():
    print("\n── Prediction (validation errors) ─────────────")

    # Missing required field
    missing = {k: v for k, v in BASE_PAYLOAD.items() if k != "Gender"}
    check("POST /predict  (Missing Gender)", "post", "/predict",
          expected_status=422, json=missing)

    # Invalid Credit_Score
    bad_score = {**BASE_PAYLOAD, "Credit_Score": 9999}
    check("POST /predict  (Bad Credit_Score)", "post", "/predict",
          expected_status=422, json=bad_score)

    # Negative loan_amount
    bad_amount = {**BASE_PAYLOAD, "loan_amount": -50000}
    check("POST /predict  (Negative amount)", "post", "/predict",
          expected_status=422, json=bad_amount)


# ═══════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  LOAN DEFAULT API — TEST SUITE")
    print("=" * 55)

    test_info()
    test_users()
    test_predict_valid()
    test_predict_invalid()

    total = _passed + _failed
    print("\n" + "=" * 55)
    print(f"  RESULTS:  {_passed}/{total} passed", end="")
    if _failed:
        print(f"  |  {_failed} FAILED")
    else:
        print("  🎉  ALL PASSED")
    print("=" * 55 + "\n")

    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()