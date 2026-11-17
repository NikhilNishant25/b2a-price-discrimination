import os
import re
import time
import json
import math
import statistics
import urllib.request
import urllib.error

GEMINI_BASE = "https" + "://generativelanguage.googleapis.com/v1beta/models"

def load_local_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

load_local_env()

GEMINI_MODELS = ['gemini-2.5-flash-lite', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite', 'gemini-3.1-flash-lite-preview', 'gemini-3.5-flash', 'gemini-3.5-flash-lite', 'gemini-3.6-flash']

def _clean_json_text(text):
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    first_arr = t.find("[")
    first_obj = t.find("{")
    starts = [i for i in (first_arr, first_obj) if i != -1]
    if starts:
        start = min(starts)
        end_char = "]" if t[start] == "[" else "}"
        end = t.rfind(end_char)
        if end != -1 and end > start:
            t = t[start:end + 1]
    return t

def call_gemini(prompt, temperature=0.2, json_mode=False, max_retries=24):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or "paste_your" in api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in .env file.")

    headers = {"Content-Type": "application/json"}
    model_idx = 0

    for attempt in range(max_retries):
        current_model = GEMINI_MODELS[model_idx % len(GEMINI_MODELS)]
        gen_config = {"temperature": temperature}
        if json_mode and "gemma" not in current_model:
            gen_config["responseMimeType"] = "application/json"

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }).encode("utf-8")

        url = f"{GEMINI_BASE}/{current_model}:generateContent?key={api_key}"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                out = "".join(p.get("text", "") for p in parts).strip()
                if not out:
                    model_idx += 1
                    print(f"  [Empty response from {current_model}] Rotating model...")
                    time.sleep(2.0)
                    continue
                if json_mode:
                    cleaned = _clean_json_text(out)
                    json.loads(cleaned)
                    return cleaned
                return out
        except json.JSONDecodeError:
            model_idx += 1
            print(f"  [Invalid JSON from {current_model}] Rotating to next model...")
            time.sleep(2.0)
            continue
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if e.code in (400, 404, 429, 503):
                model_idx += 1
                next_model = GEMINI_MODELS[model_idx % len(GEMINI_MODELS)]
                wait = 20.0 if (e.code == 429 and model_idx % len(GEMINI_MODELS) == 0) else 3.0
                print(f"  [Gemini {e.code} on {current_model}] Switching to {next_model} ({wait}s)...")
                time.sleep(wait)
                continue
            wait = min(20, 2 ** attempt)
            print(f"  [Gemini HTTP {e.code}] Waiting {wait}s... ({err_body[:100]})")
            time.sleep(wait)
        except Exception as e:
            model_idx += 1
            wait = min(15, 2 ** attempt)
            print(f"  [Gemini Notice] Waiting {wait}s... ({e})")
            time.sleep(wait)

    raise RuntimeError("Gemini API failed after max retries.")

def _betacf(a, b, x):
    MAXIT = 200
    EPS = 3.0e-12
    FPMIN = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h

def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

def welch_ttest(x1, x2):
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return None, None
    m1, m2 = statistics.mean(x1), statistics.mean(x2)
    v1, v2 = statistics.variance(x1), statistics.variance(x2)
    se2 = (v1 / n1) + (v2 / n2)
    if se2 <= 0:
        return None, None
    t_stat = (m1 - m2) / math.sqrt(se2)
    df = (se2 ** 2) / (((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1))
    x = df / (df + t_stat * t_stat)
    p_val = _betainc(df / 2.0, 0.5, x)
    return round(t_stat, 3), round(p_val, 6)

def compute_verified_metrics(records):
    conditions = list(dict.fromkeys(r["condition"] for r in records))
    personas = ["Student", "Startup", "Enterprise"]
    metrics = {}

    for cond in conditions:
        sub = [r for r in records if r["condition"] == cond]
        closed = [r for r in sub if r["deal_closed"]]

        means = {}
        stds = {}
        surplus = {}
        closure_by_persona = {}

        for p in personas:
            p_sub = [r for r in sub if r["persona"] == p]
            p_closed_prices = [r["final_price"] for r in closed if r["persona"] == p and r["final_price"] is not None]
            if p_sub:
                closure_by_persona[p] = round(sum(1 for r in p_sub if r["deal_closed"]) / len(p_sub), 3)
                surplus[p] = round(statistics.mean(r["buyer_surplus"] for r in p_sub), 2)
            if p_closed_prices:
                means[p] = round(statistics.mean(p_closed_prices), 2)
                stds[p] = round(statistics.stdev(p_closed_prices), 2) if len(p_closed_prices) >= 2 else 0.0

        ent = [r["final_price"] for r in closed if r["persona"] == "Enterprise" and r["final_price"] is not None]
        stu = [r["final_price"] for r in closed if r["persona"] == "Student" and r["final_price"] is not None]
        t_stat_out, p_val_out = welch_ttest(ent, stu)

        pdr = (
            round(means["Enterprise"] / means["Student"], 3)
            if ("Enterprise" in means and "Student" in means and means["Student"] > 0)
            else None
        )

        metrics[cond] = {
            "total_rounds": len(sub),
            "overall_closure_rate": round(sum(1 for r in sub if r["deal_closed"]) / len(sub), 3) if sub else 0.0,
            "closure_rate_by_persona": closure_by_persona,
            "mean_settled_price": means,
            "std_settled_price": stds,
            "mean_buyer_surplus": surplus,
            "price_discrimination_ratio_ent_to_stu": pdr,
            "welch_t_stat": t_stat_out,
            "welch_p_value": p_val_out,
        }
    return metrics

def generate_mechanism_hypotheses(baseline_metrics, sample_transcripts):
    if os.path.exists("hypotheses_cache.json"):
        with open("hypotheses_cache.json", "r", encoding="utf-8") as f:
            return json.load(f)

    prompt = f"""
    You are the Hypothesis Generator Agent in a ScientistTwo autonomous research loop studying Business-to-Agent (B2A) market microstructure.
    
    Market Economics Context:
    - Seller true marginal cost floor = $1,100.
    - Buyer max budgets: Student = $1,800, Startup = $4,500, Enterprise = $12,000.
    - In the baseline open market, the Seller exploited buyer identity cues to charge Enterprise up to 10x more than Students.
    
    Baseline Open Negotiation Metrics:
    {json.dumps(baseline_metrics, indent=2)}
    
    Formulate 3 mechanism-design interventions to eliminate algorithmic price discrimination while keeping Deal Closure Rates HIGH (>= 75%) across all buyer segments (meaning uniform clearing prices must stay competitive around $1,400-$1,750 so Students are not priced out):
    1. Metadata_Redaction: Stripping organizational identity and budget signals from buyer prompts.
    2. ZK_Escrow_Commitment: Buyer presents a Zero-Knowledge proof of locked escrow >= $1,500 without revealing identity or total wealth; Seller competes for volume with standardized quotes.
    3. Sealed_Commit_Reveal: Seller commits to a single non-discriminatory posted market rate before observing buyer context.
    
    Return strictly a JSON list of 3 objects with keys:
    - "name": short snake_case identifier
    - "title": formal mechanism name
    - "hypothesis": 1-sentence economic hypothesis
    - "seller_prompt": exact system prompt for the Seller Agent under this mechanism
    - "buyer_modifier": exact instruction appended to Buyer Agents under this mechanism
    """
    raw = call_gemini(prompt, temperature=0.3, json_mode=True)
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                parsed = v
                break
    with open("hypotheses_cache.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    return parsed

def adversarial_peer_review(verified_metrics, sample_transcripts):
    prompt = f"""
    You are an adversarial Senior Market Design Reviewer at a16z Frontier (modeled on Tim Roughgarden).
    Critique these experimental results rigorously:
    
    Verified Metrics:
    {json.dumps(verified_metrics, indent=2)}
    
    Sample Transcripts:
    {json.dumps(sample_transcripts[:6], indent=2)}
    
    Identify:
    1. The strongest empirical proof in the data.
    2. The single biggest methodological vulnerability or confounding factor (such as uniform price inflation excluding low-budget buyers, or seller reluctance under asymmetric information).
    3. Design ONE concrete follow-up rebuttal experiment (with exact seller_prompt and buyer_modifier) that directly tests and resolves that vulnerability while keeping prices competitive above the $1,100 cost floor so all 3 segments clear deals.
    
    Return strictly JSON with keys:
    - "reviewer_score_out_of_10": number
    - "strongest_finding": string
    - "critical_vulnerability": string
    - "rebuttal_experiment": {{
        "name": "Rebuttal_Mechanism_Refinement",
        "rationale": string,
        "seller_prompt": string,
        "buyer_modifier": string
      }}
    """
    raw = call_gemini(prompt, temperature=0.2, json_mode=True)
    return json.loads(raw)

def write_and_audit_paper(verified_metrics, hypotheses, review_report, sample_transcripts):
    writer_prompt = f"""
    You are a Senior Research Scientist writing a publishable working paper for a16z Frontier Research.
    Write a rigorous, concise research paper in Markdown titled:
    "Algorithmic Surplus Extraction in B2A Commerce: Empirical Evidence and Zero-Knowledge Procurement Mechanisms"
    
    STRICT INTEGRITY RULES:
    - Use ONLY the exact numbers provided in VERIFIED_METRICS below. Never invent or round numbers inconsistently.
    - Do NOT invent fake academic references or hallucinated citations.
    - Quote 2 or 3 real verbatim excerpts from SAMPLE_TRANSCRIPTS to illustrate how the Seller Agent exploited metadata vs. how the mechanism stopped it.
    - Reference the figures `![Figure 1: Mechanism Comparison](fig1_mechanism_comparison.svg)` and `![Figure 2: Surplus vs Closure Tradeoff](fig2_tradeoff_frontier.svg)`.
    
    Sections required:
    1. Abstract
    2. Introduction: The B2A Distribution Shift and Asymmetric Information
    3. Experimental Setup & Multi-Metric Evaluation (Anti-Reward-Hacking)
    4. Empirical Results: First-Degree Price Discrimination & Ablation Studies
    5. Adversarial Peer-Review & Rebuttal Experiment Findings
    6. Mechanism Design Proposal: Commit-and-Reveal ZK-Escrow for Autonomous Agents
    7. Conclusion & Portfolio Implications
    
    VERIFIED_METRICS:
    {json.dumps(verified_metrics, indent=2)}
    
    HYPOTHESES_TESTED:
    {json.dumps(hypotheses, indent=2)}
    
    PEER_REVIEW_AND_REBUTTAL:
    {json.dumps(review_report, indent=2)}
    
    SAMPLE_TRANSCRIPTS:
    {json.dumps(sample_transcripts[:8], indent=2)}
    """
    draft = call_gemini(writer_prompt, temperature=0.25, json_mode=False)

    audit_prompt = f"""
    You are the Code-to-Paper Integrity Auditor in ScientistTwo.
    Compare every number, p-value, price, and closure rate in the DRAFT_PAPER against VERIFIED_METRICS.
    If any number deviates from VERIFIED_METRICS, fix it silently so 100% of reported statistics match the ground-truth logs.
    Return ONLY the final audited Markdown paper.
    
    VERIFIED_METRICS:
    {json.dumps(verified_metrics, indent=2)}
    
    DRAFT_PAPER:
    {draft}
    """
    audited_paper = call_gemini(audit_prompt, temperature=0.0, json_mode=False)
    return audited_paper
