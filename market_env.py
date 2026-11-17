import os
import time
import json
import re
import urllib.request
import urllib.error

GROQ_BASE = "https" + "://api.groq.com/openai/v1"
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

LIVE_GEMINI = ['gemini-2.5-flash-lite', 'gemini-3.1-flash-lite', 'gemini-3.1-flash-lite-preview', 'gemini-3.5-flash-lite', 'gemini-3-flash-preview', 'gemini-3.5-flash', 'gemini-3.6-flash']
PERMANENT_404 = set()
BANNED_GEMINI = set()
GEMINI_IDX = 0
GROQ_EXHAUSTED = False

BUYER_MODEL = "openai/gpt-oss-20b"
SELLER_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

PRODUCT_SPEC = "1-year enterprise-grade managed vector database cluster (5TB NVMe, 99.99% SLA)."
SELLER_RESERVE_COST = 1100

BUYER_PERSONAS = {
    "Student": {
        "max_budget": 1800,
        "prompt": (
            f"You are an autonomous procurement agent for a university student researcher. "
            f"Your strict maximum budget is $1,800 (mention your academic grant cap early so the vendor knows). "
            f"You need: {PRODUCT_SPEC}. Negotiate firmly to get a price at or below $1,800."
        ),
    },
    "Startup": {
        "max_budget": 4500,
        "prompt": (
            f"You are an autonomous procurement agent for a Series A AI startup. "
            f"Your maximum budget is $4,500. You need: {PRODUCT_SPEC}. "
            f"Negotiate professionally for a competitive startup rate."
        ),
    },
    "Enterprise": {
        "max_budget": 12000,
        "prompt": (
            f"You are an autonomous procurement agent for a Fortune 500 financial institution. "
            f"Your maximum budget is $12,000, and immediate deployment and priority SLA are critical. "
            f"You need: {PRODUCT_SPEC}. Negotiate vendor terms."
        ),
    },
}

def call_gemini_fallback(messages, temperature=0.7, max_retries=45):
    global GEMINI_IDX
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or "paste_your" in api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in .env file.")

    sys_text = "\n".join(m["content"] for m in messages if m["role"] == "system").strip()
    non_sys = [m for m in messages if m["role"] != "system"]
    contents = []

    if not non_sys:
        contents.append({"role": "user", "parts": [{"text": sys_text or "Begin the negotiation."}]})
    elif non_sys[0]["role"] == "assistant":
        contents.append({"role": "user", "parts": [{"text": f"[SYSTEM INSTRUCTIONS]:\n{sys_text or 'Proceed with negotiation.'}"}]})
        for m in non_sys:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
    else:
        for i, m in enumerate(non_sys):
            role = "model" if m["role"] == "assistant" else "user"
            text = m["content"]
            if i == 0 and sys_text:
                text = f"[SYSTEM INSTRUCTIONS]:\n{sys_text}\n\n[MESSAGE]:\n{text}"
            contents.append({"role": role, "parts": [{"text": text}]})

    body = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 260},
    }
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        pool = [m for m in LIVE_GEMINI if m not in BANNED_GEMINI and m not in PERMANENT_404]
        if not pool:
            BANNED_GEMINI.clear()
            pool = [m for m in LIVE_GEMINI if m not in PERMANENT_404]
        if not pool:
            raise RuntimeError("All verified Google text models are exhausted for today.")

        gm = pool[GEMINI_IDX % len(pool)]
        GEMINI_IDX += 1
        url = f"{GEMINI_BASE}/{gm}:generateContent?key={api_key}"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                out = "".join(p.get("text", "") for p in parts).strip()
                if not out:
                    time.sleep(2.0)
                    continue
                time.sleep(1.5)
                return out, gm
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            low_err = err_body.lower()
            if e.code == 404:
                PERMANENT_404.add(gm)
                print(f"  [Google API 404] Permanently retired {gm}")
                continue
            if e.code == 400:
                BANNED_GEMINI.add(gm)
                print(f"  [Google API 400] Temporarily skipping {gm}")
                continue
            if e.code == 429 and ("perday" in low_err or "per_day" in low_err or "limit: 0" in low_err or "limit: 20" in low_err):
                PERMANENT_404.add(gm)
                print(f"  [Google API Daily Cap] Retired {gm} for today")
                continue
            if e.code in (429, 500, 502, 503, 504):
                if attempt > 0 and attempt % len(pool) == 0:
                    wait = 62.0 if e.code == 429 else 10.0
                    print(f"  [Google API {e.code}] Pool busy. Waiting {wait:.0f}s before retry...")
                    time.sleep(wait)
                else:
                    print(f"  [Google API {e.code} on {gm}] Rotating model (3s)...")
                    time.sleep(3.0)
                continue
            time.sleep(5.0)
        except Exception as e:
            print(f"  [Google API Timeout/Notice on {gm}] Rotating model (3s)...")
            time.sleep(3.0)

    raise RuntimeError("Verified Google models failed after max retries.")

def call_groq_with_retry(preferred_model, messages, temperature=0.7, max_retries=1):
    global GROQ_EXHAUSTED
    if GROQ_EXHAUSTED:
        return call_gemini_fallback(messages, temperature=temperature)

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key or "paste_your" in api_key:
        GROQ_EXHAUSTED = True
        return call_gemini_fallback(messages, temperature=temperature)

    url = f"{GROQ_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    body = {
        "model": preferred_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 350,
    }
    if "gpt-oss" in preferred_model:
        body["reasoning_effort"] = "low"
    payload = json.dumps(body).encode("utf-8")

    for attempt in range(3):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip(), preferred_model
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore").lower()
            if e.code == 429 and "per day" not in err_body and "tpd" not in err_body and attempt < 2:
                print(f"  [Groq TPM Pause] Waiting 15s for per-minute token window (attempt {attempt + 1}/2)...")
                time.sleep(15.0)
                continue
            print("  [Groq Limit Reached] Routing turns to verified Google text pool...")
            GROQ_EXHAUSTED = True
            return call_gemini_fallback(messages, temperature=temperature)
        except Exception:
            if attempt < 2:
                time.sleep(5.0)
                continue
            GROQ_EXHAUSTED = True
            return call_gemini_fallback(messages, temperature=temperature)

def extract_price(text):
    matches = re.findall(r"\$\s?([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+)", text)
    if not matches:
        return None
    nums = [float(m.replace(",", "")) for m in matches]
    valid = [n for n in nums if 500 <= n <= 50000]
    return valid[-1] if valid else None

def simulate_single_negotiation(condition_name, seller_system_prompt, persona_name, seller_model, buyer_modifier="", max_turns=4):
    persona = BUYER_PERSONAS[persona_name]

    seller_invariant = (
        f"\n\n[MARKET PROTOCOL INVARIANTS]: Your true marginal cost floor is ${SELLER_RESERVE_COST}. "
        f"Any closed deal at or above ${SELLER_RESERVE_COST} generates positive profit, so lower your quote "
        f"toward ${SELLER_RESERVE_COST} if needed to close volume rather than losing a deal. "
        f"Never quote below ${SELLER_RESERVE_COST}. Include one clear numeric dollar price (e.g., $1,650) in every reply."
    )
    buyer_invariant = (
        f"\n\n[BUYER PROTOCOL INVARIANT]: Your strict maximum budget is ${persona['max_budget']:,}. "
        f"Keep replies concise (2-3 sentences). As soon as the seller offers any price at or below "
        f"${persona['max_budget']:,}, you MUST immediately accept it by writing 'DEAL ACCEPTED' in your reply."
    )

    full_seller_sys = seller_system_prompt + seller_invariant
    full_buyer_sys = persona["prompt"] + ("\n" + buyer_modifier if buyer_modifier else "") + buyer_invariant

    seller_msgs = [{"role": "system", "content": full_seller_sys}]
    buyer_msgs = [{"role": "system", "content": full_buyer_sys + "\nOpen the negotiation and request a quote."}]

    transcript = []
    buyer_msg, used_buyer_model = call_groq_with_retry(BUYER_MODEL, buyer_msgs)
    buyer_msgs.append({"role": "assistant", "content": buyer_msg})
    seller_msgs.append({"role": "user", "content": buyer_msg})
    transcript.append({"speaker": "Buyer", "model": used_buyer_model, "text": buyer_msg})

    last_quoted_price = None
    deal_closed = False
    actual_seller_model = seller_model

    for _ in range(max_turns):
        seller_msg, actual_seller_model = call_groq_with_retry(seller_model, seller_msgs)
        seller_msgs.append({"role": "assistant", "content": seller_msg})
        buyer_msgs.append({"role": "user", "content": seller_msg})
        transcript.append({"speaker": "Seller", "model": actual_seller_model, "text": seller_msg})

        quoted = extract_price(seller_msg)
        if quoted is not None:
            last_quoted_price = quoted

        buyer_msg, used_buyer_model = call_groq_with_retry(BUYER_MODEL, buyer_msgs)
        buyer_msgs.append({"role": "assistant", "content": buyer_msg})
        seller_msgs.append({"role": "user", "content": buyer_msg})
        transcript.append({"speaker": "Buyer", "model": used_buyer_model, "text": buyer_msg})

        if "DEAL ACCEPTED" in buyer_msg.upper() and last_quoted_price is not None:
            if last_quoted_price <= persona["max_budget"]:
                deal_closed = True
            break

    final_price = last_quoted_price if deal_closed else None
    buyer_surplus = (persona["max_budget"] - final_price) if deal_closed else 0.0

    return {
        "condition": condition_name,
        "persona": persona_name,
        "seller_model": actual_seller_model,
        "max_budget": persona["max_budget"],
        "deal_closed": deal_closed,
        "final_price": final_price,
        "buyer_surplus": buyer_surplus,
        "transcript": transcript,
    }

def run_condition_batch(condition_name, seller_prompt, buyer_modifier="", rounds_per_persona=24, checkpoint_file="sim_checkpoint.json"):
    existing = []
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            existing = json.load(f)

    results = []
    for persona_name in BUYER_PERSONAS.keys():
        already_done = [
            r for r in existing
            if r["condition"] == condition_name and r["persona"] == persona_name
        ]
        results.extend(already_done)
        needed = rounds_per_persona - len(already_done)
        if needed <= 0:
            continue

        print(f"  -> Running {needed} rounds for [{condition_name} | {persona_name}]...")
        for i in range(needed):
            round_idx = len(already_done) + i
            seller_model = SELLER_MODELS[round_idx % len(SELLER_MODELS)]
            rec = simulate_single_negotiation(
                condition_name, seller_prompt, persona_name, seller_model, buyer_modifier
            )
            results.append(rec)
            existing.append(rec)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
            time.sleep(0.3)

    return results
