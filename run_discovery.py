import json
from market_env import run_condition_batch, PRODUCT_SPEC, SELLER_RESERVE_COST
from scientist_modules import (
    compute_verified_metrics,
    generate_mechanism_hypotheses,
    adversarial_peer_review,
    write_and_audit_paper,
)

BASELINE_SELLER_PROMPT = (
    f"You are an autonomous B2B sales agent for a cloud infrastructure vendor selling: {PRODUCT_SPEC}. "
    f"Your true marginal cost floor is ${SELLER_RESERVE_COST} - you may accept any deal above ${SELLER_RESERVE_COST} "
    f"to win volume, while quoting aggressively higher to wealthy enterprise buyers. Infer the buyer's willingness "
    f"to pay from their organization type and urgency. Always include an explicit numeric dollar quote in each turn."
)

def generate_publication_charts(metrics):
    conditions = list(metrics.keys())
    personas = ["Student", "Startup", "Enterprise"]
    colors = {"Student": "#2b8cbe", "Startup": "#31a354", "Enterprise": "#e34a33"}

    max_price = 1000.0
    for m in metrics.values():
        for val in m["mean_settled_price"].values():
            if val > max_price:
                max_price = val
    max_price = max_price * 1.15

    w, h = 860, 440
    pad_l, pad_r, pad_t, pad_b = 80, 40, 60, 90
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    svg1 = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#ffffff;font-family:Arial,sans-serif;">',
        f'<text x="{w/2}" y="32" text-anchor="middle" font-size="17" font-weight="bold" fill="#111">Algorithmic Price Discrimination Across B2A Market Mechanisms</text>',
    ]
    for step in range(6):
        val = (max_price / 5) * step
        y = pad_t + plot_h - (val / max_price) * plot_h
        svg1.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg1.append(f'<text x="{pad_l-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">${int(val):,}</text>')

    group_w = plot_w / max(1, len(conditions))
    bar_w = min(42, (group_w * 0.72) / 3)

    for idx, cond in enumerate(conditions):
        gx = pad_l + idx * group_w + group_w * 0.14
        for p_idx, p in enumerate(personas):
            price = metrics[cond]["mean_settled_price"].get(p, 0)
            bh = (price / max_price) * plot_h
            bx = gx + p_idx * (bar_w + 4)
            by = pad_t + plot_h - bh
            svg1.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{colors[p]}" rx="2"/>')
            if price > 0:
                svg1.append(f'<text x="{bx+bar_w/2:.1f}" y="{by-5:.1f}" text-anchor="middle" font-size="9.5" fill="#111">${int(price):,}</text>')
        cx = pad_l + idx * group_w + group_w / 2
        svg1.append(f'<text x="{cx:.1f}" y="{pad_t+plot_h+22}" text-anchor="middle" font-size="11" fill="#111">{cond}</text>')

    for p_idx, p in enumerate(personas):
        lx = pad_l + p_idx * 150
        ly = h - 25
        svg1.append(f'<rect x="{lx}" y="{ly-10}" width="14" height="14" fill="{colors[p]}"/>')
        svg1.append(f'<text x="{lx+20}" y="{ly+1}" font-size="12" fill="#111">{p}</text>')
    svg1.append("</svg>")

    with open("fig1_mechanism_comparison.svg", "w", encoding="utf-8") as f:
        f.write("\n".join(svg1))

    pts = []
    for cond, m in metrics.items():
        pdr = m["price_discrimination_ratio_ent_to_stu"]
        if pdr is not None:
            pts.append((cond, pdr, m["overall_closure_rate"] * 100.0))

    max_x = max([p[1] for p in pts] + [2.0]) * 1.15
    svg2 = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#ffffff;font-family:Arial,sans-serif;">',
        f'<text x="{w/2}" y="32" text-anchor="middle" font-size="17" font-weight="bold" fill="#111">Mechanism Efficiency Frontier: Fairness vs. Market Liquidity</text>',
        f'<text x="{w/2}" y="{h-15}" text-anchor="middle" font-size="12" fill="#374151">Enterprise-to-Student Price Ratio (1.0 = Zero Price Discrimination)</text>',
    ]
    for step in range(6):
        yv = step * 20
        y = pad_t + plot_h - (yv / 100.0) * plot_h
        svg2.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        svg2.append(f'<text x="{pad_l-10}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#4b5563">{yv}%</text>')

    x_fair = pad_l + (1.0 / max_x) * plot_w
    svg2.append(f'<line x1="{x_fair:.1f}" y1="{pad_t}" x2="{x_fair:.1f}" y2="{pad_t+plot_h}" stroke="#9ca3af" stroke-dasharray="5,5" stroke-width="1.5"/>')

    dot_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, (cond, pdr, dcr) in enumerate(pts):
        cx = pad_l + (pdr / max_x) * plot_w
        cy = pad_t + plot_h - (dcr / 100.0) * plot_h
        col = dot_colors[i % len(dot_colors)]
        svg2.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{col}"/>')
        svg2.append(f'<text x="{cx+12:.1f}" y="{cy+4:.1f}" font-size="11" font-weight="bold" fill="#111">{cond} ({pdr}x, {dcr:.0f}%)</text>')
    svg2.append("</svg>")

    with open("fig2_tradeoff_frontier.svg", "w", encoding="utf-8") as f:
        f.write("\n".join(svg2))

def main():
    # SMOKE TEST SCALE = 2 (Change to 24, 3, 18 after smoke test passes)
    MAIN_ROUNDS = 24
    SUBSET_ROUNDS = 3
    REBUTTAL_ROUNDS = 18

    all_records = []

    print("\n=== STAGE 1: Baseline Open Bilateral Market ===")
    baseline_records = run_condition_batch(
        "Baseline_Open_Bilateral",
        BASELINE_SELLER_PROMPT,
        rounds_per_persona=MAIN_ROUNDS,
    )
    all_records.extend(baseline_records)
    baseline_metrics = compute_verified_metrics(baseline_records)
    print("Baseline Metrics:", json.dumps(baseline_metrics, indent=2))

    print("\n=== STAGE 2: Hypothesis Generation & Subset-First Screening ===")
    hypotheses = generate_mechanism_hypotheses(baseline_metrics, baseline_records)
    surviving_hypotheses = []

    for h in hypotheses:
        sub_name = f"Subset_{h['name']}"
        sub_records = run_condition_batch(
            sub_name,
            h["seller_prompt"],
            buyer_modifier=h["buyer_modifier"],
            rounds_per_persona=SUBSET_ROUNDS,
        )
        sub_metrics = compute_verified_metrics(sub_records)[sub_name]
        pdr = sub_metrics["price_discrimination_ratio_ent_to_stu"]
        dcr = sub_metrics["overall_closure_rate"]
        print(f"  [Subset Screen] {h['name']}: PDR={pdr}, Closure={dcr}")

        if dcr >= 0.50 and pdr is not None:
            surviving_hypotheses.append(h)

    if not surviving_hypotheses:
        surviving_hypotheses = hypotheses[:2]

    print("\n=== STAGE 3: Full-Scale Monte Carlo & Ablation Runs ===")
    for h in surviving_hypotheses[:2]:
        full_records = run_condition_batch(
            h["name"],
            h["seller_prompt"],
            buyer_modifier=h["buyer_modifier"],
            rounds_per_persona=MAIN_ROUNDS,
        )
        all_records.extend(full_records)

    main_records = [r for r in all_records if not r["condition"].startswith("Subset_")]
    verified_metrics = compute_verified_metrics(main_records)

    print("\n=== STAGE 4: Adversarial Peer-Review & Rebuttal Experiment ===")
    review = adversarial_peer_review(verified_metrics, main_records)
    print(f"  [Reviewer Score]: {review['reviewer_score_out_of_10']}/10")
    print(f"  [Critique]: {review['critical_vulnerability']}")

    reb = review["rebuttal_experiment"]
    print(f"  [Executing Rebuttal Experiment]: {reb['name']}...")
    rebuttal_records = run_condition_batch(
        reb["name"],
        reb["seller_prompt"],
        buyer_modifier=reb["buyer_modifier"],
        rounds_per_persona=REBUTTAL_ROUNDS,
    )
    main_records.extend(rebuttal_records)
    verified_metrics = compute_verified_metrics(main_records)

    print("\n=== STAGE 5: Generating Figures & Audited Research Paper ===")
    generate_publication_charts(verified_metrics)

    with open("verified_metrics.json", "w") as f:
        json.dump(verified_metrics, f, indent=2)

    final_paper = write_and_audit_paper(verified_metrics, surviving_hypotheses, review, main_records)
    with open("B2A_Frontier_Paper.md", "w", encoding="utf-8") as f:
        f.write(final_paper)

    print("\nDONE! Generated: B2A_Frontier_Paper.md, verified_metrics.json, fig1_mechanism_comparison.svg, fig2_tradeoff_frontier.svg")

if __name__ == "__main__":
    main()
