#!/usr/bin/env python3
"""
Integration test for the linguistic fingerprinter.

Tests the full pipeline end-to-end without external API calls:
  1. Creates a small synthetic dataset (known human + AI patterns)
  2. Fine-tunes DistilBERT for 1 epoch (fast smoke test)
  3. Runs inference on test posts
  4. Validates predictions and output format
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

# ── Synthetic training data ───────────────────────────────────────────

# Human-style: casual, typos, slang, first person, emojis
HUMAN_TEXTS = [
    "lmao just yolo'd my rent money into GME calls, wife's boyfriend is gonna be so mad 🚀🚀",
    "bought the dip today on AMC, averaging down like a true retard. diamond hands baby 💎🙌",
    "my portfolio is down 60% this month lol. at this point im just holding for the tax write off",
    "anyone else notice the unusual options activity on PLTR? something big is coming, can feel it in my bones",
    "just sold all my boomer stocks and went all in on meme stocks, my financial advisor blocked my calls lmao",
    "been bagholding BB since $25, at this point its a long term investment right? right?? 😭",
    "accidentally bought puts instead of calls, somehow made money. maybe inverse myself more often",
    "working behind wendy's to fund my trading account, this is the way 🦍",
    "I genuinely believe SPY will hit 500 by end of year, dont @ me. positions or ban",
    "saw some dd about short interest on WISH, looks like a potential squeeze setup. thoughts?",
    "my analysis of the charts shows a clear cup and handle forming on TSLA, breakout imminent imo",
    "Paper handed my NVDA calls this morning and it ripped 10%. FML as usual. this game hates me",
    "ngl the market has been boring af this week. theta gang eating good while the rest of us suffer",
    "just discovered options trading last week, already down 2k. can someone explain what theta decay is",
    "real talk tho, the fed is gonna print us into oblivion. inflation is already insane at the grocery store",
    "loaded up on SPY puts for the FOMC meeting, either lambo or food stamps, no in between",
    "i swear every time i buy it dips, and every time i sell it moons. am i the market maker's algo??",
    "spent 4 hours doing technical analysis on GME only to realize none of it matters with meme stocks lol",
    "my buddy at work told me to buy DOGE at 60 cents... we don't talk anymore",
    "holding AMC through the squeeze, not selling til i see phone numbers in my account 📱🚀🌕",
]

# AI-style: formal, structured, polished, analytical
AI_TEXTS = [
    "GameStop (GME) presents a compelling investment opportunity with significant upside potential. The company's transformation under Ryan Cohen's leadership, combined with its expanding e-commerce presence and strong brand recognition, positions it for substantial growth. Current short interest remains elevated at approximately 20%, creating conditions for a potential short squeeze. My price target is $85, representing a 300% upside from current levels.",
    "AMC Entertainment Holdings (AMC) stands at a pivotal inflection point. Post-pandemic recovery in theatrical attendance, combined with strategic diversification into cryptocurrency payments and exclusive content partnerships, provides a robust foundation for sustained revenue growth. The company's strong retail investor base and significant short interest create asymmetric upside potential.",
    "A comprehensive analysis of Tesla's (TSLA) competitive positioning reveals underappreciated catalysts. The company's vertical integration strategy, encompassing battery production, autonomous driving technology, and energy storage solutions, creates a formidable competitive moat. Revenue diversification beyond automotive sales, particularly in energy generation and storage, suggests the current valuation may not fully capture long-term growth potential.",
    "Palantir Technologies (PLTR) represents a differentiated play on the enterprise AI revolution. The company's Gotham and Foundry platforms provide mission-critical analytics capabilities with high switching costs. Government contract renewals and expanding commercial adoption demonstrate strong product-market fit. The addressable market for enterprise AI platforms is projected to reach $500 billion by 2028.",
    "NVIDIA Corporation (NVDA) maintains its dominant position in the accelerated computing ecosystem. The company's GPU architecture has become the de facto standard for artificial intelligence training and inference workloads. Strong demand visibility from hyperscale cloud providers and enterprise customers, combined with the upcoming product cycle, supports continued revenue momentum.",
    "Due diligence on Apple Inc. (AAPL) reveals several underappreciated growth vectors. The Services segment continues to expand at double-digit rates, driving margin expansion and recurring revenue growth. The company's installed base of over 2 billion active devices creates a powerful platform for monetization through subscriptions, advertising, and financial services.",
    "Microsoft Corporation (MSFT) is strategically positioned to capitalize on the secular shift toward cloud computing and artificial intelligence. Azure's market share gains, combined with the integration of AI capabilities across the productivity suite, create a compelling growth narrative. Enterprise digital transformation spending remains robust, providing strong demand visibility for cloud infrastructure.",
    "A thorough examination of NIO Inc. (NIO) reveals an emerging leader in China's electric vehicle market. The company's battery-as-a-service model represents a differentiated approach to addressing range anxiety and reducing upfront vehicle costs. Strategic investments in autonomous driving capabilities and expansion into the European market provide additional growth catalysts.",
    "Rivian Automotive (RIVN) occupies a unique position in the electric vehicle landscape with its focus on the adventure and commercial segments. The Amazon delivery van contract provides revenue visibility and validates the company's manufacturing capabilities. R1T and R1S consumer vehicles have received favorable reviews, and the order backlog remains substantial.",
    "SoFi Technologies (SOFI) is building a comprehensive digital financial services ecosystem. The company's bank charter acquisition enables deposit-taking capabilities and improved lending economics. Cross-selling opportunities across lending, investing, and banking products drive customer lifetime value expansion. The total addressable market for digital banking exceeds $1 trillion.",
    "The SPY ETF tracking the S&P 500 index offers investors broad market exposure with exceptional liquidity. Current macroeconomic indicators suggest continued economic expansion, supported by strong corporate earnings growth and accommodative monetary policy. Technical analysis reveals a sustained uptrend with strong support at key moving averages.",
    "BlackBerry Limited (BB) is executing a strategic transformation from consumer hardware to enterprise cybersecurity and IoT software. The company's QNX operating system powers over 215 million vehicles globally, representing a significant competitive advantage in the automotive software market. CylancePROTECT and CylanceOPTICS provide AI-powered endpoint security solutions.",
    "Dogecoin (DOGE) has evolved from a meme cryptocurrency to a legitimate digital payment medium with growing merchant adoption. Network transaction volumes have increased substantially, and the upcoming protocol upgrade will improve transaction throughput and reduce fees. Community-driven development and high-profile endorsements continue to drive ecosystem growth.",
    "Nokia Corporation (NOK) is positioned to benefit from the global 5G infrastructure deployment cycle. The company's end-to-end portfolio spanning radio access networks, core networks, and enterprise solutions provides a comprehensive offering. Strategic partnerships with major telecommunications operators and government contracts for critical infrastructure modernization support sustained revenue growth.",
    "QQQ, the Invesco NASDAQ-100 ETF, provides concentrated exposure to technology and growth-oriented companies. The underlying index's weighting toward artificial intelligence beneficiaries, cloud computing leaders, and digital advertising platforms creates favorable positioning for secular growth trends. Strong earnings momentum among top holdings supports continued appreciation.",
    "A detailed analysis of WISH (ContextLogic) reveals a turnaround opportunity. The company's marketplace platform serves price-conscious consumers with a mobile-first shopping experience. Operational improvements including faster shipping times, enhanced product curation, and improved seller quality metrics are driving improved customer satisfaction.",
    "Advanced Micro Devices (AMD) continues to gain market share across CPU and GPU segments. The company's EPYC server processors have achieved design wins at major cloud providers, while Radeon GPUs compete effectively in both gaming and data center applications. The acquisition of Xilinx expands AMD's addressable market into adaptive computing.",
    "Virgin Galactic (SPCE) represents a pioneering investment in commercial space tourism. The company has successfully completed crewed test flights and begun commercial operations. The addressable market for space tourism is projected to reach $8 billion annually, with Virgin Galactic positioned as the category leader.",
    "Lucid Group (LCID) is manufacturing premium electric vehicles that compete directly with established luxury brands. The Lucid Air has received accolades for its industry-leading range and performance specifications. Saudi Arabia's Public Investment Fund provides strategic financial backing and access to emerging market opportunities.",
    "Coinbase Global (COIN) serves as critical infrastructure for the digital asset ecosystem. As the largest regulated cryptocurrency exchange in the United States, the company benefits from increasing institutional adoption and regulatory clarity. Revenue diversification through staking, custody, and blockchain infrastructure services reduces dependence on trading volumes.",
]


def create_dataset(output_dir: str) -> str:
    """Create a small synthetic training dataset."""
    import random

    data = [{"text": t, "label": 0} for t in HUMAN_TEXTS] + [
        {"text": t, "label": 1} for t in AI_TEXTS
    ]
    random.seed(42)
    random.shuffle(data)

    split = int(len(data) * 0.8)
    dataset = {"train": data[:split], "test": data[split:]}

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "dataset.json")
    with open(path, "w") as fh:
        json.dump(dataset, fh)
    return path


def create_test_jsonl(data_dir: str) -> str:
    """Create test JSONL posts to run inference on."""
    os.makedirs(data_dir, exist_ok=True)
    posts = [
        {
            "post_id": "test_001",
            "author_id": "human_user",
            "account_age_days": 500,
            "text": "lol bought more GME today, wife is gonna kill me 🚀🚀 diamond hands or bust",
            "timestamp_utc": "2026-06-23T12:00:00+00:00",
            "platform": "reddit",
            "ticker": "TEST",
            "upvotes": 100,
            "cashtags": ["GME"],
        },
        {
            "post_id": "test_002",
            "author_id": "ai_user",
            "account_age_days": 3,
            "text": (
                "A comprehensive analysis of GameStop's strategic transformation reveals "
                "significant upside potential. The company's pivot toward e-commerce, "
                "combined with elevated short interest and strong retail investor engagement, "
                "creates conditions for asymmetric returns. My price target of $120 represents "
                "approximately 400% upside from current levels."
            ),
            "timestamp_utc": "2026-06-23T12:05:00+00:00",
            "platform": "stocktwits",
            "ticker": "TEST",
            "upvotes": 5,
            "cashtags": ["GME"],
        },
        {
            "post_id": "test_003",
            "author_id": "ambiguous_user",
            "account_age_days": 100,
            "text": "Thinking about buying some TSLA shares. What do you all think?",
            "timestamp_utc": "2026-06-23T12:10:00+00:00",
            "platform": "twitter",
            "ticker": "TEST",
            "upvotes": 10,
            "cashtags": ["TSLA"],
        },
    ]
    path = os.path.join(data_dir, "TEST_reddit_2026-06-23.jsonl")
    with open(path, "w") as fh:
        for p in posts:
            fh.write(json.dumps(p) + "\n")
    return path


def main() -> None:
    import config

    # Use temp dir for isolation
    tmp = tempfile.mkdtemp(prefix="classifier_test_",
                           dir=os.path.dirname(__file__))

    dataset_dir = os.path.join(tmp, "training")
    model_dir = os.path.join(tmp, "model")
    raw_dir = os.path.join(tmp, "raw")
    analyzed_dir = os.path.join(tmp, "analyzed")

    # Temporarily override config
    orig_classifier_dir = config.CLASSIFIER_DIR
    orig_data_dir = config.DATA_DIR
    orig_analyzed_dir = config.ANALYZED_DIR

    config.CLASSIFIER_DIR = model_dir
    config.DATA_DIR = raw_dir
    config.ANALYZED_DIR = analyzed_dir

    try:
        # ── Step 1: Create dataset ────────────────────────────────
        print("━━━ Creating synthetic dataset ━━━")
        dataset_path = create_dataset(dataset_dir)
        with open(dataset_path) as fh:
            ds = json.load(fh)
        print(f"✓ Dataset: {len(ds['train'])} train, {len(ds['test'])} test")

        # ── Step 2: Train (1 epoch for speed) ─────────────────────
        print("\n━━━ Training DistilBERT (1 epoch, smoke test) ━━━")
        from classifier.trainer import FingerprinterTrainer

        trainer = FingerprinterTrainer(
            output_dir=model_dir,
            epochs=1,
            batch_size=8,
            lr=2e-5,
        )
        metrics = trainer.train(dataset_path)

        print(f"✓ Training complete")
        for k, v in metrics.items():
            if "f1" in k or "auc" in k or "loss" in k:
                print(f"    {k}: {v}")

        # Verify model files exist
        final_dir = os.path.join(model_dir, "final")
        assert os.path.exists(final_dir), "Model dir not created"
        assert os.path.exists(
            os.path.join(final_dir, "config.json")
        ), "config.json missing"
        assert os.path.exists(
            os.path.join(final_dir, "tokenizer_config.json")
        ), "tokenizer missing"
        print("✓ Model saved correctly")

        # Verify metrics file
        metrics_file = os.path.join(model_dir, "eval_metrics.json")
        assert os.path.exists(metrics_file), "eval_metrics.json missing"
        with open(metrics_file) as fh:
            saved_metrics = json.load(fh)
        assert "eval_f1_macro" in saved_metrics, "F1 not in metrics"
        assert "eval_auc_roc" in saved_metrics, "AUC-ROC not in metrics"
        print(f"✓ Metrics file valid: F1={saved_metrics['eval_f1_macro']}, "
              f"AUC={saved_metrics['eval_auc_roc']}")

        # ── Step 3: Inference ─────────────────────────────────────
        print("\n━━━ Running inference ━━━")
        create_test_jsonl(raw_dir)

        from classifier.predictor import LinguisticFingerprinter

        fp = LinguisticFingerprinter(
            model_dir=final_dir,
            threshold=0.7,
        )

        # Direct prediction
        test_texts = [
            "yolo GME calls, diamond hands 🚀💎 tendies incoming",
            (
                "A comprehensive fundamental analysis of GameStop Corporation reveals "
                "substantial upside potential driven by strategic transformation "
                "initiatives and elevated short interest dynamics."
            ),
        ]
        preds = fp.predict(test_texts)
        assert len(preds) == 2, "Wrong number of predictions"

        for i, (text, pred) in enumerate(zip(test_texts, preds)):
            snippet = text[:60] + "…"
            print(
                f"    [{i}] ai_prob={pred['ai_prob']:.3f}  "
                f"is_ai={pred['is_ai_generated']}  "
                f"text={snippet}"
            )
            assert 0.0 <= pred["ai_prob"] <= 1.0, "ai_prob out of range"
            assert isinstance(pred["is_ai_generated"], bool), "Bad type"

        print("✓ Direct prediction OK")

        # JSONL pipeline
        out_path = fp.process_jsonl("TEST", data_dir=raw_dir, output_dir=analyzed_dir)
        assert out_path is not None, "No output file"
        assert out_path.exists(), "Output file not found"

        with open(out_path) as fh:
            records = [json.loads(l) for l in fh if l.strip()]
        assert len(records) == 3, f"Expected 3 records, got {len(records)}"

        for r in records:
            assert "ai_prob" in r, f"Missing ai_prob in {r['post_id']}"
            assert "is_ai_generated" in r, f"Missing is_ai_generated in {r['post_id']}"
            assert "text" in r, "Original fields missing"
            assert "post_id" in r, "Original fields missing"

        print(f"✓ JSONL pipeline OK → {out_path.name}")
        for r in records:
            print(
                f"    {r['post_id']}: ai_prob={r['ai_prob']:.3f}  "
                f"is_ai={r['is_ai_generated']}"
            )

        print("\n══════════════════════════════════════════")
        print("  ALL CLASSIFIER TESTS PASSED ✓")
        print("══════════════════════════════════════════")

    finally:
        config.CLASSIFIER_DIR = orig_classifier_dir
        config.DATA_DIR = orig_data_dir
        config.ANALYZED_DIR = orig_analyzed_dir
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
