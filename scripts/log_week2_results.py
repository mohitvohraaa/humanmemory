"""
scripts/log_week2_results.py
───────────────────────────────
Logs Week 1 + Week 2 results to W&B in one run.
Run once to populate the dashboard with everything so far.
"""

import sys
import json
sys.path.insert(0, '.')

import wandb
from src.memory.affective.emotion_classifier import EmotionClassifier
from src.memory.affective.topic_tagger import TopicTagger
from src.memory.affective.affective_store import AffectiveStore
from src.evaluation.affective_eval import evaluate_affective_accuracy, accuracy_summary

wandb.init(project="humanmemory", name="week2_affective_layer")

# ── Week 1 results (already known, logging for the record) ──────────────────
wandb.log({
    "week1/naive_recall_at_10": 0.019,
    "week1/semantic_recall_at_10": 0.068,
    "week1/semantic_mrr": 0.012,
    "week1/semantic_p95_ms": 10.6,
})

# ── Week 2 — affective accuracy (the clean 5-topic run) ──────────────────────
with open('data/synthetic/sentence_pool.json') as f:
    pool = json.load(f)

emotion_clf = EmotionClassifier()
store = AffectiveStore(user_id='wandb_log_run')
store.clear()

topic_turns = {}
for topic, sentences in pool.items():
    selected = sentences[:8]
    topic_turns[topic] = selected
    for text in selected:
        result = emotion_clf.classify(text)
        store.update_from_turn([topic], result.group_scores, result.intensity)

results = evaluate_affective_accuracy(topic_turns, store, emotion_clf)
summary = accuracy_summary(results)

wandb.log({
    "week2/affective_accuracy": summary["accuracy"],
    "week2/n_topics_evaluated": summary["n_topics"],
    "week2/margin_threshold": 0.1,
    "week2/min_intensity_gate": 0.2,
    "week2/min_mentions_required": 3,
    "week2/high_valence_threshold": 0.3,
})

# Log per-topic breakdown as a table
table = wandb.Table(columns=["topic", "predicted", "expected", "correct", "mentions"])
for r in results:
    table.add_data(r.topic, r.predicted_emotion, r.expected_emotion, r.correct, r.mention_count)
wandb.log({"week2/affective_results_table": table})

wandb.finish()
print("Logged Week 1 + Week 2 results to W&B")
