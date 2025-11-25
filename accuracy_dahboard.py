import streamlit as st
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import nltk

# Ensure NLTK tokenization is available
nltk.download('punkt')

def calculate_metrics(generated, reference):
    # ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)

    # BLEU Score
    smoothie = SmoothingFunction().method4
    bleu_score_val = sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)

    # BERT Score
    bert_p, bert_r, bert_f1 = bert_score([generated], [reference], lang="en", verbose=False)

    return {
        "rouge1": rouge_scores['rouge1'].fmeasure,
        "rouge2": rouge_scores['rouge2'].fmeasure,
        "rougeL": rouge_scores['rougeL'].fmeasure,
        "bleu": bleu_score_val,
        "bert": float(bert_f1[0])
    }


# STREAMLIT UI
st.title("📊 Summary Accuracy Evaluation Dashboard")

st.write("Evaluate the accuracy of generated summaries using ROUGE, BLEU, and BERTScore.")

generated_summary = st.text_area("✍️ Model Generated Summary:")
reference_summary = st.text_area("📌 Human Reference Summary:")

if st.button("Evaluate Accuracy"):
    if generated_summary and reference_summary:
        metrics = calculate_metrics(generated_summary, reference_summary)

        st.subheader("📈 Evaluation Results")

        st.metric("ROUGE-1 Score", f"{metrics['rouge1']:.4f}")
        st.progress(metrics['rouge1'])

        st.metric("ROUGE-2 Score", f"{metrics['rouge2']:.4f}")
        st.progress(metrics['rouge2'])

        st.metric("ROUGE-L Score", f"{metrics['rougeL']:.4f}")
        st.progress(metrics['rougeL'])

        st.metric("BLEU Score", f"{metrics['bleu']:.4f}")
        st.progress(min(metrics['bleu'], 1.0))

        st.metric("BERTScore F1", f"{metrics['bert']:.4f}")
        st.progress(metrics['bert'])

        st.success("🎉 Accuracy evaluation complete!")
    else:
        st.error("Please enter both generated and reference summaries.")

