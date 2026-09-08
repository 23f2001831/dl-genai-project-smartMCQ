"""
Smart MCQ Solver — Ensemble (TF-IDF + BiLSTM)
Streamlit App for Community Cloud Deployment
"""
import os, re, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import streamlit as st

# ─── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")

# ─── Model definitions (must match training) ─────────────
class DualBranchScorer(nn.Module):
    def __init__(self, word_vocab, char_vocab, word_hidden, char_hidden,
                 merged_hidden, dropout=0.3):
        super().__init__()
        self.word_branch = nn.Sequential(
            nn.Linear(word_vocab, word_hidden),
            nn.BatchNorm1d(word_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.char_branch = nn.Sequential(
            nn.Linear(char_vocab, char_hidden),
            nn.BatchNorm1d(char_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(word_hidden + char_hidden, merged_hidden),
            nn.BatchNorm1d(merged_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(merged_hidden, 1),
        )

    def forward(self, x_word, x_char):
        w = self.word_branch(x_word)
        c = self.char_branch(x_char)
        return self.head(torch.cat([w, c], dim=1)).squeeze(1)


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, x):
        return self.embedding(x)


class BiLSTMEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                     enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True,
                                     total_length=x.size(1))
        return out


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out, lengths):
        scores = self.v(torch.tanh(self.attn(lstm_out))).squeeze(-1)
        seq_len = lstm_out.size(1)
        mask = torch.arange(seq_len, device=lengths.device)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class ScoreHead(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, context):
        return self.net(context).squeeze(-1)


class PairScorer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout, pad_idx):
        super().__init__()
        self.embedding = WordEmbedding(vocab_size, embed_dim, pad_idx)
        self.encoder = BiLSTMEncoder(embed_dim, hidden_dim, num_layers, dropout)
        self.attention = Attention(hidden_dim * 2)
        self.score_head = ScoreHead(hidden_dim * 2, dropout)

    def forward(self, ids, lengths):
        x = self.embedding(ids)
        lstm_out = self.encoder(x, lengths)
        context, weights = self.attention(lstm_out, lengths)
        score = self.score_head(context)
        return score, weights


# ─── Load artifacts (cached so it only runs once) ────────
@st.cache_resource
def load_models():
    deploy_dir = os.path.join(os.path.dirname(__file__), "ensemble_deploy")

    with open(f"{deploy_dir}/config.pkl", "rb") as f:
        cfg = pickle.load(f)
    with open(f"{deploy_dir}/word_tfidf.pkl", "rb") as f:
        w_tfidf = pickle.load(f)
    with open(f"{deploy_dir}/char_tfidf.pkl", "rb") as f:
        c_tfidf = pickle.load(f)
    with open(f"{deploy_dir}/vocab.pkl", "rb") as f:
        voc = pickle.load(f)

    dev = torch.device("cpu")

    tfidf_m = DualBranchScorer(
        word_vocab=len(w_tfidf.vocabulary_), char_vocab=len(c_tfidf.vocabulary_),
        word_hidden=cfg["tfidf_word_hidden"], char_hidden=cfg["tfidf_char_hidden"],
        merged_hidden=cfg["tfidf_merged_hidden"], dropout=cfg["tfidf_dropout"],
    ).to(dev)
    tfidf_m.load_state_dict(
        torch.load(f"{deploy_dir}/best_tfidf_model.pt", map_location=dev, weights_only=True)
    )
    tfidf_m.eval()

    bilstm_m = PairScorer(
        vocab_size=cfg["vocab_size"], embed_dim=cfg["bilstm_embed_dim"],
        hidden_dim=cfg["bilstm_hidden_dim"], num_layers=cfg["bilstm_num_layers"],
        dropout=cfg["bilstm_dropout"], pad_idx=cfg["pad_idx"],
    ).to(dev)
    bilstm_m.load_state_dict(
        torch.load(f"{deploy_dir}/best_bilstm_model.pt", map_location=dev, weights_only=True)
    )
    bilstm_m.eval()

    return cfg, w_tfidf, c_tfidf, voc, tfidf_m, bilstm_m, dev


CFG, word_tfidf, char_tfidf, vocab, tfidf_model, bilstm_model, device = load_models()

LABELS = ["A", "B", "C", "D", "E"]
PAD_IDX = CFG["pad_idx"]
BEST_ALPHA = CFG["best_alpha"]


# ─── Inference helpers ────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text)


def encode_bilstm(question, option):
    text = f"question: {question} answer: {option}".lower()
    tokens = tokenize(text)
    max_len = CFG["bilstm_max_len"]
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_len]]
    length = len(ids)
    ids += [PAD_IDX] * (max_len - length)
    return ids, length


def softmax(scores):
    e = np.exp(scores - np.max(scores))
    return e / e.sum()


@torch.no_grad()
def predict(question, options):
    # TF-IDF model
    texts = [f"Question: {question} Answer: {opt}" for opt in options.values()]
    Xw = torch.tensor(word_tfidf.transform(texts).toarray(), dtype=torch.float32)
    Xc = torch.tensor(char_tfidf.transform(texts).toarray(), dtype=torch.float32)
    tfidf_logits = tfidf_model(Xw, Xc).numpy()
    tfidf_probs = softmax(torch.sigmoid(torch.tensor(tfidf_logits)).numpy())

    # BiLSTM model
    all_ids, all_lengths = [], []
    for opt_text in options.values():
        ids, length = encode_bilstm(question, opt_text)
        all_ids.append(ids)
        all_lengths.append(length)

    ids_t = torch.tensor(all_ids, dtype=torch.long).unsqueeze(0)
    len_t = torch.tensor(all_lengths, dtype=torch.long).unsqueeze(0)
    b, n_opts, seq_len = ids_t.shape
    flat_scores, _ = bilstm_model(ids_t.view(b * n_opts, seq_len),
                                   len_t.view(b * n_opts))
    bilstm_logits = flat_scores.view(b, n_opts).squeeze(0).numpy()
    bilstm_probs = softmax(bilstm_logits)

    # Ensemble
    ensemble = BEST_ALPHA * tfidf_probs + (1 - BEST_ALPHA) * bilstm_probs

    return tfidf_probs, bilstm_probs, ensemble


# ─── Streamlit UI ─────────────────────────────────────────
st.title("🧠 Smart MCQ Solver")
st.markdown(
    "Ensemble of **TF-IDF Dual-Branch Scorer** + **BiLSTM with Attention**. "
    "Enter a question with 5 options to get the predicted answer."
)

st.divider()

question = st.text_area("📝 Question", height=100,
                        placeholder="Type your MCQ question here...")

col1, col2 = st.columns(2)
with col1:
    opt_a = st.text_input("Option A")
    opt_b = st.text_input("Option B")
    opt_c = st.text_input("Option C")
with col2:
    opt_d = st.text_input("Option D")
    opt_e = st.text_input("Option E")

if st.button("🔍 Predict Answer", type="primary", use_container_width=True):
    if not question or not all([opt_a, opt_b, opt_c, opt_d, opt_e]):
        st.warning("Please fill in the question and all 5 options.")
    else:
        options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d, "E": opt_e}

        with st.spinner("Running ensemble inference..."):
            tfidf_probs, bilstm_probs, ensemble_probs = predict(question, options)

        ranked_idx = np.argsort(ensemble_probs)[::-1]
        top_letter = LABELS[ranked_idx[0]]
        top3 = [LABELS[i] for i in ranked_idx[:3]]

        # Result banner
        st.success(f"**Predicted Answer: {top_letter} — {options[top_letter]}**")
        st.markdown(f"**Top 3**: {', '.join(top3)}")
        st.caption(f"Ensemble weight: α = {BEST_ALPHA:.2f} "
                   f"(TF-IDF = {BEST_ALPHA:.0%}, BiLSTM = {1-BEST_ALPHA:.0%})")

        st.divider()

        # Score table
        st.subheader("Score Breakdown")
        for i in ranked_idx:
            letter = LABELS[i]
            ens_score = ensemble_probs[i]
            col_label, col_bar, col_scores = st.columns([1, 4, 3])
            with col_label:
                if letter == top_letter:
                    st.markdown(f"**✅ {letter}**")
                else:
                    st.markdown(f"**{letter}**")
            with col_bar:
                st.progress(float(ens_score))
                st.caption(options[letter])
            with col_scores:
                st.caption(
                    f"TF-IDF: {tfidf_probs[i]:.3f} · "
                    f"BiLSTM: {bilstm_probs[i]:.3f} · "
                    f"Ensemble: {ens_score:.3f}"
                )

st.divider()
st.caption("Built with PyTorch + Streamlit · DLP GenAI Project")
