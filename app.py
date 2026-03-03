import os, json, time


# Render Query A and Query B inputs vertically (one under the other)
INPUT_LAYOUT_VERTICAL = True
import streamlit as st
# Keep A and B outputs separate and persistent across reruns.
# Streamlit re-runs on every interaction; session_state prevents one run from overwriting the other.
if "outA" not in st.session_state:
    st.session_state.outA = {"resp": None, "debug": None, "report": None}
if "outB" not in st.session_state:
    st.session_state.outB = {"resp": None, "debug": None, "report": None}

from rag_core import (
    pretty_query_A, pretty_query_B,
    answer_query_A_single_pass, answer_query_B_single_pass,
    answer_query_A_with_verification, answer_query_B_with_verification
)

st.set_page_config(page_title="ESC RAG Runner", layout="wide")
st.title("ESC RAG Runner — Query A / Query B")

# -------------------------
# Settings
# -------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Model", value=os.environ.get("ESC_RAG_MODEL", "gpt-5.2"))

    top_k_A = st.slider("Top-K chunks (A)", 8, 30, 16, 1)
    top_k_B = st.slider("Top-K chunks (B)", 8, 40, 18, 1)

    use_verification = st.checkbox("Use verification loop (Draft → Verify → Revise)", value=True)
    max_rounds = st.slider("Max verification rounds", 1, 4, 2, 1)

    show_retrieval = st.checkbox("Show retrieval bundle", value=False)
    show_verifier = st.checkbox("Show verifier reports", value=False)
    save_outputs = st.checkbox("Save outputs to /runs", value=True)

# -------------------------
# Inputs
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Query A input (pre-diagnosis)")
    case_text_A = st.text_area(
        "Paste Case Text A (your exact style text)",
        height=420,
        placeholder="Paste the pre-diagnosis case here..."
    )
    run_A = st.button("Run Query A", type="primary", use_container_width=True)

with col2:
    st.subheader("Query B input (post-diagnosis add-on)")
    case_text_B_addon = st.text_area(
        "Paste Post-diagnosis info (diagnosis + echo + your questions)",
        height=420,
        placeholder="Paste the post-diagnosis add-on here..."
    )
    run_B = st.button("Run Query B", type="primary", use_container_width=True)

# -------------------------
# Helpers
# -------------------------
def ensure_runs_dir():
    os.makedirs("runs", exist_ok=True)

def save_run(tag: str, payload: dict, report: str):
    ensure_runs_dir()
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = f"runs/{ts}_{tag}"
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(report)

# -------------------------
# Run A
# -------------------------
if run_A:
    if not case_text_A.strip():
        st.error("Please paste Query A case text.")
    else:
        with st.spinner("Running Query A..."):
            if use_verification:
                respA, debugA = answer_query_A_with_verification(
                    case_text_A,
                    model=model_name,
                    max_rounds=max_rounds,
                    top_k_round0=top_k_A,
                    top_k_reretrieval=max(top_k_A, 18),
                )
            else:
                respA, debugA = answer_query_A_single_pass(
                    case_text_A,
                    model=model_name,
                    top_k_final=top_k_A,
                )

            reportA = pretty_query_A(respA)

        st.success("Query A done.")
        st.subheader("Readable Report (A)")
        st.code(reportA, language="text")

        with st.expander("Raw JSON (A)", expanded=False):
            st.json(respA)

        if show_retrieval:
            with st.expander("Retrieval bundle (A)", expanded=False):
                st.write("Final queries:")
                st.write(debugA.get("final_queries", debugA.get("queries")))
                top_table = debugA.get("top_table") or debugA.get("bundle", {}).get("top_table")
                if top_table is not None:
                    st.dataframe(top_table[["rerank_score","source_index","doc_title","year","section_title","page_start","chunk_id"]])

        if show_verifier and use_verification:
            with st.expander("Verifier reports (A)", expanded=False):
                for i, rep in enumerate(debugA.get("verifier_reports", []), 1):
                    st.markdown(f"**Round {i}** — ok={rep.get('ok')}")
                    st.json(rep)

        if save_outputs:
            payload = {"mode": "verified" if use_verification else "single_pass", "debug": debugA, "response": respA}
            save_run("A", payload, reportA)
            st.info("Saved to /runs")

# -------------------------
# Run B
# -------------------------
if run_B:
    if not case_text_A.strip():
        st.error("Paste Query A case text on the left first (Query B builds on it).")
    elif not case_text_B_addon.strip():
        st.error("Please paste Query B add-on text.")
    else:
        case_text_post = case_text_A.strip() + "\n\n" + case_text_B_addon.strip()

        with st.spinner("Running Query B..."):
            if use_verification:
                respB, debugB = answer_query_B_with_verification(
                    case_text_post,
                    model=model_name,
                    max_rounds=max_rounds,
                    top_k_round0=top_k_B,
                    top_k_reretrieval=max(top_k_B, 20),
                )
            else:
                respB, debugB = answer_query_B_single_pass(
                    case_text_post,
                    model=model_name,
                    top_k_final=top_k_B,
                )

            reportB = pretty_query_B(respB)

        st.success("Query B done.")
        st.subheader("Readable Report (B)")
        st.code(reportB, language="text")

        with st.expander("Raw JSON (B)", expanded=False):
            st.json(respB)

        if show_retrieval:
            with st.expander("Retrieval bundle (B)", expanded=False):
                st.write("Final queries:")
                st.write(debugB.get("final_queries", debugB.get("queries")))
                top_table = debugB.get("top_table") or debugB.get("bundle", {}).get("top_table")
                if top_table is not None:
                    st.dataframe(top_table[["rerank_score","source_index","doc_title","year","section_title","page_start","chunk_id"]])

        if show_verifier and use_verification:
            with st.expander("Verifier reports (B)", expanded=False):
                for i, rep in enumerate(debugB.get("verifier_reports", []), 1):
                    st.markdown(f"**Round {i}** — ok={rep.get('ok')}")
                    st.json(rep)

        if save_outputs:
            payload = {"mode": "verified" if use_verification else "single_pass", "debug": debugB, "response": respB}
            save_run("B", payload, reportB)
            st.info("Saved to /runs")