from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOTS = [
    REPO_ROOT / "output" / "experiments",
    REPO_ROOT / "outputs" / "experiments",
]


st.set_page_config(
    page_title="MFA Cluster Labels",
    page_icon="",
    layout="wide",
)


def find_label_files() -> list[Path]:
    """Find cluster_labels.json files in the usual experiment output folders."""
    paths: list[Path] = []
    for root in DEFAULT_SEARCH_ROOTS:
        if root.exists():
            paths.extend(root.glob("**/cluster_labels.json"))
    return sorted(set(paths))


@st.cache_data(show_spinner=False)
def load_json_from_path(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_json_from_upload(uploaded_file) -> dict[str, Any]:
    return json.loads(uploaded_file.getvalue().decode("utf-8"))


def normalize_clusters(data: dict[str, Any]) -> list[dict[str, Any]]:
    clusters = data.get("clusters", {})
    out = []
    for cluster_id, cluster in clusters.items():
        top_tokens = cluster.get("top_tokens") or []
        examples = cluster.get("examples") or []
        haystack = " ".join(
            str(part)
            for part in [
                cluster_id,
                cluster.get("label"),
                cluster.get("description"),
                cluster.get("evidence"),
                " ".join(str(t.get("token", "")) for t in top_tokens),
                " ".join(str(ex.get("snippet", "")) for ex in examples),
            ]
            if part is not None
        ).lower()
        out.append({
            "id": str(cluster_id),
            "id_num": int(cluster_id) if str(cluster_id).isdigit() else 10**18,
            "label": cluster.get("label") or "(unlabeled)",
            "description": cluster.get("description") or "",
            "evidence": cluster.get("evidence") or "",
            "cluster_size": cluster.get("cluster_size"),
            "top_tokens": top_tokens,
            "examples": examples,
            "raw_response": cluster.get("raw_response") or "",
            "haystack": haystack,
        })
    return sorted(out, key=lambda c: c["id_num"])


def choose_data_source() -> tuple[dict[str, Any] | None, str | None]:
    st.sidebar.header("Data")

    label_files = find_label_files()
    choices = ["Upload a file"] + [str(path.relative_to(REPO_ROOT)) for path in label_files]
    selected = st.sidebar.selectbox("cluster_labels.json", choices)

    if selected == "Upload a file":
        uploaded = st.sidebar.file_uploader("Choose cluster_labels.json", type="json")
        if uploaded is None:
            return None, None
        return load_json_from_upload(uploaded), uploaded.name

    path = REPO_ROOT / selected
    return load_json_from_path(str(path)), selected


def sort_clusters(clusters: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "Cluster size":
        return sorted(clusters, key=lambda c: (c["cluster_size"] or 0, -c["id_num"]), reverse=True)
    if sort_by == "Label":
        return sorted(clusters, key=lambda c: (c["label"].lower(), c["id_num"]))
    if sort_by == "Example count":
        return sorted(clusters, key=lambda c: (len(c["examples"]), -c["id_num"]), reverse=True)
    return sorted(clusters, key=lambda c: c["id_num"])


def highlight_target(snippet: str) -> str:
    escaped = html.escape(snippet or "")
    return (
        escaped
        .replace("&lt;target&gt;", '<mark class="target-token">')
        .replace("&lt;/target&gt;", "</mark>")
    )


def render_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        .cluster-card {
            border: 1px solid #d9e1ea;
            border-radius: 8px;
            padding: 0.85rem;
            background: white;
            margin-bottom: 0.7rem;
        }
        .muted { color: #607083; font-size: 0.88rem; }
        .chip {
            display: inline-block;
            border: 1px solid #d9e1ea;
            border-radius: 999px;
            padding: 0.16rem 0.5rem;
            margin: 0.12rem;
            background: #eef3f8;
            font-size: 0.84rem;
        }
        .snippet {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            line-height: 1.55;
            font-size: 0.9rem;
        }
        .target-token {
            background: #ffe9a8;
            border: 1px solid #d39700;
            border-radius: 4px;
            padding: 0.05rem 0.18rem;
            font-weight: 700;
            color: #402900;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metadata(data: dict[str, Any], source_name: str) -> None:
    meta = data.get("metadata", {})
    with st.sidebar.expander("Metadata", expanded=True):
        st.caption(source_name)
        for key in ["layer", "assignments_path", "windows_dataset", "top_index_path"]:
            if key in meta:
                st.text(f"{key}: {meta[key]}")


def cluster_summary_rows(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for cluster in clusters:
        top = ", ".join(
            f"{item.get('token')} x{item.get('count')}"
            for item in cluster["top_tokens"][:5]
        )
        rows.append({
            "cluster": cluster["id"],
            "size": cluster["cluster_size"],
            "label": cluster["label"],
            "description": cluster["description"],
            "examples": len(cluster["examples"]),
            "top_tokens": top,
        })
    return rows


def render_cluster(cluster: dict[str, Any]) -> None:
    size = cluster["cluster_size"]
    size_text = f"{size:,}" if isinstance(size, int) else "unknown"

    st.subheader(f"Cluster {cluster['id']}: {cluster['label']}")
    st.caption(f"cluster size: {size_text} · examples shown: {len(cluster['examples'])}")

    if cluster["description"]:
        st.write(cluster["description"])
    if cluster["evidence"]:
        st.markdown("**Evidence**")
        st.write(cluster["evidence"])

    st.markdown("**Top tokens**")
    if cluster["top_tokens"]:
        chips = []
        for item in cluster["top_tokens"][:80]:
            token = html.escape(str(item.get("token", "")))
            count = html.escape(str(item.get("count", "")))
            chips.append(f'<span class="chip">{token} x{count}</span>')
        st.markdown("".join(chips), unsafe_allow_html=True)
    else:
        st.caption("No top-token data saved.")

    st.markdown("**Examples**")
    for example in cluster["examples"]:
        rank = example.get("rank")
        token = example.get("token")
        resp = example.get("responsibility")
        row = example.get("global_row")
        pos = example.get("tok_pos")
        resp_text = f"{float(resp):.6f}" if isinstance(resp, (int, float)) else str(resp)
        st.markdown(
            f"""
            <div class="cluster-card">
              <div class="muted">rank {html.escape(str(rank))} · resp {html.escape(resp_text)} · token {html.escape(str(token))} · row {html.escape(str(row))} · pos {html.escape(str(pos))}</div>
              <div class="snippet">{highlight_target(example.get("snippet", ""))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if cluster["raw_response"]:
        with st.expander("Raw LLM response"):
            st.code(cluster["raw_response"], language="json")


def main() -> None:
    render_css()
    st.title("MFA Cluster Label Dashboard")

    data, source_name = choose_data_source()
    if data is None:
        st.info("Select or upload a `cluster_labels.json` file from the sidebar.")
        return

    render_metadata(data, source_name or "uploaded file")

    clusters = normalize_clusters(data)
    query = st.sidebar.text_input("Search", placeholder="label, token, snippet...")
    sort_by = st.sidebar.selectbox("Sort clusters", ["Cluster id", "Cluster size", "Label", "Example count"])

    filtered = clusters
    if query.strip():
        needle = query.strip().lower()
        filtered = [cluster for cluster in clusters if needle in cluster["haystack"]]
    filtered = sort_clusters(filtered, sort_by)

    st.sidebar.metric("Clusters", f"{len(filtered):,}", f"of {len(clusters):,}")

    if not filtered:
        st.warning("No clusters match the current search.")
        return

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        st.markdown("#### Cluster Summary")
        st.dataframe(
            cluster_summary_rows(filtered),
            use_container_width=True,
            hide_index=True,
            height=560,
        )

    with right:
        ids = [cluster["id"] for cluster in filtered]
        selected_id = st.selectbox("Inspect cluster", ids)
        selected = next(cluster for cluster in filtered if cluster["id"] == selected_id)
        render_cluster(selected)


if __name__ == "__main__":
    main()
