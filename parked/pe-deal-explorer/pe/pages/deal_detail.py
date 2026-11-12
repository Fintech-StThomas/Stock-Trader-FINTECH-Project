"""Evidence-first Deal Detail page, selected through stable deal_id handoff."""
from __future__ import annotations
from typing import Any
import pandas as pd
import streamlit as st
from pe.analytics import load_deals


def render_deal_detail(dataset: Any, evidence: list[dict[str, Any]] | None = None) -> None:
    deals = load_deals(dataset)
    selected_id = st.session_state.get("pe_selected_deal_id")
    st.subheader("Deal Detail")
    if deals.empty:
        st.info("No published deal records are available.")
        return
    ordered=deals.sort_values('publication_date',ascending=False,na_position='last')
    ids=ordered['deal_id'].astype(str).tolist()
    labels={str(r['deal_id']):f"{r['buyer_or_investor'] if pd.notna(r['buyer_or_investor']) else 'Unknown buyer'} → {r['target_company'] if pd.notna(r['target_company']) else 'Unknown target'} · {r['publisher_pe_firm'] if pd.notna(r['publisher_pe_firm']) else 'Unknown firm'}" for _,r in ordered.iterrows()}
    wanted=str(selected_id) if selected_id and str(selected_id) in ids else ids[0]
    if st.session_state.get('pe_detail_choice') != wanted:
        st.session_state['pe_detail_choice']=wanted
    def choose():
        st.session_state['pe_selected_deal_id']=st.session_state['pe_detail_choice']
    selected_id=st.selectbox('Choose a saved deal',ids,format_func=lambda x:labels[x],key='pe_detail_choice',on_change=choose)
    st.session_state['pe_selected_deal_id']=selected_id
    row = deals.loc[deals["deal_id"].astype(str).eq(str(selected_id))].iloc[0]
    st.title(f"{row['buyer_or_investor'] if pd.notna(row['buyer_or_investor']) else 'Unknown buyer'} → {row['target_company'] if pd.notna(row['target_company']) else 'Unknown target'}")
    st.caption(f"Publication date: {row['publication_date'].date().isoformat() if pd.notna(row['publication_date']) else 'Unknown'}")
    cols = st.columns(3)
    for col, label, field in [(cols[0], "Transaction type", "transaction_type"), (cols[1], "Status", "transaction_status"), (cols[2], "Publisher / PE firm", "publisher_pe_firm")]:
        value = row[field]
        col.metric(label, str(value) if pd.notna(value) and str(value).strip() else "Unknown")
    st.markdown("#### Transaction facts")
    facts=[]
    for label,field in [('Buyer / investor','buyer_or_investor'),('Target','target_company'),('Target industry','target_industry'),('Target geography','target_geography'),('Disclosed value','price'),('Currency','currency'),('Review status','review_status')]:
        value=row[field]
        facts.append({'Fact':label,'Available information':str(value) if pd.notna(value) and str(value).strip() else 'Not disclosed'})
    st.dataframe(facts,hide_index=True,width='stretch')
    if pd.notna(row.get('review_notes')):
        st.caption(str(row['review_notes']))
    st.markdown("#### Rationale and supporting evidence")
    tags = row["rationale_tags"] if isinstance(row["rationale_tags"], list) else []
    st.write(", ".join(tags) if tags else "Unknown")
    excerpt = row["evidence_excerpt"]
    st.write(f"> {excerpt}" if pd.notna(excerpt) and str(excerpt).strip() else "No supporting excerpt is available.")
    if pd.notna(row["is_inferred"]) and bool(row["is_inferred"]):
        st.warning("Some classifications are inferred or rule-generated. Verify against the source before relying on them.")
    evidence_rows = [item for item in (evidence or []) if str(item.get("deal_id")) == str(selected_id)]
    if evidence_rows:
        st.markdown("#### Field-level evidence log")
        st.dataframe(pd.DataFrame(evidence_rows), width="stretch", hide_index=True)
    urls = row["source_url"] if isinstance(row["source_url"], list) else []
    st.markdown("#### Source announcements")
    if urls:
        for url in urls: st.markdown(f"- [{url}]({url})")
    else:
        st.write("No source link has been saved.")
    st.caption("Unknown values remain unknown. Stated strategic rationale is not evidence that a transaction improved operating results or investment returns.")
