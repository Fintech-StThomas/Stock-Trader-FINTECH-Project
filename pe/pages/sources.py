"""Read-only public source coverage. No network, refresh actions, or private text."""
import json
from datetime import datetime
from pathlib import Path
import streamlit as st
from pe.collectors.config import load_sources
from pe.storage import DEFAULT_COVERAGE
from pe.published import published_records, published_inventory


def _display_time(value):
    if not value:
        return 'Unknown'
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).strftime('%b %d, %Y at %H:%M UTC')
    except (ValueError, AttributeError):
        return 'Unknown'


def render_sources(coverage=None):
    """Accept an injected public manifest, or load the team's published coverage JSON."""
    st.subheader('Know where the information comes from')
    st.write('Track the firms included in collection, the dates observed, and any gaps before using the research.')
    if coverage is None:
        try:
            coverage = json.loads(Path(DEFAULT_COVERAGE).read_text())
        except FileNotFoundError:
            coverage = {'sources': [dict(s.to_dict(), published_coverage=None, latest_attempt=None,
                                        last_successful_run=None) for s in load_sources()]}
        except (OSError, ValueError):
            st.error('The coverage file could not be read. The team needs to republish a valid snapshot.')
            return
    if not isinstance(coverage, dict) or not isinstance(coverage.get('sources'), list):
        st.error('The coverage snapshot is invalid. Please ask the team to republish it.')
        return
    records = published_records()
    inventory = published_inventory(records)
    st.markdown('### Firms available in Deal Explorer')
    st.write('Choose these firms in the explorer to investigate competing investment activity. Counts reflect saved records, not complete firm activity.')
    st.dataframe(inventory, hide_index=True, width='stretch')
    st.caption('Curated samples are source-linked research entries awaiting team review. They are separate from automated collection and its validation results.')
    st.markdown('### Announcement collection coverage')
    sources = coverage['sources']
    st.info('Saved snapshot · This page uses cached coverage. Opening it never starts a collection.')
    st.caption('Announcement counts are not deal counts. The Deal Explorer has its own dataset; these raw inputs still need extraction and validation.')
    published = [s for s in sources if s.get('published_coverage')]
    cols = st.columns(3)
    cols[0].metric('Firms with saved inputs', len(published))
    cols[1].metric('Saved announcements', sum(s['published_coverage']['counts']['announcements'] for s in published))
    cols[2].metric('Candidate firms', sum(not s.get('enabled', False) for s in sources))
    for source in sources:
        with st.container(border=True):
            st.markdown(f"### {source['publisher_pe_firm']}")
            st.link_button('Open source newsroom', source['listing_url'])
            st.caption(source.get('access_note', ''))
            saved = source.get('published_coverage')
            latest = source.get('latest_attempt')
            curated = [r for r in records if r['publisher_pe_firm'] == source['publisher_pe_firm'] and r.get('data_status') == 'curated']
            if curated:
                st.success(f'{len(curated)} curated announcement samples available in Deal Explorer.')
                for row in curated:
                    st.markdown(f"- [{row['headline']}]({row['source_url'][0]})")
                st.caption('Short evidence and classified facts only; no full article archive or automated refresh for these samples.')
            if not source.get('enabled'):
                st.warning('Automated collection is disabled; source-linked samples may still be available above.')
            if not saved:
                st.write('No successful collection has been published for this firm.')
            else:
                st.write(f"**Last successful collection (UTC):** {_display_time(source.get('last_successful_run'))}")
                st.caption(f"Collection mode: {saved['mode']} · Website delivery: cached · Version: {source.get('published_run_id', 'Unknown')}")
                requested, observed = saved['requested_period'], saved['observed_period']
                st.write(f"**Requested period:** {requested['start']} to {requested['end']}")
                st.write(f"**Observed publication dates:** {observed['start'] or 'Unknown'} to {observed['end'] or 'Unknown'}")
                counts = saved['counts']
                st.write(f"{counts['announcements']} announcements · {counts['unknown_dates']} with unknown dates · "
                         f"{counts['duplicate_content']} identical-content duplicates grouped")
                if saved['truncated']:
                    st.warning('Limited pilot coverage: ' + '; '.join(saved['stop_reasons']))
                st.caption(saved['coverage_note'])
            if latest:
                st.write(f"**Latest attempt:** {latest['status'].title()} · {_display_time(latest['finished_at'])}")
                c = latest['counts']
                st.caption(f"Pages attempted: {c['attempted_pages']} · successful: {c['successful_pages']} · failed: {c['failed_pages']}")
                if latest['status'] != 'success' and saved:
                    st.warning('The latest attempt did not replace the previous successful snapshot.')
                if latest['errors']:
                    st.dataframe(latest['errors'], hide_index=True, width='stretch')
    st.caption('Full article text and raw pages stay in the private team archive. Public visitors cannot run refreshes or download that archive here.')
