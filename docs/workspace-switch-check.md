# Workspace switch verification

This test branch adds a short manual check for the Stock Research and PE Deal Intelligence switch.

1. Run `streamlit run app.py` from the repository root.
2. Confirm Stock Research opens by default and displays the stock tabs and ticker controls.
3. Select **PE Deal Intelligence** in the centered switch.
4. Confirm the PE landing page appears and the stock tabs, sidebar controls and results disappear.
5. Refresh the page. The `?workspace=pe` URL should reopen the PE workspace.
6. Select **Stock Research** to return to the stock tools.

The PE page is a foundation for future work. Deal collection and analysis are not connected yet, and the page states this explicitly.

Automated routing checks:

```sh
python -m pytest tests/test_workspaces.py -q
```
