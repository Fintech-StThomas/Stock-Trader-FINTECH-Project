"""
Paper trading at Alpaca (free paper account): mirror Lighthouse's tickets as real orders on a simulated account, manage
them with the ticket's own rules, and compare the fills with what the backtest assumes.

Paper only: the base URL is Alpaca's paper endpoint and there is no switch to live trading. Keys come from the
environment (ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY; free at alpaca.markets). Without keys every call is a dry run
that returns the orders it would send, so the daily job can show them.

How a ticket becomes orders
  entry        a limit order at the ticket's do-not-chase price (the most to pay),
               good for the day, with the stop-loss attached (an OTO order: the stop is live as soon as the entry fills).
               It is sent after the close, so it can fill at the open, as the backtest assumes, or later that day if the
               price comes back to the limit; the reconciliation measures the difference.
  targets plan two bracket orders, one per target (target 1's shares and the rest), each with the stop; after target 1
               fills, the daily job moves the other stop to breakeven, then trails it (the ticket's chandelier rule)
  trend plan   the entry and its stop only; at each review close where the replay says "sell at the next open" (the
               200-day SMA minus 8% or a technical score of 35 or lower), the job cancels the stop and sends a market
               order for the open (time in force "opg")
Orders carry client order ids "lh-<journal decision id>-<leg>", so every fill is tied back to its journal row.

Alpaca's free market data is IEX only and its paper fills are simulated; reconciliation numbers describe the paper
engine, not a real exchange queue. Limits: 200 API requests per minute per key (lighthouse.http paces at 3/s).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PAPER_URL = "https://paper-api.alpaca.markets"
KEY_ENV, SECRET_ENV = "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"


@dataclass
class OrderRequest:
    """One order as sent to Alpaca's POST /v2/orders (the payload) and why."""
    payload: Dict[str, object]
    reason: str
    decision_id: Optional[int] = None


@dataclass
class AlpacaPaper:
    key_id: Optional[str] = None
    secret: Optional[str] = None
    base_url: str = PAPER_URL
    sent: List[OrderRequest] = field(default_factory=list)       # what this session sent (or would send, dry run)

    def __post_init__(self) -> None:
        self.key_id = self.key_id or os.environ.get(KEY_ENV)
        self.secret = self.secret or os.environ.get(SECRET_ENV)
        if self.base_url.rstrip("/") != PAPER_URL:
            raise ValueError("Lighthouse trades on Alpaca's paper endpoint only")

    @property
    def dry_run(self) -> bool:
        return not (self.key_id and self.secret)

    # ------------------------------------------------------------------ REST
    def _call(self, method: str, path: str, params: Optional[dict] = None, body: Optional[dict] = None):
        from .http import fetch
        headers = {"APCA-API-KEY-ID": self.key_id or "", "APCA-API-SECRET-KEY": self.secret or "",
                   "Accept": "application/json"}
        r = fetch(self.base_url + path, params=params, headers=headers, method=method, json_body=body, retries=2)
        return r.json() if r.text else {}

    def account(self) -> dict:
        return {} if self.dry_run else self._call("GET", "/v2/account")

    def clock(self) -> dict:
        return {} if self.dry_run else self._call("GET", "/v2/clock")

    def positions(self) -> pd.DataFrame:
        if self.dry_run:
            return pd.DataFrame(columns=["symbol", "qty", "avg_entry_price", "side"])
        return pd.DataFrame(self._call("GET", "/v2/positions"))

    def orders(self, status: str = "all", after: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        if self.dry_run:
            return pd.DataFrame()
        params = {"status": status, "limit": limit, "nested": "true", "direction": "asc"}
        if after:
            params["after"] = after
        return pd.DataFrame(self._call("GET", "/v2/orders", params=params))

    def submit(self, req: OrderRequest) -> dict:
        self.sent.append(req)
        if self.dry_run:
            return {"dry_run": True, **req.payload}
        return self._call("POST", "/v2/orders", body=req.payload)

    def cancel(self, order_id: str) -> None:
        if not self.dry_run:
            self._call("DELETE", f"/v2/orders/{order_id}")

    def replace(self, order_id: str, **changes) -> dict:
        if self.dry_run:
            return {"dry_run": True, "id": order_id, **changes}
        return self._call("PATCH", f"/v2/orders/{order_id}", body={k: _num(v) for k, v in changes.items()})


def _num(v):
    return round(float(v), 2) if isinstance(v, (float, np.floating)) else v


# --------------------------------------------------------------------------- #
# Tickets -> orders
# --------------------------------------------------------------------------- #
def ticket_orders(plan, decision_id: Optional[int] = None) -> List[OrderRequest]:
    """The orders that put one actionable ticket on (see the module docstring). Empty for a ticket with no shares."""
    if not plan.is_actionable or plan.stop is None:
        return []
    base = {"symbol": plan.ticker, "side": "buy", "type": "limit", "time_in_force": "day",
            "limit_price": _num(plan.entry_limit)}
    tag = f"lh-{decision_id if decision_id is not None else 'x'}"
    if plan.is_trend_plan or not plan.targets:
        return [OrderRequest({**base, "qty": str(plan.shares), "order_class": "oto",
                              "stop_loss": {"stop_price": _num(plan.stop)}, "client_order_id": f"{tag}-entry"},
                             f"enter {plan.shares} at most ${plan.entry_limit:,.2f} with the stop at ${plan.stop:,.2f}",
                             decision_id)]
    out = []
    for k, t in enumerate(plan.targets, 1):
        if t.shares <= 0:
            continue
        out.append(OrderRequest({**base, "qty": str(t.shares), "order_class": "bracket",
                                 "take_profit": {"limit_price": _num(t.price)}, "stop_loss": {"stop_price": _num(plan.stop)},
                                 "client_order_id": f"{tag}-t{k}"},
                                f"{t.shares} shares to target {k} (${t.price:,.2f}), stop ${plan.stop:,.2f}", decision_id))
    return out


def review_exit_order(ticker: str, side: str, qty: int, decision_id: Optional[int], why: str) -> OrderRequest:
    """A market order for the next open (time in force 'opg'): the trend plan's review exit."""
    return OrderRequest({"symbol": ticker, "side": "sell", "type": "market",
                         "time_in_force": "opg", "qty": str(int(qty)),
                         "client_order_id": f"lh-{decision_id if decision_id is not None else 'x'}-review"},
                        f"review exit at the next open: {why}", decision_id)


def mirror(broker: AlpacaPaper, plans_by_decision: Dict[int, object], held: Optional[set] = None) -> List[dict]:
    """Send the orders for every actionable ticket whose symbol has no position or open order yet."""
    held = set(held or ())
    if not broker.dry_run:
        pos = broker.positions()
        held |= set(pos["symbol"]) if len(pos) else set()
        op = broker.orders(status="open")
        held |= set(op["symbol"]) if len(op) else set()
    out = []
    for did, plan in plans_by_decision.items():
        if plan.ticker in held:
            continue
        for req in ticket_orders(plan, did):
            out.append(broker.submit(req))
        held.add(plan.ticker)
    return out


# --------------------------------------------------------------------------- #
# Fills vs the backtest's assumptions
# --------------------------------------------------------------------------- #
def reconcile(orders: pd.DataFrame, opens: Dict[str, pd.Series]) -> pd.DataFrame:
    """Every filled Lighthouse order (client id 'lh-...') next to what the backtest assumes: an entry or review exit
    at the session's open, a stop at the stop price. Slippage in basis points, signed so that positive = worse for the
    investor."""
    if orders is None or orders.empty:
        return pd.DataFrame()
    rows = []
    flat = []
    for _, o in orders.iterrows():
        flat.append(o.to_dict())
        for leg in (o.get("legs") or []):
            flat.append(leg)
    for o in flat:
        cid = str(o.get("client_order_id") or "")
        if not o.get("filled_at") or not o.get("filled_avg_price"):
            continue
        filled_at = pd.Timestamp(o["filled_at"]).tz_convert("America/New_York")
        day = pd.Timestamp(filled_at.date())
        px = float(o["filled_avg_price"])
        sym, side, typ = o.get("symbol"), o.get("side"), o.get("type")
        if typ == "stop" or (o.get("stop_price") and typ in {"stop", "stop_limit"}):
            ref, ref_kind = float(o["stop_price"]), "stop price"
        else:
            op = opens.get(sym)
            ref = float(op.get(day, np.nan)) if op is not None else np.nan
            ref_kind = "session open (backtest fill)"
        sign = 1.0 if side == "buy" else -1.0
        slip = sign * (px / ref - 1.0) * 1e4 if np.isfinite(ref) and ref > 0 else np.nan
        did = cid.split("-")[1] if cid.startswith("lh-") and len(cid.split("-")) > 2 else None
        rows.append({"decision_id": did, "client_order_id": cid or None, "symbol": sym, "side": side, "type": typ,
                     "qty": float(o.get("filled_qty") or o.get("qty") or 0), "filled_at": filled_at, "fill": px,
                     "assumed": ref, "assumed_as": ref_kind, "slippage_bps": slip})
    return pd.DataFrame(rows)


__all__ = ["AlpacaPaper", "OrderRequest", "ticket_orders", "review_exit_order", "mirror", "reconcile", "PAPER_URL"]
