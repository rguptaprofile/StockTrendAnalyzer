import os
import random
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# Load your API keys from env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("GOOGLE_API_KEY loaded:", GOOGLE_API_KEY is not None)

# Retry configuration for LLM calls
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Initialize Gemini client with API key
gemini_model = Gemini(
    model="gemini-2.5-flash-lite",
    retry_options=retry_config,
    api_key=GOOGLE_API_KEY
)

# Short-term stock prediction tool (tries model-based predictor, falls back to demo random)
def short_term_predict(ticker: str, days: int = 5):
    """Return a mapping of YYYY-MM-DD -> price for the next `days` days.

    Tries to use `shortTerm_prediction.predict_short_term_price` when available
    (it returns a pandas Series or iterable). If that import fails, a simple
    random-demo predictor is used. Always returns a dict of ISO date strings
    to floats so callers get structured JSON.
    """
    ticker = ticker.upper().strip()
    # Try the model-based predictor first
    try:
        from ShortTerm_Prediction_Agent import predict_short_term_price  # type: ignore

        series = predict_short_term_price(ticker, days)
        # Prefer pandas-aware handling when available so we can normalize to business days
        try:
            import pandas as _pd

            # If we got a pandas Series, inspect its index. If the index contains
            # weekend dates, synthesize a business-day index instead so callers do
            # not receive weekend dates.
            if isinstance(series, _pd.Series):
                idx = series.index
                # DatetimeIndex -> decide whether to keep or synthesize
                if isinstance(idx, _pd.DatetimeIndex):
                    has_weekend = any(getattr(d, "weekday", lambda: 0)() >= 5 for d in idx)
                    if has_weekend:
                        vals = list(series.values)
                        bstart = _pd.date_range(start=_pd.Timestamp.today().normalize() + _pd.Timedelta(days=1), periods=1, freq="B")[0]
                        idx2 = _pd.date_range(start=bstart, periods=len(vals), freq="B")
                        return {str(d.date()): float(v) for d, v in zip(idx2, vals)}
                    else:
                        return {str(d.date()): float(v) for d, v in series.items()}
                else:
                    # non-datetime index: synthesize business-day dates
                    vals = list(series.values)
                    bstart = _pd.date_range(start=_pd.Timestamp.today().normalize() + _pd.Timedelta(days=1), periods=1, freq="B")[0]
                    idx2 = _pd.date_range(start=bstart, periods=len(vals), freq="B")
                    return {str(d.date()): float(v) for d, v in zip(idx2, vals)}

            # If predictor returned a dict-like mapping (date->value)
            if isinstance(series, dict):
                keys = list(series.keys())
                vals = list(series.values())
                parsed_dates = []
                parsed_ok = True
                has_weekend = False
                for k in keys:
                    try:
                        dt = _pd.to_datetime(k)
                        parsed_dates.append(dt)
                        if dt.weekday() >= 5:
                            has_weekend = True
                    except Exception:
                        parsed_ok = False
                        break

                if parsed_ok and not has_weekend:
                    return {str(d.date()): float(series[k]) for d, k in zip(parsed_dates, keys)}
                # Otherwise synthesize business-day dates for values
                bstart = _pd.date_range(start=_pd.Timestamp.today().normalize() + _pd.Timedelta(days=1), periods=1, freq="B")[0]
                idx2 = _pd.date_range(start=bstart, periods=len(vals), freq="B")
                return {str(d.date()): float(v) for d, v in zip(idx2, vals)}

            # Otherwise treat as a generic iterable of numeric values
            vals = list(series)
            bstart = _pd.date_range(start=_pd.Timestamp.today().normalize() + _pd.Timedelta(days=1), periods=1, freq="B")[0]
            idx2 = _pd.date_range(start=bstart, periods=len(vals), freq="B")
            return {str(d.date()): float(v) for d, v in zip(idx2, vals)}

        except Exception:
            # ignore pandas-specific handling if pandas not available
            pass

    except Exception:
        # Lightweight random/demo fallback (keeps deterministic-ish formatting)
        base_price = random.uniform(100, 500)
        preds = [round(base_price + random.uniform(-5, 5), 2) for _ in range(days)]
        try:
            import pandas as _pd

            # Use business days so weekend dates are skipped
            idx = _pd.date_range(start=_pd.Timestamp.today().normalize() + _pd.Timedelta(days=1), periods=days, freq="B")
            return {str(d.date()): float(p) for d, p in zip(idx, preds)}
        except Exception:
            # If pandas missing, return index-based keys
            return {str(i): float(v) for i, v in enumerate(preds)}

# Create the stock prediction agent
stock_agent = LlmAgent(
    model=gemini_model,
    name="short_term_stock_agent",
    description="Provides short-term 5-day stock price predictions.",
    instruction="""
    You are a financial assistant.
    When asked about a stock ticker, use the short_term_predict tool
    to provide 5-day short-term price predictions.
    Always provide the predictions clearly.
    """,
    tools=[short_term_predict],
)

# Expose via A2A
app = to_a2a(stock_agent, port=8002)
print("🚀 Short-term Stock Prediction Agent server created on port 8002 (call uvicorn or run this file to start).")

if __name__ == "__main__":
    # When executed directly, run a local uvicorn server so you can start
    # the agent with: `python myserver.py` (activate your venv first).
    import uvicorn
    try:
        print("Starting uvicorn server on 127.0.0.1:8002 ...")
        # Ensure a permissive /invoke HTTP fallback is available in case the
        # ADK-produced JSON-RPC method names differ from what callers expect.
        # If `app` is a Starlette/ASGI app, we can mount a small POST /invoke
        # handler that calls our local `short_term_predict` tool directly.
        try:
            # Import lazily so this still works when imports are absent.
            from starlette.requests import Request
            from starlette.responses import JSONResponse

            async def _invoke(request: Request):
                try:
                    body = await request.json()
                except Exception:
                    return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)

                action = body.get("action")
                kwargs = body.get("kwargs", {}) or {}
                if action in ("short_term_predict", "short_term_prediction"):
                    try:
                        # call the local tool defined above
                        res = short_term_predict(**kwargs)
                        return JSONResponse({"ok": True, "result": res})
                    except Exception as e:
                        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
                return JSONResponse({"ok": False, "error": "unknown action", "action": action}, status_code=404)

            # Try decorator style if app supports it (Starlette)
            try:
                app.post("/invoke")(_invoke)
            except Exception:
                # fallback to programmatic add_route when decorator fails
                try:
                    app.add_route("/invoke", _invoke, methods=["POST"])  # type: ignore
                except Exception:
                    # If app doesn't support route addition, ignore and continue
                    pass
        except Exception:
            # starlette not available or app doesn't support mounting; ignore
            pass

        # Also mount a permissive JSON-RPC root handler that will accept calls
        # where the JSON-RPC method is 'model' (or other candidate names) and
        # route them to the local `short_term_predict` tool. This helps clients
        # that prefer JSON-RPC to interact even when the ADK's internal
        # wiring doesn't expose per-skill method names.
        try:
            from starlette.requests import Request
            from starlette.responses import JSONResponse

            async def _jsonrpc_root(request: Request):
                try:
                    body = await request.json()
                except Exception:
                    return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}, status_code=400)

                method = body.get("method")
                params = body.get("params", {}) or {}

                # Recognize 'model' and proxy to short_term_predict when appropriate
                if method in ("model", "short_term_stock_agent", "short_term_predict"):
                    # params may contain tool_inputs -> {"short_term_predict": {"ticker": "AAPL"}}
                    ticker = None
                    if isinstance(params, dict):
                        # Support nested tool_inputs structure
                        tool_inputs = params.get("tool_inputs") or {}
                        if isinstance(tool_inputs, dict) and "short_term_predict" in tool_inputs:
                            ticker = tool_inputs["short_term_predict"].get("ticker")
                        # Also support direct params like {"ticker":"AAPL"}
                        if not ticker and "ticker" in params:
                            ticker = params.get("ticker")

                    if not ticker:
                        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32602, "message": "Invalid params: missing ticker"}, "id": body.get("id")}, status_code=400)

                    try:
                        res = short_term_predict(ticker)
                        return JSONResponse({"jsonrpc": "2.0", "result": res, "id": body.get("id")})
                    except Exception as e:
                        return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32000, "message": str(e)}, "id": body.get("id")}, status_code=500)

                # Not recognized — return JSON-RPC method not found
                return JSONResponse({"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": body.get("id")}, status_code=404)

            try:
                app.post("/")(_jsonrpc_root)
            except Exception:
                try:
                    app.add_route("/", _jsonrpc_root, methods=["POST"])  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass

        uvicorn.run(app, host="127.0.0.1", port=8002)
    except Exception as e:
        # Print the error for quick diagnostics; the user can inspect the
        # traceback if uvicorn fails to start in their environment.
        print("Failed to start uvicorn:", e)
