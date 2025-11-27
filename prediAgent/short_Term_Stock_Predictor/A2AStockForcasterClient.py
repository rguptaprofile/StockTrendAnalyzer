import requests
import json

AGENT_URL = "http://127.0.0.1:8002"
INVOKE_PATH = "/invoke"
AGENT_CARD_PATH = "/.well-known/agent-card.json"

def get_stock_prediction(ticker: str):
    """
    Call the root model 'model' and pass the tool_inputs dictionary for short_term_predict.
    """
    # Fallback HTTP /invoke expects: {"action": "<action_name>", "kwargs": {...}}
    # Use the tool name `short_term_predict` as the action and pass the ticker as kwargs.
    payload = {
        "action": "short_term_predict",
        "kwargs": {"ticker": ticker}
    }

    base = AGENT_URL.rstrip("/")

    # Try to fetch the agent-card to discover preferred transport and method names
    agent_card = None
    try:
        card_url = base + AGENT_CARD_PATH
        print(f"Fetching agent card from {card_url} ...")
        r = requests.get(card_url, timeout=3)
        if r.status_code == 200:
            try:
                agent_card = r.json()
                print("Agent card fetched; preferredTransport=", agent_card.get("preferredTransport"))
            except Exception:
                agent_card = None
        else:
            print(f"Agent card request returned {r.status_code}")
    except requests.exceptions.RequestException:
        agent_card = None

    # If agent prefers JSON-RPC, construct the JSON-RPC payload and POST to root '/'
    if agent_card and agent_card.get("preferredTransport", "").upper() == "JSONRPC":
        # Build a list of candidate method names to try. ADK/agent implementations
        # sometimes expose the root method as 'model', sometimes as the agent name,
        # or the tool name. We'll probe until one succeeds.
        skills = agent_card.get("skills") or []
        candidates = ["model", agent_card.get("name")]
        # add skill names and skill ids
        for s in skills:
            if isinstance(s, dict):
                candidates.append(s.get("name"))
                candidates.append(s.get("id"))
            else:
                candidates.append(s)

        # Normalize and filter
        seen = set()
        candidates = [c for c in (x for x in candidates if x) if not (c in seen or seen.add(c))]

        def call_jsonrpc(method_name: str):
            payload_rpc = {
                "jsonrpc": "2.0",
                "method": method_name,
                "params": {"tool_inputs": {"short_term_predict": {"ticker": ticker}}},
                "id": 1,
            }
            url = base + "/"
            print(f"Calling JSON-RPC method '{method_name}' on {url}")
            try:
                resp = requests.post(url, json=payload_rpc, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as e:
                print(f"  -> JSON-RPC request failed for {method_name}: {e}")
                return None

        result = None
        for method in candidates:
            if not method:
                continue
            candidate_result = call_jsonrpc(method)
            if candidate_result is None:
                continue
            # If we got a JSON-RPC response with an error indicating method not found,
            # try the next candidate. The JSON-RPC error for method not found is -32601.
            if isinstance(candidate_result, dict) and candidate_result.get("error"):
                err = candidate_result.get("error")
                code = err.get("code") if isinstance(err, dict) else None
                if code == -32601:
                    print(f"  -> method '{method}' not found; trying next candidate")
                    continue
            # Otherwise accept this result
            result = candidate_result
            break

        # If JSON-RPC probing failed to find a usable method, fall back to HTTP /invoke
        if result is None:
            print("JSON-RPC probing did not find a usable method; falling back to HTTP /invoke")
            payload = {"action": "short_term_predict", "kwargs": {"ticker": ticker}}
            invoke_url = base + INVOKE_PATH
            try:
                print(f"Calling HTTP /invoke at {invoke_url} ...")
                resp = requests.post(invoke_url, json=payload, timeout=10)
                if resp.status_code == 404:
                    print("/invoke returned 404; server may expose a different RPC method or path.")
                    result = None
                else:
                    resp.raise_for_status()
                    try:
                        result = resp.json()
                    except Exception:
                        result = resp.text
            except requests.exceptions.RequestException as e:
                print("HTTP /invoke call failed:", e)
                result = None

    else:
        # Fall back to HTTP /invoke calling style
        payload = {"action": "short_term_predict", "kwargs": {"ticker": ticker}}
        invoke_url = base + INVOKE_PATH
        try:
            print(f"Calling HTTP /invoke at {invoke_url} ...")
            resp = requests.post(invoke_url, json=payload, timeout=10)
            if resp.status_code == 404:
                print("/invoke returned 404; server may expose JSON-RPC at root or a different path.")
                result = None
            else:
                resp.raise_for_status()
                try:
                    result = resp.json()
                except Exception:
                    result = resp.text
        except requests.exceptions.RequestException as e:
            print("HTTP /invoke call failed:", e)
            result = None

    if result is None:
        print("No usable response from agent; check that the server is running and review the agent card or server logs.")
        return

    # Print the result in a friendly way
    if isinstance(result, dict) and ("result" in result or "output" in result or "ok" in result):
        if "result" in result:
            print(f"\n✅ Stock Prediction for {ticker}:\n{result['result']}")
        elif "output" in result:
            print(f"\n✅ Stock Prediction for {ticker}:\n{result['output']}")
        else:
            print(f"\nℹ️ Response for {ticker}:")
            print(json.dumps(result, indent=2))
    else:
        print(f"\n✅ Stock Prediction for {ticker}:\n{json.dumps(result, indent=2) if not isinstance(result, str) else result}")

if __name__ == "__main__":
    ticker_symbol = "TNA"
    print(f"Fetching short-term prediction for {ticker_symbol}...")
    get_stock_prediction(ticker_symbol)
