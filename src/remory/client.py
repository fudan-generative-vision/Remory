"""Small synchronous HTTP client, independent of any agent framework."""
import httpx


class Client:
    def __init__(self, url: str, *, session_id: str, timeout: float = 600,
                 transport=None):
        self.http = httpx.Client(base_url=url.rstrip("/") + "/", timeout=timeout,
                                 headers={"X-Remory-Session": session_id}, transport=transport)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.http.close()

    def _post(self, route, body):
        response = self.http.post(route, json=body)
        response.raise_for_status()
        return response.json()

    def compact(self, *, prefix_ids, history_ids, summary_ids, previous=None, branches=()):
        body = dict(prefix_ids=list(prefix_ids), history_ids=list(history_ids),
                    summary_ids=list(summary_ids), previous=previous)
        if branches:
            body["branches"] = list(branches)
        return self._post("v1/compact", body)

    def generate(self, *, max_new_tokens=1024, handle=None, input_ids=None,
                 continuation_ids=(), sampling=None):
        if handle is not None and input_ids is not None:
            raise ValueError("use continuation_ids with a handle, not full input_ids")
        if handle is None and continuation_ids:
            raise ValueError("continuation_ids requires a memory handle")
        body = {"input_ids": list(continuation_ids) if handle else input_ids,
                "sampling_params": {"temperature": 0.0, **(sampling or {}), "max_new_tokens": max_new_tokens}}
        if handle:
            body["remory"] = {"handle": handle}
        result = self._post("generate", body)
        reason = result.get("meta_info", {}).get("finish_reason")
        if isinstance(reason, dict) and reason.get("type") in {"abort", "error"}:
            raise RuntimeError("SGLang aborted generation")
        return result

    def delete(self, handle):
        response = self.http.delete("v1/memories/" + handle)
        response.raise_for_status()
        return response.json()
