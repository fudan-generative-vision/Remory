"""Admission checks for ordinary, request-local SGLang decoding slots."""


def validate_count(count):
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError("concurrency must be an integer from 1 through 32")
    return count


def assert_discrete_concurrency(info, count):
    validate_count(count)
    limit = info.get("max_running_requests")
    if type(limit) is not int or limit < count:
        raise RuntimeError(
            f"{count} discrete task slots require SGLang MAX_RUNNING_REQUESTS>={count}; "
            "no task claimed. Use fewer slots or configure the server on an idle endpoint."
        )


def summary_concurrency(args):
    """Opt native compact into discrete or attested residual task slots."""
    count = validate_count(getattr(args, "concurrency", 1))
    if count > 1 and (getattr(args, "mode", None), getattr(args, "compressor", None)) not in {
        ("native_compact", "none"), ("native_compact", "residual")
    }:
        raise ValueError("AutomationBench concurrency > 1 requires native_compact with compressor none or residual")
    return count
