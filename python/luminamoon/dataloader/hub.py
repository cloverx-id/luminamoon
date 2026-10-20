from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
from typing import Any
import urllib.parse
import urllib.request

from ._ffi import HfHubUploader, serialize_table_to_bytes
from .batch import DataBatch, MultimodalBatch
from .constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_HF_ENDPOINT,
    DEFAULT_HF_REVISION,
    HF_DISCOVERY_EXTENSIONS,
    HF_ENDPOINT_ENV_KEYS,
    HF_ORG_ENV_KEYS,
    HF_TOKEN_ENV_KEYS,
    HF_TREE_REQUEST_TIMEOUT_SECS,
    HF_USERNAME_ENV_KEYS,
    KNOWN_SHARD_EXTENSIONS,
    LOOPBACK_HTTP_PREFIXES,
    MAX_HF_TREE_PAGES,
    TABULAR_MODULE_PREFIXES,
    VALID_EXPAND_HF_SOURCES_KWARGS,
    VALID_PUSH_TO_HUB_KWARGS,
)
from .diagnostics import (
    _diagnose_uri_typo_py,
    _format_did_you_mean,
    _validate_known_kwargs,
)


def _lookup_secret_key(keys: Sequence[str]) -> str | None:
    for key in keys:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()

    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        for key in keys:
            try:
                val = userdata.get(key)
                if val and str(val).strip():
                    return str(val).strip()
            except BaseException:
                continue
    except BaseException:
        pass

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        client = UserSecretsClient()
        for key in keys:
            try:
                val = client.get_secret(key)
                if val and str(val).strip():
                    return str(val).strip()
            except BaseException:
                continue
    except BaseException:
        pass

    return None


def resolve_hf_secrets(
    token: str | None = None,
    username: str | None = None,
    org: str | None = None,
    endpoint: str | None = None,
) -> dict[str, str | None]:
    resolved_token = (
        token.strip()
        if token and token.strip()
        else _lookup_secret_key(HF_TOKEN_ENV_KEYS)
    )
    resolved_username = (
        username.strip()
        if username and username.strip()
        else _lookup_secret_key(HF_USERNAME_ENV_KEYS)
    )
    resolved_org = (
        org.strip() if org and org.strip() else _lookup_secret_key(HF_ORG_ENV_KEYS)
    )
    resolved_endpoint = (
        endpoint.strip()
        if endpoint and endpoint.strip()
        else _lookup_secret_key(HF_ENDPOINT_ENV_KEYS)
    )
    return {
        "token": resolved_token,
        "username": resolved_username,
        "org": resolved_org,
        "endpoint": resolved_endpoint,
    }


def _load_sqlite_uri_to_columns(uri: str) -> list[tuple[str, list[Any]]]:
    rest = uri.strip()
    if rest.lower().startswith("sqlite:///"):
        body = rest[len("sqlite:///") :]
    elif rest.lower().startswith("sqlite://"):
        body = rest[len("sqlite://") :]
    else:
        raise ValueError(f"Invalid SQLite URI: '{uri}'")

    query_str = ""
    if "#" in body:
        body, fragment = body.split("#", 1)
        query_str = urllib.parse.unquote(fragment).strip()
    if "?" in body:
        db_path_str, qs = body.split("?", 1)
    else:
        db_path_str, qs = body, ""

    params = urllib.parse.parse_qs(qs)
    table_name = params.get("table", [None])[0]
    sql_param = params.get("query", [None])[0] or params.get("sql", [None])[0]
    if sql_param:
        query_str = sql_param

    db_path = Path(db_path_str)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database file not found: '{db_path}'")

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        if not query_str:
            if not table_name:
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
                tables = [str(r[0]) for r in cur.fetchall()]
                if not tables:
                    raise ValueError(f"No user tables found in SQLite database '{db_path}'")
                table_name = tables[0]
            else:
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [str(r[0]) for r in cur.fetchall()]
                if table_name not in tables:
                    hint = _format_did_you_mean(table_name, tables)
                    raise ValueError(
                        f"Table '{table_name}' not found in SQLite database '{db_path}'.{hint} "
                        f"Available tables: {tables}"
                    )
            safe_table = table_name.replace('"', '""')
            query_str = f'SELECT * FROM "{safe_table}"'

        cur.execute(query_str)
        col_names = [desc[0] for desc in (cur.description or [])]
        rows = cur.fetchall()
        return [
            (col_names[idx], [row[idx] for row in rows])
            for idx in range(len(col_names))
        ]
    finally:
        conn.close()


def _materialize_columns_to_cached_arrow(
    ordered_cols: list[tuple[str, list[Any]]],
    prefix: str = "inmem",
) -> str:
    payload = bytes(
        serialize_table_to_bytes(
            ordered_cols,
            path_in_repo="cached.arrow",
            hf_image_struct=True,
        )
    )
    digest = hashlib.sha256(payload).hexdigest()[:16]
    cache_dir = Path(DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"{prefix}_{digest}.arrow"
    if not target_path.exists():
        target_path.write_bytes(payload)
    return str(target_path)


def _is_tabular_object(obj: Any) -> bool:
    if isinstance(obj, (str, Path, bytes, bytearray)):
        return False
    if isinstance(obj, Mapping):
        return True
    cls_mod = getattr(obj.__class__, "__module__", "")
    if cls_mod.startswith(TABULAR_MODULE_PREFIXES):
        return True
    if hasattr(obj, "to_pydict") or hasattr(obj, "to_pandas") or hasattr(obj, "to_arrow"):
        return True
    return False


def _extract_tabular_object_columns(obj: Any) -> list[tuple[str, list[Any]]]:
    if isinstance(obj, Mapping):
        return [
            (str(k), list(v) if isinstance(v, Iterable) and not isinstance(v, (str, bytes)) else [v])
            for k, v in obj.items()
        ]
    if hasattr(obj, "to_pydict"):
        d = obj.to_pydict()
        if isinstance(d, Mapping):
            return [(str(k), list(v)) for k, v in d.items()]
    if hasattr(obj, "to_dict"):
        try:
            d = obj.to_dict(orient="list")
        except TypeError:
            d = obj.to_dict()
        if isinstance(d, Mapping):
            first_val = next(iter(d.values()), None)
            if isinstance(first_val, Mapping):
                return [(str(k), list(v.values())) for k, v in d.items()]
            if isinstance(first_val, Sequence) and not isinstance(first_val, (str, bytes)):
                return [(str(k), list(v)) for k, v in d.items()]
    rows = list(obj)
    return _normalize_table_columns(rows)


def _normalize_table_columns(
    data: Any,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
) -> list[tuple[str, list[Any]]]:
    ordered: list[tuple[str, list[Any]]] = []

    if isinstance(data, DataBatch):
        ordered = [("text", list(data.records))]
    elif isinstance(data, MultimodalBatch):
        raw_dict = data.to_dict()
        ordered = [(col, list(raw_dict[col])) for col in data.column_names()]
    elif isinstance(data, Mapping):
        ordered = [(str(k), list(v)) for k, v in data.items()]
    elif _is_tabular_object(data):
        ordered = _extract_tabular_object_columns(data)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        if len(data) == 0:
            ordered = []
        elif isinstance(data[0], Mapping):
            col_names: list[str] = []
            col_set: set[str] = set()
            for row in data:
                if not isinstance(row, Mapping):
                    raise ValueError("All elements in row-oriented data list must be dicts")
                for k in row.keys():
                    ks = str(k)
                    if ks not in col_set:
                        col_set.add(ks)
                        col_names.append(ks)
            ordered = [
                (name, [row.get(name, None) for row in data]) for name in col_names
            ]
        else:
            for item in data:
                if not isinstance(item, Sequence) or len(item) != 2:
                    raise ValueError(
                        "Expected (column_name, values) pairs or list[dict[str, Any]]"
                    )
                ordered.append((str(item[0]), list(item[1])))
    else:
        raise TypeError(
            "Unsupported data type for push_to_hub; expected DataBatch, MultimodalBatch, dict, DataFrame, or list of dicts"
        )

    if columns is not None:
        wanted = [str(c) for c in columns]
        lookup = dict(ordered)
        avail_cols = list(lookup.keys())
        missing = [c for c in wanted if c not in lookup]
        if missing:
            first_missing = missing[0]
            hint = _format_did_you_mean(first_missing, avail_cols)
            raise ValueError(
                f"Requested columns {missing} not found in data.{hint} "
                f"Available columns: {avail_cols}"
            )
        ordered = [(c, lookup[c]) for c in wanted]

    if rename:
        rename_map = {str(k): str(v) for k, v in rename.items()}
        existing_cols = {col for col, _ in ordered}
        for src_col in rename_map:
            if src_col not in existing_cols:
                hint = _format_did_you_mean(src_col, list(existing_cols))
                if hint:
                    raise ValueError(
                        f"Rename source column '{src_col}' not found in data.{hint} "
                        f"Available columns: {sorted(existing_cols)}"
                    )
        ordered = [(rename_map.get(col, col), vals) for col, vals in ordered]

    return ordered


def _normalize_sources_list(sources: Any) -> list[str]:
    if isinstance(sources, (str, Path)):
        s_str = str(sources).strip()
        typo_msg = _diagnose_uri_typo_py(s_str)
        if typo_msg is not None:
            raise ValueError(typo_msg)
        if s_str.lower().startswith("sqlite://"):
            cols = _load_sqlite_uri_to_columns(s_str)
            return [_materialize_columns_to_cached_arrow(cols, prefix="sqlite")]
        return [s_str]

    if _is_tabular_object(sources):
        cols = _extract_tabular_object_columns(sources)
        return [_materialize_columns_to_cached_arrow(cols, prefix="table")]

    if isinstance(sources, Sequence):
        if len(sources) > 0 and isinstance(sources[0], Mapping):
            cols = _normalize_table_columns(sources)
            return [_materialize_columns_to_cached_arrow(cols, prefix="rows")]
        out: list[str] = []
        for s in sources:
            s_str = str(s).strip()
            typo_msg = _diagnose_uri_typo_py(s_str)
            if typo_msg is not None:
                raise ValueError(typo_msg)
            if s_str.lower().startswith("sqlite://"):
                cols = _load_sqlite_uri_to_columns(s_str)
                out.append(_materialize_columns_to_cached_arrow(cols, prefix="sqlite"))
            else:
                out.append(s_str)
        return out

    if isinstance(sources, Iterable):
        items = list(sources)
        if items and isinstance(items[0], Mapping):
            cols = _normalize_table_columns(items)
            return [_materialize_columns_to_cached_arrow(cols, prefix="iter")]
        return [str(x).strip() for x in items]

    raise TypeError(
        f"Unsupported sources type: {type(sources).__name__}. Expected path, URI, list of paths, "
        f"dict of columns, list of row dicts, DataFrame, Arrow Table, or HF Dataset."
    )


def _is_loopback_endpoint(endpoint: str | None) -> bool:
    if not endpoint:
        return False
    ep = endpoint.strip().lower()
    return ep.startswith(LOOPBACK_HTTP_PREFIXES)


def _parse_next_link_header(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        sections = part.split(";")
        if len(sections) >= 2 and any(
            s.strip() in ('rel="next"', "rel=next") for s in sections[1:]
        ):
            target = sections[0].strip()
            if target.startswith("<") and target.endswith(">"):
                return target[1:-1].strip()
    return None


def _list_hf_repo_files(
    org: str,
    repo: str,
    revision: str,
    token: str | None,
    endpoint: str | None,
) -> list[str]:
    base_ep = (endpoint or DEFAULT_HF_ENDPOINT).strip().rstrip("/")
    repo_id = f"{org}/{repo}"

    if not _is_loopback_endpoint(base_ep):
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            api = HfApi(token=token, endpoint=base_ep)
            files = api.list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
            )
            return [str(f) for f in files]
        except ImportError:
            pass
        except BaseException:
            pass

    next_url: str | None = (
        f"{base_ep}/api/datasets/{org}/{repo}/tree/{revision}?recursive=true"
    )
    out: list[str] = []
    pages_fetched = 0

    while next_url and pages_fetched < MAX_HF_TREE_PAGES:
        pages_fetched += 1
        headers: dict[str, str] = {"Accept": "application/json"}
        if token and (
            next_url.lower().startswith("https://") or _is_loopback_endpoint(next_url)
        ):
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(next_url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=HF_TREE_REQUEST_TIMEOUT_SECS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            link_hdr = resp.headers.get("Link") or resp.headers.get("link")

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, Mapping) and "path" in item:
                    if item.get("type", "file") != "directory":
                        out.append(str(item["path"]))
                elif isinstance(item, str):
                    out.append(item)

        next_url = _parse_next_link_header(link_hdr)

    return out


def _extract_path_tokens(paths: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for p in paths:
        for segment in p.replace("\\", "/").split("/"):
            stem = segment.split(".", 1)[0]
            for part in stem.replace("-", "_").split("_"):
                clean = part.strip().lower()
                if clean and not clean.isdigit() and clean not in seen:
                    seen.add(clean)
                    tokens.append(clean)
    return tokens


def expand_hf_sources(
    sources: Any,
    *,
    split: str | None = None,
    subset: str | None = None,
    token: str | None = None,
    endpoint: str | None = None,
    **kwargs: Any,
) -> list[str]:
    _validate_known_kwargs(
        "expand_hf_sources",
        kwargs,
        VALID_EXPAND_HF_SOURCES_KWARGS,
    )
    normalized = _normalize_sources_list(sources)
    secrets = resolve_hf_secrets(token=token, endpoint=endpoint)
    resolved_token = secrets["token"]
    resolved_endpoint = secrets["endpoint"]

    expanded: list[str] = []
    for raw_src in normalized:
        trimmed = raw_src.strip()
        if not trimmed.startswith("hf://datasets/"):
            expanded.append(trimmed)
            continue

        rest = trimmed[len("hf://datasets/") :].strip("/")
        parts = [p for p in rest.split("/") if p]
        if len(parts) < 2:
            raise ValueError(
                f"Invalid HuggingFace dataset URI '{trimmed}': expected 'hf://datasets/<org>/<repo>[/@revision][/path]'"
            )

        org = parts[0]
        repo_and_rev = parts[1]
        if "@" in repo_and_rev:
            repo, rev = repo_and_rev.split("@", 1)
        else:
            repo, rev = repo_and_rev, DEFAULT_HF_REVISION

        subpath = "/".join(parts[2:])
        clean_sub = subpath.split("?", 1)[0].lower()
        if clean_sub.endswith(KNOWN_SHARD_EXTENSIONS):
            expanded.append(trimmed)
            continue

        all_files = _list_hf_repo_files(
            org=org,
            repo=repo,
            revision=rev,
            token=resolved_token,
            endpoint=resolved_endpoint,
        )

        dataset_files = [
            p for p in all_files if p.lower().endswith(HF_DISCOVERY_EXTENSIONS)
        ]

        has_literal_default = any(
            "default" in p.lower().split("/") for p in dataset_files
        )
        candidates: list[str] = []
        for path in dataset_files:
            low_path = path.lower()
            if subpath and not (
                path == subpath or path.startswith(subpath.rstrip("/") + "/")
            ):
                continue
            if subset:
                low_sub = subset.lower()
                if low_sub == "default" and not has_literal_default:
                    if not (
                        low_path.startswith("data/") or "/" not in low_path
                    ):
                        continue
                elif low_sub not in low_path:
                    continue
            if split and split.lower() not in low_path:
                continue
            candidates.append(path)

        parquet_candidates = [
            p for p in candidates if p.lower().endswith((".parquet", ".pq"))
        ]
        arrow_candidates = [
            p for p in candidates if p.lower().endswith((".arrow", ".feather", ".ipc"))
        ]
        if parquet_candidates:
            selected = sorted(parquet_candidates)
        elif arrow_candidates:
            selected = sorted(arrow_candidates)
        else:
            selected = sorted(candidates)

        if not selected:
            avail_tokens = _extract_path_tokens(dataset_files)
            hints: list[str] = []
            if split:
                s_hint = _format_did_you_mean(split, avail_tokens)
                if s_hint:
                    hints.append(f"For split='{split}':{s_hint}")
            if subset:
                sub_hint = _format_did_you_mean(subset, avail_tokens)
                if sub_hint:
                    hints.append(f"For subset='{subset}':{sub_hint}")
            hint_suffix = (" " + " ".join(hints)) if hints else ""
            sample_files = ", ".join(dataset_files[:8])
            raise ValueError(
                f"No matching dataset shards found in {trimmed} "
                f"(split={split!r}, subset={subset!r}).{hint_suffix} "
                f"Available dataset files: [{sample_files}]"
            )
        for rel_path in selected:
            expanded.append(f"hf://datasets/{org}/{repo}@{rev}/{rel_path}")

    return expanded


def push_to_hub(
    repo: str,
    data: Any,
    path_in_repo: str = "data/part-00000.parquet",
    *,
    columns: Sequence[str] | None = None,
    rename: Mapping[str, str] | None = None,
    private: bool = True,
    revision: str = DEFAULT_HF_REVISION,
    commit_message: str = "Upload dataset shard via LuminaMoon",
    hf_image_struct: bool = True,
    token: str | None = None,
    username: str | None = None,
    org: str | None = None,
    endpoint: str | None = None,
    **kwargs: Any,
) -> str:
    _validate_known_kwargs("push_to_hub", kwargs, VALID_PUSH_TO_HUB_KWARGS)
    secrets = resolve_hf_secrets(
        token=token,
        username=username,
        org=org,
        endpoint=endpoint,
    )
    ordered_cols = _normalize_table_columns(data, columns=columns, rename=rename)
    payload_bytes = bytes(
        serialize_table_to_bytes(
            ordered_cols,
            path_in_repo=path_in_repo,
            hf_image_struct=hf_image_struct,
        )
    )

    resolved_endpoint = secrets["endpoint"]
    if not _is_loopback_endpoint(resolved_endpoint):
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]

            trimmed_repo = repo.strip().strip("/")
            if "/" in trimmed_repo:
                full_repo_id = trimmed_repo
            else:
                owner = secrets["org"] or secrets["username"]
                if not owner:
                    raise ValueError(
                        "Cannot resolve full repo_id without org/username or 'owner/repo' format"
                    )
                full_repo_id = f"{owner}/{trimmed_repo}"

            clean_path = path_in_repo.strip().lstrip("/")
            api = HfApi(token=secrets["token"], endpoint=resolved_endpoint)
            api.create_repo(
                repo_id=full_repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
            api.upload_file(
                path_or_fileobj=io.BytesIO(payload_bytes),
                path_in_repo=clean_path,
                repo_id=full_repo_id,
                repo_type="dataset",
                revision=revision,
                commit_message=commit_message,
            )
            return f"hf://datasets/{full_repo_id}@{revision}/{clean_path}"
        except ImportError:
            pass

    uploader = HfHubUploader(
        token=secrets["token"],
        username=secrets["username"],
        org=secrets["org"],
        endpoint=resolved_endpoint,
    )
    uploader.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    return uploader.upload_bytes(
        repo,
        path_in_repo,
        payload_bytes,
        revision=revision,
        commit_message=commit_message,
    )


__all__ = [
    "_extract_tabular_object_columns",
    "_is_loopback_endpoint",
    "_is_tabular_object",
    "_load_sqlite_uri_to_columns",
    "_materialize_columns_to_cached_arrow",
    "_normalize_sources_list",
    "_normalize_table_columns",
    "expand_hf_sources",
    "push_to_hub",
    "resolve_hf_secrets",
]
