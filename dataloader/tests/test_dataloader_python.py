from __future__ import annotations

from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT_PYTHON = Path(__file__).resolve().parents[2] / "python"
if str(ROOT_PYTHON) not in sys.path:
    sys.path.insert(0, str(ROOT_PYTHON))

from luminamoon import (
    DataBatch,
    DataStreamLoader,
    HfHubUploader,
    MultimodalBatch,
    MultimodalParquetLoader,
    StreamCoordinate,
    dataloader_cpu_isa,
)
from luminamoon.dataloader import dataloader_cpu_isa as subpkg_cpu_isa


def test_python_dataloader_e2e_crash_recovery_and_coord_bytes() -> None:
    isa = dataloader_cpu_isa()
    assert isa in ("AVX-512", "AVX2", "NEON", "SCALAR")
    assert subpkg_cpu_isa() == isa
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        shard_0 = tmp_path / "shard_0.jsonl"
        shard_1 = tmp_path / "shard_1.jsonl"
        meta_path = tmp_path / "checkpoint.meta"

        shard_0.write_text(
            '{"text": "row_0"}\n'
            '{"text": "row_0"}\n'
            '{"text": "row_1"}\n'
            '{"text": "row_2"}\n',
            encoding="utf-8",
        )
        shard_1.write_text(
            '{"text": "row_3"}\n'
            '{"text": "row_4"}\n'
            '{"text": "row_5"}\n',
            encoding="utf-8",
        )

        with DataStreamLoader(
            sources=[shard_0, shard_1],
            checkpoint_path=meta_path,
            batch_size=2,
            preram_batches=4,
            refill_trigger=1,
            json_extract_key="text",
        ) as loader:
            filled = loader.wait_for_refill()
            assert filled == 3

            b1 = next(loader)
            assert isinstance(b1, DataBatch)
            assert len(b1) == 2
            assert b1.batch_id == 1
            assert b1.records == ["row_0", "row_1"]

            committed_coord = loader.mark_in_gpu(b1.batch_id)
            assert isinstance(committed_coord, StreamCoordinate)
            assert committed_coord == loader.committed_coordinate()

            raw_bytes = committed_coord.to_bytes()
            assert len(raw_bytes) == 32
            restored_coord = StreamCoordinate.from_bytes(raw_bytes)
            assert restored_coord == committed_coord

            invalid_raised = False
            try:
                StreamCoordinate.from_bytes(b"short")
            except ValueError:
                invalid_raised = True
            assert invalid_raised

            b2 = next(loader)
            assert b2.batch_id == 2
            assert b2.records == ["row_2", "row_3"]

        with DataStreamLoader(
            sources=[shard_0, shard_1],
            checkpoint_path=meta_path,
            batch_size=2,
            preram_batches=4,
            refill_trigger=1,
            json_extract_key="text",
        ) as resumed:
            assert resumed.committed_batch_id() == 1
            batches = list(resumed)
            assert len(batches) == 2
            assert batches[0].batch_id == 2
            assert batches[0].records == ["row_2", "row_3"]
            assert batches[1].batch_id == 3
            assert batches[1].records == ["row_4", "row_5"]
            assert batches[1].is_eof
            resumed.commit_gpu_batch(batches[0].batch_id)
            resumed.commit_gpu_batch(batches[1].batch_id)


def test_python_multimodal_dynamic_schema_and_hf_uploader() -> None:
    import base64
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json
    import threading

    captured_files: dict[str, bytes] = {}

    class MockHfHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            for line in body.splitlines():
                obj = json.loads(line)
                if obj.get("key") == "file":
                    val = obj["value"]
                    captured_files[val["path"]] = base64.b64decode(val["content"])
            resp = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), MockHfHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        uploader = HfHubUploader(
            token="hf_py_test_token",
            username="silvermoon",
            org="luminamoon-org",
            endpoint=f"http://127.0.0.1:{port}",
        )
        columns = [
            ("prompt", ["hello", "hello", "world"]),
            ("image", [b"\x89PNG_A", b"\x89PNG_A", b"\x89PNG_B"]),
            ("meta", [{"k": 1}, {"k": 1}, {"k": 2}]),
            ("step", [10, 10, 20]),
            ("score", [0.5, 0.5, 1.25]),
            ("active", [True, True, False]),
        ]
        pq_uri = uploader.upload_table("dynamic-ds", "train/part.parquet", columns)
        assert pq_uri == "hf://datasets/luminamoon-org/dynamic-ds@main/train/part.parquet"
        jsonl_uri = uploader.upload_jsonl("dynamic-ds", "train/part.jsonl", dict(columns))
        assert jsonl_uri.endswith("/train/part.jsonl")

        with tempfile.TemporaryDirectory() as tmp_dir:
            pq_file = Path(tmp_dir) / "part.parquet"
            pq_file.write_bytes(captured_files["train/part.parquet"])

            loader = MultimodalParquetLoader(pq_file, batch_size=10, deduplicate=True)
            batch = next(loader)
            assert isinstance(batch, MultimodalBatch)
            assert len(batch) == 2
            assert batch.column_names() == [
                "prompt",
                "image",
                "meta",
                "step",
                "score",
                "active",
            ]
            d = batch.to_dict()
            assert list(d.keys()) == batch.column_names()
            assert d["prompt"] == ["hello", "world"]
            assert d["image"] == [b"\x89PNG_A", b"\x89PNG_B"]
            assert d["step"] == [10, 20]
            assert d["score"] == [0.5, 1.25]
            assert d["active"] == [True, False]
            assert isinstance(d["active"][0], bool)
            assert isinstance(d["step"][0], int)
    finally:
        server.shutdown()
        server.server_close()


def test_python_step8_dev_friendly_stream_and_push_to_hub() -> None:
    import base64
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json
    import os
    import shutil
    import threading
    import luminamoon as lm

    captured_files: dict[str, bytes] = {}
    repo_created_calls: list[dict[str, object]] = []

    class Step8MockHfHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            if self.path == "/api/repos/create":
                repo_created_calls.append(json.loads(body))
            else:
                for line in body.splitlines():
                    obj = json.loads(line)
                    if obj.get("key") == "file":
                        val = obj["value"]
                        captured_files[val["path"]] = base64.b64decode(val["content"])
            resp = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Step8MockHfHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        os.environ["HF_TOKEN"] = "hf_step8_env_token"
        os.environ["HF_ORG"] = "luminamoon-step8"
        os.environ["HF_ENDPOINT"] = f"http://127.0.0.1:{port}"

        secrets = lm.resolve_hf_secrets()
        assert secrets["token"] == "hf_step8_env_token"
        assert secrets["org"] == "luminamoon-step8"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shard = tmp_path / "auto_stream.jsonl"
            shard.write_text(
                '{"text": "item_0"}\n'
                '{"text": "item_1"}\n'
                '{"text": "item_2"}\n'
                '{"text": "item_3"}\n',
                encoding="utf-8",
            )

            auto_ckpt: Path | None = None
            stream_obj = lm.stream(
                shard,
                batch_size=2,
                download_workers=4,
                json_extract_key="text",
                auto_commit=True,
            )
            assert isinstance(stream_obj, lm.DataStreamLoader)
            with stream_obj as s:
                assert s.download_workers() == 4
                auto_ckpt = s.checkpoint_path
                assert auto_ckpt.name.startswith("stream_")
                b1 = next(s)
                assert isinstance(b1, lm.DataBatch)
                assert b1.records == ["item_0", "item_1"]
                assert b1[0] == "item_0"
                assert b1["text"] == ["item_0", "item_1"]
                assert list(b1) == ["item_0", "item_1"]
                coord_1 = b1.commit()
                assert isinstance(coord_1, StreamCoordinate)
                assert s.committed_batch_id() == 1

                b2 = next(s)
                assert isinstance(b2, lm.DataBatch)
                assert b2.records == ["item_2", "item_3"]
                batch_hub_uri = b2.push_to_hub("step8-repo")
                assert batch_hub_uri.endswith("/data/part-00002.parquet")

            stream_resumed = lm.stream(
                shard,
                batch_size=2,
                json_extract_key="text",
                auto_commit=True,
            )
            assert isinstance(stream_resumed, lm.DataStreamLoader)
            with stream_resumed as s_resumed:
                assert s_resumed.committed_batch_id() == 2
                assert list(s_resumed) == []

            if auto_ckpt is not None and auto_ckpt.parent.exists():
                shutil.rmtree(auto_ckpt.parent, ignore_errors=True)

            rows = [
                {
                    "old_text": "row_a",
                    "img": b"\x89PNG_ROW_A",
                    "meta": {"id": 1},
                    "step": 100,
                    "score": 0.75,
                    "flag": True,
                },
                {
                    "old_text": "row_b",
                    "img": b"\x89PNG_ROW_B",
                    "meta": {"id": 2},
                    "step": 200,
                    "score": 1.25,
                    "flag": False,
                },
            ]

            pq_uri = lm.push_to_hub(
                "step8-repo",
                rows,
                path_in_repo="train/rows.parquet",
                rename={"old_text": "prompt", "img": "image"},
            )
            assert pq_uri == "hf://datasets/luminamoon-step8/step8-repo@main/train/rows.parquet"
            assert len(repo_created_calls) >= 2

            jsonl_uri = lm.push_to_hub(
                "step8-repo",
                rows,
                path_in_repo="train/rows.jsonl",
                rename={"old_text": "prompt", "img": "image"},
            )
            assert jsonl_uri.endswith("/train/rows.jsonl")
            jsonl_lines = captured_files["train/rows.jsonl"].decode("utf-8").splitlines()
            assert len(jsonl_lines) == 2
            first_row_json = json.loads(jsonl_lines[0])
            assert first_row_json["prompt"] == "row_a"
            assert first_row_json["step"] == 100
            assert first_row_json["flag"] is True

            pq_local = tmp_path / "rows.parquet"
            pq_local.write_bytes(captured_files["train/rows.parquet"])

            mm_batches = list(
                lm.stream(
                    pq_local,
                    columns=["prompt", "image", "meta", "step", "score", "flag"],
                    batch_size=10,
                    download_workers=4,
                )
            )
            assert len(mm_batches) == 1
            mm_b = mm_batches[0]
            assert mm_b["prompt"] == ["row_a", "row_b"]
            assert mm_b[0]["prompt"] == "row_a"
            assert mm_b[-1]["step"] == 200
            assert len(list(mm_b)) == 2
            mm_dict = mm_b.to_dict()
            assert mm_dict["prompt"] == ["row_a", "row_b"]
            assert mm_dict["image"] == [b"\x89PNG_ROW_A", b"\x89PNG_ROW_B"]
            assert mm_dict["step"] == [100, 200]
            assert mm_dict["score"] == [0.75, 1.25]
            assert mm_dict["flag"] == [True, False]
    finally:
        for k in ("HF_TOKEN", "HF_ORG", "HF_ENDPOINT"):
            os.environ.pop(k, None)
        server.shutdown()
        server.server_close()


def test_python_step9_official_hf_pipeline_and_tokenizer() -> None:
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import json
    import threading
    import luminamoon as lm

    class Step9MockTreeHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:
            server_addr: Any = self.server.server_address
            host_port = server_addr[1]
            if self.path == "/api/datasets/luminamoon/mock-corpus/tree/main?recursive=true":
                tree_payload = json.dumps(
                    [
                        {"type": "file", "path": "README.md"},
                        {"type": "file", "path": "train/part-00001.parquet"},
                    ]
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header(
                    "Link",
                    f'<http://127.0.0.1:{host_port}/api/datasets/luminamoon/mock-corpus/tree/main?cursor=page2&recursive=true>; rel="next"',
                )
                self.send_header("Content-Length", str(len(tree_payload)))
                self.end_headers()
                self.wfile.write(tree_payload)
                return
            if self.path == "/api/datasets/luminamoon/mock-corpus/tree/main?cursor=page2&recursive=true":
                tree_payload = json.dumps(
                    [
                        {"type": "file", "path": "train/part-00000.parquet"},
                        {"type": "file", "path": "test/eval-00000.parquet"},
                    ]
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(tree_payload)))
                self.end_headers()
                self.wfile.write(tree_payload)
                return
            self.send_response(404)
            self.end_headers()

    tree_server = HTTPServer(("127.0.0.1", 0), Step9MockTreeHandler)
    tree_port = tree_server.server_port
    tree_thread = threading.Thread(target=tree_server.serve_forever, daemon=True)
    tree_thread.start()

    try:
        expanded = lm.expand_hf_sources(
            "hf://datasets/luminamoon/mock-corpus@main",
            split="train",
            endpoint=f"http://127.0.0.1:{tree_port}",
        )
        assert expanded == [
            "hf://datasets/luminamoon/mock-corpus@main/train/part-00000.parquet",
            "hf://datasets/luminamoon/mock-corpus@main/train/part-00001.parquet",
        ]
    finally:
        tree_server.shutdown()
        tree_server.server_close()

    mock_hf_dataset = [
        {"text": "  Alpha sample \r\n"},
        {"text": "Alpha sample"},
        {"text": "Beta sample\r\n"},
        {"text": "   "},
        {"text": "Gamma sample"},
    ]

    def dummy_tokenizer(texts: list[str], **kwargs: object) -> dict[str, object]:
        return {
            "input_ids": [[len(t), len(t.split())] for t in texts],
            "padding": kwargs.get("padding", False),
        }

    hook_seen_ids: list[int] = []

    def custom_hook(batch: lm.DataBatch) -> None:
        hook_seen_ids.append(batch.batch_id)

    batches = list(
        lm.from_hf_dataset(
            mock_hf_dataset,
            column="text",
            batch_size=2,
            deduplicate=True,
            tokenizer=dummy_tokenizer,
            postprocess_fn=custom_hook,
        )
    )
    assert len(batches) == 2
    assert hook_seen_ids == [1, 2]
    assert batches[0].records == ["Alpha sample", "Beta sample"]
    assert batches[0].tokens is not None
    assert batches[0].tokens["input_ids"] == [[12, 2], [11, 2]]
    assert batches[0]["tokens"]["padding"] is True
    assert batches[0].is_eof is False
    assert batches[1].records == ["Gamma sample"]
    assert batches[1].is_eof is True

    non_dedup_batches = list(
        lm.from_hf_dataset(
            mock_hf_dataset,
            column="text",
            batch_size=10,
            deduplicate=False,
        )
    )
    assert len(non_dedup_batches) == 1
    assert non_dedup_batches[0].records == [
        "Alpha sample",
        "Alpha sample",
        "Beta sample",
        "Gamma sample",
    ]

    pq_bytes = lm.serialize_table_to_bytes(
        [{"prompt": "hello", "score": 1.5, "ok": True}],
        path_in_repo="part.parquet",
    )
    assert pq_bytes.startswith(b"PAR1") and pq_bytes.endswith(b"PAR1")

    jsonl_bytes = lm.serialize_table_to_bytes(
        [{"prompt": "hello", "score": 1.5, "ok": True}],
        path_in_repo="part.jsonl",
    )
    parsed_jsonl = json.loads(jsonl_bytes.decode("utf-8").strip())
    assert parsed_jsonl == {"prompt": "hello", "score": 1.5, "ok": True}

    cleaned, hashes = lm.postprocess_and_dedup_records(
        ["  first\r\n", "first", "second\n"]
    )
    assert cleaned == ["first", "second"]
    assert len(hashes) == 2


if __name__ == "__main__":
    test_python_dataloader_e2e_crash_recovery_and_coord_bytes()
    test_python_multimodal_dynamic_schema_and_hf_uploader()
    test_python_step8_dev_friendly_stream_and_push_to_hub()
    test_python_step9_official_hf_pipeline_and_tokenizer()
    print("All Python DataStreamLoader and Multimodal integration tests passed.")

