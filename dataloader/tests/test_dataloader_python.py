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


def test_python_custom_columns_multishard_dedup_and_score_filter() -> None:
    import shutil
    import luminamoon as lm

    tmp_dir = Path(tempfile.mkdtemp(prefix="luminamoon_cols_"))
    try:
        shard_0 = tmp_dir / "train-00000.parquet"
        shard_1 = tmp_dir / "train-00001.parquet"

        rows_0 = [
            {
                "no": 1,
                "en": "Hello world.",
                "id": "Halo dunia.",
                "score": 0.92,
            },
            {
                "no": 2,
                "en": "Good morning.",
                "id": "Selamat pagi.",
                "score": 0.75,
            },
        ]
        rows_1 = [
            {
                "no": 1,
                "en": "Hello world.",
                "id": "Halo dunia.",
                "score": 0.92,
            },
            {
                "no": 3,
                "en": "Thank you.",
                "id": "Terima kasih.",
                "score": 0.98,
            },
        ]
        shard_0.write_bytes(
            lm.serialize_table_to_bytes(rows_0, path_in_repo="data/train-00000.parquet")
        )
        shard_1.write_bytes(
            lm.serialize_table_to_bytes(rows_1, path_in_repo="data/train-00001.parquet")
        )

        with lm.stream(shard_0, batch_size=10) as default_stream:
            en_batches = list(default_stream)
        assert len(en_batches) == 1
        assert en_batches[0].records == [
            "Hello world.",
            "Good morning.",
        ]
        assert en_batches[0]["text"] == en_batches[0].records

        with lm.stream(shard_0, columns=["id"], batch_size=10) as id_stream:
            id_batches = list(id_stream)
        assert len(id_batches) == 1
        assert id_batches[0]["id"] == [
            "Halo dunia.",
            "Selamat pagi.",
        ]
        assert id_batches[0]["text"] == id_batches[0]["id"]

        with lm.stream(
            shard_0,
            columns="en",
            score_column="score",
            min_score=0.90,
            batch_size=10,
        ) as filtered_stream:
            filtered_batches = list(filtered_stream)
        assert len(filtered_batches) == 1
        assert filtered_batches[0].records == ["Hello world."]

        with lm.stream(
            [shard_0, shard_1],
            columns=[],
            score_column="score",
            min_score=0.90,
            batch_size=2,
            deduplicate=True,
        ) as mm_stream:
            mm_batches = list(mm_stream)
        assert len(mm_batches) == 2
        assert mm_batches[0].batch_id == 1
        assert mm_batches[0].is_eof is False
        assert mm_batches[0].column_names() == ["no", "en", "id", "score"]
        assert mm_batches[0]["en"] == ["Hello world."]
        assert mm_batches[1].batch_id == 2
        assert mm_batches[1].is_eof is True
        assert mm_batches[1]["en"] == ["Thank you."]
        assert abs(mm_batches[1]["score"][0] - 0.98) < 1e-6
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(Path(".luminamoon_cache"), ignore_errors=True)


def test_python_universal_formats_sources_and_typo_diagnostics() -> None:
    import shutil
    import sqlite3
    import tarfile
    import io
    import luminamoon as lm

    tmp_dir = Path(tempfile.mkdtemp(prefix="luminamoon_univ_"))
    try:
        rows = [
            {
                "prompt": "Explain quantum computing.",
                "score": 0.95,
                "tags": ["physics", "quantum"],
                "active": True,
            },
            {
                "prompt": "Write a Rust parser.",
                "score": 0.88,
                "tags": ["rust", "systems"],
                "active": False,
            },
            {
                "prompt": "Optimize matrix multiplication.",
                "score": 0.99,
                "tags": ["hpc", "simd"],
                "active": True,
            },
        ]

        # 1. Arrow IPC (.arrow), CSV (.csv), and compressed JSONL (.jsonl.gz, .jsonl.zst)
        arrow_file = tmp_dir / "dataset.arrow"
        csv_file = tmp_dir / "dataset.csv"
        gz_file = tmp_dir / "dataset.jsonl.gz"
        zst_file = tmp_dir / "dataset.jsonl.zst"

        arrow_file.write_bytes(
            lm.serialize_table_to_bytes(rows, path_in_repo="train/dataset.arrow")
        )
        csv_file.write_bytes(
            lm.serialize_table_to_bytes(rows, path_in_repo="train/dataset.csv")
        )
        gz_file.write_bytes(
            lm.serialize_table_to_bytes(rows, path_in_repo="train/dataset.jsonl.gz")
        )
        zst_file.write_bytes(
            lm.serialize_table_to_bytes(rows, path_in_repo="train/dataset.jsonl.zst")
        )

        # Multimodal read from Arrow IPC including native ListArray ("tags")
        with lm.stream(arrow_file, multimodal=True, batch_size=10) as arrow_mm:
            ab = list(arrow_mm)
        assert len(ab) == 1
        assert ab[0]["prompt"] == [r["prompt"] for r in rows]
        assert ab[0]["tags"] == [
            ["physics", "quantum"],
            ["rust", "systems"],
            ["hpc", "simd"],
        ]
        np_dict = ab[0].to_numpy()
        assert len(np_dict["score"]) == 3

        # Single-column Pre-RAM text stream from CSV and compressed .jsonl.gz / .jsonl.zst
        with lm.stream(csv_file, columns="prompt", batch_size=10) as csv_s:
            csv_b = list(csv_s)
        assert len(csv_b) == 1
        assert csv_b[0].records == [r["prompt"] for r in rows]

        with lm.stream(
            gz_file, json_extract_key="prompt", batch_size=10, deduplicate=False
        ) as gz_s:
            gz_b = list(gz_s)
        assert gz_b[0].records == [r["prompt"] for r in rows]

        with lm.stream(
            zst_file, json_extract_key="prompt", batch_size=10, deduplicate=False
        ) as zst_s:
            zst_b = list(zst_s)
        assert zst_b[0].records == [r["prompt"] for r in rows]

        # 2. Nested JSON path extraction (e.g. ChatML messages.0.content)
        chat_file = tmp_dir / "chat.jsonl"
        chat_file.write_text(
            '{"messages": [{"role": "user", "content": "First nested message"}]}\n'
            '{"messages": [{"role": "user", "content": "Second nested message"}]}\n',
            encoding="utf-8",
        )
        with lm.stream(
            chat_file, json_extract_key="messages.0.content", batch_size=10
        ) as chat_s:
            chat_b = list(chat_s)
        assert chat_b[0].records == ["First nested message", "Second nested message"]

        # 3. WebDataset (.tar) streaming
        tar_path = tmp_dir / "webdataset_000.tar"
        with tarfile.open(tar_path, "w") as tf:
            for idx, caption in enumerate(["cat photo", "dog photo"]):
                txt_bytes = caption.encode("utf-8")
                ti_txt = tarfile.TarInfo(name=f"sample_{idx:03d}.txt")
                ti_txt.size = len(txt_bytes)
                tf.addfile(ti_txt, io.BytesIO(txt_bytes))

                img_bytes = f"\x89PNG_{idx}".encode("latin1")
                ti_img = tarfile.TarInfo(name=f"sample_{idx:03d}.png")
                ti_img.size = len(img_bytes)
                tf.addfile(ti_img, io.BytesIO(img_bytes))

        with lm.stream(tar_path, batch_size=10) as tar_s:
            tar_b = list(tar_s)
        assert len(tar_b) == 1
        assert isinstance(tar_b[0], lm.MultimodalBatch)
        assert tar_b[0]["txt"] == ["cat photo", "dog photo"]
        assert tar_b[0]["png"] == [b"\x89PNG_0", b"\x89PNG_1"]

        # 4. In-memory dict/list and SQLite database URI streaming
        with lm.stream(rows, columns="prompt", batch_size=2, deduplicate=False) as mem_s:
            mem_b = list(mem_s)
        assert len(mem_b) == 2
        assert mem_b[0].records + mem_b[1].records == [r["prompt"] for r in rows]

        db_path = tmp_dir / "local_corpus.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE train_samples (id INTEGER, prompt TEXT, score REAL)")
        conn.executemany(
            "INSERT INTO train_samples VALUES (?, ?, ?)",
            [(1, "SQL row 1", 0.9), (2, "SQL row 2", 0.95)],
        )
        conn.commit()
        conn.close()

        sqlite_uri = f"sqlite:///{db_path.as_posix()}?table=train_samples"
        with lm.stream(sqlite_uri, columns="prompt", batch_size=10) as sql_s:
            sql_b = list(sql_s)
        assert sql_b[0].records == ["SQL row 1", "SQL row 2"]

        # 5. Cross-shard coalescing, DDP sharding, and interleave()
        with lm.stream(
            [arrow_file, csv_file],
            columns=["prompt", "score"],
            batch_size=2,
            deduplicate=False,
            coalesce_batches=True,
        ) as coal_s:
            coal_b = list(coal_s)
        assert [len(b) for b in coal_b] == [2, 2, 2]

        s_a = lm.from_hf_dataset(["a1", "a2"], batch_size=1)
        s_b = lm.from_hf_dataset(["b1", "b2"], batch_size=1)
        interleaved = list(
            lm.interleave(
                [s_a, s_b],
                probabilities=[0.5, 0.5],
                seed=7,
                stopping_strategy="all_exhausted",
            )
        )
        assert len(interleaved) == 4

        # 6. Developer-friendly error handling & typo diagnostics ("Did you mean ...?")
        assert lm.suggest_closest_matches("prmopt", ["prompt", "score", "image"]) == [
            "prompt"
        ]

        # 6a. Unknown kwarg typo in stream()
        kw_err = ""
        try:
            lm.stream(arrow_file, btach_size=16)
        except TypeError as e:
            kw_err = str(e)
        assert "Did you mean 'batch_size'?" in kw_err

        # 6b. Column typo in MultimodalParquetLoader
        col_err = ""
        try:
            MultimodalParquetLoader(arrow_file, columns=["prmopt"])
        except ValueError as e:
            col_err = str(e)
        assert "Did you mean 'prompt'?" in col_err

        # 6c. Column typo on MultimodalBatch[key] and DataBatch[key]
        mm_key_err = ""
        try:
            _ = ab[0]["prmopt"]
        except KeyError as e:
            mm_key_err = str(e)
        assert "Did you mean 'prompt'?" in mm_key_err

        db_key_err = ""
        try:
            _ = csv_b[0]["prmopt"]
        except KeyError as e:
            db_key_err = str(e)
        assert "Did you mean 'prompt'?" in db_key_err

        # 6d. URI scheme typo
        uri_err = ""
        try:
            lm.stream("hf://dataset/org/repo/train.parquet")
        except ValueError as e:
            uri_err = str(e)
        assert "hf://datasets/" in uri_err

        # 6e. File extension typo in serialize_table_to_bytes
        ext_err = ""
        try:
            lm.serialize_table_to_bytes(rows, path_in_repo="train/part.parqet")
        except ValueError as e:
            ext_err = str(e)
        assert "Did you mean" in ext_err

        # 6f. Column typo in from_hf_dataset
        hf_col_err = ""
        try:
            list(lm.from_hf_dataset(rows, column="prmopt"))
        except KeyError as e:
            hf_col_err = str(e)
        assert "Did you mean 'prompt'?" in hf_col_err
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(Path(".luminamoon_cache"), ignore_errors=True)


if __name__ == "__main__":
    test_python_dataloader_e2e_crash_recovery_and_coord_bytes()
    test_python_multimodal_dynamic_schema_and_hf_uploader()
    test_python_step8_dev_friendly_stream_and_push_to_hub()
    test_python_step9_official_hf_pipeline_and_tokenizer()
    test_python_custom_columns_multishard_dedup_and_score_filter()
    test_python_universal_formats_sources_and_typo_diagnostics()
    print("All Python DataStreamLoader and Multimodal integration tests passed.")

