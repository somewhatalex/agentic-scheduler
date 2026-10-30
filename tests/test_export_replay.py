"""Contracts for adapting recorded calls to a program-aware replay harness."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_replay import export, request_payload
from scripts.prepare_data import prepare_trajectory, run


FIXTURE = json.loads((Path(__file__).parent / "fixtures/nemotron.json").read_text())


class RequestReconstruction(unittest.TestCase):
    def test_request_contains_only_the_prefix_and_restores_the_tool_call_envelope(self):
        row = FIXTURE["tool_calling"][0]
        prepared = prepare_trajectory(row, "tool_calling")
        request = request_payload(prepared.events, json.loads(prepared.trajectory["tools_json"]), 5, "test-model")
        self.assertEqual(len(request["messages"]), 5)
        self.assertEqual(request["messages"][2]["tool_calls"][0], {
            "id": "a", "type": "function", "function": {"name": "add", "arguments": '{"a":2,"b":2}'},
        })
        self.assertEqual(request["messages"][3]["tool_call_id"], "b")
        self.assertNotIn("Four and six.", json.dumps(request))
        self.assertNotIn("reasoning_content", request["messages"][2])

    def test_json_encoded_tool_parameters_are_restored_to_schema_objects(self):
        row = copy.deepcopy(FIXTURE["tool_calling"][0])
        row["tools"][0]["function"]["parameters"] = '{"type":"object"}'
        prepared = prepare_trajectory(row, "tool_calling")
        request = request_payload(prepared.events, row["tools"], 2, "test-model")
        self.assertEqual(request["tools"][0]["function"]["parameters"], {"type": "object"})


class ReplayExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        raw = self.root / "raw"
        raw.mkdir()
        for subset, rows in FIXTURE.items():
            (raw / f"{subset}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.processed = self.root / "processed"
        run(raw, self.processed)

    def test_only_complete_programs_are_exported_with_session_boundaries(self):
        output = self.root / "replay"
        report = export(self.processed, output, model="test-model")
        records = [json.loads(line) for line in (output / "replay.jsonl").read_text().splitlines()]
        self.assertEqual(report["programs"], 3)
        self.assertEqual(report["calls"], 5)
        self.assertEqual(report["skipped_incomplete_programs"], 1)
        conversation = [r for r in records if r["subset"] == "interactive_agent"]
        self.assertEqual(len({r["program_id"] for r in conversation}), 1)
        self.assertEqual([r["session_start"] for r in conversation], [True, False])
        self.assertEqual([r["session_end"] for r in conversation], [False, True])
        self.assertEqual(conversation[1]["parent_call_ids"], [conversation[0]["call_id"]])
        self.assertEqual(conversation[1]["interrupts_before"][0]["role"], "user")
        self.assertTrue(all(r["service_time_s"] is None for r in records))
        self.assertTrue((output / "_SUCCESS").is_file())

    def test_program_limit_keeps_each_selected_program_whole(self):
        output = self.root / "limited"
        report = export(self.processed, output, model="test-model", limit_programs=1)
        records = [json.loads(line) for line in (output / "replay.jsonl").read_text().splitlines()]
        self.assertEqual(report["programs"], 1)
        self.assertEqual(len(records), 2)
        self.assertTrue(records[-1]["session_end"])

    def test_export_requires_a_complete_processed_run_and_a_new_output_path(self):
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            export(self.processed, output, model="test-model")
        (self.processed / "_SUCCESS").unlink()
        with self.assertRaisesRegex(ValueError, "_SUCCESS"):
            export(self.processed, self.root / "unfinished", model="test-model")


if __name__ == "__main__":
    unittest.main()
