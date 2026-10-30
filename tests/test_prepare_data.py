"""Specifications for converting recorded conversations into scheduling examples."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from scripts.prepare_data import FEATURE_COLUMNS, prepare_trajectory, run


FIXTURE = json.loads((Path(__file__).parent / "fixtures/nemotron.json").read_text())


class ResponseEpisodeExamples(unittest.TestCase):
    def setUp(self):
        self.row = copy.deepcopy(FIXTURE["tool_calling"][0])

    def test_parallel_calls_count_individually_and_only_predict_after_all_results(self):
        prepared = prepare_trajectory(self.row, "tool_calling")
        self.assertEqual([r["prefix_end"] for r in prepared.examples], [2, 5])
        self.assertEqual([r["y_remaining_tool_calls"] for r in prepared.examples], [2, 0])
        self.assertEqual([r["y_remaining_assistant_turns"] for r in prepared.examples], [2, 1])
        tool_events = [r for r in prepared.events if r["role"] == "tool"]
        self.assertEqual([r["tool_name"] for r in tool_events], ["add", "add"])
        self.assertEqual([r["assistant_message_index"] for r in tool_events], [2, 2])

    def test_tokens_align_to_messages_and_include_assistant_reasoning(self):
        examples = prepare_trajectory(self.row, "tool_calling").examples
        self.assertEqual(examples[0]["prefix_recorded_tokens"], 17)
        self.assertEqual(examples[0]["prefix_assistant_recorded_tokens"], 0)
        self.assertEqual(examples[0]["y_remaining_recorded_assistant_tokens"], 11)
        self.assertEqual(examples[1]["prefix_recorded_tokens"], 22)
        self.assertEqual(examples[1]["y_remaining_recorded_assistant_tokens"], 8)

    def test_future_changes_cannot_change_features_at_the_same_prefix(self):
        original = prepare_trajectory(self.row, "tool_calling").examples[0]
        self.row["messages"][-1]["content"] = "A completely different future answer."
        self.row["metadata"]["turn_token_count"][-1]["content"] = 90
        self.row["metadata"]["num_tool_calls"] = 999
        self.row["metadata"]["ground_truth"] = "secret future label"
        changed = prepare_trajectory(self.row, "tool_calling").examples[0]
        self.assertEqual({k: original[k] for k in FEATURE_COLUMNS},
                         {k: changed[k] for k in FEATURE_COLUMNS})
        self.assertNotEqual(original["y_remaining_recorded_assistant_tokens"],
                            changed["y_remaining_recorded_assistant_tokens"])

    def test_each_user_request_has_its_own_target_and_parent_split(self):
        prepared = prepare_trajectory(FIXTURE["interactive_agent"][0], "interactive_agent")
        self.assertEqual([r["prefix_end"] for r in prepared.examples], [2, 4])
        self.assertEqual([r["y_remaining_assistant_turns"] for r in prepared.examples], [1, 1])
        self.assertEqual([r["step_index"] for r in prepared.examples], [0, 0])
        self.assertEqual(len({r["split"] for r in prepared.examples}), 1)
        self.assertEqual(len({r["trajectory_id"] for r in prepared.examples}), 1)

    def test_program_history_and_targets_continue_across_user_interrupts(self):
        prepared = prepare_trajectory(FIXTURE["interactive_agent"][0], "interactive_agent")
        first, second = prepared.calls
        self.assertEqual(first["program_id"], second["program_id"])
        self.assertEqual(first["parent_call_ids"], [])
        self.assertEqual(second["parent_call_ids"], [first["call_id"]])
        self.assertEqual(second["interrupt_message_indices"], [3])
        self.assertEqual([r["program_step_index"] for r in prepared.examples], [0, 1])
        self.assertEqual([r["y_program_remaining_assistant_turns"] for r in prepared.examples], [2, 1])
        self.assertEqual([r["y_program_remaining_recorded_assistant_tokens"] for r in prepared.examples], [6, 3])
        self.assertEqual(second["recorded_attained_assistant_tokens"], 3)
        self.assertIsNone(second["service_time_s"])

    def test_a_batch_of_tools_does_not_fabricate_parallel_llm_threads(self):
        prepared = prepare_trajectory(self.row, "tool_calling")
        first, second = prepared.calls
        self.assertEqual([r["thread_id"] for r in prepared.calls], [0, 0])
        self.assertEqual(second["parent_call_ids"], [first["call_id"]])
        self.assertEqual(second["interrupt_message_indices"], [3, 4])
        self.assertEqual(prepared.examples[0]["call_id"], first["call_id"])

    def test_unresolved_calls_are_preserved_without_completion_labels(self):
        prepared = prepare_trajectory(FIXTURE["tool_calling"][1], "tool_calling")
        self.assertEqual(prepared.examples, [])
        self.assertEqual(len(prepared.events), 2)
        self.assertEqual(prepared.exclusions["unresolved_tool_calls"], 1)

    def test_missing_or_misaligned_token_counts_do_not_become_zero_labels(self):
        for metadata in ({}, {"turn_token_count": [{"content": 7, "reasoning": 0}]}):
            with self.subTest(metadata=metadata):
                self.row["metadata"] = metadata
                examples = prepare_trajectory(self.row, "tool_calling").examples
                self.assertEqual(examples[0]["y_remaining_tool_calls"], 2)
                self.assertIsNone(examples[0]["y_remaining_recorded_assistant_tokens"])
                self.assertIsNone(examples[0]["prefix_recorded_tokens"])

    def test_repeated_request_with_a_different_completion_keeps_the_same_split_group(self):
        first = prepare_trajectory(self.row, "tool_calling").trajectory
        self.row["messages"][-1]["content"] = "Another completion."
        second = prepare_trajectory(self.row, "tool_calling").trajectory
        self.assertEqual(first["split_group_id"], second["split_group_id"])
        self.assertEqual(first["split"], second["split"])

    def test_json_encoded_messages_and_arguments_are_normalized(self):
        self.row["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"a": 2, "b": 2}
        self.row["messages"] = json.dumps([json.dumps(m) for m in self.row["messages"]])
        prepared = prepare_trajectory(self.row, "tool_calling")
        self.assertEqual(prepared.examples[0]["y_remaining_tool_calls"], 2)

    def test_a_new_assistant_turn_before_all_results_invalidates_the_episode(self):
        self.row["messages"].insert(4, {"role": "assistant", "content": "Premature answer."})
        prepared = prepare_trajectory(self.row, "tool_calling")
        self.assertEqual(prepared.examples, [])
        self.assertEqual(prepared.exclusions["assistant_before_tool_results"], 1)

    def test_a_malformed_call_list_is_rejected_instead_of_becoming_zero_work(self):
        self.row["messages"][2]["tool_calls"] = {}
        with self.assertRaisesRegex(ValueError, "invalid_tool_calls"):
            prepare_trajectory(self.row, "tool_calling")

    def test_an_unfinished_later_request_does_not_invalidate_an_earlier_response(self):
        row = copy.deepcopy(FIXTURE["interactive_agent"][0])
        row["messages"][-1] = copy.deepcopy(FIXTURE["tool_calling"][1]["messages"][-1])
        prepared = prepare_trajectory(row, "interactive_agent")
        self.assertEqual([r["episode_index"] for r in prepared.examples], [0])
        self.assertEqual(prepared.examples[0]["y_remaining_tool_calls"], 0)
        self.assertIsNone(prepared.examples[0]["y_program_remaining_tool_calls"])
        self.assertFalse(prepared.trajectory["program_complete"])
        self.assertTrue(prepared.calls[0]["episode_complete"])
        self.assertFalse(prepared.calls[-1]["episode_complete"])

    def test_resolved_call_ids_can_be_reused_and_repeated_calls_are_prefix_features(self):
        call = copy.deepcopy(self.row["messages"][2])
        call["tool_calls"] = call["tool_calls"][:1]
        result = copy.deepcopy(self.row["messages"][4])
        self.row["messages"][5:5] = [call, result]
        self.row.pop("metadata")
        prepared = prepare_trajectory(self.row, "tool_calling")
        self.assertEqual(len(prepared.examples), 3)
        self.assertEqual([r["last_call_repeated"] for r in prepared.examples], [False, False, True])


class StreamingIngestion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.output = self.root / "processed"

    def write_rows(self, rows):
        path = self.raw / "tool_calling.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_end_to_end_outputs_are_joinable_deduplicated_and_marked_complete(self):
        complete, incomplete = FIXTURE["tool_calling"]
        self.write_rows([complete, complete, incomplete])
        report = run(self.raw, self.output, batch_size=1)
        self.assertEqual(report["counts"]["duplicates"], 1)
        self.assertEqual(report["counts"]["trajectories"], 2)
        self.assertEqual(report["counts"]["examples"], 2)
        trajectories = pq.read_table(self.output / "trajectories.parquet").to_pylist()
        events = pq.read_table(self.output / "events.parquet").to_pylist()
        examples = pq.read_table(self.output / "examples.parquet").to_pylist()
        calls = pq.read_table(self.output / "calls.parquet").to_pylist()
        ids = {r["trajectory_id"] for r in trajectories}
        self.assertTrue(all(r["trajectory_id"] in ids for r in events + examples))
        self.assertTrue(all(r["program_id"] in ids for r in calls))
        self.assertEqual(len(calls), 3)
        self.assertTrue((self.output / "_SUCCESS").exists())
        self.assertEqual(json.loads((self.output / "audit.json").read_text())["status"], "complete")

    def test_bad_and_oversized_lines_are_audited_and_do_not_hide_later_rows(self):
        valid = json.dumps(FIXTURE["tool_calling"][0]).encode() + b"\n"
        path = self.raw / "tool_calling.jsonl"
        path.write_bytes(b'{bad json}\n' + b'x' * 5000 + b'\n' + valid)
        report = run(self.raw, self.output, max_line_bytes=4000)
        self.assertEqual(report["counts"]["rows_seen"], 3)
        self.assertEqual(report["counts"]["rejected"], 2)
        self.assertEqual(report["counts"]["examples"], 2)
        self.assertEqual(report["rejection_reasons"], {"invalid_json": 1, "line_too_large": 1})

    def test_limit_applies_to_input_records_in_each_file(self):
        self.write_rows(FIXTURE["tool_calling"])
        report = run(self.raw, self.output, limit=1)
        self.assertEqual(report["counts"]["rows_seen"], 1)
        self.assertFalse(report["files"][0]["scan_complete"])

    def test_existing_output_is_never_overwritten(self):
        self.write_rows([FIXTURE["tool_calling"][0]])
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing experiment")
        with self.assertRaises(FileExistsError):
            run(self.raw, self.output)
        self.assertEqual(marker.read_text(), "existing experiment")

    def test_strict_failure_leaves_no_success_marker(self):
        (self.raw / "tool_calling.jsonl").write_text("invalid\n")
        with self.assertRaises(ValueError):
            run(self.raw, self.output, strict=True)
        self.assertFalse((self.output / "_SUCCESS").exists())
        self.assertEqual(json.loads((self.output / "audit.json").read_text())["status"], "failed")

    def test_an_audit_without_labeled_examples_has_no_success_marker(self):
        self.write_rows([FIXTURE["tool_calling"][1]])
        report = run(self.raw, self.output)
        self.assertEqual(report["status"], "no_examples")
        self.assertFalse((self.output / "_SUCCESS").exists())


if __name__ == "__main__":
    unittest.main()
