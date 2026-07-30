"""Reading files and the command line: where a run meets the real world.

A malformed line in a JSONL export is the most common failure in this pipeline,
and the only acceptable behaviour is to say which line. Silent skipping loses
records from a denominator nobody rechecks.
"""
import contextlib
import io as std_io
import json
import tempfile
import unittest
from pathlib import Path

from conv_eval.cli import main
from conv_eval.io import DataError, load_json, load_jsonl, write_json, write_jsonl


def run_cli(*argv: str) -> tuple[int, str]:
    out = std_io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = main(list(argv))
    return code, out.getvalue()


class Reading(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_jsonl_round_trip(self) -> None:
        records = [{"id": 1, "text": "русский текст"}, {"id": 2, "text": "ok"}]
        path = self.dir / "records.jsonl"
        write_jsonl(path, records)
        self.assertEqual(load_jsonl(path), records)

    def test_json_round_trip(self) -> None:
        payload = {"run_id": "SYN", "nested": {"value": 1.5}}
        path = self.dir / "run.json"
        write_json(path, payload)
        self.assertEqual(load_json(path), payload)

    def test_blank_lines_are_skipped(self) -> None:
        path = self.write("a.jsonl", '{"id": 1}\n\n{"id": 2}\n')
        self.assertEqual(len(load_jsonl(path)), 2)

    def test_broken_line_names_its_number(self) -> None:
        # "invalid JSON somewhere in a 40 MB file" is not an error message.
        path = self.write("b.jsonl", '{"id": 1}\n{oops}\n')
        with self.assertRaises(DataError) as ctx:
            load_jsonl(path)
        self.assertIn("2", str(ctx.exception))

    def test_non_object_line_is_rejected(self) -> None:
        path = self.write("c.jsonl", '["not", "an", "object"]\n')
        with self.assertRaises(DataError):
            load_jsonl(path)

    def test_missing_file_is_a_data_error(self) -> None:
        with self.assertRaises((DataError, OSError)):
            load_jsonl(self.dir / "nothing.jsonl")

    def test_broken_json_document_is_reported(self) -> None:
        path = self.write("d.json", "{not json}")
        with self.assertRaises(DataError):
            load_json(path)

    def test_written_files_are_utf8_without_escapes(self) -> None:
        path = self.dir / "utf8.jsonl"
        write_jsonl(path, [{"text": "проверка"}])
        self.assertIn("проверка", path.read_text(encoding="utf-8"))


class CommandLine(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def generate(self) -> Path:
        code, output = run_cli("generate", "--out", str(self.dir), "--deals-per-family", "4",
                               "--seed", "7")
        self.assertEqual(code, 0, output)
        return self.dir

    def test_generate_writes_the_bundle(self) -> None:
        directory = self.generate()
        for name in ("golden.jsonl", "predictions.candidate.jsonl", "run.candidate.json"):
            with self.subTest(file=name):
                self.assertTrue((directory / name).exists())

    def test_generate_is_deterministic(self) -> None:
        first = self.generate()
        content = (first / "golden.jsonl").read_text(encoding="utf-8")
        second = Path(self.enterContext(tempfile.TemporaryDirectory()))
        run_cli("generate", "--out", str(second), "--deals-per-family", "4", "--seed", "7")
        self.assertEqual(content, (second / "golden.jsonl").read_text(encoding="utf-8"))

    def test_evaluate_writes_a_report(self) -> None:
        directory = self.generate()
        report = self.dir / "report.json"
        code, output = run_cli(
            "evaluate", "--gold", str(directory / "golden.jsonl"),
            "--predictions", str(directory / "predictions.candidate.jsonl"),
            "--manifest", str(directory / "run.candidate.json"),
            "--json-out", str(report), "--bootstrap", "20", "--seed", "1")
        self.assertEqual(code, 0, output)
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertIn("families", payload)

    def test_evaluate_can_also_write_markdown(self) -> None:
        directory = self.generate()
        markdown = self.dir / "report.md"
        code, output = run_cli(
            "evaluate", "--gold", str(directory / "golden.jsonl"),
            "--predictions", str(directory / "predictions.candidate.jsonl"),
            "--manifest", str(directory / "run.candidate.json"),
            "--json-out", str(self.dir / "r.json"), "--markdown-out", str(markdown),
            "--bootstrap", "10", "--seed", "1")
        self.assertEqual(code, 0, output)
        self.assertTrue(markdown.read_text(encoding="utf-8").lstrip().startswith("#"))

    def test_gate_exits_two_when_a_rule_blocks(self) -> None:
        directory = self.generate()
        for name, predictions in (("baseline", "predictions.baseline.jsonl"),
                                  ("candidate", "predictions.candidate.jsonl")):
            run_cli("evaluate", "--gold", str(directory / "golden.jsonl"),
                    "--predictions", str(directory / predictions),
                    "--manifest", str(directory / f"run.{name}.json"),
                    "--json-out", str(self.dir / f"{name}.json"),
                    "--bootstrap", "10", "--seed", "1")

        impossible = self.dir / "policy.json"
        impossible.write_text(json.dumps({"rules": [
            {"id": "impossible", "metric": "overall.schema_validity",
             "operator": "gte", "value": 1.01}]}), encoding="utf-8")
        code, _ = run_cli("gate", "--baseline", str(self.dir / "baseline.json"),
                          "--candidate", str(self.dir / "candidate.json"),
                          "--config", str(impossible), "--out", str(self.dir / "gate.json"))
        self.assertEqual(code, 2)

    def test_gate_exits_zero_when_every_rule_passes(self) -> None:
        directory = self.generate()
        run_cli("evaluate", "--gold", str(directory / "golden.jsonl"),
                "--predictions", str(directory / "predictions.candidate.jsonl"),
                "--manifest", str(directory / "run.candidate.json"),
                "--json-out", str(self.dir / "candidate.json"),
                "--bootstrap", "10", "--seed", "1")
        easy = self.dir / "easy.json"
        easy.write_text(json.dumps({"rules": [
            {"id": "trivial", "metric": "overall.schema_validity",
             "operator": "gte", "value": 0.0}]}), encoding="utf-8")
        code, _ = run_cli("gate", "--baseline", str(self.dir / "candidate.json"),
                          "--candidate", str(self.dir / "candidate.json"),
                          "--config", str(easy), "--out", str(self.dir / "gate.json"))
        self.assertEqual(code, 0)

    def test_missing_input_file_fails_cleanly(self) -> None:
        code, output = run_cli("evaluate", "--gold", str(self.dir / "nope.jsonl"),
                               "--predictions", str(self.dir / "nope.jsonl"),
                               "--manifest", str(self.dir / "nope.json"),
                               "--json-out", str(self.dir / "out.json"))
        self.assertNotEqual(code, 0)
        self.assertNotIn("Traceback", output)


if __name__ == "__main__":
    unittest.main()
