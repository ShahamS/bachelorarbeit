import importlib.util
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "evaluation"
    / "run_distributed_token_replay_experiments.py"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "run_distributed_token_replay_experiments",
        RUNNER_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DistributedTokenReplayExperimentRunnerTest(TestCase):
    def test_location_key_is_required_for_distributed_runner(self):
        runner = _load_runner_module()

        with patch("sys.argv", ["runner"]):
            with self.assertRaises(SystemExit):
                runner.parse_args()

    def test_fieldnames_keep_union_across_distributed_and_central_rows(self):
        runner = _load_runner_module()
        rows = [
            {"method": "token_replay", "strategy": "distributed", "participants": 2},
            {"method": "token_replay", "strategy": "central", "route_calls": 0},
        ]

        fieldnames = runner._fieldnames(rows)

        self.assertEqual(
            fieldnames,
            ["method", "strategy", "participants", "route_calls"],
        )

    def test_main_loads_dataset_once_for_all_splits(self):
        runner = _load_runner_module()
        output_path = Path("/tmp/distributed-token-replay-test.csv")
        calls = []

        class FakeLog:
            def __init__(self, name):
                self.name = name

            def __len__(self):
                return 2

            def iter_traces(self):
                return iter(
                    [
                        ("case-1", [object()]),
                        ("case-2", [object()]),
                    ]
                )

        class FakeSplitter:
            def __init__(self, file_path=None, event_log=None, training_split=0.8, **kwargs):
                calls.append({"file_path": file_path, "event_log": event_log, "split": training_split})
                self.log = event_log or FakeLog("base")

            def split(self):
                return FakeLog("training"), FakeLog("test")

        class FakeSummary:
            def to_dict(self):
                return {"method": "token_replay", "strategy": "distributed"}

        with patch("sys.argv", [
            "runner",
            "--dataset",
            "dummy.xes",
            "--location-key",
            "org:group",
            "--training-splits",
            "0.2",
            "0.4",
            "--output",
            str(output_path),
        ]):
            with patch.object(runner, "EventLogSplitter", FakeSplitter):
                with patch.object(runner, "evaluate_distributed_token_replay", return_value=FakeSummary()):
                    runner.main()

        file_path_calls = [call for call in calls if call["file_path"] is not None]
        event_log_calls = [call for call in calls if call["event_log"] is not None]

        self.assertEqual(len(file_path_calls), 1)
        self.assertEqual(len(event_log_calls), 2)
