import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from simulate import plas_thresholds, simulate  # noqa: E402


def program(arrival, *calls):
    return {"arrival": arrival, "calls": list(calls)}


class Simulate(unittest.TestCase):
    def test_single_program_takes_its_service_plus_waits(self):
        programs = [program(3.0, (0.0, 2.0), (5.0, 1.0))]
        for policy in ("fcfs", "plas", "oracle"):
            self.assertAlmostEqual(simulate(programs, policy, 1, [1.0])[0], 11.0)

    def test_fcfs_runs_calls_in_ready_order(self):
        programs = [program(0.0, (0.0, 10.0)), program(1.0, (0.0, 1.0))]
        self.assertEqual(simulate(programs, "fcfs", 1), [10.0, 11.0])

    def test_plas_preempts_a_program_that_left_the_top_queue(self):
        programs = [program(0.0, (0.0, 10.0)), program(1.0, (0.0, 0.4))]
        # The long program crosses the 0.5 s threshold before the short one arrives.
        finish = simulate(programs, "plas", 1, [0.5])
        self.assertAlmostEqual(finish[1], 1.4)
        self.assertAlmostEqual(finish[0], 10.4)

    def test_plas_is_fcfs_within_a_queue_level(self):
        # The newcomer also crosses 0.5 s, then waits behind the earlier program.
        programs = [program(0.0, (0.0, 10.0)), program(1.0, (0.0, 1.0))]
        finish = simulate(programs, "plas", 1, [0.5])
        self.assertAlmostEqual(finish[0], 10.5)
        self.assertAlmostEqual(finish[1], 11.0)

    def test_plas_counts_service_across_calls_of_a_program(self):
        # A's second call is short, but A has already used 3 s, so newcomer B goes first.
        programs = [program(0.0, (0.0, 3.0), (2.0, 1.0)), program(5.0, (0.0, 2.0))]
        finish = simulate(programs, "plas", 1, [2.5])
        self.assertAlmostEqual(finish[1], 7.0)
        self.assertAlmostEqual(finish[0], 8.0)

    def test_external_wait_releases_the_slot(self):
        programs = [program(0.0, (0.0, 1.0), (10.0, 1.0)), program(0.0, (0.0, 4.0))]
        finish = simulate(programs, "fcfs", 1)
        self.assertAlmostEqual(finish[1], 5.0)
        self.assertAlmostEqual(finish[0], 12.0)

    def test_oracle_runs_the_shortest_remaining_program_first(self):
        programs = [program(0.0, (0.0, 5.0)), program(0.0, (0.0, 1.0), (0.0, 1.0))]
        self.assertEqual(simulate(programs, "oracle", 1), [7.0, 2.0])

    def test_slots_run_calls_concurrently(self):
        programs = [program(0.0, (0.0, 4.0)), program(0.0, (0.0, 4.0))]
        self.assertEqual(simulate(programs, "fcfs", 2), [4.0, 4.0])

    def test_thresholds_are_geometric(self):
        self.assertEqual(plas_thresholds(2.0, 2.0, 4), [2.0, 4.0, 8.0])


if __name__ == "__main__":
    unittest.main()
