import unittest

from scripts.bootstrap_database import seed_action


class SeedActionTests(unittest.TestCase):
    def test_empty_tables_are_seeded(self) -> None:
        self.assertEqual(seed_action(record_count=0, evidence_count=0), "seed")

    def test_populated_tables_are_preserved(self) -> None:
        self.assertEqual(seed_action(record_count=1451, evidence_count=451), "preserve")

    def test_partial_dataset_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "parcial"):
            seed_action(record_count=1, evidence_count=0)


if __name__ == "__main__":
    unittest.main()
