import json
import unittest
from unittest.mock import MagicMock

from adapters.mock_adapter import MockAdapter

class TestMockAdapterRules(unittest.TestCase):
    def setUp(self):
        self.vocab = MagicMock()
        self.vocab.action.tokens = ["PAD", "UNK", "peek", "fire", "throw_grenade", "trade_kill"]
        self.vocab.location.tokens = ["PAD", "UNK", "mid_window", "mid", "connector", "a_site", "b_site"]
        self.vocab.outcome.tokens = ["PAD", "UNK", "EnemySpoted", "EnemyDamaged", "Kill", "Death"]
        self.vocab.impact.tokens = ["PAD", "UNK", "MapInformation", "Pressure", "ZoneControl", "Initiation"]
        self.vocab.weapon.tokens = ["PAD", "UNK", "usp_s", "he_grenade", "m4a1_s", "awp"]

        self.norm = MagicMock()
        self.norm.timestamp = MagicMock(mode="minmax", min=0.0, max=10.0)
        self.norm.damage_sum = MagicMock(mode="clip_minmax", min=0.0, max=150.0)

        self.T = 4  # Window length
        self.k_multi = 3  # Max labels per event (K)
        self.adapter = MockAdapter(self.vocab, self.norm, self.T, self.k_multi)

        # Mock JSON input
        self.mock_json = '''
        {
            "match_id": "M0001",
            "map": "de_mirage",
            "rounds": [
                {
                    "round_number": 1,
                    "players": [
                        {
                            "player_id": "76561198000000000",
                            "name": "PlayerA",
                            "team": "CT",
                            "trajectory": [
                                {
                                    "timestamp": 0.4,
                                    "action": "peek",
                                    "location": "mid_window",
                                    "result": {
                                        "outcome": ["EnemySpoted"],
                                        "impact": ["MapInformation"],
                                        "weapon": ["usp_s"],
                                        "damage": [0]
                                    }
                                },
                                {
                                    "timestamp": 1.2,
                                    "action": "fire",
                                    "location": "mid_window",
                                    "result": {
                                        "outcome": ["EnemyDamaged"],
                                        "impact": ["Pressure"],
                                        "weapon": ["usp_s"],
                                        "damage": [34]
                                    }
                                },
                                {
                                    "timestamp": 2.8,
                                    "action": "throw_grenade",
                                    "location": "mid",
                                    "result": {
                                        "outcome": ["EnemyDamaged"],
                                        "impact": ["ZoneControl"],
                                        "weapon": ["he_grenade"],
                                        "damage": [52]
                                    }
                                },
                                {
                                    "timestamp": 4.1,
                                    "action": "trade_kill",
                                    "location": "connector",
                                    "result": {
                                        "outcome": ["Kill"],
                                        "impact": ["Initiation"],
                                        "weapon": ["m4a1_s"],
                                        "damage": [100]
                                    }
                                }
                            ]
                        }
                    ]
                }
            ],
            "metadata": {
                "source": "mock",
                "schema_version": "v0.1",
                "created_at": "2025-08-20T12:00:00Z"
            }
        }
        '''

    def test_valid_input(self):
        """Parse valid input: verify fields, windowing, and normalization."""
        data = json.loads(self.mock_json)
        samples = self.adapter.parse_obj(data)
        
        self.assertEqual(len(samples), 1, "Should produce one sample (single player, single window)")
        sample = samples[0]

        # Verify hard constraints: relative timestamps with normalization
        # timestamp: [0.4, 1.2, 2.8, 4.1] -> relative [0.0, 0.8, 2.4, 3.7] -> normalized by (10.0 - 0.0)
        expected_ts = [0.0, 0.08, 0.24, 0.37]
        for i, (actual, expected) in enumerate(zip(sample.timestamp_rel, expected_ts)):
            self.assertAlmostEqual(actual, expected, places=5, msg=f"timestamp_rel[{i}] mismatch")

        # Verify that location uses the standardized vocabulary
        self.assertEqual(sample.loc_idx, [2, 2, 3, 4], "location indices should be [mid_window=2, mid_window=2, mid=3, connector=4]")

        # Verify outcome/impact multi-hot encoding with K=3 cap
        expected_outcome = [
            [0, 0, 1, 0, 0, 0],  # EnemySpoted=2
            [0, 0, 0, 1, 0, 0],  # EnemyDamaged=3
            [0, 0, 0, 1, 0, 0],  # EnemyDamaged=3
            [0, 0, 0, 0, 1, 0]   # Kill=4
        ]
        self.assertEqual(sample.outcome_multi, expected_outcome, "outcome_multi mismatch")
        
        expected_impact = [
            [0, 0, 1, 0, 0, 0],  # MapInformation=2
            [0, 0, 0, 1, 0, 0],  # Pressure=3
            [0, 0, 0, 0, 1, 0],  # ZoneControl=4
            [0, 0, 0, 0, 0, 1]   # Initiation=5
        ]
        self.assertEqual(sample.impact_multi, expected_impact, "impact_multi mismatch")

        # Verify mask
        self.assertEqual(sample.mask, [1, 1, 1, 1], "mask should be all 1s (no padding)")

        # Verify weapon_top1_idx and damage aggregation
        self.assertEqual(sample.weapon_top1_idx, [2, 2, 3, 4], "weapon_top1_idx should be [usp_s=2, usp_s=2, he_grenade=3, m4a1_s=4]")
        self.assertEqual(sample.damage_sum, [0.0, 34.0, 52.0, 100.0], "damage_sum mismatch")
        self.assertEqual(sample.damage_mean, [0.0, 34.0, 52.0, 100.0], "damage_mean mismatch")
        self.assertEqual(sample.damage_max, [0.0, 34.0, 52.0, 100.0], "damage_max mismatch")
        self.assertEqual(sample.is_lethal, [0, 0, 0, 1], "is_lethal should be 1 only when damage=100")

        # Verify meta
        self.assertEqual(sample.meta["match_id"], "M0001", "meta.match_id mismatch")
        self.assertEqual(sample.meta["player_id"], "76561198000000000", "meta.player_id mismatch")
        self.assertEqual(sample.meta["round_number"], 1, "meta.round_number mismatch")
        self.assertAlmostEqual(sample.meta["window_start_s"], 0.4, places=5, msg="meta.window_start_s mismatch")

    def test_invalid_timestamp(self):
        """Invalid timestamps (negative or missing) are filtered."""
        invalid_json = '''
        {
            "match_id": "M0002",
            "rounds": [
                {
                    "round_number": 1,
                    "players": [
                        {
                            "player_id": "76561198000000001",
                            "team": "T",
                            "trajectory": [
                                {
                                    "timestamp": -0.1,
                                    "action": "peek",
                                    "location": "mid_window",
                                    "result": {"outcome": ["EnemySpoted"], "impact": [], "weapon": [], "damage": []}
                                },
                                {
                                    "action": "fire",
                                    "location": "mid",
                                    "result": {"outcome": [], "impact": [], "weapon": [], "damage": []}
                                },
                                {
                                    "timestamp": 1.0,
                                    "action": "fire",
                                    "location": "mid",
                                    "result": {"outcome": ["EnemyDamaged"], "impact": [], "weapon": ["usp_s"], "damage": [50]}
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        '''
        data = json.loads(invalid_json)
        samples = self.adapter.parse_obj(data)
        
        self.assertEqual(len(samples), 1, "Should produce one sample (only one valid event)")
        sample = samples[0]
        self.assertEqual(sample.timestamp_rel, [0.0, 0.0, 0.0, 0.0], "Invalid timestamps should be filtered; zeros padded")
        self.assertEqual(sample.mask, [1, 0, 0, 0], "mask should indicate a single valid event")
        self.assertEqual(sample.loc_idx, [3, 0, 0, 0], "location should be mid=3, rest padded")

    def test_unknown_location(self):
        """Unknown location should map to UNK."""
        json_with_unknown = '''
        {
            "match_id": "M0003",
            "rounds": [
                {
                    "round_number": 1,
                    "players": [
                        {
                            "player_id": "76561198000000002",
                            "team": "CT",
                            "trajectory": [
                                {
                                    "timestamp": 0.5,
                                    "action": "peek",
                                    "location": "invalid_loc",
                                    "result": {"outcome": [], "impact": [], "weapon": [], "damage": []}
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        '''
        data = json.loads(json_with_unknown)
        samples = self.adapter.parse_obj(data)
        
        self.assertEqual(len(samples), 1, "Should produce one sample")
        sample = samples[0]
        self.assertEqual(sample.loc_idx, [1, 0, 0, 0], "Unknown location should map to UNK=1")

    def test_outcome_impact_dedupe_and_cap(self):
        """Outcome/impact should be de-duplicated and capped at K=3."""
        json_with_duplicates = '''
        {
            "match_id": "M0004",
            "rounds": [
                {
                    "round_number": 1,
                    "players": [
                        {
                            "player_id": "76561198000000003",
                            "team": "CT",
                            "trajectory": [
                                {
                                    "timestamp": 0.5,
                                    "action": "fire",
                                    "location": "mid_window",
                                    "result": {
                                        "outcome": ["EnemyDamaged", "EnemyDamaged", "Kill", "Death", "EnemySpoted"],
                                        "impact": ["Pressure", "Pressure", "Initiation", "ZoneControl"],
                                        "weapon": ["usp_s"],
                                        "damage": [50]
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        '''
        data = json.loads(json_with_duplicates)
        samples = self.adapter.parse_obj(data)
        
        self.assertEqual(len(samples), 1, "Should produce one sample")
        sample = samples[0]
        expected_outcome = [
            [0, 0, 0, 1, 1, 1],  # EnemyDamaged=3, Kill=4, Death=5 (deduplicated, first 3)
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
        expected_impact = [
            [0, 0, 0, 1, 1, 1],  # Pressure=3, Initiation=5, ZoneControl=4 (deduplicated, first 3)
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
        self.assertEqual(sample.outcome_multi, expected_outcome, "outcome should be deduplicated and capped at K=3")
        self.assertEqual(sample.impact_multi, expected_impact, "impact should be deduplicated and capped at K=3")

    def test_window_split(self):
        """Events exceeding T=4 should be split into multiple windows."""
        json_long_traj = '''
        {
            "match_id": "M0005",
            "rounds": [
                {
                    "round_number": 1,
                    "players": [
                        {
                            "player_id": "76561198000000004",
                            "team": "CT",
                            "trajectory": [
                                {"timestamp": 0.1, "action": "peek", "location": "mid_window", "result": {"outcome": [], "impact": [], "weapon": [], "damage": []}},
                                {"timestamp": 0.2, "action": "fire", "location": "mid_window", "result": {"outcome": [], "impact": [], "weapon": [], "damage": []}},
                                {"timestamp": 0.3, "action": "throw_grenade", "location": "mid", "result": {"outcome": [], "impact": [], "weapon": [], "damage": []}},
                                {"timestamp": 0.4, "action": "trade_kill", "location": "connector", "result": {"outcome": [], "impact": [], "weapon": [], "damage": []}},
                                {"timestamp": 0.5, "action": "peek", "location": "a_site", "result": {"outcome": [], "impact": [], "weapon": [], "damage": []}}
                            ]
                        }
                    ]
                }
            ]
        }
        '''
        data = json.loads(json_long_traj)
        samples = self.adapter.parse_obj(data)
        
        self.assertEqual(len(samples), 2, "Should produce two windows (5 events -> [4, 1])")
        self.assertEqual(samples[0].loc_idx, [2, 2, 3, 4], "First window locations should be [mid_window=2, mid_window=2, mid=3, connector=4]")
        self.assertEqual(samples[1].loc_idx, [5, 0, 0, 0], "Second window locations should be [a_site=5, PAD, PAD, PAD]")

    def test_missing_fields(self):
        """Missing fields handling: player_id, team, etc."""
        json_missing = '''
        {
            "match_id": "M0006",
            "rounds": [
                {
                    "round_number": 1,
                    "players": [
                        {
                            "trajectory": [
                                {
                                    "timestamp": 0.5,
                                    "action": "peek",
                                    "location": "mid_window",
                                    "result": {"outcome": [], "impact": [], "weapon": [], "damage": []}
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        '''
        data = json.loads(json_missing)
        samples = self.adapter.parse_obj(data)
        
        self.assertEqual(len(samples), 1, "Should produce one sample")
        sample = samples[0]
        self.assertEqual(sample.meta["player_id"], "unknown", "Missing player_id should default to 'unknown'")
        self.assertEqual(sample.team_idx, 0, "Missing team should default to CT (team_idx=0)")

if __name__ == '__main__':
    unittest.main()
