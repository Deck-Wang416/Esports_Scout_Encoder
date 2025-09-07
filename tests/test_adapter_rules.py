import json
import unittest
from unittest.mock import MagicMock
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

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

        self.T = 4  # 窗口大小
        self.k_multi = 3  # 多标签最大数量
        self.adapter = MockAdapter(self.vocab, self.norm, self.T, self.k_multi)

        # Mock JSON 数据
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
        """测试正常输入的解析：验证字段正确性、窗口切分、归一化等"""
        data = json.loads(self.mock_json)
        samples = self.adapter.parse_obj(data)
        
        self.assertEqual(len(samples), 1, "应生成一个样本（单个玩家，单个窗口）")
        sample = samples[0]

        # 验证硬约束：timestamp 相对时间且归一化
        # timestamp: [0.4, 1.2, 2.8, 4.1] -> 相对 [0.0, 0.8, 2.4, 3.7] -> 归一化 / (10.0 - 0.0)
        expected_ts = [0.0, 0.08, 0.24, 0.37]
        for i, (actual, expected) in enumerate(zip(sample.timestamp_rel, expected_ts)):
            self.assertAlmostEqual(actual, expected, places=5, msg=f"timestamp_rel[{i}] 不匹配")

        # 验证 location 使用标准化词表
        self.assertEqual(sample.loc_idx, [2, 2, 3, 4], "location 索引应为 [mid_window=2, mid_window=2, mid=3, connector=4]")

        # 验证 outcome/impact 多热编码，最大 K=3
        expected_outcome = [
            [0, 0, 1, 0, 0, 0],  # EnemySpoted=2
            [0, 0, 0, 1, 0, 0],  # EnemyDamaged=3
            [0, 0, 0, 1, 0, 0],  # EnemyDamaged=3
            [0, 0, 0, 0, 1, 0]   # Kill=4
        ]
        self.assertEqual(sample.outcome_multi, expected_outcome, "outcome_multi 不匹配")
        
        expected_impact = [
            [0, 0, 1, 0, 0, 0],  # MapInformation=2
            [0, 0, 0, 1, 0, 0],  # Pressure=3
            [0, 0, 0, 0, 1, 0],  # ZoneControl=4
            [0, 0, 0, 0, 0, 1]   # Initiation=5
        ]
        self.assertEqual(sample.impact_multi, expected_impact, "impact_multi 不匹配")

        # 验证 mask
        self.assertEqual(sample.mask, [1, 1, 1, 1], "mask 应全为 1，无 padding")

        # 验证 weapon_top1_idx 和 damage 聚合
        self.assertEqual(sample.weapon_top1_idx, [2, 2, 3, 4], "weapon_top1_idx 应为 [usp_s=2, usp_s=2, he_grenade=3, m4a1_s=4]")
        self.assertEqual(sample.damage_sum, [0.0, 34.0, 52.0, 100.0], "damage_sum 不匹配")
        self.assertEqual(sample.damage_mean, [0.0, 34.0, 52.0, 100.0], "damage_mean 不匹配")
        self.assertEqual(sample.damage_max, [0.0, 34.0, 52.0, 100.0], "damage_max 不匹配")
        self.assertEqual(sample.is_lethal, [0, 0, 0, 1], "is_lethal 应仅在 damage=100 时为 1")

        # 验证 meta
        self.assertEqual(sample.meta["match_id"], "M0001", "meta.match_id 不匹配")
        self.assertEqual(sample.meta["player_id"], "76561198000000000", "meta.player_id 不匹配")
        self.assertEqual(sample.meta["round_number"], 1, "meta.round_number 不匹配")
        self.assertAlmostEqual(sample.meta["window_start_s"], 0.4, places=5, msg="meta.window_start_s 不匹配")

    def test_invalid_timestamp(self):
        """测试非法 timestamp（负值或缺失）被过滤"""
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
        
        self.assertEqual(len(samples), 1, "应生成一个样本（仅一个有效事件）")
        sample = samples[0]
        self.assertEqual(sample.timestamp_rel, [0.0, 0.0, 0.0, 0.0], "非法 timestamp 应被过滤，填充 0")
        self.assertEqual(sample.mask, [1, 0, 0, 0], "mask 应仅有一个有效事件")
        self.assertEqual(sample.loc_idx, [3, 0, 0, 0], "location 应为 mid=3，填充 PAD")

    def test_unknown_location(self):
        """测试未知 location 被映射为 UNK"""
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
        
        self.assertEqual(len(samples), 1, "应生成一个样本")
        sample = samples[0]
        self.assertEqual(sample.loc_idx, [1, 0, 0, 0], "未知 location 应映射为 UNK=1")

    def test_outcome_impact_dedupe_and_cap(self):
        """测试 outcome/impact 去重和 K=3 限制"""
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
        
        self.assertEqual(len(samples), 1, "应生成一个样本")
        sample = samples[0]
        expected_outcome = [
            [0, 0, 0, 1, 1, 1],  # EnemyDamaged=3, Kill=4, Death=5（去重后前 3 个）
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
        expected_impact = [
            [0, 0, 0, 1, 1, 1],  # Pressure=3, Initiation=5, ZoneControl=4（去重后前 3 个）
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]
        self.assertEqual(sample.outcome_multi, expected_outcome, "outcome 应去重并限制 K=3")
        self.assertEqual(sample.impact_multi, expected_impact, "impact 应去重并限制 K=3")

    def test_window_split(self):
        """测试超过 T=4 的事件被切分为多个窗口"""
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
        
        self.assertEqual(len(samples), 2, "应生成两个窗口（5 事件切分为 [4, 1]）")
        self.assertEqual(samples[0].loc_idx, [2, 2, 3, 4], "第一个窗口 location 应为 [mid_window=2, mid_window=2, mid=3, connector=4]")
        self.assertEqual(samples[1].loc_idx, [5, 0, 0, 0], "第二个窗口 location 应为 [a_site=5, PAD, PAD, PAD]")

    def test_missing_fields(self):
        """测试缺失字段的处理：player_id, team 等"""
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
        
        self.assertEqual(len(samples), 1, "应生成一个样本")
        sample = samples[0]
        self.assertEqual(sample.meta["player_id"], "unknown", "缺失 player_id 应使用默认值 'unknown'")
        self.assertEqual(sample.team_idx, 0, "缺失 team 应默认 CT，team_idx=0")

if __name__ == '__main__':
    unittest.main()
