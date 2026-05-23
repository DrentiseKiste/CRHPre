
# -*- coding: utf-8 -*-
"""
简化版过滤模块
"""

import pandas as pd
from simple_topology import create_simple_topology


class SimplePredictionFilter:
    """简化的预测过滤器"""
    
    def __init__(self, topology):
        self.topology = topology
        self.existing_segments = topology.get_existing_segments()
    
    def explain_filtering(self, city_a, city_b):
        """解释过滤原因"""
        pair = tuple(sorted([city_a, city_b]))
        
        if pair in self.existing_segments:
            return {
                "should_filter": True,
                "reason": "existing_segment",
                "description": "线段 %s-%s 已存在于现有线路中" % (city_a, city_b)
            }
        
        conflicts = self.topology.check_conflicts(city_a, city_b)
        if conflicts:
            return {
                "should_filter": True,
                "reason": "topology_conflict",
                "conflicts": conflicts
            }
        
        return {
            "should_filter": False,
            "reason": "no_conflict",
            "description": "未发现 %s-%s 的拓扑冲突" % (city_a, city_b)
        }
    
    def filter_predictions(self, predictions, strategy="moderate"):
        """过滤预测"""
        filtered = []
        
        for _, row in predictions.iterrows():
            city_a = row["city_a"]
            city_b = row["city_b"]
            
            result = self.explain_filtering(city_a, city_b)
            
            if result["should_filter"]:
                continue
            
            if strategy == "strict":
                if self.topology.is_pair_connected(city_a, city_b):
                    continue
            
            filtered.append(row)
        
        return pd.DataFrame(filtered) if filtered else pd.DataFrame(columns=predictions.columns)

