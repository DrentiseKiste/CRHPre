
# -*- coding: utf-8 -*-
"""
高铁线路冲突检测与过滤模块
集成拓扑结构，避免生成冗余或重复的线路预测
"""

import pandas as pd
from hsr_topology import HSRTopology, create_default_topology


class PredictionFilter:
    """预测结果过滤器"""
    
    def __init__(self, topology):
        self.topology = topology
        self.existing_segments = topology.get_existing_segments()
        self.filter_stats = {
            "total_candidates": 0,
            "filtered_by_segment": 0,
            "filtered_by_path": 0,
            "filtered_by_connected": 0,
            "passed": 0
        }
    
    def filter_predictions(self, predictions, filter_strategy="moderate", keep_top=None):
        """
        过滤预测结果
        
        Args:
            predictions: 预测结果DataFrame
            filter_strategy: 过滤策略 (strict/moderate/lenient)
            keep_top: 保留的预测数量
        
        Returns:
            (过滤后的DataFrame, 统计信息)
        """
        self.filter_stats = {
            "total_candidates": len(predictions),
            "filtered_by_segment": 0,
            "filtered_by_path": 0,
            "filtered_by_connected": 0,
            "passed": 0
        }
        
        filtered_rows = []
        conflict_details = []
        
        for _, row in predictions.iterrows():
            city_a = row["city_a"]
            city_b = row["city_b"]
            pair = tuple(sorted([city_a, city_b]))
            
            if pair in self.existing_segments:
                self.filter_stats["filtered_by_segment"] += 1
                conflict_details.append({
                    "city_a": city_a,
                    "city_b": city_b,
                    "prob": row.get("prob", 0),
                    "reason": "existing_segment",
                    "lines": []
                })
                continue
            
            conflicts = self.topology.check_conflicts(city_a, city_b)
            
            if conflicts:
                should_filter = self._should_filter_by_conflicts(conflicts, filter_strategy)
                
                if should_filter:
                    for conflict in conflicts:
                        if conflict.conflict_type == "redundant_segment":
                            self.filter_stats["filtered_by_segment"] += 1
                        elif conflict.conflict_type == "existing_path":
                            self.filter_stats["filtered_by_path"] += 1
                    
                    conflict_details.append({
                        "city_a": city_a,
                        "city_b": city_b,
                        "prob": row.get("prob", 0),
                        "reason": "topology_conflict",
                        "conflicts": [
                            {
                                "type": c.conflict_type,
                                "line": c.existing_line,
                                "description": c.description,
                                "severity": c.severity
                            }
                            for c in conflicts
                        ]
                    })
                    continue
            
            if filter_strategy == "strict":
                if self.topology.is_pair_connected(city_a, city_b):
                    self.filter_stats["filtered_by_connected"] += 1
                    conflict_details.append({
                        "city_a": city_a,
                        "city_b": city_b,
                        "prob": row.get("prob", 0),
                        "reason": "already_connected",
                        "lines": []
                    })
                    continue
            
            self.filter_stats["passed"] += 1
            filtered_rows.append(row)
        
        result_df = pd.DataFrame(filtered_rows) if filtered_rows else pd.DataFrame(columns=predictions.columns)
        
        if keep_top and len(result_df) &gt; keep_top:
            result_df = result_df.head(keep_top)
        
        filter_rate = 1.0 - (float(self.filter_stats["passed"]) / max(1, self.filter_stats["total_candidates"]))
        
        stats = {
            **self.filter_stats,
            "filter_rate": filter_rate,
            "strategy": filter_strategy,
            "conflict_details": conflict_details
        }
        
        return result_df, stats
    
    def _should_filter_by_conflicts(self, conflicts, strategy):
        """根据策略决定是否过滤"""
        if strategy == "strict":
            return len(conflicts) &gt; 0
        elif strategy == "moderate":
            return any(c.severity &gt;= 4 for c in conflicts)
        elif strategy == "lenient":
            return any(c.conflict_type == "redundant_segment" for c in conflicts)
        else:
            return len(conflicts) &gt; 0
    
    def explain_filtering(self, city_a, city_b):
        """
        解释为什么某个城市对会被过滤
        
        Returns:
            包含过滤原因的详细信息
        """
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
                "conflicts": [
                    {
                        "type": c.conflict_type,
                        "line": c.existing_line,
                        "description": c.description,
                        "severity": c.severity
                    }
                    for c in conflicts
                ]
            }
        
        if self.topology.is_pair_connected(city_a, city_b):
            return {
                "should_filter": True,
                "reason": "already_connected",
                "description": "%s 和 %s 已通过现有网络连通" % (city_a, city_b)
            }
        
        return {
            "should_filter": False,
            "reason": "no_conflict",
            "description": "未发现 %s-%s 的拓扑冲突" % (city_a, city_b)
        }


def build_filter_from_history(history_csv_path="hsr_history.csv"):
    """
    从历史数据构建预测过滤器
    
    Args:
        history_csv_path: 历史数据CSV文件路径
    
    Returns:
        PredictionFilter实例
    """
    try:
        history_df = pd.read_csv(history_csv_path, encoding="utf-8-sig")
        history_data = history_df.to_dict("records")
    except:
        history_data = None
    
    topology = create_default_topology(history_data)
    return PredictionFilter(topology)


def filter_predictions_with_topology(predictions, history_csv_path="hsr_history.csv", strategy="moderate"):
    """
    使用拓扑结构过滤预测结果的快捷函数
    
    Args:
        predictions: 预测结果DataFrame
        history_csv_path: 历史数据文件路径
        strategy: 过滤策略
    
    Returns:
        (过滤后的预测, 统计信息)
    """
    filter_obj = build_filter_from_history(history_csv_path)
    return filter_obj.filter_predictions(predictions, filter_strategy=strategy)

