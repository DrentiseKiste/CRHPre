"""
大中华区高铁线路未来预测模型 (China HSR Future Prediction Model)
===============================================================
任务：预测中国（含大陆所有省份及香港、澳门、台湾）未来可能建设的高铁线路。
模型类型：二分类问题 —— 预测任意两个城市对之间“通高铁(1)”或“不通(0)”。

数据说明：
-----------
当前版本使用模拟数据集。当获取真实数据后，请按以下方式替换：
1. 城市基础数据：替换 CITIES_DATA 中的模拟数据，需包含字段：
    - name: 城市名称
    - province: 所属省级行政区
    - lat, lon: 经纬度坐标（WGS84）
    - gdp: 年度GDP（亿元人民币）
    - population: 常住人口（万人）
    - admin_level: 行政级别编码（见 ADMIN_LEVEL_MAP）
    - urban_agglomeration: 所属城市群名称（如"长三角"、"珠三角"等，无则填 None）
    - is_east_of_hu: 是否在胡焕庸线以东（True/False）

2. 现有高铁线路：替换 EXISTING_HSR_ROUTES 中的城市对列表，
    格式为 [(城市A名称, 城市B名称), ...]，代表目前已通车或已开工的高铁线路。

3. 跨海通道：在 CROSS_SEA_PAIRS 中维护需要特殊处理的跨海城市对。

特征工程逻辑（修订版）：
--------------
- gdp_sum: 两地GDP之和。经济总量越大，高铁建设的经济驱动力越强。
- pop_product: 两地人口乘积。人口规模反映潜在客流量，乘积体现双向流动需求。
- distance_km: 两地Haversine地理距离，跨海通道已乘以路径修正系数1.6。
    距离过短（<100km）高铁优势不明显，过长（>1500km）则航空竞争力强，
    中等距离（300-800km）是高铁黄金走廊。
- economic_gravity: 经济引力 = log1p(gdp_sum * pop_product) - log1p(distance_km)，
    综合反映经济人口规模与距离衰减的耦合效果。
- admin_level_max: 两地行政级别较高者。级别越高，政策资源倾斜越大。
- same_agglomeration: 是否同属一个城市群。城市群内部一体化程度高，互联互通需求强。
- hu_category: 胡焕庸线类别。2=两市都在东侧，1=一东一西，0=皆在西侧。
    替代原固定权重，让模型自行学习地理差异。
- cross_sea: 是否为跨海通道。跨海工程成本极高，但作为战略通道有额外政策加成。

可视化说明：
-----------
- 城市用"圆"(CircleMarker)表示，圆的大小映射人口权重，颜色映射行政级别。
- 预测线路用 PolyLine 绘制，线条粗细映射建设概率。
- 特别标注跨海通道（台湾海峡、琼州海峡）。
"""

import math
import random
import itertools
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

# 机器学习库
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[警告] scikit-learn 未安装，将使用随机概率作为占位。请运行: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[警告] xgboost 未安装，将回退到 RandomForest。请运行: pip install xgboost")

# 可视化库
try:
    import folium
    from folium import CircleMarker, PolyLine
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("[警告] folium 未安装，地图将无法生成。请运行: pip install folium")

# =============================================================================
# 1. 模拟数据集（待替换为真实数据）
# =============================================================================

ADMIN_LEVEL_MAP = {
    "直辖市": 4,
    "特别行政区": 4,
    "省会/副省级": 3,
    "地级市": 2,
    "县级": 1,
}

# 数据格式说明（见文档字符串，此处省略重复注释）

# 模拟城市数据：覆盖主要省会、直辖市、特别行政区及台湾主要城市
CITIES_DATA: List[Dict[str, Any]] = [
    # 直辖市
    {"name": "北京", "province": "北京市", "lat": 39.9042, "lon": 116.4074, "gdp": 43760, "population": 2189, "admin_level": "直辖市", "urban_agglomeration": "京津冀", "is_east_of_hu": True},
    {"name": "上海", "province": "上海市", "lat": 31.2304, "lon": 121.4737, "gdp": 47218, "population": 2489, "admin_level": "直辖市", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "天津", "province": "天津市", "lat": 39.0842, "lon": 117.2010, "gdp": 16311, "population": 1386, "admin_level": "直辖市", "urban_agglomeration": "京津冀", "is_east_of_hu": True},
    {"name": "重庆", "province": "重庆市", "lat": 29.5630, "lon": 106.5516, "gdp": 30145, "population": 3212, "admin_level": "直辖市", "urban_agglomeration": None, "is_east_of_hu": True},
    # 特别行政区
    {"name": "香港", "province": "香港特别行政区", "lat": 22.3193, "lon": 114.1694, "gdp": 24280, "population": 741, "admin_level": "特别行政区", "urban_agglomeration": "珠三角", "is_east_of_hu": True},
    {"name": "澳门", "province": "澳门特别行政区", "lat": 22.1987, "lon": 113.5439, "gdp": 1929, "population": 68, "admin_level": "特别行政区", "urban_agglomeration": "珠三角", "is_east_of_hu": True},
    # 台湾主要城市
    {"name": "台北", "province": "台湾省", "lat": 25.0330, "lon": 121.5654, "gdp": 5500, "population": 260, "admin_level": "省会/副省级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "台中", "province": "台湾省", "lat": 24.1477, "lon": 120.6736, "gdp": 3200, "population": 282, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "高雄", "province": "台湾省", "lat": 22.6273, "lon": 120.3014, "gdp": 2800, "population": 273, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    # 各省会及重点城市（部分列举）
    {"name": "广州", "province": "广东省", "lat": 23.1291, "lon": 113.2644, "gdp": 28839, "population": 1881, "admin_level": "省会/副省级", "urban_agglomeration": "珠三角", "is_east_of_hu": True},
    {"name": "深圳", "province": "广东省", "lat": 22.5431, "lon": 114.0579, "gdp": 34606, "population": 1768, "admin_level": "地级市", "urban_agglomeration": "珠三角", "is_east_of_hu": True},
    {"name": "成都", "province": "四川省", "lat": 30.5728, "lon": 104.0668, "gdp": 22074, "population": 2126, "admin_level": "省会/副省级", "urban_agglomeration": "成渝", "is_east_of_hu": True},
    {"name": "武汉", "province": "湖北省", "lat": 30.5928, "lon": 114.3055, "gdp": 20011, "population": 1364, "admin_level": "省会/副省级", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "西安", "province": "陕西省", "lat": 34.3416, "lon": 108.9398, "gdp": 12010, "population": 1316, "admin_level": "省会/副省级", "urban_agglomeration": "关中平原", "is_east_of_hu": False},
    {"name": "郑州", "province": "河南省", "lat": 34.7466, "lon": 113.6253, "gdp": 13617, "population": 1282, "admin_level": "省会/副省级", "urban_agglomeration": "中原", "is_east_of_hu": True},
    {"name": "南京", "province": "江苏省", "lat": 32.0603, "lon": 118.7969, "gdp": 17421, "population": 949, "admin_level": "省会/副省级", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "杭州", "province": "浙江省", "lat": 30.2741, "lon": 120.1551, "gdp": 20059, "population": 1237, "admin_level": "省会/副省级", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "福州", "province": "福建省", "lat": 26.0745, "lon": 119.2965, "gdp": 12698, "population": 842, "admin_level": "省会/副省级", "urban_agglomeration": "海峡西岸", "is_east_of_hu": True},
    {"name": "厦门", "province": "福建省", "lat": 24.4798, "lon": 118.0894, "gdp": 7802, "population": 528, "admin_level": "地级市", "urban_agglomeration": "海峡西岸", "is_east_of_hu": True},
    {"name": "昆明", "province": "云南省", "lat": 25.0389, "lon": 102.7183, "gdp": 7541, "population": 862, "admin_level": "省会/副省级", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "贵阳", "province": "贵州省", "lat": 26.6470, "lon": 106.6302, "gdp": 5154, "population": 610, "admin_level": "省会/副省级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "南宁", "province": "广西壮族自治区", "lat": 22.8170, "lon": 108.3665, "gdp": 5120, "population": 883, "admin_level": "省会/副省级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "海口", "province": "海南省", "lat": 20.0440, "lon": 110.1999, "gdp": 2057, "population": 290, "admin_level": "省会/副省级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "兰州", "province": "甘肃省", "lat": 36.0611, "lon": 103.8343, "gdp": 3387, "population": 438, "admin_level": "省会/副省级", "urban_agglomeration": "兰西", "is_east_of_hu": False},
    {"name": "乌鲁木齐", "province": "新疆维吾尔自治区", "lat": 43.8256, "lon": 87.6168, "gdp": 3893, "population": 405, "admin_level": "省会/副省级", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "拉萨", "province": "西藏自治区", "lat": 29.6500, "lon": 91.1000, "gdp": 850, "population": 86, "admin_level": "省会/副省级", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "哈尔滨", "province": "黑龙江省", "lat": 45.8038, "lon": 126.5350, "gdp": 5351, "population": 988, "admin_level": "省会/副省级", "urban_agglomeration": "哈长", "is_east_of_hu": True},
    {"name": "长春", "province": "吉林省", "lat": 43.8171, "lon": 125.3235, "gdp": 6638, "population": 906, "admin_level": "省会/副省级", "urban_agglomeration": "哈长", "is_east_of_hu": True},
    {"name": "沈阳", "province": "辽宁省", "lat": 41.8057, "lon": 123.4315, "gdp": 8122, "population": 911, "admin_level": "省会/副省级", "urban_agglomeration": "辽中南", "is_east_of_hu": True},
    {"name": "大连", "province": "辽宁省", "lat": 38.9140, "lon": 121.6147, "gdp": 8752, "population": 745, "admin_level": "地级市", "urban_agglomeration": "辽中南", "is_east_of_hu": True},
    {"name": "济南", "province": "山东省", "lat": 36.6512, "lon": 117.1201, "gdp": 12027, "population": 941, "admin_level": "省会/副省级", "urban_agglomeration": "山东半岛", "is_east_of_hu": True},
    {"name": "青岛", "province": "山东省", "lat": 36.0671, "lon": 120.3826, "gdp": 14920, "population": 1025, "admin_level": "地级市", "urban_agglomeration": "山东半岛", "is_east_of_hu": True},
    {"name": "石家庄", "province": "河北省", "lat": 38.0428, "lon": 114.5149, "gdp": 7100, "population": 1122, "admin_level": "省会/副省级", "urban_agglomeration": "京津冀", "is_east_of_hu": True},
    {"name": "太原", "province": "山西省", "lat": 37.8706, "lon": 112.5489, "gdp": 5121, "population": 530, "admin_level": "省会/副省级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "合肥", "province": "安徽省", "lat": 31.8206, "lon": 117.2272, "gdp": 12013, "population": 963, "admin_level": "省会/副省级", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "南昌", "province": "江西省", "lat": 28.6820, "lon": 115.8579, "gdp": 7203, "population": 653, "admin_level": "省会/副省级", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "长沙", "province": "湖南省", "lat": 28.2282, "lon": 112.9388, "gdp": 14331, "population": 1042, "admin_level": "省会/副省级", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "呼和浩特", "province": "内蒙古自治区", "lat": 40.8414, "lon": 111.7519, "gdp": 3800, "population": 344, "admin_level": "省会/副省级", "urban_agglomeration": "呼包鄂", "is_east_of_hu": False},
    {"name": "银川", "province": "宁夏回族自治区", "lat": 38.4872, "lon": 106.2309, "gdp": 2260, "population": 285, "admin_level": "省会/副省级", "urban_agglomeration": "兰西", "is_east_of_hu": False},
    {"name": "西宁", "province": "青海省", "lat": 36.6171, "lon": 101.7782, "gdp": 1644, "population": 246, "admin_level": "省会/副省级", "urban_agglomeration": "兰西", "is_east_of_hu": False},
    # 以下为新增城市（用于匹配更完整的线路数据）
    {"name": "包头", "province": "内蒙古自治区", "lat": 40.6582, "lon": 109.8403, "gdp": 3750, "population": 279, "admin_level": "地级市", "urban_agglomeration": "呼包鄂", "is_east_of_hu": False},
    {"name": "鄂尔多斯", "province": "内蒙古自治区", "lat": 39.6083, "lon": 109.7810, "gdp": 5800, "population": 215, "admin_level": "地级市", "urban_agglomeration": "呼包鄂", "is_east_of_hu": False},
    {"name": "唐山", "province": "河北省", "lat": 39.6292, "lon": 118.1802, "gdp": 8900, "population": 771, "admin_level": "地级市", "urban_agglomeration": "京津冀", "is_east_of_hu": True},
    {"name": "苏州", "province": "江苏省", "lat": 31.2989, "lon": 120.5853, "gdp": 23958, "population": 1275, "admin_level": "地级市", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "宁波", "province": "浙江省", "lat": 29.8683, "lon": 121.5440, "gdp": 15704, "population": 961, "admin_level": "地级市", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "嘉兴", "province": "浙江省", "lat": 30.7461, "lon": 120.7555, "gdp": 7062, "population": 543, "admin_level": "地级市", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "佛山", "province": "广东省", "lat": 23.0218, "lon": 113.1219, "gdp": 12698, "population": 961, "admin_level": "地级市", "urban_agglomeration": "珠三角", "is_east_of_hu": True},
    {"name": "东莞", "province": "广东省", "lat": 23.0473, "lon": 113.7518, "gdp": 11200, "population": 1054, "admin_level": "地级市", "urban_agglomeration": "珠三角", "is_east_of_hu": True},
    {"name": "珠海", "province": "广东省", "lat": 22.2707, "lon": 113.5670, "gdp": 4233, "population": 246, "admin_level": "地级市", "urban_agglomeration": "珠三角", "is_east_of_hu": True},
    {"name": "肇庆", "province": "广东省", "lat": 23.0467, "lon": 112.4659, "gdp": 2700, "population": 413, "admin_level": "地级市", "urban_agglomeration": "珠三角", "is_east_of_hu": True},
    {"name": "绵阳", "province": "四川省", "lat": 31.4675, "lon": 104.6796, "gdp": 3600, "population": 486, "admin_level": "地级市", "urban_agglomeration": "成渝", "is_east_of_hu": True},
    {"name": "乐山", "province": "四川省", "lat": 29.5823, "lon": 103.7665, "gdp": 2200, "population": 316, "admin_level": "地级市", "urban_agglomeration": "成渝", "is_east_of_hu": True},
    {"name": "内江", "province": "四川省", "lat": 29.5808, "lon": 105.0580, "gdp": 1650, "population": 314, "admin_level": "地级市", "urban_agglomeration": "成渝", "is_east_of_hu": True},
    {"name": "淄博", "province": "山东省", "lat": 36.8135, "lon": 118.0550, "gdp": 4400, "population": 470, "admin_level": "地级市", "urban_agglomeration": "山东半岛", "is_east_of_hu": True},
    {"name": "潍坊", "province": "山东省", "lat": 36.7089, "lon": 119.1619, "gdp": 7300, "population": 942, "admin_level": "地级市", "urban_agglomeration": "山东半岛", "is_east_of_hu": True},
    {"name": "烟台", "province": "山东省", "lat": 37.4638, "lon": 121.4481, "gdp": 8712, "population": 706, "admin_level": "地级市", "urban_agglomeration": "山东半岛", "is_east_of_hu": True},
    {"name": "大庆", "province": "黑龙江省", "lat": 46.5893, "lon": 125.1036, "gdp": 2600, "population": 278, "admin_level": "地级市", "urban_agglomeration": "哈长", "is_east_of_hu": True},
    {"name": "吉林", "province": "吉林省", "lat": 43.8378, "lon": 126.5494, "gdp": 1550, "population": 362, "admin_level": "地级市", "urban_agglomeration": "哈长", "is_east_of_hu": True},
    {"name": "四平", "province": "吉林省", "lat": 43.1703, "lon": 124.3503, "gdp": 580, "population": 181, "admin_level": "地级市", "urban_agglomeration": "哈长", "is_east_of_hu": True},
    {"name": "莆田", "province": "福建省", "lat": 25.4541, "lon": 119.0077, "gdp": 3100, "population": 321, "admin_level": "地级市", "urban_agglomeration": "海峡西岸", "is_east_of_hu": True},
    {"name": "泉州", "province": "福建省", "lat": 24.8744, "lon": 118.6757, "gdp": 12103, "population": 878, "admin_level": "地级市", "urban_agglomeration": "海峡西岸", "is_east_of_hu": True},
    {"name": "开封", "province": "河南省", "lat": 34.7972, "lon": 114.3076, "gdp": 2500, "population": 482, "admin_level": "地级市", "urban_agglomeration": "中原", "is_east_of_hu": True},
    {"name": "洛阳", "province": "河南省", "lat": 34.6187, "lon": 112.4540, "gdp": 5800, "population": 706, "admin_level": "地级市", "urban_agglomeration": "中原", "is_east_of_hu": True},
    {"name": "新乡", "province": "河南省", "lat": 35.3030, "lon": 113.9268, "gdp": 3400, "population": 625, "admin_level": "地级市", "urban_agglomeration": "中原", "is_east_of_hu": True},
    {"name": "黄石", "province": "湖北省", "lat": 30.1997, "lon": 115.0389, "gdp": 1900, "population": 247, "admin_level": "地级市", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "孝感", "province": "湖北省", "lat": 30.9279, "lon": 113.9109, "gdp": 2600, "population": 427, "admin_level": "地级市", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "株洲", "province": "湖南省", "lat": 27.8274, "lon": 113.1340, "gdp": 3400, "population": 390, "admin_level": "地级市", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "湘潭", "province": "湖南省", "lat": 27.8297, "lon": 112.9443, "gdp": 2700, "population": 273, "admin_level": "地级市", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "九江", "province": "江西省", "lat": 29.7051, "lon": 116.0019, "gdp": 4000, "population": 460, "admin_level": "地级市", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "咸阳", "province": "陕西省", "lat": 34.3296, "lon": 108.7089, "gdp": 2800, "population": 421, "admin_level": "地级市", "urban_agglomeration": "关中平原", "is_east_of_hu": True},
    {"name": "宝鸡", "province": "陕西省", "lat": 34.3630, "lon": 107.2370, "gdp": 2600, "population": 332, "admin_level": "地级市", "urban_agglomeration": "关中平原", "is_east_of_hu": True},
    {"name": "渭南", "province": "陕西省", "lat": 34.4996, "lon": 109.5098, "gdp": 1900, "population": 469, "admin_level": "地级市", "urban_agglomeration": "关中平原", "is_east_of_hu": True},
    {"name": "白银", "province": "甘肃省", "lat": 36.5450, "lon": 104.1385, "gdp": 620, "population": 151, "admin_level": "地级市", "urban_agglomeration": "兰西", "is_east_of_hu": False},
    {"name": "北海", "province": "广西壮族自治区", "lat": 21.4813, "lon": 109.1002, "gdp": 1700, "population": 185, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "钦州", "province": "广西壮族自治区", "lat": 21.9797, "lon": 108.6542, "gdp": 1650, "population": 330, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "防城港", "province": "广西壮族自治区", "lat": 21.6869, "lon": 108.3547, "gdp": 900, "population": 105, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "遵义", "province": "贵州省", "lat": 27.7254, "lon": 106.9274, "gdp": 4500, "population": 661, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "安顺", "province": "贵州省", "lat": 26.2533, "lon": 105.9476, "gdp": 1100, "population": 236, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "玉溪", "province": "云南省", "lat": 24.3505, "lon": 102.5439, "gdp": 2300, "population": 225, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "曲靖", "province": "云南省", "lat": 25.4900, "lon": 103.7961, "gdp": 3600, "population": 577, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "昌吉", "province": "新疆维吾尔自治区", "lat": 44.0142, "lon": 87.2675, "gdp": 1700, "population": 161, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "石河子", "province": "新疆维吾尔自治区", "lat": 44.3059, "lon": 86.0806, "gdp": 750, "population": 62, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "连云港", "province": "江苏省", "lat": 34.6004, "lon": 119.1792, "gdp": 4000, "population": 460, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "徐州", "province": "江苏省", "lat": 34.2610, "lon": 117.1848, "gdp": 8900, "population": 908, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "湛江", "province": "广东省", "lat": 21.2707, "lon": 110.3594, "gdp": 3700, "population": 703, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "赣州", "province": "江西省", "lat": 25.8311, "lon": 114.9348, "gdp": 4200, "population": 897, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "常德", "province": "湖南省", "lat": 29.0316, "lon": 111.6985, "gdp": 4200, "population": 528, "admin_level": "地级市", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "张家界", "province": "湖南省", "lat": 29.1171, "lon": 110.4792, "gdp": 600, "population": 152, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "黔江", "province": "重庆市", "lat": 29.5335, "lon": 108.7707, "gdp": 280, "population": 49, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "桂林", "province": "广西壮族自治区", "lat": 25.2742, "lon": 110.2993, "gdp": 2400, "population": 495, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "百色", "province": "广西壮族自治区", "lat": 23.9023, "lon": 106.6183, "gdp": 1850, "population": 357, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "金华", "province": "浙江省", "lat": 29.0791, "lon": 119.6494, "gdp": 6000, "population": 706, "admin_level": "地级市", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "衢州", "province": "浙江省", "lat": 28.9359, "lon": 118.8743, "gdp": 2000, "population": 228, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "安庆", "province": "安徽省", "lat": 30.5319, "lon": 117.1151, "gdp": 2800, "population": 417, "admin_level": "地级市", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "荆州", "province": "湖北省", "lat": 30.3352, "lon": 112.2396, "gdp": 3000, "population": 523, "admin_level": "地级市", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "宜昌", "province": "湖北省", "lat": 30.6919, "lon": 111.2864, "gdp": 6100, "population": 391, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "临沂", "province": "山东省", "lat": 35.1046, "lon": 118.3564, "gdp": 6100, "population": 1102, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "淮安", "province": "江苏省", "lat": 33.6104, "lon": 119.0158, "gdp": 4800, "population": 456, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "扬州", "province": "江苏省", "lat": 32.3932, "lon": 119.4210, "gdp": 7100, "population": 456, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "黄山", "province": "安徽省", "lat": 29.7147, "lon": 118.3375, "gdp": 1000, "population": 133, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "绥芬河", "province": "黑龙江省", "lat": 44.4123, "lon": 131.1514, "gdp": 150, "population": 7, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "牡丹江", "province": "黑龙江省", "lat": 44.5514, "lon": 129.6330, "gdp": 900, "population": 229, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "齐齐哈尔", "province": "黑龙江省", "lat": 47.3543, "lon": 123.9179, "gdp": 1300, "population": 407, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "海拉尔", "province": "内蒙古自治区", "lat": 49.2122, "lon": 119.7658, "gdp": 200, "population": 36, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "满洲里", "province": "内蒙古自治区", "lat": 49.5978, "lon": 117.3786, "gdp": 170, "population": 15, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "雄安", "province": "河北省", "lat": 39.0500, "lon": 115.9000, "gdp": 300, "population": 120, "admin_level": "县级", "urban_agglomeration": "京津冀", "is_east_of_hu": True},
    # 新增城市（补充历史数据中缺失的）
    {"name": "万州", "province": "重庆市", "lat": 30.8300, "lon": 108.3200, "gdp": 1000, "population": 175, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "三亚", "province": "海南省", "lat": 18.2208, "lon": 109.5028, "gdp": 835, "population": 103, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "丹东", "province": "辽宁省", "lat": 40.1272, "lon": 124.3764, "gdp": 854, "population": 218, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "佳木斯", "province": "黑龙江省", "lat": 46.8139, "lon": 130.3467, "gdp": 810, "population": 225, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "利川", "province": "湖北省", "lat": 30.2943, "lon": 108.9388, "gdp": 245, "population": 75, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "十堰", "province": "湖北省", "lat": 32.6500, "lon": 110.7800, "gdp": 2300, "population": 320, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "商丘", "province": "河南省", "lat": 34.4422, "lon": 115.6478, "gdp": 3262, "population": 773, "admin_level": "地级市", "urban_agglomeration": "中原", "is_east_of_hu": True},
    {"name": "太仓", "province": "江苏省", "lat": 31.5167, "lon": 121.1167, "gdp": 1650, "population": 71, "admin_level": "县级", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "宜宾", "province": "四川省", "lat": 28.7667, "lon": 104.6167, "gdp": 3400, "population": 459, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "庄寨", "province": "山东省", "lat": 35.1300, "lon": 115.4800, "gdp": 50, "population": 8, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "延安", "province": "陕西省", "lat": 36.6056, "lon": 109.4644, "gdp": 2200, "population": 228, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "张家口", "province": "河北省", "lat": 40.8250, "lon": 114.8850, "gdp": 1700, "population": 411, "admin_level": "地级市", "urban_agglomeration": "京津冀", "is_east_of_hu": True},
    {"name": "弥勒", "province": "云南省", "lat": 24.4000, "lon": 103.4333, "gdp": 474, "population": 54, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "怀化", "province": "湖南省", "lat": 27.5800, "lon": 109.9800, "gdp": 1800, "population": 459, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "承德", "province": "河北省", "lat": 40.9700, "lon": 117.9400, "gdp": 1700, "population": 336, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "日照", "province": "山东省", "lat": 35.4200, "lon": 119.4600, "gdp": 2300, "population": 297, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "曲阜", "province": "山东省", "lat": 35.5900, "lon": 116.9800, "gdp": 400, "population": 65, "admin_level": "县级", "urban_agglomeration": "山东半岛", "is_east_of_hu": True},
    {"name": "汉口", "province": "湖北省", "lat": 30.5800, "lon": 114.2800, "gdp": 0, "population": 0, "admin_level": "县级", "urban_agglomeration": "长江中游", "is_east_of_hu": True},
    {"name": "汕头", "province": "广东省", "lat": 23.3500, "lon": 116.6800, "gdp": 3000, "population": 553, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "汕尾", "province": "广东省", "lat": 22.8000, "lon": 115.3500, "gdp": 1300, "population": 268, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "池州", "province": "安徽省", "lat": 30.6600, "lon": 117.4800, "gdp": 1000, "population": 134, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "温州", "province": "浙江省", "lat": 28.0200, "lon": 120.6500, "gdp": 8000, "population": 964, "admin_level": "地级市", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "湖州", "province": "浙江省", "lat": 30.8600, "lon": 120.1100, "gdp": 3800, "population": 341, "admin_level": "地级市", "urban_agglomeration": "长三角", "is_east_of_hu": True},
    {"name": "滨海新区", "province": "天津市", "lat": 39.0200, "lon": 117.7200, "gdp": 8700, "population": 207, "admin_level": "县级", "urban_agglomeration": "京津冀", "is_east_of_hu": True},
    {"name": "濮阳", "province": "河南省", "lat": 35.7100, "lon": 115.0100, "gdp": 1760, "population": 374, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "玉林", "province": "广西壮族自治区", "lat": 22.6300, "lon": 110.1500, "gdp": 2000, "population": 581, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "珲春", "province": "吉林省", "lat": 42.8700, "lon": 130.3500, "gdp": 160, "population": 23, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "秦皇岛", "province": "河北省", "lat": 39.9300, "lon": 119.5900, "gdp": 1800, "population": 313, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "荣成", "province": "山东省", "lat": 37.1000, "lon": 122.4100, "gdp": 1000, "population": 72, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "蒙自", "province": "云南省", "lat": 23.3300, "lon": 103.4300, "gdp": 400, "population": 46, "admin_level": "县级", "urban_agglomeration": None, "is_east_of_hu": False},
    {"name": "衡阳", "province": "湖南省", "lat": 26.8900, "lon": 112.5900, "gdp": 3800, "population": 664, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "襄阳", "province": "湖北省", "lat": 32.0000, "lon": 112.1500, "gdp": 5800, "population": 526, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
    {"name": "阜阳", "province": "安徽省", "lat": 32.8900, "lon": 115.8100, "gdp": 3200, "population": 820, "admin_level": "地级市", "urban_agglomeration": None, "is_east_of_hu": True},
]


# 现有高铁线路（更完整的八纵八横及区域干线）
EXISTING_HSR_ROUTES: List[Tuple[str, str]] = [
    # ===== 八纵通道 =====
    # 1. 京沪通道
    ("北京", "天津"), ("天津", "济南"), ("济南", "南京"), ("南京", "上海"),
    ("济南", "青岛"), ("南京", "杭州"),

    # 2. 京港（台）通道
    ("北京", "石家庄"), ("石家庄", "郑州"), ("郑州", "武汉"), ("武汉", "长沙"),
    ("长沙", "广州"), ("广州", "深圳"), ("深圳", "香港"),
    ("武汉", "南昌"), ("南昌", "深圳"),

    # 3. 京哈-京港澳通道
    ("北京", "沈阳"), ("沈阳", "长春"), ("长春", "哈尔滨"),
    ("北京", "石家庄"), ("石家庄", "郑州"), ("郑州", "武汉"), ("武汉", "长沙"),
    ("长沙", "广州"), ("广州", "深圳"),
    ("石家庄", "太原"), ("太原", "西安"),

    # 4. 包（银）海通道
    ("呼和浩特", "包头"), ("包头", "银川"), ("银川", "兰州"),
    ("兰州", "成都"), ("成都", "贵阳"), ("贵阳", "南宁"), ("南宁", "海口"),

    # 5. 沿海通道
    ("大连", "沈阳"), ("沈阳", "天津"), ("天津", "济南"), ("济南", "青岛"),
    ("青岛", "连云港"), ("连云港", "上海"), ("上海", "杭州"), ("杭州", "宁波"),
    ("宁波", "福州"), ("福州", "厦门"), ("厦门", "深圳"), ("深圳", "广州"),
    ("广州", "湛江"), ("湛江", "海口"),

    # 6. 京沪二通道（辅助）
    ("北京", "天津"), ("天津", "潍坊"), ("潍坊", "临沂"), ("临沂", "淮安"),
    ("淮安", "扬州"), ("扬州", "上海"),

    # 7. 京昆通道
    ("北京", "石家庄"), ("石家庄", "太原"), ("太原", "西安"), ("西安", "成都"),
    ("成都", "昆明"),

    # 8. 兰（西）广通道
    ("兰州", "西宁"), ("西宁", "成都"), ("成都", "贵阳"), ("贵阳", "桂林"),
    ("桂林", "广州"),

    # ===== 八横通道 =====
    # 1. 绥满通道
    ("绥芬河", "牡丹江"), ("牡丹江", "哈尔滨"), ("哈尔滨", "齐齐哈尔"),
    ("齐齐哈尔", "海拉尔"), ("海拉尔", "满洲里"),

    # 2. 京兰通道
    ("北京", "呼和浩特"), ("呼和浩特", "包头"), ("包头", "银川"),
    ("银川", "兰州"),

    # 3. 青银通道
    ("青岛", "济南"), ("济南", "石家庄"), ("石家庄", "太原"),
    ("太原", "银川"),

    # 4. 陆桥通道
    ("连云港", "徐州"), ("徐州", "郑州"), ("郑州", "西安"),
    ("西安", "兰州"), ("兰州", "西宁"), ("兰州", "乌鲁木齐"),

    # 5. 沿江通道
    ("上海", "南京"), ("南京", "合肥"), ("合肥", "武汉"), ("武汉", "宜昌"),
    ("宜昌", "重庆"), ("重庆", "成都"), ("武汉", "荆州"), ("荆州", "常德"),
    ("南京", "安庆"), ("安庆", "九江"), ("九江", "武汉"),

    # 6. 沪昆通道
    ("上海", "杭州"), ("杭州", "南昌"), ("南昌", "长沙"), ("长沙", "贵阳"),
    ("贵阳", "昆明"), ("杭州", "金华"), ("金华", "衢州"),

    # 7. 厦渝通道
    ("厦门", "赣州"), ("赣州", "长沙"), ("长沙", "常德"), ("常德", "张家界"),
    ("张家界", "黔江"), ("黔江", "重庆"),

    # 8. 广昆通道
    ("广州", "南宁"), ("南宁", "昆明"), ("南宁", "百色"),

    # ===== 其他重要区域干线 =====
    # 京津冀内部
    ("北京", "天津"), ("天津", "唐山"), ("北京", "雄安"),
    # 长三角内部
    ("上海", "苏州"), ("苏州", "南京"), ("南京", "杭州"), ("杭州", "宁波"),
    ("合肥", "南京"), ("合肥", "杭州"), ("上海", "嘉兴"),
    # 珠三角内部
    ("广州", "佛山"), ("佛山", "肇庆"), ("广州", "东莞"), ("东莞", "深圳"),
    ("广州", "珠海"), ("珠海", "澳门"),
    # 成渝城市群
    ("成都", "绵阳"), ("成都", "乐山"), ("成都", "内江"), ("内江", "重庆"),
    # 山东半岛
    ("济南", "淄博"), ("淄博", "潍坊"), ("潍坊", "青岛"), ("潍坊", "烟台"),
    # 哈长城市群
    ("哈尔滨", "大庆"), ("长春", "吉林"), ("长春", "四平"),
    # 海峡西岸
    ("福州", "莆田"), ("莆田", "泉州"), ("泉州", "厦门"),
    # 中原城市群
    ("郑州", "开封"), ("郑州", "洛阳"), ("郑州", "新乡"),
    # 长江中游
    ("武汉", "黄石"), ("武汉", "孝感"), ("长沙", "株洲"), ("长沙", "湘潭"),
    ("南昌", "九江"),
    # 关中平原
    ("西安", "咸阳"), ("西安", "宝鸡"), ("西安", "渭南"),
    # 兰西城市群
    ("兰州", "西宁"), ("兰州", "白银"),
    # 北部湾
    ("南宁", "北海"), ("南宁", "钦州"), ("钦州", "防城港"),
    # 黔中城市群
    ("贵阳", "遵义"), ("贵阳", "安顺"),
    # 滇中城市群
    ("昆明", "玉溪"), ("昆明", "曲靖"),
    # 呼包鄂榆
    ("呼和浩特", "包头"), ("包头", "鄂尔多斯"),
    # 天山北坡
    ("乌鲁木齐", "昌吉"), ("乌鲁木齐", "石河子"),
    # 其他省会连接线
    ("福州", "合肥"), ("杭州", "黄山"), ("南昌", "杭州"),
    ("广州", "长沙"), ("南宁", "广州"), ("贵阳", "南宁"),
    ("昆明", "贵阳"), ("兰州", "西安"), ("西宁", "兰州"),
    ("银川", "兰州"), ("呼和浩特", "北京"), ("太原", "石家庄"),
    ("济南", "郑州"), ("青岛", "济南"), ("大连", "沈阳"),
    ("长春", "哈尔滨"), ("沈阳", "大连"), ("石家庄", "济南"),
    # 港澳
    ("广州", "澳门"), ("深圳", "香港"),
]


# 跨海通道特殊城市对
CROSS_SEA_PAIRS: List[Tuple[str, str]] = [
    ("福州", "台北"),
    ("厦门", "台中"),
    ("广州", "高雄"),
    ("海口", "湛江"),
]

# 胡焕庸线参数
HU_LINE = {"start": {"lat": 50.245, "lon": 127.528}, "end": {"lat": 24.500, "lon": 98.500}}

# 跨海路径修正系数（直线距离乘以该系数以近似实际工程线路长度）
CROSS_SEA_DISTANCE_FACTOR = 1.6


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两点间地球表面距离（公里）"""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def point_east_of_hu_line(lat: float, lon: float) -> bool:
    """判断点是否在胡焕庸线以东"""
    start = HU_LINE["start"]
    end = HU_LINE["end"]
    ratio = (lat - end["lat"]) / (start["lat"] - end["lat"])
    ratio = max(0.0, min(1.0, ratio))
    lon_threshold = end["lon"] + ratio * (start["lon"] - end["lon"])
    return lon > lon_threshold


# =============================================================================
# 2. 数据预处理与特征工程
# =============================================================================

def build_city_df(cities_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """构建城市DataFrame，自动计算胡焕庸线位置（若缺失）"""
    df = pd.DataFrame(cities_data)
    if "is_east_of_hu" not in df.columns:
        df["is_east_of_hu"] = df.apply(lambda row: point_east_of_hu_line(row["lat"], row["lon"]), axis=1)
    df["admin_level_code"] = df["admin_level"].map(ADMIN_LEVEL_MAP)
    return df


def generate_pair_features(
    city_df: pd.DataFrame,
    city_pairs: List[Tuple[str, str]],
    label: Optional[int] = None,
) -> pd.DataFrame:
    """
    为给定的城市对列表生成特征。
    新增经济引力特征、胡焕庸线类别，去除距离分箱，跨海距离修正。
    """
    records = []
    city_map = {row["name"]: row for _, row in city_df.iterrows()}

    for a_name, b_name in city_pairs:
        if a_name not in city_map or b_name not in city_map:
            continue
        a = city_map[a_name]
        b = city_map[b_name]

        # 基础经济人口特征
        gdp_sum = a["gdp"] + b["gdp"]
        pop_product = a["population"] * b["population"]

        # 跨海判定（双向检查）
        cross_sea = 1 if (a_name, b_name) in CROSS_SEA_PAIRS or (b_name, a_name) in CROSS_SEA_PAIRS else 0

        # 地理距离，跨海时乘以修正系数
        raw_distance = haversine(a["lat"], a["lon"], b["lat"], b["lon"])
        distance_km = raw_distance * CROSS_SEA_DISTANCE_FACTOR if cross_sea else raw_distance

        # 经济引力（替换原固定 weight，使用修正后距离）
        economic_gravity = np.log1p(gdp_sum * pop_product) - np.log1p(distance_km)

        # 行政级别：取较高者
        admin_level_max = max(a["admin_level_code"], b["admin_level_code"])

        # 是否同属一个城市群
        same_agglomeration = 1 if (a["urban_agglomeration"] == b["urban_agglomeration"] and pd.notna(a["urban_agglomeration"])) else 0

        # 胡焕庸线类别（替代原先的固定权重）
        a_east = a["is_east_of_hu"]
        b_east = b["is_east_of_hu"]
        if a_east and b_east:
            hu_category = 2  # 两市都在东侧
        elif a_east or b_east:
            hu_category = 1  # 一东一西
        else:
            hu_category = 0  # 皆在西侧

        record = {
            "city_a": a_name,
            "city_b": b_name,
            "gdp_sum": gdp_sum,
            "pop_product": pop_product,
            "distance_km": distance_km,
            "economic_gravity": economic_gravity,
            "admin_level_max": admin_level_max,
            "same_agglomeration": same_agglomeration,
            "hu_category": hu_category,
            "cross_sea": cross_sea,
        }
        if label is not None:
            record["label"] = label
        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# 3. 训练集构建
# =============================================================================

def build_training_data(city_df: pd.DataFrame) -> pd.DataFrame:
    """
    构建训练数据集：
    - 正样本：EXISTING_HSR_ROUTES 中已知高铁线路（去重，仅保留单向）
    - 负样本：从所有未连接城市对中随机抽取，数量为正样本的3倍
    """
    all_cities = city_df["name"].tolist()
    existing_set = set()
    for a, b in EXISTING_HSR_ROUTES:
        if a in all_cities and b in all_cities:
            # 统一排序后存储，避免方向重复
            existing_set.add(tuple(sorted([a, b])))

    # 正样本（已去重）
    positive_pairs = list(existing_set)
    positive_df = generate_pair_features(city_df, positive_pairs, label=1)

    # 所有可能的城市对（无向，避免重复）
    all_possible_pairs = []
    for i in range(len(all_cities)):
        for j in range(i + 1, len(all_cities)):
            pair = (all_cities[i], all_cities[j])
            if pair not in existing_set:
                all_possible_pairs.append(pair)

    # 负样本：随机抽取，数量与正样本倍数（可调整）
    n_positive = len(positive_df)
    n_negative = min(n_positive * 3, len(all_possible_pairs))
    negative_pairs = random.sample(all_possible_pairs, n_negative)
    negative_df = generate_pair_features(city_df, negative_pairs, label=0)

    train_df = pd.concat([positive_df, negative_df], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_df


# =============================================================================
# 4. 模型训练与预测
# =============================================================================

def train_model(train_df: pd.DataFrame, use_xgboost: bool = True):
    """
    训练二分类模型。优先使用XGBoost，并自动设置样本平衡权重。
    特征列已更新：新增 economic_gravity, hu_category，移除 hu_weight, distance_bucket。
    """
    feature_cols = [
        "gdp_sum", "pop_product", "distance_km", "admin_level_max",
        "same_agglomeration", "cross_sea", "economic_gravity", "hu_category"
    ]
    X = train_df[feature_cols]
    y = train_df["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = None
    if use_xgboost and XGBOOST_AVAILABLE:
        # 计算正样本权重以处理不平衡
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
            n_jobs=4,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif SKLEARN_AVAILABLE:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=4,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
    else:
        raise RuntimeError("scikit-learn 和 xgboost 均未安装，无法训练模型。")

    # 验证集评估
    if SKLEARN_AVAILABLE:
        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        print(f"[模型验证] AUC-ROC: {auc:.4f}")
        print("[模型验证] Classification Report:")
        print(classification_report(y_val, (val_pred > 0.5).astype(int)))

    return model, feature_cols


def predict_all_pairs(city_df: pd.DataFrame, model, feature_cols: List[str]) -> pd.DataFrame:
    """对所有未连接城市对进行预测，强制包含跨海通道"""
    all_cities = city_df["name"].tolist()
    existing_set = set()
    for a, b in EXISTING_HSR_ROUTES:
        if a in all_cities and b in all_cities:
            existing_set.add(tuple(sorted([a, b])))

    # 构造所有未连接对 + 跨海通道
    candidate_pairs = set()
    for i in range(len(all_cities)):
        for j in range(i + 1, len(all_cities)):
            pair = (all_cities[i], all_cities[j])
            if pair not in existing_set:
                candidate_pairs.add(pair)

    # 确保跨海通道在候选集中
    for pair in CROSS_SEA_PAIRS:
        a, b = pair
        if a in all_cities and b in all_cities:
            if tuple(sorted([a, b])) not in existing_set:
                candidate_pairs.add(pair)

    candidate_pairs = list(candidate_pairs)
    predict_df = generate_pair_features(city_df, candidate_pairs)
    X_pred = predict_df[feature_cols]
    predict_df["prob"] = model.predict_proba(X_pred)[:, 1]
    return predict_df.sort_values("prob", ascending=False).reset_index(drop=True)


# =============================================================================
# 5. 可视化
# =============================================================================

def get_color_by_admin_level(admin_level: str) -> str:
    color_map = {
        "直辖市": "#d62728",
        "特别行政区": "#9467bd",
        "省会/副省级": "#ff7f0e",
        "地级市": "#2ca02c",
        "县级": "#1f77b4",
    }
    return color_map.get(admin_level, "#7f7f7f")


def create_map(
    city_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    top_n: int = 10,
    output_path: str = "china_hsr_prediction_map.html",
    show_existing: bool = True,
):
    """生成交互式Folium地图"""
    if not FOLIUM_AVAILABLE:
        print("[错误] folium 未安装，无法生成地图。请运行: pip install folium")
        return

    m = folium.Map(location=[35.0, 105.0], zoom_start=5, tiles="CartoDB positron")
    city_map = {row["name"]: (row["lat"], row["lon"]) for _, row in city_df.iterrows()}

    # 已有线路
    if show_existing:
        existing_group = folium.FeatureGroup(name="已有高铁线路", show=True)
        drawn_existing = set()
        for a_name, b_name in EXISTING_HSR_ROUTES:
            pair_key = tuple(sorted([a_name, b_name]))
            if pair_key in drawn_existing:
                continue
            drawn_existing.add(pair_key)
            if a_name not in city_map or b_name not in city_map:
                continue
            coords = [city_map[a_name], city_map[b_name]]
            PolyLine(
                locations=coords,
                color="#7f7f7f",
                weight=2,
                opacity=0.5,
                popup=folium.Popup(f"已有线路: {a_name} ↔ {b_name}", max_width=200),
            ).add_to(existing_group)
        existing_group.add_to(m)
        print(f"[地图] 已加载 {len(drawn_existing)} 条已有线路")

    # 城市圆
    max_pop = city_df["population"].max()
    for _, row in city_df.iterrows():
        radius = 3 + (row["population"] / max_pop) * 15
        color = get_color_by_admin_level(row["admin_level"])
        popup_html = f"""
        <b>{row['name']}</b><br>
        省份: {row['province']}<br>
        GDP: {row['gdp']} 亿元<br>
        人口: {row['population']} 万人<br>
        行政级别: {row['admin_level']}<br>
        城市群: {row['urban_agglomeration'] if pd.notna(row['urban_agglomeration']) else '无'}
        """
        CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    # 预测线路
    predict_group = folium.FeatureGroup(name="预测潜在线路", show=True)
    top_predictions = predict_df.head(top_n)

    for _, row in top_predictions.iterrows():
        a_name = row["city_a"]
        b_name = row["city_b"]
        prob = row["prob"]
        is_cross_sea = row["cross_sea"]

        if a_name not in city_map or b_name not in city_map:
            continue

        coords = [city_map[a_name], city_map[b_name]]
        weight = 2 + prob * 8

        if is_cross_sea:
            PolyLine(
                locations=coords,
                color="#e41a1c",
                weight=weight,
                opacity=0.9,
                dash_array="10, 10",
                popup=folium.Popup(
                    f"<b>跨海通道预测</b><br>{a_name} ↔ {b_name}<br>建设概率: {prob:.2%}",
                    max_width=250,
                ),
            ).add_to(predict_group)
        else:
            PolyLine(
                locations=coords,
                color="#377eb8",
                weight=weight,
                opacity=0.7,
                popup=folium.Popup(
                    f"{a_name} ↔ {b_name}<br>建设概率: {prob:.2%}",
                    max_width=250,
                ),
            ).add_to(predict_group)
    predict_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 9999; background-color: white;
                padding: 10px; border: 2px solid grey; border-radius: 5px; font-size: 12px;">
        <b>图例</b><br>
        <i style="background:#d62728;width:12px;height:12px;display:inline-block;"></i> 直辖市<br>
        <i style="background:#9467bd;width:12px;height:12px;display:inline-block;"></i> 特别行政区<br>
        <i style="background:#ff7f0e;width:12px;height:12px;display:inline-block;"></i> 省会/副省级<br>
        <i style="background:#2ca02c;width:12px;height:12px;display:inline-block;"></i> 地级市<br>
        <i style="background:#1f77b4;width:12px;height:12px;display:inline-block;"></i> 县级<br>
        <hr style="margin:5px 0;">
        <span style="color:#7f7f7f;">&#8212;&#8212;</span> 已有高铁线路<br>
        <span style="color:#377eb8;">&#8212;&#8212;</span> 陆地预测线路<br>
        <span style="color:#e41a1c;">- - -</span> 跨海通道预测<br>
        <span style="font-size:10px;color:grey;">线条粗细 ∝ 建设概率</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(output_path)
    print(f"[地图已保存] {output_path}")
    print(f"[预测结果] Top {top_n} 潜在高铁线路（按建设概率排序）：")
    for idx, row in top_predictions.iterrows():
        tag = " [跨海通道]" if row["cross_sea"] else ""
        print(f"  {idx+1}. {row['city_a']} ↔ {row['city_b']} : {row['prob']:.2%}{tag}")


# =============================================================================
# 6. 主流程
# =============================================================================

def main():
    print("=" * 60)
    print("大中华区高铁线路未来预测模型")
    print("=" * 60)

    # 1. 构建城市数据
    city_df = build_city_df(CITIES_DATA)
    print(f"[数据加载] 共加载 {len(city_df)} 个城市")

    # 2. 构建训练数据
    train_df = build_training_data(city_df)
    print(f"[训练数据] 正样本: {(train_df['label']==1).sum()}, 负样本: {(train_df['label']==0).sum()}")

    # 3. 训练模型
    use_xgb = XGBOOST_AVAILABLE
    model, feature_cols = train_model(train_df, use_xgboost=use_xgb)
    print(f"[模型训练] 使用算法: {'XGBoost' if use_xgb else 'RandomForest'}")

    # 4. 预测所有未连接城市对
    predict_df = predict_all_pairs(city_df, model, feature_cols)
    print(f"[预测完成] 共评估 {len(predict_df)} 个城市对")

    # 5. 生成交互式地图
    create_map(city_df, predict_df, top_n=10, output_path="china_hsr_prediction_map_1.html")

    # 6. 跨海通道专项输出
    print("\n[跨海通道专项预测]")
    sea_df = predict_df[predict_df["cross_sea"] == 1].sort_values("prob", ascending=False)
    for _, row in sea_df.iterrows():
        print(f"  {row['city_a']} ↔ {row['city_b']} : {row['prob']:.2%}")

    print("\n[完成] 所有结果已输出。请查看 china_hsr_prediction_map_1.html 文件。")


if __name__ == "__main__":
    main()