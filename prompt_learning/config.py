import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch


# Sampling weights for WeightedRandomSampler: coarser scales get higher weights.
_SCALE_WEIGHTS = {
    "street_L1": 5.0, "street_L2": 4.0, "street_L3": 3.0,
    "street_L4": 2.0, "street_L5": 1.0, "street_L11": 5.0,
    "grid_1km": 2.0, "grid_3km": 3.0, "grid_5km": 5.0,
}
_TASK_FAMILIES = ["population", "PM25", "CO", "CO2", "GDP", "HFP_tx", "HP", "bike"]
_SAMPLE_WEIGHTS = {
    f"{fam}_{scale}": w
    for fam in _TASK_FAMILIES
    for scale, w in _SCALE_WEIGHTS.items()
}


class Config:
    # ---- Paths ----
    ROOT_DIR = "/data/fyf/code/CityScale"
    CITYNAME = "shenzhen"

    # Pretrained embedding; must match the one used to train the downstream model.
    EMBEDDING_PATH = os.path.join(ROOT_DIR, "pre_train/output/shenzhen/shenzhen_embedding.pkl")

    # Prompt text encoder
    BERT_PATH = "/data/fyf/models/bge-base-zh-v1.5"

    # Spatial mapping cache
    MAPPING_CACHE_DIR = os.path.join(ROOT_DIR, "cache")
    SPATIAL_MAPPING_CACHE_FILE = os.path.join(MAPPING_CACHE_DIR, "spatial_mappings_cache.pkl")

    # City boundary for visualization (optional; skipped if missing).
    BOUNDARY_PATH = os.path.join(ROOT_DIR, f"boundary/WGS84/boundaries/{CITYNAME}_wgs84.geojson")

    OUTPUT_DIR = os.path.join(ROOT_DIR, f"prompt_learning/output/train/{CITYNAME}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Split and sampling ----
    SPLIT_SEED = 2026
    K_FOLDS = 5
    TRAIN_RATIO = 0.7

    # Base scale for train/validation partitioning ('h3', 'L1'..'L5', 'L11', '1km', '3km', '5km').
    BASE_SCALE = 'L11'

    SAMPLE_WEIGHTS = _SAMPLE_WEIGHTS
    GRID_THRESHOLD = 0.5

    # ---- Model hyperparameters ----
    RAW_INPUT_DIM = 512
    HIDDEN_DIM = 256
    PROMPT_EMB_DIM = 768
    OUTPUT_DIM = 1

    DROPOUT = 0.3
    GPU_ID = 0
    DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"

    # ---- Training parameters ----
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    WEIGHT_DECAY = 1e-3
    LR_PATIENCE = 5
    LR_FACTOR = 0.5

    # ---- Prompt component library ----
    SCALE_MAP = {
        "h3": {"cn": "尺度：H3六边形规则网格，面积约0.7平方公里，精细尺度的基础空间单元，适合捕捉局部微观特征", "en": "Scale: H3 hexagonal regular grid, approximately 0.7 square kilometers, fine-scale basic spatial unit suitable for capturing local micro-level features"},
        "1km": {"cn": "尺度：1公里规则网格，精细尺度的空间单元，适合分析街区级别的空间异质性", "en": "Scale: 1km regular grid, fine-scale spatial unit suitable for analyzing neighborhood-level spatial heterogeneity"},
        "2km": {"cn": "尺度：2公里规则网格，精细至中等尺度的空间单元，适合分析街区组团级别的空间异质性", "en": "Scale: 2km regular grid, fine-to-medium scale spatial unit suitable for analyzing neighborhood-cluster-level spatial heterogeneity"},
        "3km": {"cn": "尺度：3公里规则网格，中等尺度的空间单元，适合分析社区级别的空间模式", "en": "Scale: 3km regular grid, medium-scale spatial unit suitable for analyzing community-level spatial patterns"},
        "4km": {"cn": "尺度：4公里规则网格，中等至较大尺度的空间单元，适合分析社区组团级别的空间模式", "en": "Scale: 4km regular grid, medium-to-large scale spatial unit suitable for analyzing community-cluster-level spatial patterns"},
        "5km": {"cn": "尺度：5公里规则网格，较大尺度的空间单元，适合分析城市功能区级别的宏观格局", "en": "Scale: 5km regular grid, larger-scale spatial unit suitable for analyzing urban functional zone-level macro patterns"},
        "street_normal": {"cn": "尺度：行政街道不规则划分，基于行政管理边界的空间单元", "en": "Scale: Administrative street irregular division, spatial units based on administrative management boundaries"},
        "L1": {"cn": "尺度：A级路网划分的大型城市功能区，面积约10至50平方公里，由高速公路和主干道围合形成", "en": "Scale: Large urban functional zones divided by A-level road network, approximately 10-50 square kilometers, enclosed by highways and arterial roads"},
        "L2": {"cn": "尺度：B级路网划分的中型城市片区，面积约3至10平方公里，由主干道和次干道围合形成", "en": "Scale: Medium urban districts divided by B-level road network, approximately 3-10 square kilometers, enclosed by arterial and secondary roads"},
        "L3": {"cn": "尺度：C级路网划分的城市组团，面积约1至3平方公里，由次干道和支路围合形成", "en": "Scale: Urban clusters divided by C-level road network, approximately 1-3 square kilometers, enclosed by secondary roads and branch roads"},
        "L4": {"cn": "尺度：D级路网划分的城市街区，面积约0.3至1平方公里，由支路和地方道路围合形成", "en": "Scale: Urban blocks divided by D-level road network, approximately 0.3-1 square kilometers, enclosed by branch roads and local roads"},
        "L5": {"cn": "尺度：E级路网划分的微观地块，面积约0.05至0.3平方公里，由地方道路和小路围合形成的最小路网单元", "en": "Scale: Micro parcels divided by E-level road network, approximately 0.05-0.3 square kilometers, the smallest road network units enclosed by local roads and paths"},
        "L11": {"cn": "尺度：一级社区行政边界划分，面积较大的基层行政管理单元，对应街道办事处级别", "en": "Scale: Level-1 community administrative boundary division, larger grassroots administrative units corresponding to sub-district office level"},
    }

    # Task descriptions
    TASK_TYPE_MAP = {
        "population": {"cn": "任务：预测区域常住人口数量，反映城市人口空间分布密度，与住宅用地和公共服务设施密切相关", "en": "Task: Predict regional resident population count, reflecting urban population spatial density, closely related to residential land use and public service facilities"},
        "PM2.5": {"cn": "任务：预测细颗粒物PM2.5的空间浓度分布，该指标反映大气污染程度，与交通排放、工业活动和气象条件密切相关", "en": "Task: Predict spatial concentration of fine particulate matter PM2.5, reflecting air pollution level, closely related to traffic emissions, industrial activities and meteorological conditions"},
        "CO": {"cn": "任务：预测一氧化碳CO的空间浓度分布，该指标主要来源于不完全燃烧过程，与机动车密度和道路交通强度相关", "en": "Task: Predict spatial concentration of carbon monoxide CO, mainly from incomplete combustion, related to vehicle density and road traffic intensity"},
        "CO2": {"cn": "任务：预测二氧化碳CO2的空间浓度分布，该指标反映区域碳排放强度，与能源消耗、交通和建筑密度相关", "en": "Task: Predict spatial concentration of carbon dioxide CO2, reflecting regional carbon emission intensity, related to energy consumption, traffic and building density"},
        "GDP": {"cn": "任务：预测区域生产总值GDP，反映经济活动的空间集聚程度，与商业用地、就业密度和基础设施水平相关", "en": "Task: Predict regional gross domestic product GDP, reflecting spatial agglomeration of economic activities, related to commercial land use, employment density and infrastructure level"},
        "HFP_tx": {"cn": "任务：预测腾讯人类足迹指数，基于移动设备定位数据反映人类活动的时空分布强度", "en": "Task: Predict Tencent Human Footprint Index, reflecting spatiotemporal distribution intensity of human activities based on mobile device location data"},
        "HP": {"cn": "任务：预测区域房价水平，反映城市土地价值和居住吸引力的空间分异，与交通可达性、公共服务和环境品质相关", "en": "Task: Predict regional housing price level, reflecting spatial differentiation of urban land value and residential attractiveness, related to transportation accessibility, public services and environmental quality"},
        "bike": {"cn": "任务：预测区域共享单车骑行量，反映短距离出行需求和慢行交通活力的空间分布，与地铁站点、商业区和居住区密切相关", "en": "Task: Predict regional shared bike ridership, reflecting short-distance travel demand and active transportation vitality, closely related to metro stations, commercial and residential areas"},
    }

    # ---- Active tasks ----
    ACTIVE_TASKS = [
        # ---------- population ----------
        "population_street_L1",
        "population_street_L2",
        "population_street_L3",
        "population_street_L4",
        "population_street_L5",
        "population_street_L11",
        "population_grid_1km",
        "population_grid_3km",
        "population_grid_5km",

        # ---------- PM2.5 ----------
        "PM25_street_L1",
        "PM25_street_L2",
        "PM25_street_L3",
        "PM25_street_L4",
        "PM25_street_L5",
        "PM25_street_L11",
        "PM25_grid_1km",
        "PM25_grid_3km",
        "PM25_grid_5km",

        # ---------- CO ----------
        "CO_street_L1",
        "CO_street_L2",
        "CO_street_L3",
        "CO_street_L4",
        "CO_street_L5",
        "CO_street_L11",
        "CO_grid_1km",
        "CO_grid_3km",
        "CO_grid_5km",

        # ---------- CO2 ----------
        "CO2_street_L1",
        "CO2_street_L2",
        "CO2_street_L3",
        "CO2_street_L4",
        "CO2_street_L5",
        "CO2_street_L11",
        "CO2_grid_1km",
        "CO2_grid_3km",
        "CO2_grid_5km",

        # ---------- GDP ----------
        "GDP_street_L1",
        "GDP_street_L2",
        "GDP_street_L3",
        "GDP_street_L4",
        "GDP_street_L5",
        "GDP_street_L11",
        "GDP_grid_1km",
        "GDP_grid_3km",
        "GDP_grid_5km",

        # ---------- HFP_tx (Tencent Human Footprint) ----------
        "HFP_tx_street_L1",
        "HFP_tx_street_L2",
        "HFP_tx_street_L3",
        "HFP_tx_street_L4",
        "HFP_tx_street_L5",
        "HFP_tx_street_L11",
        "HFP_tx_grid_1km",
        "HFP_tx_grid_3km",
        "HFP_tx_grid_5km",

        # ---------- HP (housing price) ----------
        "HP_street_L1",
        "HP_street_L2",
        "HP_street_L3",
        "HP_street_L4",
        "HP_street_L5",
        "HP_street_L11",
        "HP_grid_1km",
        "HP_grid_3km",
        "HP_grid_5km",

        # ---------- bike (shared bike ridership) ----------
        "bike_street_L1",
        "bike_street_L2",
        "bike_street_L3",
        "bike_street_L4",
        "bike_street_L5",
        "bike_street_L11",
        "bike_grid_1km",
        "bike_grid_3km",
        "bike_grid_5km",
    ]
