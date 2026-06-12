# config.py
import torch


class Config:
    # ================= Paths =================
    ROOT_DIR = "/data/fyf/code/CityScale"
    DATA_DIR = "/data/fyf/data"
    H3_resolution = 8
    CITYNAME = "shenzhen"

    # Input files
    H3_ID_FILE = f"{ROOT_DIR}/h3/{CITYNAME}_h3_{H3_resolution}_only_h3ids.csv"
    SVI_MAPPING_FILE = f"{ROOT_DIR}/h3/mapping_svi/{CITYNAME}_h3_{H3_resolution}_with_svi.csv"
    POI_MAPPING_FILE = f"{ROOT_DIR}/h3/mapping_poi/{CITYNAME}_h3_{H3_resolution}_with_pois.csv"
    POI_TEXT_FILE = f"{ROOT_DIR}/data/POI/{CITYNAME}/poi_text_chinese_{CITYNAME}.csv"
    AE_EMBED_FILE = f"{ROOT_DIR}/data/AE/{CITYNAME}_h3_{H3_resolution}_AE_embeddings.pkl"
    SVI_IMAGE_DIR = f"{DATA_DIR}/SVI/{CITYNAME}/image_normal"

    # Output directory
    PRE_TRAIN_OUTPUT = f"{ROOT_DIR}/pre_train/output/{CITYNAME}"

    # ================= Training hyperparameters =================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BATCH_SIZE = 4
    NUM_WORKERS = 8

    # Learning rates
    LR_ENCODER = 1e-4       # CLIP / text encoder fine-tuning rate
    LR_HEAD = 1e-4          # GAT and projection head rate
    WEIGHT_DECAY = 1e-4
    EPOCHS = 40

    # ================= Sampling limits =================
    MAX_IMAGES_PER_GRID = 16
    MAX_POIS_PER_GRID = 32

    GAT_NEIGHBOR_DEPTH = 2

    # ================= Model structure =================
    CLIP_MODEL_NAME = "/data/fyf/models/clip-vit-base-patch32"
    text_MODEL_NAME = "/data/fyf/models/chinese-macbert-base"

    EMBED_DIM = 512         # common contrastive space dimension
    AE_INPUT_DIM = 64       # AlphaEarth raw dimension
    AE_LIFTING_DIM = 256    # AE lifting dimension (64 -> 256)

    # GAT parameters
    GAT_LAYERS = 2
    GAT_HEADS = 4 
    GAT_DROPOUT = 0.1

    # Contrastive loss weights
    LAMBDA_SVI_POI = 1.0
    LAMBDA_SVI_AE = 1.0
    LAMBDA_POI_AE = 1.0
