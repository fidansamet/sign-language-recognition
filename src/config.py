DATASET_NAME ="MSASL"
RESIZED_NAME ="MSASL_resized"
MSASL_RGB_PATH = "../data/MSASL/rgb"
RESIZED_RGB_PATH = "../data/MSASL_resized/rgb"
MSASL_FLOW_PATH = "../data/MSASL/flow"
TRAIN_JSON_PATH = "../data/MSASL/MSASL_train.json"
VAL_JSON_PATH = "../data/MSASL/MSASL_val.json"
TEST_JSON_PATH = "../data/MSASL/MSASL_test.json"
TRAIN_MODEL_PATH = "model/early_fusion_sgd"
INIT_MODEL_PATH = "model/"
CLASSES = ["hello", "nice", "teacher", "eat", "no", "happy", "like", "orange", "want", "deaf"]
MIN_RESIZE = 256

''' TRAIN PARAMS '''
SPATIAL_IN_CHANNEL = 3
TEMPORAL_IN_CHANNEL = 20
IM_RESIZE = 256
IM_CROP = 224
LEARNING_RATE = 2.5e-5
BATCH_SIZE = 64
EPOCH_COUNT = 20
SAVE_PERIOD_IN_EPOCHS = 5
LOG_STEP = 10
NUM_WORKERS = 8
LATE = 0
EARLY = 1
SPATIAL_FLATTEN = 25088
TEMPORAL_FLATTEN = 32768
PRETRAINED_SPATIAL_PATH = "model/spatial_lr=1.0e-3/model-5.pkl"
PRETRAINED_TEMPORAL_PATH = "model/temporal_lr=1.0e-3/model-5.pkl"
