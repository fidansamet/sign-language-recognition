DATASET_NAME ="MSASL"
RESIZED_NAME ="MSASL_resized"
MSASL_RGB_PATH = "../data/MSASL/rgb"
RESIZED_RGB_PATH = "../data/MSASL_resized/rgb"
MSASL_FLOW_PATH = "../data/MSASL/flow"
TRAIN_JSON_PATH = "../data/MSASL/MSASL_train.json"
VAL_JSON_PATH = "../data/MSASL/MSASL_val.json"
TEST_JSON_PATH = "../data/MSASL/MSASL_test.json"
TRAIN_MODEL_PATH = "model/temporal_lr=1.0e-3"
INIT_MODEL_PATH = "model/"
CLASSES = ["hello", "nice", "teacher", "eat", "no", "happy", "like", "orange", "want", "deaf"]
MIN_RESIZE = 256

''' TRAIN PARAMS '''
SPATIAL_IN_CHANNEL = 3
TEMPORAL_IN_CHANNEL = 20
IM_RESIZE = 256
IM_CROP = 224
LEARNING_RATE = 1.0e-3    #  ogrenme orani, eger loss azalmazsa bunu e-5 ya da e-6 ya dusurebilirsiniz.
BATCH_SIZE = 64          #   her seferde kac adet resmi isleyecek model. eger train i calistirirken memory hatasi alirsaniz bunu azaltmaniz lazim.
EPOCH_COUNT = 100       #     tum veri kumesi uzerinden kac kez gecilecek. baktiniz loss cok dusmemeye basladiysa 40-50 epoch sonra train i durdurabilirsiniz.
SAVE_PERIOD_IN_EPOCHS = 5  # kac epoch da save edilecek model.
LOG_STEP = 10  # kac adimda ekrana log bilgisi basilacak.
NUM_WORKERS = 8  # data loader icin worker sayisi, core sayiniz az ise bunu azaltabilirsiniz.
LOAD_TRAINED_MODEL = 0  # train asamasinda eski bir model yuklemek isterseniz bunu 1 yapin.
LOAD_MODEL_NAME = "base_model-100.pkl" # LOAD_TRAINED_MODEL 1 olursa load edilecek model adi.
LATE = 0
EARLY = 1
SPATIAL_FLATTEN = 25088
TEMPORAL_FLATTEN = 32768
PRETRAINED_SPATIAL_PATH = "model/spatial_1/spatial_model-55.pkl"
PRETRAINED_TEMPORAL_PATH = "model/temporal_1/spatial_model-5.pkl"
