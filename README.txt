This program recognizes and classifies MS-ASL dataset videos into one of the 10 classes which are hello, nice, teacher, eat, no, happy, like, orange, want and deaf by using Two-stream Convolutional Neural Networks. To compute optical flows, it uses DualTVL1OpticalFlow_create from opencv_contrib package. The classification can be done by one of 4 models: SpatialNet, TemporalNet, Late Fusion and Early Fusion. It takes 2 arguments: one for model name and one for set name. Model name can be spatial, temporal, late_fusion or early_fusion. Set name can be train which runs the train or test which runs test. To run pre-trained model, PRETRAINED value in main.py file must be 1 and TRAIN_MODEL_PATH value in config.py must be updated according to saved model's path.

To run the program run the following command:

python3 main.py <model_name> <set_name>



Classes:

SpatialDataset: This class is used for SpatialNet data loading. It reads images, resizes, converts and normalizes them so that they are ready for training. Returns transformed image and its label.

TemporalDataset: This class is used for TemporalNet data loading. It reads 10 consecutive frames' optical flows and concatenates them. Centerally crops them so that they are ready for training. Returns stacked optical flows and its label.

BaseModel: This class contains the base network that is used accross all experiments.

FusedModel: This class loads the fused models by loading SpatialNet and TemporalNet with necessary flatten sizes. If pretrained option is selected, loads saved model.

FinalFcLayer: This class is used for early fusion. It concatenates feature vectors of SpatialNet and TemporalNet before the last fully connected layer. After concatenation, gives the concatenated feature vector to the last fc layer and returns the output.


