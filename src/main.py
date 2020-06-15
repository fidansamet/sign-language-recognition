from dataset.compute_flow import save_optical_flow
from dataset.download import download_dataset
import separate
import fused
import sys
import config as cfg

PRETRAINED = 0

if __name__ == '__main__':
    # build dataset
    # download_dataset()
    # save_optical_flow()

    if len(sys.argv) != 3:
        print("Please enter model and set names")
        sys.exit()

    model_name = sys.argv[1]
    set = sys.argv[2]

    for i in ["late_fusion", "early_fusion"]:
        model_name = i
        cfg.TRAIN_MODEL_PATH = "model/" + model_name + "_sgd"

        if model_name in ["spatial", "temporal"]:
            if set == "train":
                separate.train(model_name)
            elif set == "test":
                separate.run_test(model_name)
            else:
                print("Please enter one of the followings to run: train, test")
                sys.exit()

        elif model_name in ["late_fusion", "early_fusion"]:
            if set == "train":
                fused.train(model_name, PRETRAINED)
            elif set == "test":
                fused.run_test(model_name, PRETRAINED)
            else:
                print("Please enter one of the followings to run: train, test")
                sys.exit()

        else:
            print("Please enter one of the followings to run: spatial, temporal, late_fusion, early_fusion")
            sys.exit()
