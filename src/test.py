import torch
from torchvision import transforms
from PIL import Image
import config as cfg
import numpy as np
import os
import utils
from sklearn import metrics


"""
Test kodu. Config dosyasindan ilgili modeli okuyarak verilen resmi test ediyor. Return ettigi ise label bilgisi (0 tabanli).

"""
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    try:
        img = Image.open(image_path)
        image = img.convert('RGB')
        if transform is not None:
            image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        return None


def load_model(cls_count):
    # Build models
    dunham_model = DunhamModel(cls_count, cfg.RESNET).eval()  # eval mode (batchnorm uses moving mean/variance)
    dunham_model = dunham_model.to(device)

    # Load the trained model parameters
    dunham_model.load_state_dict(torch.load(os.path.join(cfg.TEST_MODEL_PATH, cfg.TEST_MODEL_NAME), map_location=lambda storage, loc: storage))

    return dunham_model


def main():
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((cfg.IM_CROP, cfg.IM_CROP)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])

    test_files, test_labels, cls_list = utils.prep_test_data(cfg.TEST_PATH)
    test_labels = list(map(int, test_labels))

    dream_model = load_model(len(cls_list))

    test_res = []

    for im_name in test_files:
        # Prepare an image
        image = load_image(im_name, transform)
        if image is None:
            print(im_name)
            continue

        image_tensor = image.to(device)
        # test image
        predicted_label = dream_model(image_tensor).cpu().data.numpy() # convert to numpy array

        predicted_label_id = np.argmax(predicted_label)
        test_res.append(predicted_label_id)

        print("Image Name: " + im_name + " Test Res: " + cls_list[predicted_label_id])
        print(predicted_label)

    # burada accuracy hesabi vs eklenebilir. eger coklu resim test ederseniz.
    # dogru tahmin edilenleri ve yanlis edilenleri saydirip basit bir accuracy = TP / (TP + FP)

    score = metrics.accuracy_score(test_labels, test_res)
    cls_report = metrics.classification_report(test_labels, test_res)
    conf_mat = metrics.confusion_matrix(test_labels, test_res)

    print("Accuracy = " + str(score))
    print(cls_report)
    print(conf_mat)


if __name__ == '__main__':
    main()
