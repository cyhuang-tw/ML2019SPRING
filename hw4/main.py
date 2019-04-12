import sys
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from lime import lime_image
from skimage.segmentation import slic
from skimage.color import gray2rgb, rgb2gray
from util import read_file, MyNet

def compute_saliency_maps(x, y, model):
    model.eval()
    x.requires_grad_()
    y_pred = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliency = x.grad.abs().squeeze().data
    return saliency
#def show_saliency_maps(x, y, model):
def show_saliency_maps(x, y):
    x_org = x.squeeze().numpy()
    saliency = compute_saliency_maps(x, y, model)
    saliency = saliency.detach().cpu().numpy()

    return saliency

def generate_mask(img, saliency_map):
    mask = np.zeros(img.shape)
    for i in range(img.shape[0]):
        cur_map = np.copy(saliency_map[i])
        mean = np.mean(cur_map)
        cur_map[cur_map >= mean] = 1
        cur_map[cur_map < mean] = 0
        mask[i,:,:] = img[i] * cur_map
    return mask

class VisCNN():
    def __init__(self, layer_index, filter_index=0):
        self.model = list(model.children())
        self.layer_index = layer_index
        self.filter_index = filter_index
        self.img = np.random.randn(1, 1, 48, 48).astype(np.float32)
        self.img = torch.from_numpy(self.img)
        self.img = Variable(self.img, requires_grad=True)

    def hook_layer(self):
        def hook_fn(module, input, output):
            self.features = output[0, self.filter_index]
        self.model[self.layer_index].register_forward_hook(hook_fn)

    def visualize_filter(self):
        self.hook_layer()
        optimizer = torch.optim.Adam([self.img], lr=0.1, weight_decay=1e-6)

        for i in range(250):
            optimizer.zero_grad()

            input_img = self.img
            input_img = input_img.cuda()

            for index, layer in enumerate(self.model):
                input_img = layer(input_img)

                if index == self.layer_index:
                    break

            loss = -torch.sum(self.features)
            loss.backward()
            optimizer.step()

            self.result = self.img.cpu().detach().numpy().reshape(48, 48)

        return self.result
    
    def visualze_img(self, img):
        img = torch.Tensor(img).cuda()
        self.hook_layer()

        for index, layer in enumerate(self.model):
            img = layer(img)

            if index == self.layer_index:
                break

        img = img.cpu().detach().numpy()
        img = img[:, :32, :, :].reshape(32, 24, 24)
        return img

def predict(input_img):
    input_img = rgb2gray(input_img).reshape(-1, 48, 48)
    for i in range(input_img.shape[0]):
        mean, std = np.mean(input_img[i]), np.std(input_img[i])
        input_img[i] = (input_img[i] - mean) / std

    input_img = input_img.reshape(-1, 1, 48, 48)
    input_img = torch.Tensor(input_img).cuda()

    output = model(input_img)
    output = output.cpu().detach().numpy()
    return output

def segmentation(input_img):
    return slic(input_img)

def main(data_file, output_path):
    #Load Data
    feature, label = read_file(data_file, train=False)

    selected_index = [22, 9612, 1302, 7, 321, 29, 11]

    input_img = np.zeros((0, 48 * 48))
    input_label = np.zeros((0, 1))
    
    for i in selected_index:
        img = np.copy(feature[i,:])
        mean, std = np.mean(img), np.std(img)
        img = (img - mean) / std
        input_img = np.concatenate((input_img, img.reshape(1, -1)), axis=0)
        input_label = np.concatenate((input_label, label[i].reshape(1, -1)), axis=0)

    input_img = input_img.reshape(-1, 48, 48)
    
    #Saliency Map
    saliency_map = show_saliency_maps(torch.Tensor(input_img.reshape(-1, 1, 48, 48)), torch.LongTensor(input_label).squeeze_())
    mask_img = generate_mask(input_img, saliency_map).reshape(-1, 48, 48)

    for i in range(len(selected_index)):
        plt.axis('off')
        plt.imshow(saliency_map[i], cmap=plt.cm.jet)
        file_name = os.path.join(output_path, 'fig1_' + str(i) + '.jpg')
        plt.savefig(file_name)
        plt.clf()

    #Filter Visualization
    plt.suptitle('Filters for Layer_1')
    for i in range(32):
        vis_tool = VisCNN(layer_index=0, filter_index=i)
        img = vis_tool.visualize_filter().reshape(48, 48)
        plt.axis('off')
        plt.subplot(4, 8, i + 1), plt.imshow(img, 'gray')
    plt.axis('off')
    file_name = os.path.join(output_path, 'fig2_1' + '.jpg')
    plt.savefig(file_name)

    plt.clf()

    plt.suptitle('Filter Outputs for Layer_1')
    vis_tool = VisCNN(layer_index=0)
    result = vis_tool.visualze_img(feature[22,:].reshape(1, 1, 48, 48))
    for i in range(32):
        plt.axis('off')
        plt.subplot(4, 8, i + 1), plt.imshow(result[i], 'gray')
    plt.axis('off')
    file_name = os.path.join(output_path, 'fig2_2' + '.jpg')
    plt.savefig(file_name)

    plt.clf()

    #lime
    explainer = lime_image.LimeImageExplainer()

    for i, index in enumerate(selected_index):
        img = feature[index,:].reshape(48, 48)
        img = gray2rgb(img)
        label = np.asscalar(input_label[i])
        np.random.seed(2539)
        explaination = explainer.explain_instance(
                            image=img, 
                            classifier_fn=predict,
                            segmentation_fn=segmentation
                        )
        image, mask = explaination.get_image_and_mask(
                                label=label,
                                positive_only=False,
                                hide_rest=False,
                                num_features=7,
                                min_weight=0
                            )
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        file_name = os.path.join(output_path, 'fig3_' + str(i) + '.jpg')
        plt.axis('off')
        plt.imshow(image)
        plt.savefig(file_name)
        plt.clf()



if __name__ == '__main__':
    model_file = sys.argv[1]
    model = torch.load(model_file)
    model.eval()
    main(sys.argv[2], sys.argv[3])