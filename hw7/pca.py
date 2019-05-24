import os
import sys
import argparse
import numpy as np
import skimage

parser = argparse.ArgumentParser()

parser.add_argument('--pre_computed', default=False, action='store_true')
parser.add_argument('--input_dir', default='./Aberdeen')
parser.add_argument('--input_img', default=None)
parser.add_argument('--output_img', default='./output.jpg')

def PCA(data):
    mean = np.mean(data, axis=0).reshape(-1, 1)
    data = data.T
    print(data.shape, mean.shape)
    data -= mean
    U, s, V = np.linalg.svd(data, full_matrices=False)
    s_sum = np.sum(s)
    eigenfaces = U.T
    return eigenfaces

def reconstruct(input_dir, img_file, eigen_face, mean_face, dim=5):
    img = skimage.io.imread(os.path.join(input_dir, img_file))
    img = np.array(img).astype(np.float32)
    img = img.flatten() - mean_face.flatten()
    p = np.matmul(eigen_face, img.reshape(-1, 1)).astype(np.float32)
    rec_img = np.matmul(eigen_face.T[:, :dim], p[:dim, :]) + mean_face.reshape(-1, 1)
    rec_img = rec_img.reshape(600, 600, 3)
    rec_img = np.around(rec_img)
    rec_img = np.clip(rec_img, 0, 255).astype(np.uint8)
    return rec_img

def compute_eigenface(input_dir):
    file_list = sorted(os.listdir(input_dir))
    file_list = [file for file in file_list if os.path.splitext(file)[-1] == '.jpg' and file.find('_') == -1]
    print('Number of files:', len(file_list))
    img_list = np.zeros((len(file_list), 600 * 600 * 3), dtype=np.float32)
    for i, file in enumerate(file_list):
        img = skimage.io.imread(os.path.join(input_dir, file))
        img = np.array(img).astype(np.float32)
        img_list[i, :] = img.flatten()
    mean_face = np.mean(img_list, axis=0)
    eigen_face = PCA(img_list)

    np.save('eigen_face.npy', eigen_face)
    np.save('mean_face.npy', mean_face)

    return eigen_face, mean_face

def main(pre_computed, input_dir, input_img, output_img):
    if not pre_computed:
        eigen_face, mean_face = compute_eigenface(input_dir)
    else:
        eigen_face = np.load('eigen_face.npy')
        mean_face = np.load('mean_face.npy')
    if input_img != None:
        img = reconstruct(input_dir, input_img, eigen_face, mean_face)
        skimage.io.imsave(output_img, img)

if __name__ == '__main__':
    args = parser.parse_args()
    pre_computed = args.pre_computed
    input_dir = args.input_dir
    input_img = args.input_img
    output_img = args.output_img
    main(pre_computed, input_dir, input_img, output_img)
