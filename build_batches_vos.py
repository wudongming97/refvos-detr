import sys
sys.path.append('./external/coco/PythonAPI')
import os
import argparse
import numpy as np
import glob
from PIL import Image

import text_processing



# class DAVIS_MO_Test():
#     # for multi object, do shuffling
#
#     def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
#         self.root = root
#         self.mask_dir = os.path.join(root, 'Annotations', resolution)
#         self.mask480_dir = os.path.join(root, 'Annotations', '480p')
#         self.image_dir = os.path.join(root, 'JPEGImages', resolution)
#         _imset_dir = os.path.join(root, 'ImageSets')
#         _imset_f = os.path.join(_imset_dir, imset)
#
#         self.videos = []
#         self.num_frames = {}
#         self.num_objects = {}
#         self.shape = {}
#         self.size_480p = {}
#         with open(os.path.join(_imset_f), "r") as lines:
#             for line in lines:
#                 _video = line.rstrip('\n')
#                 self.videos.append(_video)
#                 self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
#                 _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
#                 self.num_objects[_video] = np.max(_mask)
#                 self.shape[_video] = np.shape(_mask)
#                 _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
#                 self.size_480p[_video] = np.shape(_mask480)
#
#         self.K = 11
#         self.single_object = single_object
#
#     def __len__(self):
#         return len(self.videos)
#
#     def To_onehot(self, mask):
#         M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
#         for k in range(self.K):
#             M[k] = (mask == k).astype(np.uint8)
#         return M
#
#     def All_to_onehot(self, masks):
#         Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
#         for n in range(masks.shape[0]):
#             Ms[:, n] = self.To_onehot(masks[n])
#         return Ms
#
#     def __getitem__(self, index):
#         video = self.videos[index]
#         info = {}
#         info['name'] = video
#         info['num_frames'] = self.num_frames[video]
#         info['size_480p'] = self.size_480p[video]
#
#         N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32)
#         N_masks = np.empty((self.num_frames[video],) + self.shape[video], dtype=np.uint8)
#         for f in range(self.num_frames[video]):
#             img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
#             N_frames[f] = np.array(Image.open(img_file).convert('RGB')) / 255.
#             try:
#                 mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
#                 N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
#             except:
#                 # print('a')
#                 N_masks[f] = 255
#
#         # Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
#         Fs = np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()
#         Ms = self.All_to_onehot(N_masks).copy()
#         num_objects = int(self.num_objects[video])
#         # if self.single_object:
#         #     N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
#         #     Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
#         #     num_objects = torch.LongTensor([int(1)])
#         #     return Fs, Ms, num_objects, info
#         # else:
#         #     Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
#         #     num_objects = torch.LongTensor([int(self.num_objects[video])])
#         return Fs, Ms, num_objects, info

def To_onehot(mask):
    K = 11
    M = np.zeros((K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for k in range(K):
        M[k] = (mask == k).astype(np.uint8)
    return M

def build_referit_batches(setname, T, input_H, input_W):
    # data directory
    # im_dir = './data/referit/images/'
    # mask_dir = './data/referit/mask/'
    # query_file = './data/referit/referit_query_' + setname + '.json'
    # vocab_file = './data/vocabulary_referit.txt'

    im_dir = './data/DAVIS/JPEGImages/480p/'
    mask_dir = './data/DAVIS/Annotations/480p/'
    query_file = './data/DAVIS/Davis17_annot1_full_video.txt'
    vocab_file = './data/vocabulary_Gref.txt'

    # saving directory
    data_folder = './data/DAVIS/' + setname + '_batch/'
    data_prefix = setname
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    # train or val
    split_dir = './data/DAVIS/ImageSets/2017/'
    split_file = split_dir + setname + '.txt'
    set_content = open(split_file).read().split('\n')

    # load annotations
    # query_dict = json.load(open(query_file))
    query_dict = open(query_file).read().split('\n')
    vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

    # collect training samples
    samples = []
    for n_im, name in enumerate(query_dict[:-1]):
        im_name = name.split(' ', 2)[0]
        if im_name in set_content:
            object_name = name.split(' ', 2)[1]
            mask_name = name.split(' ', 2)[0]
            sent = name.split(' ', 2)[2].split('"')[1]
            # for sent in query_dict[name]:
            samples.append((im_name, mask_name, sent, object_name))

    # save batches to disk
    num_batch = len(samples)
    frame = 0
    for n_batch in range(num_batch):

        print('saving batch %d / %d' % (n_batch + 1, num_batch))
        im_name, mask_name, sent, object_name = samples[n_batch]
        # im = skimage.io.imread(im_dir + im_name)
        # mask = skimage.io.imread(mask_dir + mask_name)
        num_frames = len(glob.glob(os.path.join(im_dir, im_name, '*.jpg')))
        # _mask = np.array(Image.open(os.path.join(mask_dir, mask_name, '00000.png')).convert("P"))
        # num_objects[mask_name] = np.max(_mask)
        # shape[mask_name] = np.shape(_mask)

        # N_frames = np.empty((num_frames[im_name],) + shape[im_name] + (3,), dtype=np.float32)
        # N_masks = np.empty((num_frames[im_name],) + shape[im_name], dtype=np.uint8)

        for f in range(num_frames):
            if f % 1 == 0:
                print('saving frame %d' % (frame))

                img_file = os.path.join(im_dir, im_name, '{:05d}.jpg'.format(f))
                im = np.array(Image.open(img_file).convert('RGB').resize((320,320),Image.BILINEAR), dtype=np.uint8)
                mask_file = os.path.join(mask_dir, mask_name, '{:05d}.png'.format(f))
                mask = np.array(Image.open(mask_file).convert('P').resize((320,320),Image.BILINEAR), dtype=np.uint8)
                mask = To_onehot(mask)

                # if 'train' in setname:
                #     im = skimage.img_as_ubyte(im_processing.resize_and_pad(im, input_H, input_W))
                #     mask = im_processing.resize_and_pad(mask, input_H, input_W)
                # if im.ndim == 2:
                #     im = np.tile(im[:, :, np.newaxis], (1, 1, 3))

                text = text_processing.preprocess_sentence(sent, vocab_dict, T)

                np.savez(file = data_folder + data_prefix + '_' + str(frame) + '.npz',
                    text_batch = text,
                    im_batch = im,
                    mask_batch = mask[int(object_name)],
                    sent_batch = [sent],
                    object_batch = int(object_name),
                    )

                frame += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', type = str, default = 'Gref') # 'unc', 'unc+', 'Gref'
    parser.add_argument('-t', type = str, default = 'val') # 'test', val', 'testA', 'testB'

    args = parser.parse_args()
    T = 20
    # input_H = 320
    # input_W = 320
    input_H = 480
    input_W = 854
    build_referit_batches(setname = args.t,
            T = T, input_H = input_H, input_W = input_W)

