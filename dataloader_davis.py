import random
import glob
import os
import json
from collections import OrderedDict
import numpy as np

from PIL import Image
import torch
import torchvision as tv

import dataset_utils
import utils
import text_processing


def get_sample_bernoulli(p):
    return (lambda lst: [elem for elem in lst if random.random() < p])


def get_sample_all():
    return (lambda lst: lst)


def get_sample_k_random(k):
    return (lambda lst: sorted(random.sample(lst, min(k, len(lst)))))

def To_onehot(mask, object_name):
    M = torch.zeros((mask.shape[0], 1, mask.shape[2], mask.shape[3]), dtype=torch.int)
    for i in range(mask.shape[0]):
        M[i] = (mask[i] == object_name).int()
    return M

def To_allhot(mask):
    M = torch.zeros((mask.shape[0], 1, mask.shape[2], mask.shape[3]), dtype=torch.int)
    for i in range(mask.shape[0]):
        M[i] = (mask[i] > 0).int()
    return M


def get_anno_ids(anno_path, pic_to_tensor_function, threshold):
    pic = Image.open(anno_path)
    tensor = pic_to_tensor_function(pic)
    values = (tensor.view(-1).bincount() > threshold).nonzero().view(-1).tolist()
    if 0 in values: values.remove(0)
    if 255 in values: values.remove(255)
    return values


def get_default_image_read(size=(240, 432)):
    def image_read(path):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [tv.transforms.Resize(size, interpolation=Image.BILINEAR),
             tv.transforms.ToTensor(),
             tv.transforms.Normalize(mean=dataset_utils.IMAGENET_MEAN, std=dataset_utils.IMAGENET_STD)])
        return transform(pic)

    return image_read


def get_default_anno_read(size=(240, 432)):
    def label_read(path):
        if os.path.exists(path):
            pic = Image.open(path)
            transform = tv.transforms.Compose(
                [tv.transforms.Resize(size, interpolation=Image.NEAREST),
                 dataset_utils.LabelToLongTensor()])
            label = transform(pic)
        else:
            label = torch.LongTensor(1, *size).fill_(255)  # Put label that will be ignored
        return label

    return label_read


class DAVIS17V2(torch.utils.data.Dataset):
    def __init__(self, root_path, version, image_set, image_read=get_default_image_read(),
                 anno_read=get_default_anno_read(),
                 joint_transform=None, samplelen=4, obj_selection=get_sample_all(), min_num_obj=1,
                 start_frame='random'):
        self._min_num_objects = min_num_obj
        self._root_path = root_path
        self._version = version
        self._image_set = image_set
        self._image_read = image_read
        self._anno_read = anno_read
        self._joint_transform = joint_transform
        self._seqlen = samplelen
        self._obj_selection = obj_selection
        self._start_frame = start_frame
        assert version in ('2016', '2017')
        assert image_set in ('train', 'val', 'test-dev', 'test-challenge')
        #        assert samplelen > 1, "samplelen must be at least 2"
        assert start_frame in ('random', 'first')
        self._init_data()

    def _init_data(self):
        """ Store some metadata that needs to be known during training. In order to sample, the viable sequences
        must be known. Sequences are viable if a snippet of given sample length can be selected, starting with
        an annotated frame and containing at least one more annotated frame.
        """
        print("-- DAVIS17 dataset initialization started.")
        framework_path = os.path.join(os.path.dirname(__file__), '..')
        cache_path = os.path.join(framework_path, 'cache', 'davis17_v2_visible_objects_100px_threshold.json')

        # First find visible objects in all annotated frames
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._visible_objects = json.load(f)
                self._visible_objects = {seqname: OrderedDict((int(idx), objlst) for idx, objlst in val.items())
                                         for seqname, val in self._visible_objects.items()}
            print("Datafile {} loaded, describing {} sequences.".format(cache_path, len(self._visible_objects)))
        else:
            # Grab all sequences in dataset
            seqnames = os.listdir(os.path.join(self._root_path, 'JPEGImages', '480p'))

            # Construct meta-info
            self._visible_objects = {}
            for seqname in seqnames:
                anno_paths = sorted(glob.glob(self._full_anno_path(seqname, '*.png')))
                self._visible_objects[seqname] = OrderedDict(
                    (self._frame_name_to_idx(os.path.basename(path)),
                     get_anno_ids(path, dataset_utils.LabelToLongTensor(), 100))
                    for path in anno_paths)

            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            with open(cache_path, 'w') as f:
                json.dump(self._visible_objects, f)
            print("Datafile {} was not found, creating it with {} sequences.".format(cache_path,
                                                                                     len(self._visible_objects)))

        # Find sequences in the requested image_set
        with open(os.path.join(self._root_path, 'ImageSets', self._version, self._image_set + '.txt'), 'r') as f:
            self._all_seqs = f.read().splitlines()
            print("{} sequences found in image set \"{}\"".format(len(self._all_seqs), self._image_set))

        # Filter out sequences that are too short from first frame with object, to last annotation
        self._nonempty_frame_ids = {
            seq: [frame_idx for frame_idx, obj_ids in lst.items() if len(obj_ids) >= self._min_num_objects]
            for seq, lst in self._visible_objects.items()}
        self._viable_seqs = [seq for seq in self._all_seqs if
                             len(self._nonempty_frame_ids[seq]) > 0
                             and len(self.get_image_frame_ids(seq)[min(self._nonempty_frame_ids[seq]):
                                                                   max(self._visible_objects[seq].keys()) + 1])
                             >= self._seqlen]
        print(
            "{} sequences remaining after filtering on length (from first anno obj appearance to last anno frame.".format(
                len(self._viable_seqs)))

        # Find querys in the requested image_set
        query_file = '../data/DAVIS/Davis17_annot1_full_video.txt'
        self._query_seqs = []
        self._query = open(query_file).read().splitlines()
        for query in self._query:
            if query.split(' ')[0] in self._all_seqs:
                self._query_seqs.append(query)
        print(
            "{} querys remaining found in image set \"{}\"".format(len(self._query_seqs), self._image_set))

        vocab_file = '../data/vocabulary_Gref.txt'
        self.vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

    # def __len__(self):
    #     return len(self._viable_seqs)
    def __len__(self):
        return len(self._query_seqs)

    def _frame_idx_to_image_fname(self, idx):
        return "{:05d}.jpg".format(idx)

    def _frame_idx_to_anno_fname(self, idx):
        return "{:05d}.png".format(idx)

    def _frame_name_to_idx(self, fname):
        return int(os.path.splitext(fname)[0])

    def get_viable_seqnames(self):
        return self._viable_seqs

    def get_all_seqnames(self):
        return self._all_seqs

    def get_anno_frame_names(self, seqname):
        return os.listdir(os.path.join(self._root_path, "Annotations", "480p", seqname))

    def get_anno_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_anno_frame_names(seqname)])

    def get_image_frame_names(self, seqname):
        return os.listdir(os.path.join(self._root_path, "JPEGImages", "480p", seqname))

    def get_image_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_image_frame_names(seqname)])

    def get_frame_ids(self, seqname):
        return sorted([self._frame_name_to_idx(fname) for fname in self.get_image_frame_names(seqname)])

    def get_nonempty_frame_ids(self, seqname):
        return self._nonempty_frame_ids[seqname]

    def _full_image_path(self, seqname, image):
        if isinstance(image, int):
            image = self._frame_idx_to_image_fname(image)
        return os.path.join(self._root_path, 'JPEGImages', "480p", seqname, image)

    def _full_anno_path(self, seqname, anno):
        if isinstance(anno, int):
            anno = self._frame_idx_to_anno_fname(anno)
        return os.path.join(self._root_path, 'Annotations', "480p", seqname, anno)

    def _select_frame_ids(self, frame_ids, viable_starting_frame_ids):
        if self._start_frame == 'first':
            frame_idxidx = frame_ids.index(viable_starting_frame_ids[0])
            return frame_ids[frame_idxidx: frame_idxidx + self._seqlen]
        if self._start_frame == 'random':
            frame_idxidx = frame_ids.index(random.choice(viable_starting_frame_ids))
            return frame_ids[frame_idxidx: frame_idxidx + self._seqlen]

    def _select_object_ids(self, labels):
        assert labels.min() > -1 and labels.max() < 256, "{}".format(utils.print_tensor_statistics(labels))
        possible_obj_ids = (labels[0].view(-1).bincount() > 25).nonzero().view(-1).tolist()
        if 0 in possible_obj_ids: possible_obj_ids.remove(0)
        if 255 in possible_obj_ids: possible_obj_ids.remove(255)
        assert len(possible_obj_ids) > 0

        obj_ids = self._obj_selection(possible_obj_ids)
        bg_ids = (labels.view(-1).bincount() > 0).nonzero().view(-1).tolist()
        if 0 in bg_ids: bg_ids.remove(0)
        if 255 in bg_ids: bg_ids.remove(255)
        for idx in obj_ids:
            bg_ids.remove(idx)

        for idx in bg_ids:
            labels[labels == idx] = 0
        for new_idx, old_idx in zip(range(1, len(obj_ids) + 1), obj_ids):
            labels[labels == old_idx] = new_idx
        return labels



    def __getitem__(self, idx):
        """
        returns:
            dict (Tensors): contains 'images', 'given_segmentations', 'labels'
        """

        query = self._query_seqs[idx]

        #        assert self._version == '2017', "Only the 2017 version is supported for training as of now"
        seqname = query.split(' ', 2)[0]



        # We require to begin with a nonempty frame, and will consider all objects in that frame to be tracked.
        # A starting frame is valid if it is followed by seqlen-1 frames with corresp images
        frame_ids = self.get_frame_ids(seqname)
        viable_starting_frame_ids = [idx for idx in self.get_nonempty_frame_ids(seqname)
                                     if idx <= frame_ids[-self._seqlen]]

        frame_ids = self._select_frame_ids(frame_ids, viable_starting_frame_ids)

        images = torch.stack([self._image_read(self._full_image_path(seqname, idx))
                              for idx in frame_ids])
        segannos = torch.stack([self._anno_read(self._full_anno_path(seqname, idx))
                                for idx in frame_ids])

        if self._joint_transform is not None:
            images, segannos = self._joint_transform(images, segannos)


        object_name = int(query.split(' ', 2)[1])
        segannos_one = To_onehot(segannos, object_name)

        segannos_all = To_allhot(segannos)

        sentence = query.split(' ', 2)[2].split('"')[1]
        txt = np.array(text_processing.preprocess_sentence(sentence, self.vocab_dict, 20))
        # txt = np.tile(txt, self._seqlen).reshape(self._seqlen, -1)

        # return {'images': images, 'provides_seganno': provides_seganno, 'given_seganno': given_seganno,
        #         'segannos': segannos, 'sentence': sentence}

        return {'images': images, 'segannos': segannos_one, 'segannos_all': segannos_all, 'sentence': txt}

    def _get_snippet(self, seqname, object_name, txt, frame_ids):
        images = torch.stack([self._image_read(self._full_image_path(seqname, idx))
                              for idx in frame_ids])
        if self._image_set in ('test-dev', 'test-challenge'):
            segannos = None
            given_segannos = [self._anno_read(self._full_anno_path(seqname, idx)).unsqueeze(0)
                              if idx in anno_frame_ids else None for idx in frame_ids]
        else:
            # segannos = torch.stack([self._anno_read(self._full_anno_path(seqname, idx))
            #                         for idx in frame_ids]).squeeze().unsqueeze(0)
            segannos = torch.stack([self._anno_read(self._full_anno_path(seqname, idx))
                                    for idx in frame_ids])
            if self._version == '2016':
                segannos = (segannos != 0).long()
            given_segannos = [self._anno_read(self._full_anno_path(seqname, idx)).unsqueeze(0)
                              if idx == self.get_anno_frame_ids(seqname)[0] else None for idx in frame_ids]
        for i in range(len(given_segannos)):  # Remove dont-care from given segannos
            if given_segannos[i] is not None:
                given_segannos[i][given_segannos[i] == 255] = 0
                if self._version == '2016':
                    given_segannos[i] = (given_segannos[i] != 0).long()

        segannos = To_onehot(segannos, object_name)

        # txt = np.tile(txt, segannos.shape[0]).reshape(segannos.shape[0], -1)

        fnames = [self._frame_idx_to_anno_fname(idx) for idx in frame_ids]
        return {'images': images, 'segannos': segannos, 'sentence': txt, 'object_id': object_name, 'fnames': fnames}

    def _get_video(self, seqname, object_name, txt):
        seq_frame_ids = self.get_frame_ids(seqname)
        partitioned_frame_ids = [seq_frame_ids[start_idx: start_idx + self._seqlen]
                                 for start_idx in range(0, len(seq_frame_ids), self._seqlen)]
        for frame_ids in partitioned_frame_ids:
            yield self._get_snippet(seqname, object_name, txt, frame_ids)

    def get_video_generator(self):

        for query in self._query_seqs:
            seqname = query.split(' ', 2)[0]
            object_name = int(query.split(' ', 2)[1])

            sentence = query.split(' ', 2)[2].split('"')[1]
            txt = np.array(text_processing.preprocess_sentence(sentence, self.vocab_dict, 20))

            if seqname in self._all_seqs:
                yield (seqname, self._get_video(seqname, object_name, txt))
