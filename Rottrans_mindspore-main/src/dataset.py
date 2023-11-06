import os
import os.path as osp
import copy
import random
from collections import defaultdict
from PIL import Image
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms.transforms import Compose

from .datasets_define import PRAI1581, init_img_dataset
from .transforms import build_train_transforms


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))

    return img


def init_dataset(name, **kwargs):
    """Initializes an image dataset."""
    all_image_datasets = {
        'prai1581': PRAI1581,
    }
    avai_datasets = list(all_image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return all_image_datasets[name](**kwargs)


class RandomIdentitySampler(ds.Sampler):  # torch original
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        super().__init__()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)  # dict with list value
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[int(pid)].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length




# train dataset
def dataset_creator(
        root='',
        dataset=None,
        height=256,
        width=128,
        norm_mean=None,
        norm_std=None,
        batch_size_train=32,
        batch_size_test=32,
        workers=1,
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        mode='train',
        cfg=None
):
    '''
    create and preprocess data for train and evaluate
    '''
    if dataset is None:
        raise ValueError('dataset must not be None')
    if dataset == 'cuhk03':
        dataset_ = init_dataset(
            name=dataset,
            root=root,
            mode=mode,
            cuhk03_labeled=cuhk03_labeled,
            cuhk03_classic_split=cuhk03_classic_split
        )
    else:
        dataset_ = init_dataset(
            name=dataset,
            root=root,
            mode=mode,
        )

    num_pids = dataset_.num_train_pids
    cam_num = dataset_.num_train_cams
    view_num = dataset_.num_datasets


    # train dataset
    if mode == 'train':
        device_num, rank_id = _get_rank_info()

        if cfg.DATALOADER.SAMPLER == 'softmax_triplet':
            sampler = RandomIdentitySampler(
                dataset_, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
        else:
            sampler = ds.RandomSampler()

        device_num, rank_id = _get_rank_info()
        if isinstance(device_num, int) and device_num > 1:
            data_set = ds.GeneratorDataset(
                dataset_, 
                ['img', 'pid'],
                sampler=sampler, 
                num_parallel_workers=workers,
                num_shards=device_num, 
                shard_id=rank_id, 
                shuffle=True)
        else:
            data_set = ds.GeneratorDataset(
                dataset_, 
                ['img', 'pid'],
                sampler=sampler, 
                num_parallel_workers=workers)
        
        transforms = build_train_transforms(height=height, 
                                            width=width, 
                                            norm_mean=norm_mean, 
                                            norm_std=norm_std)
        data_set = data_set.map(operations=transforms, input_columns=['img'])
        data_set = data_set.batch(batch_size=batch_size_train, drop_remainder=True)
        return num_pids, data_set, cam_num, view_num
    
    else:
        print('Wrong mode.')
        



class ImageDataset():
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        return img, pid, camid



def create_eval_dataset(real_path, cfg):
    dataset = init_img_dataset(root=real_path, name=cfg.DATASETS.FOLDERNAMES)

    transform_test = [
        vision.Resize((cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1])),
        vision.ToTensor(),
        vision.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, is_hwc=False)
    ]
    transform_test = Compose(transform_test)

    val_loader = ds.GeneratorDataset(
        source=ImageDataset(dataset.query + dataset.gallery),
        column_names=["img", "pid", "camid"],
        shuffle=False,
        num_parallel_workers=cfg.DATALOADER.NUM_WORKERS
    )

    val_loader = val_loader.map(input_columns=['img'], operations=transform_test)
    val_loader = val_loader.batch(batch_size=32, drop_remainder=False)

    num_query = len(dataset.query)

    return val_loader, num_query, dataset.num_train_pids, dataset.num_train_cams






def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
