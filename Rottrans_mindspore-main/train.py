import random
import numpy as np
import os
import argparse
import mindspore as ms

from mindspore import context
from mindspore import nn, Model
import numpy as np

from config import cfg
from utils.logger import setup_logger



context.set_context(device_target='GPU', save_graphs=False)


def set_seed(seed):
    ms.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)



    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))


    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


rank_id = 0
rank_size = 1

from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID
__factory_T_type = {'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID}




# dataloader
from src.dataset import dataset_creator
num_classes, train_dataset, camera_num, view_num = dataset_creator(
    root=cfg.DATASETS.ROOT_DIR, 
    height=cfg.INPUT.SIZE_TRAIN[0],
    width=cfg.INPUT.SIZE_TRAIN[1], 
    dataset=cfg.DATASETS.NAMES,
    norm_mean=cfg.INPUT.PIXEL_MEAN, 
    norm_std=cfg.INPUT.PIXEL_STD,
    batch_size_train=32, 
    workers=cfg.DATALOADER.NUM_WORKERS,
    mode='train',
    cfg=cfg)

steps_per_epoch = train_dataset.get_dataset_size()




# net
from model import build_transformer_local
net = build_transformer_local(num_classes=num_classes,
                              camera_num=camera_num,
                              view_num=view_num,
                              cfg=cfg,
                              factory=__factory_T_type,
                              rearrange=cfg.MODEL.RE_ARRANGE)



# learning rate
import mindcv.scheduler as scheduler_ms
lr = scheduler_ms.create_scheduler(
    steps_per_epoch = steps_per_epoch,
    scheduler       = "cosine_decay",
    lr              = cfg.SOLVER.BASE_LR,
    min_lr          = 0.002 * cfg.SOLVER.BASE_LR,
    warmup_epochs   = cfg.SOLVER.WARMUP_EPOCHS,
    warmup_factor   = 0.01,
    decay_epochs    = cfg.SOLVER.MAX_EPOCHS - cfg.SOLVER.WARMUP_EPOCHS,
    num_epochs      = cfg.SOLVER.MAX_EPOCHS,
    num_cycles      = 1,
    cycle_decay     = 1.0,
    lr_epoch_stair  = True
)



# loss function
from src.customloss import CustomLoss
lossfunction = CustomLoss(cfg=cfg)



# optimizer
opt = nn.SGD(params=net.trainable_params(),
             learning_rate=lr,
             weight_decay=cfg.SOLVER.WEIGHT_DECAY,
             momentum=cfg.SOLVER.MOMENTUM)



# callback
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor, LossMonitor
checkpoint_out = cfg.OUTPUT_DIR
time_cb = TimeMonitor(data_size=1)
loss_cb = LossMonitor(per_print_times=100)
config_ck = CheckpointConfig(save_checkpoint_steps=steps_per_epoch*100, keep_checkpoint_max=2)
ckpoint_cb = ModelCheckpoint(prefix='RotTrans', directory=checkpoint_out, config=config_ck)
cb = [ckpoint_cb, time_cb, loss_cb]



# train
model = Model(network=net, optimizer=opt, loss_fn=lossfunction)
model.train(epoch=cfg.SOLVER.MAX_EPOCHS, train_dataset=train_dataset, callbacks=cb, dataset_sink_mode=False)


print('Train successfully.')