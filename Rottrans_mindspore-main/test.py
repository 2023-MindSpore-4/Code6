import os
import logging
import argparse

import mindspore as ms
import mindspore.ops as ops

from config import cfg
from utils.logger import setup_logger
from model import build_transformer_local
from src.metrics import R1_mAP_eval
from src.dataset import create_eval_dataset
from model.backbones.vit_pytorch import vit_base_patch16_224_TransReID

__factory_T_type = {'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID}



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    model.set_train(False)


    for n_iter, (img, pid, camid) in enumerate(val_loader):
        img = ops.stop_gradient(img)
        pid = ops.stop_gradient(pid)
        camid = ops.stop_gradient(camid)
        feat = ops.stop_gradient(model(img))
        
        evaluator.update((feat, pid, camid))

    cmc, mAP, minp , _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.2%}".format(r, cmc[r - 1]))
    logger.info("minP: {:.2%}".format(minp))
    return cmc[0], cmc[4]







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    test_path = cfg.DATASETS.ROOT_DIR
    val_loader, num_query, num_train_pids, num_train_cams = create_eval_dataset(test_path, cfg)

    net = build_transformer_local(num_classes=num_train_pids,
                                      camera_num=num_train_cams,
                                      view_num=0,
                                      cfg=cfg,
                                      factory=__factory_T_type,
                                      rearrange=cfg.MODEL.RE_ARRANGE)

    cp = cfg.TEST.WEIGHT
    param_dict = ms.load_checkpoint(cp, filter_prefix='classifier')

    # fix bugs of ckpt
    def prepend_prefix_to_numeric_keys(original_dict):
        updated_dict = {}
        for key, value in original_dict.items():
            if key[0].isdigit():
                new_key = 'base.blocks.' + key
            else:
                new_key = key
            updated_dict[new_key] = value
        return updated_dict
        
    param_dict = prepend_prefix_to_numeric_keys(param_dict)


    params_not_loaded = ms.load_param_into_net(net, param_dict, strict_load=True)
    

    print('Eval:', cp)
    do_inference(cfg,
                 net,
                 val_loader,
                 num_query)