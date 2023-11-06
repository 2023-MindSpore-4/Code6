import mindspore as ms
import mindspore.ops as P
import mindspore.dataset.vision as vision
import mindspore.common.initializer as init
from mindspore.common.initializer import HeNormal, Normal, Constant
import mindspore.nn as nn
import copy




def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Dense') != -1:
        m.weight.set_data(init.initializer(HeNormal(mode='fan_out'), m.weight.shape, m.weight.dtype))
        m.bias.set_data(init.initializer(Constant(0.0), m.bias.shape, m.bias.dtype))
    elif classname.find('Conv') != -1:
        m.weight.set_data(init.initializer(HeNormal(mode='fan_in'), m.weight.shape, m.weight.dtype))
        if m.bias is not None:
            m.bias.set_data(init.initializer(Constant(0.0), m.bias.shape, m.bias.dtype))
    elif classname.find('BatchNorm') != -1:
        m.gamma.set_data(init.initializer(Constant(1.0), m.gamma.shape, m.gamma.dtype))
        m.beta.set_data(init.initializer(Constant(0.0), m.beta.shape, m.beta.dtype))



def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Dense') != -1:
        m.weight.set_data(init.initializer(Normal(sigma=0.001), m.weight.shape, m.weight.dtype))

        if m.bias:
            m.bias.set_data(init.initializer(Constant(0.0), m.bias.shape, m.bias.dtype))



class build_transformer_local(nn.Cell):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH)

        self.num_x = self.base.num_x
        self.num_y = self.base.num_y

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))

        self.rot_number = 4
        print('using rot_number size:{}'.format(self.rot_number))
        self.rearrange = rearrange

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]

        layer_norm = self.base.norm
        self.b1 = nn.SequentialCell(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.b2 = nn.SequentialCell(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.b2_1 = nn.SequentialCell(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2_2 = nn.SequentialCell(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2_3 = nn.SequentialCell(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2_4 = nn.SequentialCell(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2_5 = nn.SequentialCell(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.representation = nn.AdaptiveAvgPool1d(1)
        self.normalization = nn.Identity()

        self.num_classes = num_classes

        self.classifier = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier_a = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
        self.classifier_a.apply(weights_init_classifier)

        self.classifier_1 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
        self.classifier_1.apply(weights_init_classifier)

        self.classifier_2 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
        self.classifier_2.apply(weights_init_classifier)

        self.classifier_3 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
        self.classifier_3.apply(weights_init_classifier)

        self.classifier_4 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.classifier_5 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
        self.classifier_5.apply(weights_init_classifier)

        self.classifier_6 = nn.Dense(self.in_planes, self.num_classes, has_bias=False)
        self.classifier_6.apply(weights_init_classifier)



        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.beta.requires_grad = False
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_a = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_a.beta.requires_grad = False
        self.bottleneck_a.apply(weights_init_kaiming)

        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.beta.requires_grad = False
        self.bottleneck_1.apply(weights_init_kaiming)

        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.beta.requires_grad = False
        self.bottleneck_2.apply(weights_init_kaiming)

        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.beta.requires_grad = False
        self.bottleneck_3.apply(weights_init_kaiming)

        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.beta.requires_grad = False
        self.bottleneck_4.apply(weights_init_kaiming)

        self.bottleneck_5 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_5.beta.requires_grad = False
        self.bottleneck_5.apply(weights_init_kaiming)

        self.bottleneck_6 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_6.beta.requires_grad = False
        self.bottleneck_6.apply(weights_init_kaiming)

    def construct(self, x, label=None, cam_label=None, view_label=None):

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features)

        global_feat = b1_feat[:, 0]  # cls token


        token = features[:, 0:1]

        if self.rearrange:
            pass
        else:
            x = features[:, 1:]
            parts = []
            for i in range(self.rot_number):
                rot = self.rotation(x, self.num_x, self.num_y)
                parts.append(rot)


        rot_features = []

        for index in range(self.rot_number):
            rot_feat = parts[index]
            if index == 0:
                rot_feat = self.b2_1(P.cat((token, rot_feat), axis=1))
            if index == 1:
                rot_feat = self.b2_2(P.cat((token, rot_feat), axis=1))
            if index == 2:
                rot_feat = self.b2_3(P.cat((token, rot_feat), axis=1))
            if index == 3:
                rot_feat = self.b2_4(P.cat((token, rot_feat), axis=1))
            if index == 4:
                rot_feat = self.b2_5(P.cat((token, rot_feat), axis=1))
            rot_feat_1 = rot_feat[:, 0]
            rot_features.append(rot_feat_1)

        feat = self.bottleneck(global_feat)

        cls_scores = []
        rot_features_bn = []
        global_score = self.classifier(feat)

        cls_scores.append(global_score)

        count = 1

        for f in rot_features:
            if count == 1:
                rot_feat_bn = self.bottleneck_1(f)
                cls_score_1 = self.classifier_1(rot_feat_bn)
            if count == 2:
                rot_feat_bn = self.bottleneck_2(f)
                cls_score_1 = self.classifier_2(rot_feat_bn)
            if count == 3:
                rot_feat_bn = self.bottleneck_3(f)
                cls_score_1 = self.classifier_3(rot_feat_bn)
            if count == 4:
                rot_feat_bn = self.bottleneck_4(f)
                cls_score_1 = self.classifier_4(rot_feat_bn)
            if count == 5:
                rot_feat_bn = self.bottleneck_5(f)
                cls_score_1 = self.classifier_5(rot_feat_bn)
            if count == 6:
                rot_feat_bn = self.bottleneck_6(f)
                cls_score_1 = self.classifier_6(rot_feat_bn)
            rot_features_bn.append(rot_feat_bn)
            cls_scores.append(cls_score_1)
            count += 1

        if self.training:
            feat_train = [global_feat]
            for i in rot_features:
                feat_train.append(i)
            return cls_scores, feat_train
        else:
            if self.neck_feat == 'bn':
                feat_test = [feat]
                return P.cat(feat_test, axis=1)
            else:
                feat_test = [global_feat]
                return P.cat(feat_test, axis=1)

    def load_param(self, trained_path):
        param_dict = ms.load_checkpoint(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = ms.load_checkpoint(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
 
    def rotation(self, features, H, W):
        batchsize = features.shape[0]
        dim = features.shape[-1]
    
        x = P.operations.Transpose()(features, (0, 2, 1))
        x = x.reshape(batchsize, dim, H, W)
    
        x = P.operations.Reshape()(x, (batchsize, dim, H*W))
        x = P.operations.Transpose()(x, (0, 2, 1))

        return x