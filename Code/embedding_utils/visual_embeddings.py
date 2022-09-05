import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2, json

from detectron2.layers import nms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from torchinfo import summary
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList
from detectron2.data import transforms as T
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs
from detectron2.structures.boxes import Boxes

cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
MIN_BOXES=10
MAX_BOXES=100


def img2bgr(img_path):
    
    img_rgblist = [img_path]
    img_bgrlist = []
    for img_file in img_rgblist:
        img = plt.imread(img_file)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgrlist.append(img_bgr)
    
    return img_bgrlist
    
############ Functions to extract visual embeddings ###########################

def load_config_and_model_weights(cfg_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"):
    '''
    Load config & model weights
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_path))

    # ROI HEADS SCORE THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Comment the next line if you're using 'cuda'
    cfg['MODEL']['DEVICE']= 'cpu'

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

    return cfg

def get_model(cfg):
    '''
    Load object detection model
    '''
    
    # build model
    model = build_model(cfg)

    # load weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # eval mode
    model.eval()
    return model

def prepare_image_inputs(cfg, img_list, model):
    
    '''
    Resize list of images (which are already converted to BGR form)
    Change to tensor forms
    '''
    
    # Resizing the image according to the configuration
    transform_gen = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
    img_list = [transform_gen.get_transform(img).apply_image(img) for img in img_list]

    # Convert to C,H,W format
    convert_to_tensor = lambda x: torch.Tensor(x.astype("float32").transpose(2, 0, 1))

    batched_inputs = [{"image":convert_to_tensor(img), "height": img.shape[0], "width": img.shape[1]} for img in img_list]

    # Normalizing the image
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    images = [normalizer(x["image"]) for x in batched_inputs]

    # Convert to ImageList
    images =  ImageList.from_tensors(images,model.backbone.size_divisibility)
    
    return images, batched_inputs

def get_features(model, images):
    '''
    Extract Resnet + FPN Features
    '''
    
    features = model.backbone(images.tensor)
    return features

def get_proposals(model, images, features):
    '''
    Get the regional proposals from RPN
    '''
    proposals, _ = model.proposal_generator(images, features)
    return proposals

def get_box_features(model, features, proposals, imglist):
    '''
    Get the boxes features from model, and reshape based on number of boxes extracted
    '''
    features_list = [features[f] for f in ['p2', 'p3', 'p4', 'p5']]
    box_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    box_features = model.roi_heads.box_head.flatten(box_features)
    box_features = model.roi_heads.box_head.fc1(box_features)
    box_features = model.roi_heads.box_head.fc_relu1(box_features)
    box_features = model.roi_heads.box_head.fc2(box_features)

    box_features = box_features.reshape(len(imglist), 1000, 1024) # depends on your config and batch size
    return box_features, features_list

def get_prediction_logits(model, features_list, proposals):
    '''
    Get Prediction logits & boxes
    '''
    cls_features = model.roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
    cls_features = model.roi_heads.box_head(cls_features)
    pred_class_logits, pred_proposal_deltas = model.roi_heads.box_predictor(cls_features)
    return pred_class_logits, pred_proposal_deltas

def get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals):
    '''
    Extract boxes coordinates and scores based on FastRCNN    
    '''
    
    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
    smooth_l1_beta = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA

    outputs = FastRCNNOutputs(
    box2box_transform,
    pred_class_logits,
    pred_proposal_deltas,
    proposals,
    smooth_l1_beta,
    )

    boxes = outputs.predict_boxes()
    scores = outputs.predict_probs()
    image_shapes = outputs.image_shapes

    return boxes, scores, image_shapes

def get_output_boxes(boxes, batched_inputs, image_size):
    '''
    Rescale boxes to original image size
    '''
    proposal_boxes = boxes.reshape(-1, 4).clone()
    scale_x, scale_y = (batched_inputs["width"] / image_size[1], batched_inputs["height"] / image_size[0])
    output_boxes = Boxes(proposal_boxes)

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(image_size)

    return output_boxes

def select_boxes(cfg, output_boxes, scores):
    '''
    Boxes selection with NMS
    '''
    test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cls_prob = scores.detach()
    cls_boxes = output_boxes.tensor.detach().reshape(1000,80,4)
    max_conf = torch.zeros((cls_boxes.shape[0]))
    for cls_ind in range(0, cls_prob.shape[1]-1):
        cls_scores = cls_prob[:, cls_ind+1]
        det_boxes = cls_boxes[:,cls_ind,:]
        keep = np.array(nms(det_boxes, cls_scores, test_nms_thresh))
        max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
    keep_boxes = torch.where(max_conf >= test_score_thresh)[0]
    return keep_boxes, max_conf

def filter_boxes(keep_boxes, max_conf, min_boxes, max_boxes):
    '''
    Limit number of boxes
    '''
    if len(keep_boxes) < min_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:min_boxes]
    elif len(keep_boxes) > max_boxes:
        keep_boxes = np.argsort(max_conf).numpy()[::-1][:max_boxes]
    return keep_boxes

def get_visual_embeds(box_features, keep_boxes):
    '''
    Extract Visual Embeddings
    '''
    return box_features[keep_boxes.copy()]

def extract_visual_embeddings(imglist):
    cfg = load_config_and_model_weights()                       ## Load Config & Model Weights
    cfg_model = get_model(cfg)                                              ## Load Object Detection Model

    images, batched_inputs = prepare_image_inputs(cfg, imglist, cfg_model)     ## Convert Image to Model input
    features = get_features(cfg_model, images)                              ## Get ResNet + FPN Features 
    proposals = get_proposals(cfg_model, images, features)                  ## Get Region Proposals from RPN 
    box_features, features_list = get_box_features(cfg_model, features, proposals, imglist)             ## Get Box features
    pred_class_logits, pred_proposal_deltas = get_prediction_logits(cfg_model, features_list, proposals)    ## Get prediction logits & boxes
    boxes, scores, image_shapes = get_box_scores(cfg, pred_class_logits, pred_proposal_deltas, proposals)          ## Get FastRCNN scores and boxes\
    output_boxes = [get_output_boxes(boxes[i], batched_inputs[i], proposals[i].image_size) for i in range(len(proposals))]  ## Rescale boxes back to original size
    temp = [select_boxes(cfg, output_boxes[i], scores[i]) for i in range(len(scores))]                  ## Select boxes with NMS 
    keep_boxes, max_conf = [],[]
    for keep_box, mx_conf in temp:
        keep_boxes.append(keep_box)
        max_conf.append(mx_conf)
    keep_boxes = [filter_boxes(keep_box, mx_conf, MIN_BOXES, MAX_BOXES) for keep_box, mx_conf in zip(keep_boxes, max_conf)]     ## Limit number of boxes
    visual_embeddings = [get_visual_embeds(box_feature, keep_box) for box_feature, keep_box in zip(box_features, keep_boxes)]       ### Get visual embeds ###
    return visual_embeddings
    
##################################################################

def img_visual_embeds(img_bgrlist, device):
    for img in img_bgrlist:
        visual_embeds = []
        imglist = [img]
        visual_embeddings = extract_visual_embeddings(imglist)
        visual_embeds += visual_embeddings
        visual_embeds = [embeds.detach().cpu().numpy().tolist() for embeds in visual_embeds]
        data_dict = {"data": visual_embeds}

    with open("eval.json", "w") as outfile:
        json.dump(data_dict, outfile)

    f = open("eval.json")
    visual_embeds = json.load(f)
    visual_embeds = torch.from_numpy(np.array(visual_embeds["data"])).float().to(device)
    f.close()
    
    return visual_embeds