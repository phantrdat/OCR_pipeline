import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from utils import AttnLabelConverter
from dataset import StreamDataset, AlignCollate
from model import SCATTER
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Params:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

opt = Params(FeatureExtraction='ResNet', PAD=False,
batch_max_length=35, batch_size=192, character='''0123456789abcdefghijklmnopqrstuvwxyz''', 
hidden_size=512, image_folder='', imgH=32, imgW=100, input_channel=1, num_fiducial=20, num_gpu=1, output_channel=512, 
rgb=False, saved_model='./cvpr20_scatter_text_recognizer/weights/scatter-case-sensitive.pth', sensitive=True, workers=4)

if opt.sensitive:
    opt.character = string.printable[:-6]

converter = AttnLabelConverter(opt.character)
opt.num_class = len(converter.character)

if opt.rgb:
    opt.input_channel = 3

# prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
model = SCATTER(opt)
print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
        opt.hidden_size, opt.num_class, opt.batch_max_length)
model = torch.nn.DataParallel(model).to(device)

# load model
print('loading pretrained model from %s' % opt.saved_model)
model.load_state_dict(torch.load(opt.saved_model, map_location=device))
model.eval()

def infer(bb_image_dict):

    demo_data = StreamDataset(opt, bb_image_dict)  # use StreamDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    
    final_preds = []
    final_conf = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            all_block_preds = []
            all_confidence_scores = []
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            predss = model(image, text_for_pred, is_train=False)[0]
            

            for i, preds in enumerate(predss):
                confidence_score_list = []
                pred_str_list = []

                # select max probability (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_str_list.append(pred)
                    pred_max_prob = pred_max_prob[:pred_EOS]
                    
                    # calculate confidence score (= multiply of pred_max_prob)
                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1].cpu().numpy()
                    except:
                        confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                    confidence_score_list.append(confidence_score)
            
                all_block_preds.append(pred_str_list)
                all_confidence_scores.append(confidence_score_list)
            
            all_confidence_scores =  np.array(all_confidence_scores)
            all_block_preds = np.array(all_block_preds)

            best_pred_index = np.argmax(all_confidence_scores, axis=0)
            best_pred_index = np.expand_dims(best_pred_index, axis=0)

            # Get max predition per image through blocks
            all_block_preds = np.take_along_axis(all_block_preds, best_pred_index, axis=0)[0]
            all_confidence_scores = np.take_along_axis(all_confidence_scores, best_pred_index, axis=0)[0]
            
            final_conf.extend(all_confidence_scores.tolist())
            final_preds.extend(all_block_preds.tolist())

    return final_preds, final_conf
