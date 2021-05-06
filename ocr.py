import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import more_itertools as mit
import string
from scipy.signal import argrelextrema

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from joblib import Parallel, delayed
import time

from craft_text_detector import *
from scatter_text_recognizer import *
from ocr_utils import copyStateDict, plot_one_box, Params, four_point_transform

class OCR:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.toTensor = transforms.ToTensor()

    def load_net(self):
        # """ Loading detection network"""
        # self.craft = CRAFT()     # initialize
        # print('Loading box detection weights from checkpoint (' + self.cfg.craft_model + ')')
        # if self.cfg.cuda:
            # self.craft.load_state_dict(copyStateDict(torch.load(self.cfg.craft_model)))
        # else:
            # self.craft.load_state_dict(copyStateDict(torch.load(self.cfg.craft_model, map_location='cpu')))
        # if self.cfg.cuda:
            # self.craft = self.craft.cuda()
            # torch.backends.cudnn.benchmark = False
        # self.craft.eval()

        # LinkRefiner
        self.refine_net = None
        if self.cfg.craft_refine:
            from refinenet import RefineNet
            self.refine_net = RefineNet()
            print('Loading weights of refiner from checkpoint (' + self.cfg.craft_refiner_model + ')')
            if self.cfg.cuda:
                self.refine_net.load_state_dict(copyStateDict(torch.load(self.cfg.craft_refiner_model)))
                self.refine_net = self.refine_net.cuda()
                # self.refine_net = torch.nn.DataParallel(self.refine_net)
            else:
                self.refine_net.load_state_dict(copyStateDict(torch.load(self.cfg.craft_refiner_model, map_location='cpu')))
            self.refine_net.eval()
            self.cfg.craft_poly = True

        """ Loading recognition network """

        if self.cfg.scatter_sensitive:
            self.cfg.scatter_character = string.printable[:-6]

        self.scatter_converter = AttnLabelConverter(self.cfg.scatter_character)
        self.cfg.scatter_num_class = len(self.scatter_converter.character)

        if self.cfg.scatter_rgb:
            self.cfg.scatter_input_channel = 3

        self.align_collate = AlignCollate(imgH=self.cfg.scatter_img_h, imgW=self.cfg.scatter_img_w, keep_ratio_with_pad=self.cfg.scatter_pad)
        self.scatter_params = Params(FeatureExtraction=self.cfg.scatter_feature_extraction, PAD=self.cfg.scatter_pad,
                                     batch_max_length=self.cfg.scatter_batch_max_length, batch_size=self.cfg.scatter_batch_size,
                                     character=self.cfg.scatter_character,
                                     hidden_size=self.cfg.scatter_hidden_size, imgH=self.cfg.scatter_img_h, imgW=self.cfg.scatter_img_w,
                                     input_channel=self.cfg.scatter_input_channel, num_fiducial=self.cfg.scatter_num_fiducial, num_gpu=self.cfg.scatter_num_gpu,
                                     output_channel=self.cfg.scatter_output_channel,
                                     rgb=self.cfg.scatter_rgb, saved_model=self.cfg.scatter_model, sensitive=self.cfg.scatter_sensitive,
                                     workers=self.cfg.scatter_workers, num_class=self.cfg.scatter_num_class)

        self.scatter = SCATTER(self.scatter_params)
        self.scatter = torch.nn.DataParallel(self.scatter).to(self.device)

        print('loading pretrained model from %s' % self.cfg.scatter_model)
        self.scatter.load_state_dict(torch.load(self.cfg.scatter_model, map_location=self.device))
        self.scatter.eval()

        # craft trt
        print('Loading text detection engine (' + self.cfg.craft_engine_path + ')')
        TRT_LOGGER = trt.Logger()
        with open(self.cfg.craft_engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            # Create engine and execution context
            self.craft_engine = runtime.deserialize_cuda_engine(f.read())
            self.craft_context = self.craft_engine.create_execution_context()

            # Create a stream in which to copy inputs/outputs and run inference.
            self.craft_stream = cuda.Stream()


    def craft_pre_process(self, image):
        def resize(image):
            height, width, _ = image.shape
            resized = cv2.resize(image, (self.cfg.craft_target_width, self.cfg.craft_target_height), interpolation=cv2.INTER_LINEAR)
            ratio_w = width / self.cfg.craft_target_width
            ratio_h = height / self.cfg.craft_target_height
            return resized, ratio_w, ratio_h

        # Resize aspect
        # img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(image, self.cfg.craft_canvas_size,
                                                                   # interpolation=cv2.INTER_LINEAR, mag_ratio=self.cfg.craft_mag_ratio)

        # Resize fix
        img_resized, ratio_w, ratio_h = resize(image)

        # Normalize and transpose
        x = imgproc.normalizeMeanVariance(img_resized)
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, axis=0)
        x = np.ascontiguousarray(x)
        return x, ratio_w, ratio_h

    def craft_trt_inference(self, h_input):
        # TODO: - remove feature output head
        #       - optimization profile ?
        # Get indexes
        # input_index = self.craft_engine.get_binding_index("input")
        output_y_index = self.craft_engine.get_binding_index("output_y")
        output_feature_index = self.craft_engine.get_binding_index("output_feature")

        # For dynamic axes
        # # Specify profile
        # self.craft_context.set_optimization_profile_async(0, self.craft_stream.handle)

        # # Specify runtime dimensions
        # self.craft_context.set_binding_shape(input_index, h_input.shape)

        # Allocate host memory for inputs and outputs.
        h_output_y = cuda.pagelocked_empty(trt.volume(self.craft_context.get_binding_shape(output_y_index)), dtype=np.float32)
        h_output_feature = cuda.pagelocked_empty(trt.volume(self.craft_context.get_binding_shape(output_feature_index)), dtype=np.float32)

        # Allocate device memory for inputs and outputs.
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output_y = cuda.mem_alloc(h_output_y.nbytes)
        d_output_feature = cuda.mem_alloc(h_output_feature.nbytes)

        # Copy input host to device
        cuda.memcpy_htod_async(d_input, h_input, self.craft_stream)

        # run inference
        bindings = [int(d_input), int(d_output_feature), int(d_output_y)]
        self.craft_context.execute_async_v2(bindings=bindings, stream_handle=self.craft_stream.handle)

        # Copy output device to host
        cuda.memcpy_dtoh_async(h_output_y, d_output_y, self.craft_stream)
        # cuda.memcpy_dtoh_async(h_output_feature, d_output_feature, self.craft_stream)

        # Sync stream
        self.craft_stream.synchronize()

        # Reshape output
        b, _, h, w = h_input.shape
        y = h_output_y.reshape((b, int(h / 2), int(w / 2), 2))

        # Make score and link map
        score_text = y[0, :, :, 0]
        score_link = y[0, :, :, 1]
        return y, score_text, score_link

    def craft_inference(self, x):
        x = torch.from_numpy(x)
        if self.cfg.cuda:
            x = x.to(self.device)
        with torch.no_grad():
            y, _ = self.craft(x)

        # Make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        return y, score_text, score_link

    def craft_post_process(self, score_text, score_link, ratio_w, ratio_h):
        # Get detection boxes
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, self.cfg.craft_text_threshold, self.cfg.craft_link_threshold,
                                               self.cfg.craft_low_text, self.cfg.craft_poly)

        # Coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]
        return boxes, polys

    def detection(self, image):
        t0 = time.time()
        x, ratio_w, ratio_h = self.craft_pre_process(image)
        t1 = time.time()
        print(f"\tDet pre: {t1 - t0}")

        # Do inference and reshape result
        t0 = time.time()
        _, score_text, score_link = self.craft_trt_inference(x)
        # _, score_text, score_link = self.craft_inference(x)
        t1 = time.time()
        print(f"\tDet infer: {t1 - t0}")

        # # refine link
        # if self.refine_net is not None:
            # with torch.no_grad():
                # y_refiner = self.refine_net(y, feature)
            # score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        # Post-processing
        t0 = time.time()
        boxes, polys = self.craft_post_process(score_text, score_link, ratio_w, ratio_h)
        t1 = time.time()
        print(f"\tDet post total: {t1 - t0}")
        return boxes, polys, score_text

    def scatter_pre_process(self, pts, clone, tensors):
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)

        cropped_box = four_point_transform(clone, pts)
        cropped_box = cv2.cvtColor(cropped_box, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(cropped_box)
        if self.scatter_params.rgb:
            img = img.convert('RGB')  # for color image
        else:
            img = img.convert('L')
        img = img.resize((self.scatter_params.imgW, self.scatter_params.imgH), Image.BICUBIC)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        tensors.append(img)
        return

    def scatter_post_process(self, predss):
        def get_all_preds_and_scores(preds, all_block_preds, all_confidence_scores):
            confidence_score_list = []
            pred_str_list = []

            # select max probability (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_index = preds_index.cpu().numpy()
            preds_str = self.scatter_converter.decode(preds_index)
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
            return

        final_preds = []
        final_conf = []
        all_block_preds = []
        all_confidence_scores = []
        jl_parallel = Parallel(n_jobs=len(predss), prefer="threads", batch_size=len(predss))
        jl_parallel(delayed(get_all_preds_and_scores)(preds, all_block_preds, all_confidence_scores) for preds in predss)

        all_confidence_scores = np.array(all_confidence_scores)
        all_block_preds = np.array(all_block_preds)

        best_pred_index = np.argmax(all_confidence_scores, axis=0)
        best_pred_index = np.expand_dims(best_pred_index, axis=0)

        # Get max predition per image through blocks
        all_block_preds = np.take_along_axis(all_block_preds, best_pred_index, axis=0)[0]
        all_confidence_scores = np.take_along_axis(all_confidence_scores, best_pred_index, axis=0)[0]

        final_conf.extend(all_confidence_scores.tolist())
        final_preds.extend(all_block_preds.tolist())
        return final_preds, final_conf

    def recognize(self, image, polys):
        # Pre-processing
        t0 = time.time()
        clone = image[:, :, ::-1].copy()
        image_tensors = []
        jl_parallel = Parallel(n_jobs=1, prefer="threads", batch_size=1)
        jl_parallel(delayed(self.scatter_pre_process)(pts, clone, image_tensors) for pts in polys)
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        batch_size = image_tensors.size(0)
        image_tensors = image_tensors.to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.scatter_params.batch_max_length + 1).fill_(0).to(self.device)
        t1 = time.time()
        print(f"\tRec pre: {t1 - t0}")

        # Forward
        t0 = time.time()
        with torch.no_grad():
            predss = self.scatter(image_tensors, text_for_pred, is_train=False)[0]
        t1 = time.time()
        print(f"\tRec infer: {t1 - t0}")

        # Post-processing
        t0 = time.time()
        final_preds, final_conf = self.scatter_post_process(predss)
        t1 = time.time()
        print(f"\tRec post: {t1 - t0}")
        return final_preds, final_conf

    def ocr(self, image):
        t0 = time.time()
        if isinstance(image, str):
            image = imgproc.loadImage(image)
        if self.cfg.transform_type == "dilation":
            kernel = np.ones(self.cfg.transform_kernel_size, np.uint8)
            # kernel = np.random.choice([0,1], size=self.cfg.transform_kernel_size).astype(np.uint8)
            transformed_image = cv2.dilate(1 - kernel, kernel, iterations=1)
        else:
            transformed_image = image.copy()
        t1 = time.time()
        print(f"Pre: {t1 - t0}")
        t0 = time.time()
        # _, polys, _, _ = self.detection(transformed_image)
        _, polys, _ = self.detection(transformed_image)
        t1 = time.time()
        print(f"Detection total: {t1 - t0}")
        t0 = time.time()
        pred_str, pred_conf = self.recognize(image, polys)
        t1 = time.time()
        print(f"Recognition total: {t1 - t0}")
        t0 = time.time()
        json_list = []
        for points, text, conf in zip(polys, pred_str, pred_conf):
            word_pred_dict = {}
            word_pred_dict['text'] = text
            if self.cfg.box_type == 'rectangle':
                x1 = max(0, int(min([p[0] for p in points])))
                y1 = max(0, int(min([p[1] for p in points])))
                x2 = max(0, int(max([p[0] for p in points])))
                y2 = max(0, int(max([p[1] for p in points])))
                word_pred_dict['x1'], word_pred_dict['y1'], word_pred_dict['x2'], word_pred_dict['y2'] = x1, y1, x2, y2
            elif self.cfg.box_type == 'polygon':
                x1, y1 = list(points[0])
                x2, y2 = list(points[1])
                x3, y3 = list(points[2])
                x4, y4 = list(points[3])
                word_pred_dict['x1'], word_pred_dict['y1'], word_pred_dict['x2'], word_pred_dict['y2'] = int(x1), int(y1), int(x2), int(y2)
                word_pred_dict['x3'], word_pred_dict['y3'], word_pred_dict['x4'], word_pred_dict['y4'] = int(x3), int(y3), int(x4), int(y4)

            word_pred_dict['confdt'] = conf
            json_list.append(word_pred_dict)

        t1 = time.time()
        print(f"Post: {t1 - t0}")
        return json_list

    def ocr_with_split(self, image, h_slide=10, v_slide=5):  # Threshold for splitting line horizontally and vertically:
        def consec(lst):
            G = mit.consecutive_groups(lst)
            G = [list(g) for g in G]
            return G

        t = time.time()
        final_output = {}
        if isinstance(image, str):
            image = imgproc.loadImage(image)

        im_height, im_width, _ = image.shape
        _, _, score_text, target_ratio = self.detection(image)

        # First split text image horizontally, then split vertically

        horizontal_line_score = np.sum(score_text, axis=1)
        horizontal_cut_lines = argrelextrema(horizontal_line_score, np.less)[0]

        h_idx = horizontal_line_score[horizontal_cut_lines].argsort()[:h_slide]
        horizontal_cut_lines = sorted(list(horizontal_cut_lines[h_idx]))

        if horizontal_cut_lines[0] != 0:
            horizontal_cut_lines.insert(0, 0)
        if horizontal_cut_lines[-1] != len(horizontal_line_score) - 1:
            horizontal_cut_lines.append(len(horizontal_line_score) - 1)

        final_horizontal_cut_lines = [int(c * 2 * (1 / target_ratio)) for c in horizontal_cut_lines]

        # Split vertically on each horizontally patches
        vertical_cut_lines = []

        for i in range(0, len(horizontal_cut_lines) - 1):
            patch_score = score_text[horizontal_cut_lines[i]:horizontal_cut_lines[i + 1]]
            vertical_patch_score = np.sum(patch_score, axis=0)

            vertical_patch_cut_lines = argrelextrema(vertical_patch_score, np.less)[0]

            v_idx = vertical_patch_score[vertical_patch_cut_lines].argsort()[:v_slide]
            vertical_patch_cut_lines = sorted(list(vertical_patch_cut_lines[v_idx]))

            if vertical_patch_cut_lines != []:
                if vertical_patch_cut_lines[0] != 0:
                    vertical_patch_cut_lines.insert(0, 0)
                if vertical_patch_cut_lines[-1] != len(vertical_patch_score) - 1:
                    vertical_patch_cut_lines.append(len(vertical_patch_score) - 1)

            vertical_cut_lines.append([int(c * 2 * (1 / target_ratio)) for c in vertical_patch_cut_lines])

        final_json_list = []
        all_parts_json = []
        for i in range(0, len(final_horizontal_cut_lines) - 1):

            line_parts = []
            if self.cfg.craft_split_vertically:
                v_l = vertical_cut_lines[i]
            else:
                v_l = [None, None]
            if len(v_l) == 0:
                v_l = [0, im_width]
            for j in range(len(v_l) - 1):
                json_list = []
                try:
                    split_im = image.copy()[final_horizontal_cut_lines[i]:final_horizontal_cut_lines[i + 1], v_l[j]:v_l[j + 1]]
                    json_list = self.ocr(split_im)
                    # cv2.imwrite(f'temp/h_{final_horizontal_cut_lines[i]}{final_horizontal_cut_lines[i+1]}-w_{v_l[j]}{v_l[j+1]}.png', split_im)
                    for k in range(len(json_list)):
                        if v_l[j] != None:
                            json_list[k]['x1'] += v_l[j]
                            json_list[k]['x2'] += v_l[j]

                        json_list[k]['y1'] += final_horizontal_cut_lines[i]
                        json_list[k]['y2'] += final_horizontal_cut_lines[i]
                except:
                    pass
                final_json_list.extend(json_list)
                line_parts.append(json_list)

            all_parts_json.append(line_parts)

        return final_json_list, all_parts_json, final_horizontal_cut_lines, vertical_cut_lines

    def split_text_vertically(self, image, s_length=5):
        def consec(lst):
            G = mit.consecutive_groups(lst)
            G = [list(g) for g in G]
            return G
        if isinstance(image, str):
            image = imgproc.loadImage(image)
        t0 = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, self.cfg.craft_canvas_size,
                                                                              interpolation=cv2.INTER_LINEAR, mag_ratio=self.cfg.craft_mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cfg.cuda:
            x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            y, feature = self.craft(x)  # CRAFT

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        vertical_cut_lines = []
        vertical_line_score = np.sum(score_text, axis=0)

        vertical_cut_lines = list(argrelextrema(vertical_line_score, np.less)[0])
        if vertical_cut_lines[0] != 0:
            vertical_cut_lines.insert(0, 0)
        if vertical_cut_lines[-1] != len(vertical_line_score) - 1:
            vertical_cut_lines.append(len(vertical_line_score) - 1)

        vertical_cut_lines = [int(c * 2 * (1 / target_ratio)) for c in vertical_cut_lines]

        final_vertical_cut_lines = []
        textbb_dict = {}
        split_to = len(vertical_cut_lines) // s_length
        for i in range(split_to):
            final_vertical_cut_lines.append(vertical_cut_lines[i * s_length])
        final_vertical_cut_lines.append(vertical_cut_lines[-1])

        # for i in range(0, len(final_vertical_cut_lines)):
        for i in range(0, len(final_vertical_cut_lines) - 1):
            try:
                split_im = image.copy()[:, final_vertical_cut_lines[i]:final_vertical_cut_lines[i + 1]]
                coor_key = '-'.join([str(final_vertical_cut_lines[i]), str(final_vertical_cut_lines[i + 1])])
                textbb_dict[coor_key] = Image.fromarray(split_im)
            except:
                pass

        return textbb_dict, vertical_line_score

    def re_regconize(self, image, json_list, min_inp_confidence=0.5, s_length=7, max_length=30, min_out_confidence=0.8):
        if isinstance(image, str):
            image = imgproc.loadImage(image)
        marked = []
        parts = {}
        for k, box in enumerate(json_list):
            try:
                if box['confdt'] < min_inp_confidence and len(box['text']) < max_length:
                    if 'x3' in box and 'y3' in box and 'x4' in box and 'y4' in box:
                        points = np.array([[box['x1'], box['y1']], [box['x2'], box['y2']], [box['x3'], box['y3']], [box['x4'], box['y4']]])
                        part = four_point_transform(image, points)
                    else:
                        part = image[box['y1']:box['y2'], box['x1']:box['x2']].copy()
                    sub_part, _ = self.split_text_vertically(part, s_length=s_length)

                    marked.append((k, len(sub_part)))
                    parts.update(sub_part)
            except:
                pass

        reg_res = self.recognize(parts)
        # Update recognize result
        ptr = 0
        for (idx, L) in marked:
            parts_of_text = []
            confdts_of_text = []
            res = reg_res[0][ptr:ptr + L]
            confident = reg_res[1][ptr:ptr + L]
            for j, sub_text in enumerate(res):
                if confident[j] > min_out_confidence:
                    parts_of_text.append(sub_text)
                    confdts_of_text.append(confident[j])
            json_list[idx]['text'] = ''.join(parts_of_text)
            if confdts_of_text != []:
                json_list[idx]['confdt'] = sum(confdts_of_text) / len(confdts_of_text)
            ptr += L
        return json_list

    def plot(self, image, json_list, color=(178, 74, 74), line_thickness=1):
        if isinstance(image, str):
            image = imgproc.loadImage(image)
        for b in json_list:

            if b != None:
                conf = b['confdt'] if 'confdt' in b else 1.0
                text = b['text']
                if self.cfg.box_type == 'rectangle':
                    c1 = (b['x1'], b['y1'])
                    c2 = (b['x2'], b['y2'])
                    c3 = None
                    c4 = None

                if self.cfg.box_type == 'polygon':
                    c1 = (int(b['x1']), int(b['y1']))
                    c2 = (int(b['x2']), int(b['y2']))
                    c3 = (int(b['x3']), int(b['y3']))
                    c4 = (int(b['x4']), int(b['y4']))

            plot_one_box(image, c1, c2, c3, c4, label=text, score=conf, color=color, line_thickness=line_thickness, box_type=self.cfg.box_type)

        return image
