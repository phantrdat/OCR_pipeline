class Config:
    def __init__(self):
        
        """General config"""
        self.cuda=True

        """ Config of detection module """
        self.craft_model ='./craft_text_detector/weights/craft_mlt_25k.pth'
        self.craft_text_threshold=0.6
        self.craft_low_text=0.4
        self.craft_link_threshold=0.4
        self.craft_canvas_size=1280
        self.craft_mag_ratio=1.0
        self.craft_poly=False
        # self.craft_show_time=False
        self.craft_refine=False
        self.craft_refiner_model='./craft_text_detector/weights/craft_refiner_CTW1500.pth'

        self.craft_padding_ratio = 8 # Extend detected boxes generated from CRAFT. Each box will be add "box_height/craft_padding_ratio" both sides

        """ Config of recognition module """
        self.scatter_feature_extraction='ResNet'
        self.scatter_pad=False
        self.scatter_batch_max_length=35
        self.scatter_batch_size=192
        self.scatter_character= '0123456789abcdefghijklmnopqrstuvwxyz'
        self.scatter_hidden_size=512
        self.scatter_img_h=32
        self.scatter_img_w=100
        self.scatter_input_channel=1
        self.scatter_num_fiducial=20
        self.scatter_num_gpu=1
        self.scatter_output_channel=512
        self.scatter_rgb=False
        self.scatter_sensitive=True
        self.scatter_workers=4
        self.scatter_model='./scatter_text_recognizer/weights/scatter-case-sensitive.pth'
        