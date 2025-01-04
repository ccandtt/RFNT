from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--premodel_path',default="./pretrained_weights/fc_weights.pth")  # our 直接将res50用作默认预训练模型
        parser.add_argument('--premodel_path', default="./checkpoints/rfnt_clip/model_epoch_best.pth")
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--predict_path', type=str, default='predict', help='')
        parser.add_argument('--test_dataset_path', type=str, default='./datasets/faceB', help='')

        self.isTrain = False
        return parser



