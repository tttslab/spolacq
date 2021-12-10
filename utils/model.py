import torch.nn as nn
import torchvision.models as models

class SimpleImageCorrNet(nn.Module):
    def __init__(self, num_class = 50, ext_v = True, ext_a = True, pre_trained=False):
        super(SimpleImageCorrNet, self).__init__()
        self.visual_net = models.resnet50(num_classes=num_class, pretrained=pre_trained)
        self.audio_net = models.resnet50(num_classes=num_class, pretrained=pre_trained)
        self.audio_net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.ext_v = ext_v
        self.ext_a = ext_a

    def _visual_extract(self, v):
        return self.visual_net(v)

    def _sound_extract(self, a):
        return self.audio_net(a[:,None,:,:])

    def forward(self, visual_feat, audio_feat):
        if self.ext_v:      visual_feat = self._visual_extract(visual_feat)
        if self.ext_a:      audio_feat = self._sound_extract(audio_feat)
        return (visual_feat, audio_feat)