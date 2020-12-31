import numpy as np
import pretrainedmodels as ptm
import torch
from PIL import Image
from torch import nn
from torchvision import datasets, models, transforms
from tqdm import tqdm

# 色及び形状特徴抽出用モデル
class ResNet50(nn.Module):
    def __init__(self, opt="", list_style=False, no_norm=False):
        super(ResNet50, self).__init__()

        self.pars = opt
        not_pretrained = False
        if not not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=None)

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, 128)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

    def forward(self, x, is_init_cluster_generation=False):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))

        for layerblock in self.layer_blocks:
            x = layerblock(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0),-1)

        mod_x = self.model.last_linear(x)
        #No Normalization is used if N-Pair Loss is the target criterion.
        return mod_x if self.pars=='triplet' else torch.nn.functional.normalize(mod_x, dim=-1)

def make_model_color(system_parameters):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    # output->model:色特徴抽出モデル
    ####################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50()
    model.to(device)
    model.load_state_dict(torch.load(system_parameters["model"]["weight"]["color"]))
    return model

def make_model_type(system_parameters):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    # output->model:形状特徴抽出モデル
    ####################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50()
    model.to(device)
    model.load_state_dict(torch.load(system_parameters["model"]["weight"]["type"]))
    return model

def predict(system_parameters, model, images):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        model:特徴抽出model
    #        images:入力画像list
    # output->特徴量ベクトル
    ####################################################
    val_transform = transforms.Compose([transforms.Resize(299),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ans = []
    
    for data in tqdm(images):
        data = val_transform(Image.fromarray(data))
        data = data.view(1,3,299,299)
        out = model(data.to(device))

        #out = model[0](data)
        #out = model[1](out)
        out = np.squeeze(out.to("cpu").detach().numpy().copy())
        ans.append(out)
    
    return np.array(ans)
