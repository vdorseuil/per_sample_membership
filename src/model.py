import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from transformers import AutoModelForSequenceClassification
from pathlib import Path

def _cache_dir():
    return os.environ.get("LEVERAGE_CACHE_DIR", os.path.abspath("./cache"))


def _bert_local_dir():
    return os.path.join(_cache_dir(), "bert_tinybert")


def _torch_hub_cache():
    return os.path.join(_cache_dir(), "torch_hub")

class LogisticRegression(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.linear(x)
        return out
    
    def get_features(self, x):
        """
        Returns the penultimate layer.
        """
        x = x.view(-1, 28 * 28)
        return x

class LinearRegression(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out
    
    def get_features(self, x):
        """
        Returns the penultimate layer.
        """
        return x


class TabularMLP(nn.Module):
    """
    A robust Multi-Layer Perceptron designed for tabular data.
    Includes Batch Normalization, Dropout, and ReLU activations.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list = [64, 32], dropout_rate: float = 0.2):
        super(TabularMLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Build the hidden layers dynamically
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            current_dim = hidden_dim
            
        # The feature extractor (useful if you need embeddings/features for your leverage code)
        self.feature_extractor = nn.Sequential(*layers)
        
        # The final output layer (no activation here, handle it in the loss function)
        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        logits_or_values = self.output_layer(features)
        return logits_or_values
        
    def get_features(self, x):
        """
        Helper method to extract the final hidden state. 
        This perfectly matches the `model.get_features(x)` requirement 
        in your multivariate leverage computation code!
        """
        with torch.no_grad():
            return self.feature_extractor(x)
        
        

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int, output_size: int, dropout_prob: float = 0.2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        
        self.head = nn.Linear(hidden_dim, output_size)
        
    def get_features(self, x):
        """Extracts the representation from the last hidden layer."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        x = self.get_features(x)
        x = self.head(x)
        return x
        

class CNN(nn.Module):
    def __init__(self, feat_dim:int=256, dropout_rate:float=0.3):
        super(CNN, self).__init__()
        
        def conv_block(in_f, out_f, dropout_rate):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout_rate)
            )

        self.features = nn.Sequential(
            conv_block(3, 64, 0.2),   # 32x32 -> 16x16
            conv_block(64, 128, 0.3), # 16x16 -> 8x8
            conv_block(128, feat_dim, 0.5) # 8x8 -> 4x4
        )
        
        self.pre_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pre_classifier(x)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        """
        Recovers the penultimate layer output.
        Returns a tensor of shape (Batch, 256).
        """
        with torch.no_grad(): # Use no_grad if just extracting for inference
            x = self.features(x)
            x = self.pre_classifier(x)
        return x
    

class BasicBlockWithDropout(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.3):
        super(BasicBlockWithDropout, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Dropout applied BEFORE the second convolution
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, dropout_rate=0.3):
        super(ResNet20, self).__init__()
        self.in_planes = 16

        # Initial layer designed specifically for 32x32 images
        # (Note: NO 7x7 conv and NO MaxPooling here)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 3 stages of residual blocks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)  # Global Average Pooling for 8x8 feature map
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def get_features(self, x):
        """Returns the penultimate layer."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)  # Global Average Pooling for 8x8 feature map
        x = x.view(x.size(0), -1)
        return x


class BertClassifier(nn.Module):
    def __init__(self, model_name="google/bert_uncased_L-2_H-128_A-2", num_classes=2, freeze=False, cache_dir="./cache"):
        super().__init__()
        
        model_cache = Path(cache_dir) / "huggingface"
        os.makedirs(model_cache, exist_ok=True)

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            cache_dir=str(model_cache),
        )

        if freeze:
            # Step 1: Freeze everything
            for param in self.bert.parameters():
                param.requires_grad = False
            
            # Step 2: Unfreeze ONLY the classifier head
            # In AutoModelForSequenceClassification, the head is named 'classifier'
            for param in self.bert.classifier.parameters():
                param.requires_grad = True
            print(f"!!! Feature Extraction Mode: Backbone {model_name} is FROZEN.")
        else:
            print(f"!!! Full Fine-Tuning Mode: Backbone {model_name} is TRAINABLE.")

    
    def forward(self, x):
        if isinstance(x, dict):
            outputs = self.bert(**x)
        else:
            outputs = self.bert(x)
        return outputs.logits

    def get_features(self, x):
        with torch.no_grad():
            if isinstance(x, dict):
                outputs = self.bert.base_model(**x)
            else:
            # 'base_model' gets us past the classification head to the embeddings
                outputs = self.bert.base_model(x)
            
        # Return the [CLS] token representation
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]


class DinoClassifier(nn.Module):
    def __init__(self, model_name="dinov2_vits14", num_classes=8, freeze=True, hub_dir=None):
        super().__init__()
        hub_dir = hub_dir or _torch_hub_cache()
        os.makedirs(hub_dir, exist_ok=True)
        torch.hub.set_dir(hub_dir)
        # If the dinov2 repo is already cached, load from the local copy so torch.hub
        # never tries to reach github.com (required on offline compute nodes).
        local_repo = os.path.join(hub_dir, "facebookresearch_dinov2_main")
        if not os.path.isdir(local_repo):
            local_repo = os.path.join(hub_dir, "facebookresearch_dinov2_master")
        if os.path.isdir(local_repo):
            self.backbone = torch.hub.load(local_repo, model_name, source="local")
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x):
        with torch.no_grad():
            return self.backbone(x)


def get_model(model_name, **kwargs):

    if model_name == "logistic_regression":
        return LogisticRegression(
            input_size=kwargs.get("dimension", 784),
            num_classes=kwargs.get("num_classes", 10),
        )
    if model_name == "linear_regression":
        return LinearRegression(
            input_dim=kwargs.get("input_dim", 784),
            output_dim=kwargs.get("output_dim", 1),
        )
    if model_name == "tabular_mlp":
        return TabularMLP(
            input_dim=kwargs.get("input_dim", 784),
            hidden_layers=kwargs.get("hidden_layers", [64, 32]),
            output_dim=kwargs.get("output_dim", 1),
            dropout_rate=kwargs.get("dropout", 0.2)
        )
    if model_name == "mlp":
        return MLP(
            input_size=kwargs.get("input_dim", 784),
            hidden_dim=kwargs.get("hidden_dim", 784),
            output_size=kwargs.get("output_dim", 1),
        )
    if model_name == "cnn":
        return CNN(
            dropout_rate=kwargs.get("dropout", 0.3)
        )
    if model_name == "bert":
        return BertClassifier(
            model_name=kwargs.get("bert_model_name", "google/bert_uncased_L-2_H-128_A-2"),
            num_classes=kwargs.get("num_classes", 2),
            freeze=kwargs.get("freeze", False),
            cache_dir=kwargs.get("cache_dir", "./cache"),
        )
    if model_name == "dino":
        return DinoClassifier(
            num_classes=kwargs.get("num_classes", 8),
            freeze=kwargs.get("freeze", False),
        )
    if model_name == "resnet20":
        return ResNet20(BasicBlockWithDropout, [3, 3, 3], num_classes=kwargs.get("num_classes", 100), dropout_rate=kwargs.get("dropout", 0.3))
    else:
        raise ValueError(f"Unknown model: {model_name}")
