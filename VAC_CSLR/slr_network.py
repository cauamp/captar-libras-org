import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import time

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True, freeze_conv2d=False, 
            conv2d_dropout_prob=0, 
            custom_resnet=False, half_it=False, chunks=1, memory_debug=False
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        if torch.__version__ >= "1.9.0":
            self.conv2d = getattr(models, c2d_type)(weights='IMAGENET1K_V1')
        else:
            self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        
        if custom_resnet:
            self.conv2d = ResNetMemoryEfficient(self.conv2d, half_it=half_it, chunks=chunks, memory_debug=memory_debug) 

        if freeze_conv2d:
            for param in self.conv2d.parameters():
                param.requires_grad = False
                                
        if c2d_type in ['resnet18', 'resnet34']:
            conv2d_output_channels = 512  # ResNet18/34 final feature size
        elif c2d_type in ['resnet50', 'resnet101', 'resnet152']:
            conv2d_output_channels = 2048  # ResNet50+ final feature size
        else:
            raise ValueError(f"Unsupported ResNet type: {c2d_type}")
        
        self.conv1d = TemporalConv(input_size=conv2d_output_channels,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        self.register_full_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            if isinstance(g, torch.Tensor):  # Ensure only Tensors are processed
                g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None)-> dict:
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x
        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])\
            
        if self.training:
            pred, prob, conf = None, None, None
            conv_pred, conv_prob, conv_conf = None, None, None
        else:
            pred, prob, conf = self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
            conv_pred, conv_prob, conv_conf = self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "conv_decoding_stats": {"prob": conv_prob, "conf": conv_conf},
            "recognized_decoding_stats": {"prob": prob, "conf": conf}
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss


class ResNetMemoryEfficient(nn.Module):
    def __init__(self, resnet, half_it=False, chunks=1, memory_debug=False):
        super().__init__()
        self.half_it = half_it
        self.chunks = chunks
        self.debug = memory_debug
        print(f"Using ResNetMemoryEfficient with {chunks} chunks {16 if half_it else 32}-bit \n Memory Allocation debugging: {memory_debug}")
        
        if self.half_it:
            self.conv1 = resnet.conv1.half()
            self.bn1 = resnet.bn1.half()
            self.relu = resnet.relu.half()
            self.maxpool = resnet.maxpool.half()
            self.layer1 = resnet.layer1.half()
            self.layer2 = resnet.layer2.half()
            self.layer3 = resnet.layer3.half()
            self.layer4 = resnet.layer4
            self.avgpool = resnet.avgpool
        else:
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
            self.avgpool = resnet.avgpool
            
    def forward(self, x):
        sleep_time = 2
        if self.half_it:
            x = x.half()
        
        chunks = torch.chunk(x, self.chunks, dim=0)  # This splits the tensor into {self.chunks} parts along the first dimension
        
        if self.debug:
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB\n\n")
        
        # Process each chunk separately
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            if self.debug:
                print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Apply conv1 to the chunk
            chunk = self.conv1(chunk)

            if self.debug:
                print(f'conv1 OK - chunk {i+1}')
                print(f'{chunk.shape =}')
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB\n\n")
                time.sleep(sleep_time)
            
            # Apply bn1 to the chunk
            chunk = self.bn1(chunk)

            if self.debug:
                print(f'bn1 OK - chunk {i+1}')
                print(f'{chunk.shape =}')
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB\n\n")
                time.sleep(sleep_time)
            
            # Apply ReLU to the chunk
            chunk = self.relu(chunk)

            if self.debug:
                print(f'relu OK - chunk {i+1}')
                print(f'{chunk.shape =}')
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB\n\n")
                time.sleep(sleep_time)
            
            # Apply maxpool to the chunk
            chunk = self.maxpool(chunk)

            if self.debug:
                print(f'maxpool OK - chunk {i+1}')
                print(f'{chunk.shape =}')
                print(f"GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB\n\n")
                time.sleep(sleep_time)
            
            processed_chunks.append(chunk.cpu())

        # Concatenate the processed chunks back together
        x = torch.cat(processed_chunks, dim=0)
        x = x.cuda()
        
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)

            if layer == self.layer3 and self.half_it:
                x = x.float()
            if self.debug:
                print(f'layer{i} OK')
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x