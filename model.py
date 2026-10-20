import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.ao.nn.quantized.modules import dropout
from torch.nn import GELU, Dropout
from transformers import BertModel, AutoModel, AutoConfig, BertPreTrainedModel

from transformers import PretrainedConfig, PreTrainedModel

class MultiModelFusionConfig(PretrainedConfig):
    """
    通用配置类，适配你的全部多BERT融合变体模型
    model_type 用来区分不同模型变体
    """
    model_type = "multi_bert_fusion"

    def __init__(
        self,
        # 公共基础参数
        num_labels=6,
        model_names=None,   # [path1,path2,path3] 三个预训练模型本地路径
        hidden_dim=128,
        mlp_hidden_dim=256,
        num_heads=8,
        dropout_rate=0.3,

        # 可选：部分消融模型用到的额外参数，给默认值
        use_cross_attention=False,
        use_mlp_weight_fusion=True,
        use_mlp_feature_fusion=True,

        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.model_names = model_names
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # 消融开关，不同变体读取不同开关
        self.use_cross_attention = use_cross_attention
        self.use_mlp_weight_fusion = use_mlp_weight_fusion
        self.use_mlp_feature_fusion = use_mlp_feature_fusion

class MASFON(PreTrainedModel):
    config_class = MultiModelFusionConfig

    def __init__(self, config: MultiModelFusionConfig):
        super().__init__(config)

        # 从config取出全部超参
        hidden_dim = config.hidden_dim
        mlp_hidden_dim = config.mlp_hidden_dim
        num_heads = config.num_heads
        dropout_rate = config.dropout_rate
        num_labels = config.num_labels
        model_name_1, model_name_2, model_name_3 = config.model_names

        self.model_config = {
            'hidden_dim': hidden_dim,
            'mlp_hidden_dim': mlp_hidden_dim,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
            'num_labels': num_labels,
            'model_names': [model_name_1, model_name_2, model_name_3]
        }

        # 三个BERT模型
        self.Bert_1 = AutoModel.from_pretrained(model_name_1)
        self.Bert_2 = AutoModel.from_pretrained(model_name_2)
        self.Bert_3 = AutoModel.from_pretrained(model_name_3)

        # 特征降维层
        self.fc_1 = nn.Linear(768, hidden_dim)
        self.fc_2 = nn.Linear(768, hidden_dim)
        self.fc_3 = nn.Linear(768, hidden_dim)

        # 自注意力特征强化层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout_rate
        )

        # MLP融合网络
        self.mlp_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, hidden_dim * 3),
        )

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, num_labels)
        )

        # 层归一化和dropout
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 3)
        self.dropout = nn.Dropout(dropout_rate)

        # 权重融合层
        self.weight_fusion = nn.Sequential(
            nn.Linear(9, 16),
            nn.GELU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=1)
        )

        # PreTrainedModel要求调用，做权重初始化
        self.post_init()

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2,
                input_ids_3, attention_mask_3, labels=None, return_weights=False, return_attention_details=False):

        # 1. 特征提取和降维
        feat1 = self.layer_norm1(F.gelu(self.fc_1(
            self.Bert_1(input_ids_1, attention_mask_1)[0][:, 0]
        )))
        feat2 = self.layer_norm1(F.gelu(self.fc_2(
            self.Bert_2(input_ids_2, attention_mask_2)[0][:, 0]
        )))
        feat3 = self.layer_norm1(F.gelu(self.fc_3(
            self.Bert_3(input_ids_3, attention_mask_3)[0][:, 0]
        )))

        # 2. 自注意力特征强化
        stacked_feats = torch.stack([feat1, feat2, feat3], dim=1)  # [batch, 3, hidden_dim]
        self_attended, self_weights = self.self_attention(
            stacked_feats, stacked_feats, stacked_feats
        )
        enhanced_feats = self.layer_norm1(self_attended + stacked_feats)  # [batch, 3, hidden_dim]

        # 3. MLP计算每个模型的融合权重
        attention_vector = self_weights.reshape(self_weights.size(0), -1)  # [batch, 9]
        combined_weights = self.weight_fusion(attention_vector)  # [batch, 3]

        # 4. 对强化后的特征进行加权融合
        enhanced_feat1, enhanced_feat2, enhanced_feat3 = enhanced_feats.unbind(dim=1)
        weighted_feat1 = combined_weights[:, 0].unsqueeze(1) * enhanced_feat1
        weighted_feat2 = combined_weights[:, 1].unsqueeze(1) * enhanced_feat2
        weighted_feat3 = combined_weights[:, 2].unsqueeze(1) * enhanced_feat3
        #等价
        # weighted_feats = enhanced_feats * combined_weights.unsqueeze(-1)  # [B, 3, H]
        # concat_features = weighted_feats.reshape(weighted_feats.size(0), -1)  # [B, 3H]

        # 5. MLP深度融合
        concat_features = torch.cat([weighted_feat1, weighted_feat2, weighted_feat3], dim=1)

        # 残差连接
        mlp_fused = self.mlp_fusion(concat_features)
        mlp_fused = self.layer_norm2(mlp_fused + concat_features)  # 残差连接
        mlp_fused = self.dropout(F.gelu(mlp_fused))

        # 6. 分类
        logits = self.classifier(mlp_fused)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            outputs = (loss, logits)
        else:
            outputs = (None, logits)

        if return_weights:
            outputs += (combined_weights,)

        # 返回监控所需的中间变量
        if return_attention_details:
            details = {
                'self_weights': self_weights,  # [batch, 3, 3]
                'feat_self': enhanced_feats,  # [batch, 3, hidden_dim]
                'fusion_weights': combined_weights,  # [batch, 3]
            }
            outputs += (details,)

        return outputs

