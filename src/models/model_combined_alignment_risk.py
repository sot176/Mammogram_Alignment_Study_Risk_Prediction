import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from  models.model_risk_prediction import RiskModelWithAttention_implicit_alignment, RiskModelWithAttention, RiskModelWithAttention_NoAlignment
from asymmetry_model.mirai_localized_dif_head import extract_mirai_backbone
from models.model_feat_alignment import FeatureAlignmentModel, SpatialTransformerBlock


class CombinedAlignmentRiskModel_Mirai(nn.Module):
    """
    Combined model for alignment and risk prediction using feature based alignment.
    """
    def __init__(self, in_channels=512):
        super().__init__()
        sys.path.append('path/to/onconet')  # Adjust or remove based on your package structure
        self.encoder = extract_mirai_backbone(
            '/scratch/project_465001915/thrunsol/mirai_pretrained_backbone/mgh_mammo_MIRAI_Base_May20_2019.p'
        )
        self.encoder.requires_grad = False  # Freeze encoder weights
        self.attention_alignment_model = FeatureAlignmentModel(in_channels)
        self.risk_prediction_model = RiskModelWithAttention()

    def forward(self, img_cur, img_pri, time_gap):
        """
        Args:
            img_cur (Tensor): Current image (B, 1, H, W)
            img_pri (Tensor): Prior image (B, 1, H, W)
            time_gap (Tensor): Time gap vector (B, T)

        Returns:
            dict: {
                'risk_prediction': Tensor,
                'deformation_field': Tensor,
                'aligned_prior_feature': Tensor,
                'prior_feature_before_alignment': Tensor,
                'current_feature': Tensor,
                'diff_feature': Tensor
            }
        """
        img_cur = img_cur.repeat(1, 3, 1, 1)
        img_pri = img_pri.repeat(1, 3, 1, 1)

        fcur = self.encoder(img_cur)
        fpri = self.encoder(img_pri)

        alignment_outputs = self.attention_alignment_model(fcur, fpri)

        return {
            'risk_prediction': self.risk_prediction_model(
                alignment_outputs['current_features'],
                alignment_outputs['prior_feature_before_alignment'],
                alignment_outputs['aligned_prior'],
                alignment_outputs['differential_feature'],
                time_gap
            ),
            'deformation_field': alignment_outputs['deformation_field'],
            'aligned_prior_feature': alignment_outputs['aligned_prior'],
            'prior_feature_before_alignment': alignment_outputs['prior_feature_before_alignment'],
            'current_feature': alignment_outputs['current_features'],
            'diff_feature': alignment_outputs['differential_feature'],
        }



class RiskModel_no_alignment_Mirai(nn.Module):
    """
    Risk prediction model using features from two timepoints without explicit alignment.
    """
    def __init__(self):
        super().__init__()
        sys.path.append('path/to/onconet')  # Adjust or remove based on your package structure

        self.encoder = extract_mirai_backbone(
            '/scratch/project_465001915/thrunsol/mirai_pretrained_backbone/mgh_mammo_MIRAI_Base_May20_2019.p'
        )
        self.encoder.requires_grad = False
        self.risk_prediction_model = RiskModelWithAttention_NoAlignment()

    def forward(self, img_cur, img_pri, time_gap):
        """
        Args:
            img_cur (Tensor): Current image (B, 1, H, W)
            img_pri (Tensor): Prior image (B, 1, H, W)
            time_gap (Tensor): Time gap vector (B, T)

        Returns:
            dict: {'risk_prediction': Tensor}
        """
        img_cur = img_cur.repeat(1, 3, 1, 1)
        img_pri = img_pri.repeat(1, 3, 1, 1)

        fcur = self.encoder(img_cur)
        fpri = self.encoder(img_pri)

        return {'risk_prediction': self.risk_prediction_model(fcur, fpri, time_gap)}


class RiskModel_implicit_alignment_Mirai(nn.Module):
    """
    Risk prediction model using implicit alignment within the attention mechanism.
    """
    def __init__(self):
        super().__init__()
        sys.path.append('path/to/onconet')  # Adjust or remove based on your package structure

        self.encoder = extract_mirai_backbone(
            '/scratch/project_465001915/thrunsol/mirai_pretrained_backbone/mgh_mammo_MIRAI_Base_May20_2019.p'
        )
        self.encoder.requires_grad = False
        self.risk_prediction_model = RiskModelWithAttention_implicit_alignment()

    def forward(self, img_cur, img_pri, time_gap):
        """
        Args:
            img_cur (Tensor): Current image (B, 1, H, W)
            img_pri (Tensor): Prior image (B, 1, H, W)
            time_gap (Tensor): Time gap vector (B, T)

        Returns:
            dict: {'risk_prediction': Tensor}
        """
        img_cur = img_cur.repeat(1, 3, 1, 1)
        img_pri = img_pri.repeat(1, 3, 1, 1)

        fcur = self.encoder(img_cur)
        fpri = self.encoder(img_pri)

        return {'risk_prediction': self.risk_prediction_model(fcur, fpri, time_gap)}


class CombinedImgAlignmentRiskModel_Mirai(nn.Module):
    """
    Combines pre-aligned image features with the risk prediction model.
    """
    def __init__(self,):
        super().__init__()
        sys.path.append('path/to/onconet')  # Adjust or remove based on your package structure

        self.encoder = extract_mirai_backbone(
            '/scratch/project_465001915/thrunsol/mirai_pretrained_backbone/mgh_mammo_MIRAI_Base_May20_2019.p'
        )
        self.encoder.requires_grad = False
        self.risk_prediction_model = RiskModelWithAttention()

    def forward(self, img_cur, img_pri, warped_pri_img, deformation_field, time_gap):
        """
        Args:
            img_cur (Tensor): Current image (B, 1, H, W)
            img_pri (Tensor): Prior image (B, 1, H, W)
            warped_pri_img (Tensor): Warped prior image after registration (B, 1, H, W)
            time_gap (Tensor): Time gap vector (B, T)
            deformation_field (Tensor): Deformation field used for warping (B, 2, H, W)

        Returns:
            dict: Prediction and feature maps
        """
        img_cur = img_cur.repeat(1, 3, 1, 1)
        img_pri = img_pri.repeat(1, 3, 1, 1)
        warped_pri_img = warped_pri_img.repeat(1, 3, 1, 1)

        fcur = self.encoder(img_cur)
        fpri = self.encoder(img_pri)
        fpri_aligned = self.encoder(warped_pri_img)
        fdiff = torch.abs(fcur - fpri_aligned)

        return {
            'risk_prediction': self.risk_prediction_model(fcur, fpri, fpri_aligned, fdiff, time_gap),
            'deformation_field': deformation_field,
            'aligned_prior_feature': fpri_aligned,
            'prior_feature_before_alignment': fpri,
            'current_feature': fcur,
            'diff_feature': fdiff,
        }


class CombinedImgAlignmentRiskModel_downsample_img_deformation_field_Mirai(nn.Module):
    """
    Combines downsampled deformation field applied to feature maps for risk prediction.
    """
    def __init__(self):
        super().__init__()
        sys.path.append('path/to/onconet')  # Adjust or remove based on your package structure
        self.encoder = extract_mirai_backbone(
            '/scratch/project_465001915/thrunsol/mirai_pretrained_backbone/mgh_mammo_MIRAI_Base_May20_2019.p'
        )
        self.encoder.requires_grad = False
        self.risk_prediction_model = RiskModelWithAttention()
        self.feat_transformer = SpatialTransformerBlock(mode='bilinear')

    def forward(self, img_cur, img_pri, warped_pri_img, deformation_field, time_gap):
        """
        Args:
            img_cur (Tensor): Current image (B, 1, H, W)
            img_pri (Tensor): Prior image (B, 1, H, W)
            warped_pri_img (Tensor): Warped prior image after registration (B, 1, H, W)
            time_gap (Tensor): Time gap vector (B, T)
            deformation_field (Tensor): Deformation field used for warping (B, 2, H, W)

        Returns:
            dict: Risk prediction and related intermediate features
        """
        img_cur = img_cur.repeat(1, 3, 1, 1)
        img_pri = img_pri.repeat(1, 3, 1, 1)

        fcur = self.encoder(img_cur)
        fpri = self.encoder(img_pri)

        # Resize deformation field to match feature map resolution
        deformation_field_downsampled = F.interpolate(
            deformation_field.detach().cpu(),
            size=(fcur.shape[2], fcur.shape[3]),
            mode='bilinear',
            align_corners=True
        ).to(fpri.device)

        scaling_factor_y = fcur.shape[2] / img_cur.shape[2]
        scaling_factor_x = fcur.shape[3] / img_cur.shape[3]

        deformation_field_downsampled[:, 0, :, :] *= scaling_factor_x  # x-direction
        deformation_field_downsampled[:, 1, :, :] *= scaling_factor_y  # y-direction

        fpri_aligned = self.feat_transformer(fpri, deformation_field_downsampled)
        fdiff = torch.abs(fcur - fpri_aligned)

        return {
            'risk_prediction': self.risk_prediction_model(fcur, fpri, fpri_aligned, fdiff, time_gap),
            'deformation_field': deformation_field,
            'aligned_prior_feature': fpri_aligned,
            'prior_feature_before_alignment': fpri,
            'current_feature': fcur,
            'diff_feature': fdiff,
        }

