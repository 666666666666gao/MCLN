# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
from .mcln_attention import MultiheadAttention
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from .backbone_module import Pointnet2Backbone
from .modules import (
    PointsObjClsModule, GeneralSamplingModule,
    ClsAgnosticPredictHead, PositionEmbeddingLearned
)
from .encoder_decoder_layers import (
    BiEncoder, BiEncoderLayer, BiDecoderLayer
)
from .source_choice_adapter import (
    build_mcln_source_choice_batch,
    compute_default_source_scores,
)
from .source_choice_selector import SourceChoiceSelector
from .source_moe import SourceMoE
from .mask_fusion import (
    BoundaryAwareSuperpointGraphMaskRefiner,
    EvidenceGeometryQuerySuperpointMaskRefiner,
    QueryMaskFusionCalibrator,
    apply_query_mask_calibration,
    apply_query_superpoint_mask_residual,
    build_query_mask_source_evidence,
    query_fusion_weight,
)
from .joint_query_quality import (
    JointQueryQualityReranker,
    build_joint_query_gate_evidence,
    summarize_joint_query_residual,
)
from .structured_slots import StructuredSlotBuilder
from .sacr_head import SACRHead
from .structured_source import (
    apply_authoritative_coverage,
    build_decomposition_masks,
    build_token_span_tensors,
)
from utils.scatter_util import deterministic_scatter_mean_dim0
import pointnet2_utils
import einops
def calc_pairwise_locs(obj_centers,  eps=1e-10):

    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
        - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps) # (b, l, l)
    norm_pairwise_dists = pairwise_dists

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2]**2, 3)+eps)

    pairwise_locs = torch.stack(
        [norm_pairwise_dists, pairwise_locs[..., 2]/pairwise_dists, 
        pairwise_dists_2d/pairwise_dists, pairwise_locs[..., 1]/pairwise_dists_2d,
        pairwise_locs[..., 0]/pairwise_dists_2d],
        dim=3
    )
    return pairwise_locs


class SWA(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        
        query = self.with_pos_embed(query, pe)
        B = query.shape[1]
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[0], k.shape[0])
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=None,attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight, src_weight = self.attn(query, k, v)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output.transpose(0,1), output_weight, src_weight # (b, n_q, d_model), (b, n_q, n_v)

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output


class DecoderQueryTextAdapter(nn.Module):
    """Zero-initialized cross-modal residual for final decoder queries."""

    def __init__(self, d_model=288, hidden_dim=288, num_heads=4,
                 dropout=0.1, max_delta=0.25):
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if num_heads <= 0:
            raise ValueError("adapter num_heads must be positive")
        if d_model % num_heads != 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "adapter d_model and hidden_dim must be divisible by num_heads"
            )
        if not math.isfinite(float(max_delta)) or float(max_delta) <= 0.0:
            raise ValueError("adapter max_delta must be finite and positive")

        self.max_delta = float(max_delta)
        self.query_norm = nn.LayerNorm(d_model)
        self.text_norm = nn.LayerNorm(d_model)
        self.text_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(4 * d_model + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.set_norm = nn.LayerNorm(hidden_dim)
        self.set_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.set_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, d_model)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _normalized_geometry(centers, sizes):
        if centers.ndim != 3 or sizes.ndim != 3:
            raise ValueError("adapter centers and sizes must be rank-three")
        if centers.shape != sizes.shape or centers.shape[-1] != 3:
            raise ValueError("adapter centers and sizes must have shape [B, Q, 3]")
        geometry = torch.cat(
            [centers, sizes.clamp(min=1e-4).log()], dim=-1
        )
        mean = geometry.mean(dim=1, keepdim=True)
        variance = geometry.var(dim=1, keepdim=True, unbiased=False)
        return (geometry - mean) * torch.rsqrt(variance + 1e-6)

    def forward(self, query, text_feats, text_padding_mask, centers, sizes):
        if query.ndim != 3 or text_feats.ndim != 3:
            raise ValueError("adapter query and text features must be rank-three")
        if query.shape[0] != text_feats.shape[0]:
            raise ValueError("adapter query/text batch sizes must match")
        if query.shape[-1] != text_feats.shape[-1]:
            raise ValueError("adapter query/text feature dimensions must match")
        if text_padding_mask.shape != text_feats.shape[:2]:
            raise ValueError("adapter text padding mask shape is invalid")

        padding_mask = text_padding_mask.bool()
        if padding_mask.all(dim=1).any():
            padding_mask = padding_mask.clone()
            padding_mask[padding_mask.all(dim=1), 0] = False

        normalized_query = self.query_norm(query)
        normalized_text = self.text_norm(text_feats)
        text_message, _ = self.text_attention(
            normalized_query,
            normalized_text,
            normalized_text,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        geometry = self.geometry_encoder(
            self._normalized_geometry(centers, sizes)
        )
        semantic = self.semantic_encoder(torch.cat([
            normalized_query,
            text_message,
            normalized_query * text_message,
            (normalized_query - text_message).abs(),
            geometry,
        ], dim=-1))
        set_input = self.set_norm(semantic)
        set_message, _ = self.set_attention(
            set_input, set_input, set_input, need_weights=False
        )
        hidden = semantic + self.set_dropout(set_message)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        residual = self.max_delta * torch.tanh(self.output(hidden))
        return query + residual, residual



class MCLN(nn.Module):
    """
    3D language grounder.

    Args:
        num_class (int): number of semantics classes to predict
        num_obj_class (int): number of object classes
        input_feature_dim (int): feat_dim of pointcloud (without xyz)
        num_queries (int): Number of queries generated
        num_decoder_layers (int): number of decoder layers
        self_position_embedding (str or None): how to compute pos embeddings
        contrastive_align_loss (bool): contrast queries and token features
        d_model (int): dimension of features
        butd (bool): use detected box stream
        pointnet_ckpt (str or None): path to pre-trained pp++ checkpoint
        self_attend (bool): add self-attention in encoder
    """

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True, use_source_choice_selector=False,
                 source_choice_selector_sources="default,mask_text",
                 source_choice_selector_hidden_dim=288,
                 use_source_moe=False,
                 source_moe_shared_source="default",
                 source_moe_top_k=2,
                 source_moe_balance_loss_weight=0.01,
                 source_moe_query_layers=1,
                 source_moe_query_heads=4,
                 source_moe_query_dropout=0.1,
                 source_moe_query_max_delta=0.25,
                 source_moe_use_fallback_gate=False,
                 source_moe_gate_hidden_dim=128,
                 source_moe_gate_candidate_top_k=8,
                 source_moe_gate_break_cost=2.0,
                 source_moe_gate_decision_margin=0.0,
                 source_moe_gate_mask_utility_weight=0.25,
                 source_moe_gate_uncertainty_weight=0.0,
                 source_moe_gate_use_evidence_features=False,
                 source_moe_gate_context_layers=0,
                 source_moe_gate_context_heads=4,
                 source_moe_gate_context_dropout=0.1,
                 source_moe_gate_action_mode="decision",
                 use_query_mask_fusion_calibrator=False,
                 query_mask_fusion_hidden_dim=128,
                 query_mask_fusion_dropout=0.0,
                 query_mask_fusion_max_delta=0.25,
                 query_mask_fusion_detach_inputs=True,
                 use_egqs_mask_refiner=False,
                 egqs_mask_refiner_arch="egqs",
                 egqs_mask_refiner_hidden_dim=32,
                 egqs_mask_refiner_max_delta=2.0,
                 egqs_mask_refiner_components="all",
                 egqs_mask_refiner_graph_mode="bilateral",
                 egqs_mask_refiner_neighbor_count=8,
                 egqs_mask_refiner_detach_inputs=True,
                 use_joint_query_quality_reranker=False,
                 joint_query_quality_hidden_dim=128,
                 joint_query_quality_heads=4,
                 joint_query_quality_layers=1,
                 joint_query_quality_dropout=0.1,
                 joint_query_quality_max_delta=1.25,
                 joint_query_quality_mask_weight=0.25,
                 joint_query_quality_score_weight=1.0,
                 joint_query_quality_direct_residual_scale=1.0,
                 joint_query_quality_use_metric_aligned_utility=False,
                 joint_query_quality_preserve_parent_score=False,
                 joint_query_quality_candidate_promotion_margin=0.0,
                 joint_query_quality_use_parent_transition_advantage=False,
                 joint_query_quality_use_decomposed_transition_advantage=False,
                 joint_query_quality_use_setwise_tier_advantage=False,
                 joint_query_quality_use_decoupled_setwise_heads=False,
                 joint_query_quality_use_factorized_setwise_safety=False,
                 joint_query_quality_use_factorized_setwise_risk_bound=False,
                 joint_query_quality_use_setwise_safety_veto_gate=False,
                 joint_query_quality_use_cost_calibrated_setwise_risk_bound=False,
                  joint_query_quality_use_setwise_safety_slack_quantile_bound=False,
                  joint_query_quality_use_setwise_safety_slack_pairwise_order=False,
                  joint_query_quality_use_proposal_conditioned_safety=False,
                  joint_query_quality_use_parent_referenced_safety=False,
                  joint_query_quality_use_coupled_safe_repair_witness=False,
                  joint_query_quality_use_bidirectional_coupled_boundary=False,
                  joint_query_quality_use_centered_coupled_separation=False,
                  joint_query_quality_use_hazard_conditioned_coupled_separation=False,
                  joint_query_quality_use_monotonic_box_safety_folding=False,
                  joint_query_quality_use_same_candidate_branchwise_witness=False,
                  joint_query_quality_use_parent_non_degradation_certificate=False,
                  joint_query_quality_use_criterion_responsible_hazard_attribution=False,
                  joint_query_quality_use_independent_joint_hazard_certificate=False,
                  joint_query_quality_use_frozen_raw_joint_hazard_features=False,
                 joint_query_quality_use_factorized_hit_advantage=False,
                 joint_query_quality_use_factorized_nested_dominance=False,
                 joint_query_quality_factorized_hit_break_cost=4.0,
                 joint_query_quality_parent_transition_break_cost=4.0,
                 joint_query_quality_parent_transition_candidate_top_k=0,
                 joint_query_quality_use_mask_calibration=False,
                 joint_query_quality_max_mask_alpha_delta=1.0,
                 joint_query_quality_max_mask_logit_bias=2.0,
                 joint_query_quality_use_source_mask_evidence=False,
                 joint_query_quality_use_gate_evidence=False,
                 joint_query_quality_use_spatial_mask_refiner=False,
                 joint_query_quality_spatial_mask_hidden_dim=32,
                 joint_query_quality_max_spatial_mask_delta=2.0,
                 joint_query_quality_use_adaptive_source_mixing=False,
                 joint_query_quality_use_source_distribution_reliability=False,
                 joint_query_quality_source_names="",
                 joint_query_quality_max_source_mix_delta=1.0,
                 joint_query_quality_source_mix_temperature=0.5,
                 joint_query_quality_detach_inputs=True,
                 use_decoder_query_adapter=False,
                 decoder_query_adapter_hidden_dim=288,
                 decoder_query_adapter_heads=4,
                 decoder_query_adapter_dropout=0.1,
                 decoder_query_adapter_max_delta=0.25,
                 use_sacr_source=False,
                 use_sacr_score_refiner=False,
                 sacr_score_max_delta=0.25,
                 sacr_hidden_dim=288,
                 sacr_max_pairs=3,
                 sacr_top_m_targets=32,
                 sacr_top_k_anchors=16,
                 sacr_geo_dim=16,
                 sacr_min_parse_confidence=0.0,
                 sacr_score_contract_audit=False,
                 sacr_residual_scale_init=0.1):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd
        self.use_source_choice_selector = bool(use_source_choice_selector)
        self.use_source_moe = bool(use_source_moe)
        self.use_query_mask_fusion_calibrator = bool(
            use_query_mask_fusion_calibrator
        )
        self.use_egqs_mask_refiner = bool(use_egqs_mask_refiner)
        self.use_joint_query_quality_reranker = bool(
            use_joint_query_quality_reranker
        )
        self.use_decoder_query_adapter = bool(use_decoder_query_adapter)
        if not isinstance(
                joint_query_quality_preserve_parent_score, bool):
            raise ValueError(
                "joint_query_quality_preserve_parent_score must be boolean"
            )
        self.joint_query_quality_preserve_parent_score = (
            joint_query_quality_preserve_parent_score
        )
        if (not isinstance(joint_query_quality_candidate_promotion_margin,
                           (int, float))
                or isinstance(
                    joint_query_quality_candidate_promotion_margin, bool
                )
                or not math.isfinite(float(
                    joint_query_quality_candidate_promotion_margin
                ))
                or float(
                    joint_query_quality_candidate_promotion_margin
                ) < 0.0):
            raise ValueError(
                "joint query candidate promotion margin must be finite and "
                "non-negative"
            )
        self.joint_query_quality_candidate_promotion_margin = float(
            joint_query_quality_candidate_promotion_margin
        )
        if (self.joint_query_quality_candidate_promotion_margin > 0.0
                and not self.joint_query_quality_preserve_parent_score):
            raise ValueError(
                "candidate promotion margin requires preserved parent score"
            )
        if not isinstance(
                joint_query_quality_use_parent_transition_advantage, bool):
            raise ValueError(
                "joint query parent transition advantage must be boolean"
            )
        self.joint_query_quality_use_parent_transition_advantage = (
            joint_query_quality_use_parent_transition_advantage
        )
        if not isinstance(
                joint_query_quality_use_decomposed_transition_advantage,
                bool):
            raise ValueError(
                "joint query decomposed transition advantage must be boolean"
            )
        self.joint_query_quality_use_decomposed_transition_advantage = (
            joint_query_quality_use_decomposed_transition_advantage
        )
        if not isinstance(
                joint_query_quality_use_setwise_tier_advantage, bool):
            raise ValueError(
                "joint query setwise tier advantage must be boolean"
            )
        self.joint_query_quality_use_setwise_tier_advantage = (
            joint_query_quality_use_setwise_tier_advantage
        )
        if not isinstance(
                joint_query_quality_use_decoupled_setwise_heads, bool):
            raise ValueError(
                "joint query decoupled setwise heads must be boolean"
            )
        self.joint_query_quality_use_decoupled_setwise_heads = (
            joint_query_quality_use_decoupled_setwise_heads
        )
        if (self.joint_query_quality_use_decoupled_setwise_heads
                and not self.joint_query_quality_use_setwise_tier_advantage):
            raise ValueError(
                "decoupled setwise heads require setwise tier advantage"
            )
        if not isinstance(
                joint_query_quality_use_factorized_setwise_safety, bool):
            raise ValueError(
                "joint query factorized setwise safety must be boolean"
            )
        self.joint_query_quality_use_factorized_setwise_safety = (
            joint_query_quality_use_factorized_setwise_safety
        )
        if (self.joint_query_quality_use_factorized_setwise_safety
                and not self.joint_query_quality_use_decoupled_setwise_heads):
            raise ValueError(
                "factorized setwise safety requires decoupled setwise heads"
            )
        if not isinstance(
                joint_query_quality_use_factorized_setwise_risk_bound, bool):
            raise ValueError(
                "joint query factorized setwise risk bound must be boolean"
            )
        self.joint_query_quality_use_factorized_setwise_risk_bound = (
            joint_query_quality_use_factorized_setwise_risk_bound
        )
        if (self.joint_query_quality_use_factorized_setwise_risk_bound
                and not self.joint_query_quality_use_factorized_setwise_safety):
            raise ValueError(
                "factorized setwise risk bound requires factorized safety"
            )
        if not isinstance(
                joint_query_quality_use_setwise_safety_veto_gate, bool):
            raise ValueError(
                "joint query setwise safety veto gate must be boolean"
            )
        self.joint_query_quality_use_setwise_safety_veto_gate = (
            joint_query_quality_use_setwise_safety_veto_gate
        )
        if (self.joint_query_quality_use_setwise_safety_veto_gate
                and not self.joint_query_quality_use_decoupled_setwise_heads):
            raise ValueError(
                "setwise safety veto gate requires decoupled setwise heads"
            )
        if not isinstance(
                joint_query_quality_use_cost_calibrated_setwise_risk_bound,
                bool):
            raise ValueError(
                "joint query cost-calibrated risk bound must be boolean"
            )
        self.joint_query_quality_use_cost_calibrated_setwise_risk_bound = (
            joint_query_quality_use_cost_calibrated_setwise_risk_bound
        )
        if (self.joint_query_quality_use_cost_calibrated_setwise_risk_bound
                and not self.joint_query_quality_use_factorized_setwise_risk_bound):
            raise ValueError(
                "cost-calibrated risk bound requires factorized risk bound"
            )
        if not isinstance(
                joint_query_quality_use_setwise_safety_slack_quantile_bound,
                bool):
            raise ValueError(
                "joint query safety-slack quantile bound must be boolean"
            )
        self.joint_query_quality_use_setwise_safety_slack_quantile_bound = (
            joint_query_quality_use_setwise_safety_slack_quantile_bound
        )
        if (self.joint_query_quality_use_setwise_safety_slack_quantile_bound
                and not self.joint_query_quality_use_factorized_setwise_risk_bound):
            raise ValueError(
                "safety-slack quantile bound requires factorized risk bound"
            )
        if (self.joint_query_quality_use_setwise_safety_slack_quantile_bound
                and self.joint_query_quality_use_cost_calibrated_setwise_risk_bound):
            raise ValueError(
                "safety-slack quantile bound and cost-calibrated risk bound "
                "are mutually exclusive"
            )
        if not isinstance(
                joint_query_quality_use_setwise_safety_slack_pairwise_order,
                bool):
            raise ValueError(
                "joint query safety-slack pairwise order must be boolean"
            )
        self.joint_query_quality_use_setwise_safety_slack_pairwise_order = (
            joint_query_quality_use_setwise_safety_slack_pairwise_order
        )
        if (self.joint_query_quality_use_setwise_safety_slack_pairwise_order
                and not self.joint_query_quality_use_setwise_safety_slack_quantile_bound):
            raise ValueError(
                "safety-slack pairwise order requires slack quantile bound"
            )
        if not isinstance(
                joint_query_quality_use_proposal_conditioned_safety, bool):
            raise ValueError(
                "joint query proposal-conditioned safety must be boolean"
            )
        self.joint_query_quality_use_proposal_conditioned_safety = (
            joint_query_quality_use_proposal_conditioned_safety
        )
        if (self.joint_query_quality_use_proposal_conditioned_safety
                and not self.joint_query_quality_use_setwise_safety_slack_pairwise_order):
            raise ValueError(
                "proposal-conditioned safety requires safety-slack pairwise "
                "order"
            )
        if (self.joint_query_quality_use_proposal_conditioned_safety
                and not self.joint_query_quality_use_setwise_safety_veto_gate):
            raise ValueError(
                "proposal-conditioned safety requires the safety veto gate"
            )
        if not isinstance(
                joint_query_quality_use_parent_referenced_safety, bool):
            raise ValueError(
                "joint query parent-referenced safety must be boolean"
            )
        self.joint_query_quality_use_parent_referenced_safety = (
            joint_query_quality_use_parent_referenced_safety
        )
        if (self.joint_query_quality_use_parent_referenced_safety
                and not self.joint_query_quality_use_setwise_safety_slack_pairwise_order):
            raise ValueError(
                "parent-referenced safety requires safety-slack pairwise "
                "order"
            )
        if (self.joint_query_quality_use_parent_referenced_safety
                and self.joint_query_quality_use_proposal_conditioned_safety):
            raise ValueError(
                "parent-referenced and proposal-conditioned safety are "
                "mutually exclusive"
            )
        if not isinstance(
                joint_query_quality_use_coupled_safe_repair_witness, bool):
            raise ValueError(
                "joint query coupled safe-repair witness must be boolean"
            )
        self.joint_query_quality_use_coupled_safe_repair_witness = (
            joint_query_quality_use_coupled_safe_repair_witness
        )
        if (self.joint_query_quality_use_coupled_safe_repair_witness
                and not self.joint_query_quality_use_parent_referenced_safety):
            raise ValueError(
                "coupled safe-repair witness requires parent-referenced "
                "safety"
            )
        if not isinstance(
                joint_query_quality_use_bidirectional_coupled_boundary, bool):
            raise ValueError(
                "joint query bidirectional coupled boundary must be boolean"
            )
        self.joint_query_quality_use_bidirectional_coupled_boundary = (
            joint_query_quality_use_bidirectional_coupled_boundary
        )
        if (self.joint_query_quality_use_bidirectional_coupled_boundary
                and not self.joint_query_quality_use_coupled_safe_repair_witness):
            raise ValueError(
                "bidirectional coupled boundary requires coupled "
                "safe-repair witness"
            )
        if not isinstance(
                joint_query_quality_use_centered_coupled_separation, bool):
            raise ValueError(
                "joint query centered coupled separation must be boolean"
            )
        self.joint_query_quality_use_centered_coupled_separation = (
            joint_query_quality_use_centered_coupled_separation
        )
        if (self.joint_query_quality_use_centered_coupled_separation
                and not self.joint_query_quality_use_bidirectional_coupled_boundary):
            raise ValueError(
                "centered coupled separation requires bidirectional "
                "coupled boundary"
            )
        if not isinstance(
                joint_query_quality_use_hazard_conditioned_coupled_separation,
                bool):
            raise ValueError(
                "joint query hazard-conditioned coupled separation must be "
                "boolean"
            )
        self.joint_query_quality_use_hazard_conditioned_coupled_separation = (
            joint_query_quality_use_hazard_conditioned_coupled_separation
        )
        if (self.joint_query_quality_use_hazard_conditioned_coupled_separation
                and not self.joint_query_quality_use_centered_coupled_separation):
            raise ValueError(
                "hazard-conditioned coupled separation requires centered "
                "coupled separation"
            )
        if not isinstance(
                joint_query_quality_use_monotonic_box_safety_folding, bool):
            raise ValueError(
                "joint query monotonic box-safety folding must be boolean"
            )
        self.joint_query_quality_use_monotonic_box_safety_folding = (
            joint_query_quality_use_monotonic_box_safety_folding
        )
        if (self.joint_query_quality_use_monotonic_box_safety_folding
                and not self.joint_query_quality_use_hazard_conditioned_coupled_separation):
            raise ValueError(
                "monotonic box-safety folding requires hazard-conditioned "
                "coupled separation"
            )
        if not isinstance(
                joint_query_quality_use_same_candidate_branchwise_witness,
                bool):
            raise ValueError(
                "joint query same-candidate branchwise witness must be boolean"
            )
        self.joint_query_quality_use_same_candidate_branchwise_witness = (
            joint_query_quality_use_same_candidate_branchwise_witness
        )
        if (self.joint_query_quality_use_same_candidate_branchwise_witness
                and not self.joint_query_quality_use_monotonic_box_safety_folding):
            raise ValueError(
                "same-candidate branchwise witness requires monotonic "
                "box-safety folding"
            )
        if not isinstance(
                joint_query_quality_use_parent_non_degradation_certificate,
                bool):
            raise ValueError(
                "joint query parent non-degradation certificate must be "
                "boolean"
            )
        self.joint_query_quality_use_parent_non_degradation_certificate = (
            joint_query_quality_use_parent_non_degradation_certificate
        )
        if (self.joint_query_quality_use_parent_non_degradation_certificate
                and not self.joint_query_quality_use_same_candidate_branchwise_witness):
            raise ValueError(
                "parent non-degradation certificate requires same-candidate "
                "branchwise witness"
            )
        if not isinstance(
                joint_query_quality_use_criterion_responsible_hazard_attribution,
                bool):
            raise ValueError(
                "joint query criterion-responsible hazard attribution must be "
                "boolean"
            )
        self.joint_query_quality_use_criterion_responsible_hazard_attribution = (
            joint_query_quality_use_criterion_responsible_hazard_attribution
        )
        if (self.joint_query_quality_use_criterion_responsible_hazard_attribution
                and not self.joint_query_quality_use_parent_non_degradation_certificate):
            raise ValueError(
                "criterion-responsible hazard attribution requires parent "
                "non-degradation certificate"
            )
        if not isinstance(
                joint_query_quality_use_independent_joint_hazard_certificate,
                bool):
            raise ValueError(
                "joint query independent joint-hazard certificate must be "
                "boolean"
            )
        self.joint_query_quality_use_independent_joint_hazard_certificate = (
            joint_query_quality_use_independent_joint_hazard_certificate
        )
        if (self.joint_query_quality_use_independent_joint_hazard_certificate
                and not self.joint_query_quality_use_parent_non_degradation_certificate):
            raise ValueError(
                "independent joint-hazard certificate requires parent "
                "non-degradation certificate"
            )
        if (self.joint_query_quality_use_independent_joint_hazard_certificate
                and self.joint_query_quality_use_criterion_responsible_hazard_attribution):
            raise ValueError(
                "independent joint-hazard certificate and criterion-"
                "responsible attribution are mutually exclusive"
            )
        if not isinstance(
                joint_query_quality_use_frozen_raw_joint_hazard_features,
                bool):
            raise ValueError(
                "joint query frozen raw joint-hazard features must be boolean"
            )
        self.joint_query_quality_use_frozen_raw_joint_hazard_features = (
            joint_query_quality_use_frozen_raw_joint_hazard_features
        )
        if (self.joint_query_quality_use_frozen_raw_joint_hazard_features
                and not self.joint_query_quality_use_independent_joint_hazard_certificate):
            raise ValueError(
                "frozen raw joint-hazard features require independent "
                "joint-hazard certificate"
            )
        if not isinstance(
                joint_query_quality_use_factorized_hit_advantage, bool):
            raise ValueError(
                "joint query factorized hit advantage must be boolean"
            )
        self.joint_query_quality_use_factorized_hit_advantage = (
            joint_query_quality_use_factorized_hit_advantage
        )
        if not isinstance(
                joint_query_quality_use_factorized_nested_dominance, bool):
            raise ValueError(
                "joint query factorized nested dominance must be boolean"
            )
        self.joint_query_quality_use_factorized_nested_dominance = (
            joint_query_quality_use_factorized_nested_dominance
        )
        if (self.joint_query_quality_use_factorized_nested_dominance
                and not self.joint_query_quality_use_factorized_hit_advantage):
            raise ValueError(
                "joint query factorized nested dominance requires "
                "factorized hit advantage"
            )
        if (not isinstance(joint_query_quality_factorized_hit_break_cost,
                           (int, float))
                or isinstance(joint_query_quality_factorized_hit_break_cost,
                              bool)
                or not math.isfinite(float(
                    joint_query_quality_factorized_hit_break_cost
                ))
                or float(joint_query_quality_factorized_hit_break_cost) <= 0):
            raise ValueError(
                "joint query factorized hit break cost must be positive"
            )
        self.joint_query_quality_factorized_hit_break_cost = float(
            joint_query_quality_factorized_hit_break_cost
        )
        if sum((
                self.joint_query_quality_use_parent_transition_advantage,
                self.joint_query_quality_use_decomposed_transition_advantage,
                self.joint_query_quality_use_setwise_tier_advantage,
                self.joint_query_quality_use_factorized_hit_advantage,
        )) > 1:
            raise ValueError(
                "transition advantage modes are mutually exclusive"
            )
        if (not isinstance(joint_query_quality_parent_transition_break_cost,
                           (int, float))
                or isinstance(
                    joint_query_quality_parent_transition_break_cost, bool
                )
                or not math.isfinite(float(
                    joint_query_quality_parent_transition_break_cost
                ))
                or float(
                    joint_query_quality_parent_transition_break_cost
                ) <= 0.0):
            raise ValueError(
                "joint query parent transition break cost must be finite "
                "and positive"
            )
        self.joint_query_quality_parent_transition_break_cost = float(
            joint_query_quality_parent_transition_break_cost
        )
        if (not isinstance(
                joint_query_quality_parent_transition_candidate_top_k, int)
                or isinstance(
                    joint_query_quality_parent_transition_candidate_top_k,
                    bool,
                )
                or joint_query_quality_parent_transition_candidate_top_k < 0):
            raise ValueError(
                "joint query parent transition candidate top k must be "
                "non-negative int"
            )
        self.joint_query_quality_parent_transition_candidate_top_k = int(
            joint_query_quality_parent_transition_candidate_top_k
        )
        if (self.joint_query_quality_use_parent_transition_advantage
                and not self.joint_query_quality_preserve_parent_score):
            raise ValueError(
                "parent transition advantage requires preserved parent score"
            )
        if (self.joint_query_quality_use_decomposed_transition_advantage
                and not self.joint_query_quality_preserve_parent_score):
            raise ValueError(
                "decomposed transition advantage requires preserved parent "
                "score"
            )
        if (self.joint_query_quality_use_setwise_tier_advantage
                and not self.joint_query_quality_preserve_parent_score):
            raise ValueError(
                "setwise tier advantage requires preserved parent score"
            )
        if (self.joint_query_quality_use_factorized_hit_advantage
                and not self.joint_query_quality_preserve_parent_score):
            raise ValueError(
                "factorized hit advantage requires preserved parent score"
            )
        if not isinstance(joint_query_quality_use_mask_calibration, bool):
            raise ValueError(
                "joint_query_quality_use_mask_calibration must be boolean"
            )
        self.joint_query_quality_use_mask_calibration = (
            joint_query_quality_use_mask_calibration
        )
        if not isinstance(
                joint_query_quality_use_source_mask_evidence, bool):
            raise ValueError(
                "joint_query_quality_use_source_mask_evidence must be boolean"
            )
        self.joint_query_quality_use_source_mask_evidence = (
            joint_query_quality_use_source_mask_evidence
        )
        if not isinstance(joint_query_quality_use_gate_evidence, bool):
            raise ValueError(
                "joint_query_quality_use_gate_evidence must be boolean"
            )
        self.joint_query_quality_use_gate_evidence = (
            joint_query_quality_use_gate_evidence
        )
        if not isinstance(
                joint_query_quality_use_spatial_mask_refiner, bool):
            raise ValueError(
                "joint_query_quality_use_spatial_mask_refiner must be boolean"
            )
        self.joint_query_quality_use_spatial_mask_refiner = (
            joint_query_quality_use_spatial_mask_refiner
        )
        if not isinstance(
                joint_query_quality_use_adaptive_source_mixing, bool):
            raise ValueError(
                "joint_query_quality_use_adaptive_source_mixing must be "
                "boolean"
            )
        self.joint_query_quality_use_adaptive_source_mixing = (
            joint_query_quality_use_adaptive_source_mixing
        )
        if not isinstance(
                joint_query_quality_use_source_distribution_reliability,
                bool):
            raise ValueError(
                "joint_query_quality_use_source_distribution_reliability "
                "must be boolean"
            )
        self.joint_query_quality_use_source_distribution_reliability = (
            joint_query_quality_use_source_distribution_reliability
        )
        source_moe_gate_action_mode = (
            source_moe_gate_action_mode or "decision"
        )
        self.source_moe_gate_use_rich_features = (
            source_moe_gate_action_mode in (
                "cascade_v19_rich_set_correction",
                "cascade_v23_dense_quality_correction",
                "cascade_v24_relative_risk_correction",
                "cascade_v25_pairwise_calibrated_correction",
                "cascade_v26_prior_restored_pairwise_correction",
                "cascade_v28_selected_abstention_correction",
                "cascade_v29_counterfactual_selected_correction",
                "cascade_v37_counterfactual_benefit_hazard_correction",
                "cascade_v38_complementary_logodds_correction",
                "cascade_v39_hazard_residual_correction",
            )
        )
        if not isinstance(source_moe_gate_use_evidence_features, bool):
            raise ValueError(
                "source_moe_gate_use_evidence_features must be boolean"
            )
        self.source_moe_gate_use_evidence_features = (
            source_moe_gate_use_evidence_features
        )
        self.source_moe_shared_source = str(source_moe_shared_source)
        self.source_choice_selector_sources = tuple(
            s.strip() for s in source_choice_selector_sources.split(',')
            if s.strip()
        )
        requested_joint_sources = tuple(
            source.strip()
            for source in str(joint_query_quality_source_names).split(',')
            if source.strip()
        )
        self.joint_query_quality_source_names = (
            requested_joint_sources
            if requested_joint_sources
            else self.source_choice_selector_sources
        )
        self.use_sacr_source = bool(use_sacr_source)
        self.use_sacr_score_refiner = bool(use_sacr_score_refiner)
        self.sacr_score_contract_audit = bool(
            sacr_score_contract_audit
        )
        if self.sacr_score_contract_audit and not self.use_sacr_score_refiner:
            raise ValueError(
                "SACR score contract audit requires score refinement"
            )
        if self.use_sacr_source and self.use_sacr_score_refiner:
            raise ValueError(
                "SACR source mixing and score-only refinement are exclusive"
            )
        if self.use_sacr_score_refiner and not (
                self.use_source_moe or self.use_source_choice_selector):
            raise ValueError(
                "SACR score refinement requires a parent score arbiter"
            )
        if (
            not isinstance(sacr_score_max_delta, (float, int))
            or isinstance(sacr_score_max_delta, bool)
            or not math.isfinite(float(sacr_score_max_delta))
            or not 0.0 < float(sacr_score_max_delta) <= 0.25
        ):
            raise ValueError(
                "sacr_score_max_delta must be finite and in (0,0.25]"
            )
        self.sacr_score_max_delta = float(sacr_score_max_delta)
        self.sacr_min_parse_confidence = float(
            sacr_min_parse_confidence
        )
        if not 0.0 <= self.sacr_min_parse_confidence < 1.0:
            raise ValueError(
                "sacr_min_parse_confidence must be in [0,1)"
            )
        if self.use_source_choice_selector and self.use_source_moe:
            raise ValueError(
                "use_source_choice_selector and use_source_moe are mutually "
                "exclusive: both write selected_source_scores"
            )
        if self.source_moe_gate_use_evidence_features and not (
                self.use_source_moe and source_moe_use_fallback_gate):
            raise ValueError(
                "source_moe_gate_use_evidence_features requires an enabled "
                "SourceMoE fallback gate"
            )
        if self.source_moe_gate_use_rich_features and not (
                self.use_source_moe and source_moe_use_fallback_gate):
            raise ValueError(
                "cascade_v19_rich_set_correction requires an enabled "
                "SourceMoE fallback gate"
            )
        if (self.source_moe_gate_use_rich_features
                and not contrastive_align_loss):
            raise ValueError(
                "cascade_v19_rich_set_correction requires "
                "contrastive_align_loss"
            )
        if self.use_joint_query_quality_reranker and not (
                self.use_source_moe or self.use_source_choice_selector):
            raise ValueError(
                "joint query quality reranking requires SourceMoE or the "
                "source-choice selector"
            )
        if (self.joint_query_quality_preserve_parent_score
                and not self.use_joint_query_quality_reranker):
            raise ValueError(
                "parent score preservation requires joint query quality"
            )
        if (self.use_joint_query_quality_reranker
                and not contrastive_align_loss):
            raise ValueError(
                "joint query quality reranking requires contrastive features"
            )
        if (self.joint_query_quality_use_mask_calibration
                and not self.use_joint_query_quality_reranker):
            raise ValueError(
                "joint query mask calibration requires joint query quality"
            )
        if (self.joint_query_quality_use_source_mask_evidence
                and not self.joint_query_quality_use_mask_calibration):
            raise ValueError(
                "source mask evidence requires joint query mask calibration"
            )
        if (self.joint_query_quality_use_spatial_mask_refiner
                and not self.joint_query_quality_use_mask_calibration):
            raise ValueError(
                "spatial mask refinement requires joint query mask calibration"
            )
        if self.joint_query_quality_use_adaptive_source_mixing:
            if not self.use_joint_query_quality_reranker:
                raise ValueError(
                    "adaptive source mixing requires joint query quality"
                )
            if (len(self.joint_query_quality_source_names) < 2
                    or self.source_moe_shared_source
                    not in self.joint_query_quality_source_names):
                raise ValueError(
                    "adaptive source mixing requires a shared source and at "
                    "least one routed source"
                )
        if (self.joint_query_quality_use_source_distribution_reliability
                and not self.joint_query_quality_use_adaptive_source_mixing):
            raise ValueError(
                "source distribution reliability requires adaptive source "
                "mixing"
            )
        if self.use_sacr_source:
            if not self.use_joint_query_quality_reranker:
                raise ValueError(
                    "SACR source requires joint query quality reranking"
                )
            if "sacr_structured" not in self.joint_query_quality_source_names:
                raise ValueError(
                    "SACR source must be present in the joint source pool"
                )
            if not self.joint_query_quality_use_adaptive_source_mixing:
                raise ValueError(
                    "SACR source requires adaptive source mixing"
                )
        if (self.joint_query_quality_use_gate_evidence and not (
                self.use_joint_query_quality_reranker
                and self.use_source_moe
                and source_moe_use_fallback_gate)):
            raise ValueError(
                "joint query gate evidence requires joint reranking and an "
                "enabled SourceMoE fallback gate"
            )
#-------------------text-head decoder-------------------------------------
        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.swa_layers = nn.ModuleList([])
        self.swa_ffn_layers = nn.ModuleList([])
        self.rel_encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 288))
        for i in range(3):
            self.swa_layers.append(SWA(d_model, nhead=8,dropout=0.2))
            self.swa_ffn_layers.append(FFN(d_model, hidden_dim=128, dropout=0.2, activation_fn='relu'))
        # Visual encoder
        self.backbone_net = Pointnet2Backbone(
            input_feature_dim=input_feature_dim,
            width=1
        )
        if input_feature_dim == 3 and pointnet_ckpt is not None:
            self.backbone_net.load_state_dict(torch.load(
                pointnet_ckpt, map_location="cpu"
            ), strict=False)

        # Text Encoder
        # # (1) online
        # t_type = "roberta-base"
        # NOTE (2) offline: load from the local folder.
        t_type = f'{data_path}roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load(
                'data/class_embeddings3d.npy', allow_pickle=True
            ))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, d_model - 128)
            self.box_embeddings = PositionEmbeddingLearned(6, 128)

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Mask Feats Generation layer
        self.x_mask = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model, 1)
            )
        self.x_query = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model, 1)
            )
        self.text_query_proj=nn.Sequential(
            nn.Linear(d_model,2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model,2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model,d_model)
            )
        self.super_grouper = pointnet2_utils.QueryAndGroup(radius=0.2, nsample=2, use_xyz=False, normalize_xyz=True)

        # Query initialization
        self.points_obj_cls = PointsObjClsModule(d_model)
        self.gsample_module = GeneralSamplingModule()
        self.decoder_query_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(
            num_class, 1, num_queries, d_model,
            objectness=False, heading=False,
            compute_sem_scores=True
        )

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                d_model, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding=self_position_embedding, butd=self.butd
            ))

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_class, 1, num_queries, d_model,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        self.decoder_query_adapter = (
            DecoderQueryTextAdapter(
                d_model=d_model,
                hidden_dim=decoder_query_adapter_hidden_dim,
                num_heads=decoder_query_adapter_heads,
                dropout=decoder_query_adapter_dropout,
                max_delta=decoder_query_adapter_max_delta,
            )
            if self.use_decoder_query_adapter else None
        )

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )

        if self.use_source_choice_selector:
            selector_feat_dim = 64 if contrastive_align_loss else d_model
            self.source_choice_selector = SourceChoiceSelector(
                d_model=selector_feat_dim,
                hidden_dim=source_choice_selector_hidden_dim,
                source_names=self.source_choice_selector_sources,
                text_dim=d_model,
            )
        else:
            self.source_choice_selector = None

        if self.use_source_moe:
            moe_feat_dim = 64 if contrastive_align_loss else d_model
            self.source_moe = SourceMoE(
                source_names=self.source_choice_selector_sources,
                shared_source=self.source_moe_shared_source,
                d_model=moe_feat_dim,
                hidden_dim=source_choice_selector_hidden_dim,
                text_dim=d_model,
                top_k=source_moe_top_k,
                balance_loss_weight=source_moe_balance_loss_weight,
                query_layers=source_moe_query_layers,
                query_heads=source_moe_query_heads,
                query_dropout=source_moe_query_dropout,
                query_max_delta=source_moe_query_max_delta,
                use_fallback_gate=source_moe_use_fallback_gate,
                gate_hidden_dim=source_moe_gate_hidden_dim,
                gate_candidate_top_k=source_moe_gate_candidate_top_k,
                gate_break_cost=source_moe_gate_break_cost,
                gate_decision_margin=source_moe_gate_decision_margin,
                gate_mask_utility_weight=source_moe_gate_mask_utility_weight,
                gate_uncertainty_weight=(
                    source_moe_gate_uncertainty_weight
                ),
                gate_use_evidence_features=(
                    self.source_moe_gate_use_evidence_features
                ),
                gate_evidence_dim=d_model,
                gate_context_layers=source_moe_gate_context_layers,
                gate_context_heads=source_moe_gate_context_heads,
                gate_context_dropout=source_moe_gate_context_dropout,
                gate_action_mode=source_moe_gate_action_mode,
            )
        else:
            self.source_moe = None

        if self.use_query_mask_fusion_calibrator:
            self.query_mask_fusion_calibrator = QueryMaskFusionCalibrator(
                d_model=d_model,
                hidden_dim=query_mask_fusion_hidden_dim,
                dropout=query_mask_fusion_dropout,
                max_delta=query_mask_fusion_max_delta,
                detach_inputs=query_mask_fusion_detach_inputs,
            )
        else:
            self.query_mask_fusion_calibrator = None

        if self.use_egqs_mask_refiner:
            if egqs_mask_refiner_arch == "egqs":
                self.egqs_mask_refiner = (
                    EvidenceGeometryQuerySuperpointMaskRefiner(
                        d_model=d_model,
                        hidden_dim=egqs_mask_refiner_hidden_dim,
                        max_delta=egqs_mask_refiner_max_delta,
                        components=egqs_mask_refiner_components,
                        detach_inputs=egqs_mask_refiner_detach_inputs,
                    )
                )
            elif egqs_mask_refiner_arch == "graph":
                self.egqs_mask_refiner = (
                    BoundaryAwareSuperpointGraphMaskRefiner(
                        d_model=d_model,
                        neighbor_count=egqs_mask_refiner_neighbor_count,
                        max_delta=egqs_mask_refiner_max_delta,
                        graph_mode=egqs_mask_refiner_graph_mode,
                        detach_inputs=egqs_mask_refiner_detach_inputs,
                    )
                )
            else:
                raise ValueError(
                    "egqs_mask_refiner_arch must be 'egqs' or 'graph'"
                )
        else:
            self.egqs_mask_refiner = None

        if self.use_sacr_source:
            if (
                not isinstance(sacr_residual_scale_init, (float, int))
                or isinstance(sacr_residual_scale_init, bool)
                or not math.isfinite(float(sacr_residual_scale_init))
                or abs(float(sacr_residual_scale_init)) > 1.0
            ):
                raise ValueError(
                    "sacr_residual_scale_init must be finite and in [-1,1]"
                )
        if self.use_sacr_source or self.use_sacr_score_refiner:
            self.structured_slot_builder = StructuredSlotBuilder(
                d_model=d_model,
                pooling="attention",
                max_pairs=sacr_max_pairs,
            )
            self.sacr_head = SACRHead(
                d_model=d_model,
                hidden_dim=sacr_hidden_dim,
                top_m_targets=sacr_top_m_targets,
                top_k_anchors=sacr_top_k_anchors,
                geo_dim=sacr_geo_dim,
            )
        else:
            self.structured_slot_builder = None
            self.sacr_head = None

        if self.use_sacr_source:
            self.sacr_residual_scale = nn.Parameter(torch.tensor([
                float(sacr_residual_scale_init)
            ]))
        else:
            self.sacr_residual_scale = None
        self.sacr_score_gate = (
            nn.Parameter(torch.zeros(1))
            if self.use_sacr_score_refiner else None
        )

        if self.use_joint_query_quality_reranker:
            self.joint_query_quality_reranker = JointQueryQualityReranker(
                input_dim=2 * 64 + 24,
                hidden_dim=joint_query_quality_hidden_dim,
                num_heads=joint_query_quality_heads,
                num_layers=joint_query_quality_layers,
                dropout=joint_query_quality_dropout,
                max_delta=joint_query_quality_max_delta,
                mask_weight=joint_query_quality_mask_weight,
                quality_score_weight=joint_query_quality_score_weight,
                direct_residual_scale=(
                    joint_query_quality_direct_residual_scale
                ),
                use_metric_aligned_utility=(
                    joint_query_quality_use_metric_aligned_utility
                ),
                preserve_parent_score=(
                    self.joint_query_quality_preserve_parent_score
                ),
                candidate_promotion_margin=(
                    self.joint_query_quality_candidate_promotion_margin
                ),
                use_parent_transition_advantage=(
                    self.joint_query_quality_use_parent_transition_advantage
                ),
                use_decomposed_transition_advantage=(
                    self.joint_query_quality_use_decomposed_transition_advantage
                ),
                use_setwise_tier_advantage=(
                    self.joint_query_quality_use_setwise_tier_advantage
                ),
                use_decoupled_setwise_heads=(
                    self.joint_query_quality_use_decoupled_setwise_heads
                ),
                use_factorized_setwise_safety=(
                    self.joint_query_quality_use_factorized_setwise_safety
                ),
                use_factorized_setwise_risk_bound=(
                    self.joint_query_quality_use_factorized_setwise_risk_bound
                ),
                use_setwise_safety_veto_gate=(
                    self.joint_query_quality_use_setwise_safety_veto_gate
                ),
                use_cost_calibrated_setwise_risk_bound=(
                    self.joint_query_quality_use_cost_calibrated_setwise_risk_bound
                ),
                use_setwise_safety_slack_quantile_bound=(
                    self.joint_query_quality_use_setwise_safety_slack_quantile_bound
                ),
                use_setwise_safety_slack_pairwise_order=(
                    self.joint_query_quality_use_setwise_safety_slack_pairwise_order
                ),
                use_proposal_conditioned_safety=(
                    self.joint_query_quality_use_proposal_conditioned_safety
                ),
                use_parent_referenced_safety=(
                    self.joint_query_quality_use_parent_referenced_safety
                ),
                use_coupled_safe_repair_witness=(
                    self.joint_query_quality_use_coupled_safe_repair_witness
                ),
                use_bidirectional_coupled_boundary=(
                    self.joint_query_quality_use_bidirectional_coupled_boundary
                ),
                use_centered_coupled_separation=(
                    self.joint_query_quality_use_centered_coupled_separation
                ),
                use_hazard_conditioned_coupled_separation=(
                    self.joint_query_quality_use_hazard_conditioned_coupled_separation
                ),
                use_monotonic_box_safety_folding=(
                    self.joint_query_quality_use_monotonic_box_safety_folding
                ),
                use_same_candidate_branchwise_witness=(
                    self.joint_query_quality_use_same_candidate_branchwise_witness
                ),
                use_parent_non_degradation_certificate=(
                    self.joint_query_quality_use_parent_non_degradation_certificate
                ),
                use_criterion_responsible_hazard_attribution=(
                    self.joint_query_quality_use_criterion_responsible_hazard_attribution
                ),
                use_independent_joint_hazard_certificate=(
                    self.joint_query_quality_use_independent_joint_hazard_certificate
                ),
                use_frozen_raw_joint_hazard_features=(
                    self.joint_query_quality_use_frozen_raw_joint_hazard_features
                ),
                use_factorized_hit_advantage=(
                    self.joint_query_quality_use_factorized_hit_advantage
                ),
                use_factorized_nested_dominance=(
                    self.joint_query_quality_use_factorized_nested_dominance
                ),
                factorized_hit_break_cost=(
                    self.joint_query_quality_factorized_hit_break_cost
                ),
                parent_transition_break_cost=(
                    self.joint_query_quality_parent_transition_break_cost
                ),
                parent_transition_candidate_top_k=(
                    self.joint_query_quality_parent_transition_candidate_top_k
                ),
                detach_inputs=joint_query_quality_detach_inputs,
                use_mask_calibration=(
                    self.joint_query_quality_use_mask_calibration
                ),
                max_mask_alpha_delta=(
                    joint_query_quality_max_mask_alpha_delta
                ),
                max_mask_logit_bias=joint_query_quality_max_mask_logit_bias,
                use_source_mask_evidence=(
                    self.joint_query_quality_use_source_mask_evidence
                ),
                use_gate_evidence=(
                    self.joint_query_quality_use_gate_evidence
                ),
                use_spatial_mask_refiner=(
                    self.joint_query_quality_use_spatial_mask_refiner
                ),
                spatial_mask_d_model=d_model,
                spatial_mask_hidden_dim=(
                    joint_query_quality_spatial_mask_hidden_dim
                ),
                max_spatial_mask_delta=(
                    joint_query_quality_max_spatial_mask_delta
                ),
                use_adaptive_source_mixing=(
                    self.joint_query_quality_use_adaptive_source_mixing
                ),
                source_count=len(self.joint_query_quality_source_names),
                shared_source_index=(
                    self.joint_query_quality_source_names.index(
                        self.source_moe_shared_source
                    )
                    if self.joint_query_quality_use_adaptive_source_mixing
                    else None
                ),
                max_source_mix_delta=(
                    joint_query_quality_max_source_mix_delta
                ),
                source_mix_temperature=(
                    joint_query_quality_source_mix_temperature
                ),
                use_source_distribution_reliability=(
                    self.joint_query_quality_use_source_distribution_reliability
                ),
            )
        else:
            self.joint_query_quality_reranker = None

        # Init
        self.init_bn_momentum()
    
    
    # BRIEF visual and text backbones.
    def _run_backbones(self, inputs):
        """Run visual and text backbones."""
        # step 1. Visual encoder
        end_points = self.backbone_net(inputs['point_clouds'], end_points={}) # 50000 points -> 1024 points
        end_points['seed_inds'] = end_points['fp2_inds']                      # [batch_size,point_num]       
        end_points['seed_xyz'] = end_points['fp2_xyz']                        # [batch_size,point_num,3]
        end_points['seed_features'] = end_points['fp2_features']              # [batch_size,feature_dim,point_num]
        
        # step 2. Text encoder
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'],
            padding="longest",
            return_tensors="pt",
            return_offsets_mapping=self.use_sacr_source,
            return_special_tokens_mask=self.use_sacr_source,
        ).to(inputs['point_clouds'].device)
        
        encoded_text = self.text_encoder(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask'],
        )
        text_feats = self.text_projector(encoded_text.last_hidden_state)

        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        end_points['text_feats'] = text_feats
        end_points['text_attention_mask'] = text_attention_mask
        end_points['tokenized'] = tokenized
        if self.structured_slot_builder is not None:
            batch_size = tokenized['input_ids'].shape[0]
            empty = [[] for _ in range(batch_size)]
            target_spans = build_token_span_tensors(
                tokenized,
                inputs.get('target_spans', empty),
                inputs['text'],
                text_feats.device,
            )
            entity_spans = build_token_span_tensors(
                tokenized,
                inputs.get('entity_spans', empty),
                inputs['text'],
                text_feats.device,
            )
            attr_spans = build_token_span_tensors(
                tokenized,
                inputs.get('attr_spans', empty),
                inputs['text'],
                text_feats.device,
            )
            rel_spans = build_token_span_tensors(
                tokenized,
                inputs.get('rel_spans', empty),
                inputs['text'],
                text_feats.device,
            )
            structured_anchor_ids = inputs.get('structured_anchor_ids')
            if structured_anchor_ids is not None:
                structured_anchor_ids = structured_anchor_ids.to(
                    device=text_feats.device, dtype=torch.long
                )
            slots = self.structured_slot_builder(
                token_feats=text_feats,
                attention_mask=tokenized['attention_mask'],
                target_spans=target_spans,
                entity_spans=entity_spans,
                attr_spans=attr_spans,
                rel_spans=rel_spans,
                anchor_ids=structured_anchor_ids,
            )
            end_points['structured_slots'] = apply_authoritative_coverage(
                inputs, slots
            )
        return end_points

    # BRIEF generate query.
    def _generate_queries(self, xyz, features, end_points):
        # kps sampling
        points_obj_cls_logits = self.points_obj_cls(features)  # [B, 1, K=1024]
        end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
        
        # top-k
        sample_inds = torch.topk(   
            torch.sigmoid(points_obj_cls_logits).squeeze(1),
            self.num_queries
        )[1].int()

        xyz, features, sample_inds = self.gsample_module(   
            xyz, features, sample_inds
        )

        end_points['query_points_xyz'] = xyz  # (B, V, 3)
        end_points['query_points_feature'] = features  # (B, F, V)
        end_points['query_points_sample_inds'] = sample_inds  # (B, V)
        return end_points
    
    # segmentation prediction
    def _seg_seeds_prediction(self, query, mask_feats, end_points, prefix=''):
        ## generate seed points masks
        pred_mask_seeds = torch.einsum('bnd,bdm->bnm', query, mask_feats)
        ## mapping seed points masks to superpoints masks
        end_points[f'{prefix}pred_mask_seeds'] = pred_mask_seeds
        return pred_mask_seeds

    def get_mask(self, query, mask_feats):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
        attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
        attn_masks = attn_masks.detach()

        return pred_masks, attn_masks

    def prediction_head(self, query, superpoint_feats):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, superpoint_feats)
        return pred_scores, pred_masks, attn_masks
    

    def avg_lang_feat(self, lang_feats, lang_masks):
        lang_len = lang_masks.sum(-1)
        lang_len = lang_len.unsqueeze(-1)
        lang_len[torch.where(lang_len == 0)] = 1
        return (lang_feats * ~lang_masks.unsqueeze(-1).expand_as(lang_feats)).sum(1) / lang_len

    # BRIEF forward.
    def forward(self, inputs):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B

                more keys if butd is enabled:
                    det_bbox_label_mask
                    det_boxes
                    det_class_ids
        Returns:
            end_points: dict
        """
        # STEP 1. vision and text encoding
        end_points = self._run_backbones(inputs)
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        text_feats = end_points['text_feats']
        text_padding_mask = end_points['text_attention_mask']
        end_points['coords'] = inputs['point_clouds'][..., 0:3].contiguous()
        
        # STEP 2. Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~inputs['det_bbox_label_mask']

            # step box position.    det_boxes ([B, 132, 6]) -->  ([B, 128, 132])
            box_embeddings = self.box_embeddings(inputs['det_boxes'])
            # step box class        det_class_ids ([B, 132])  -->  ([B, 132, 160])
            class_embeddings = self.class_embeddings(self.butd_class_embeddings(inputs['det_class_ids']))
            # step box feature     ([B, 132, 288])
            detected_feats = torch.cat([box_embeddings, class_embeddings.transpose(1, 2)]
                                        , 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None

        # STEP 3. Cross-modality encoding
        spatial_point_xyz=calc_pairwise_locs(points_xyz)
        points_features, text_feats = self.cross_encoder(
            vis_feats=points_features.transpose(1, 2).contiguous(),
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            padding_mask=torch.zeros(
                len(points_xyz), points_xyz.size(1)
            ).to(points_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask,
            spatial_point_xyz=spatial_point_xyz
        )
        points_features = points_features.transpose(1, 2)
        points_features = points_features.contiguous()
        end_points["text_memory"] = text_feats
        end_points['seed_features'] = points_features
        
        # STEP 4. text projection --> 64
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            end_points['proj_tokens'] = proj_tokens     # ([B, L, 64])

        # STEP 4.1 Mask Feats Generation
        mask_feats = self.x_mask(points_features)  # [B, 288, 1024]
        superpoint = inputs['superpoint']  # [B, 50000]
        end_points['superpoints'] = superpoint
        source_xzy = inputs['point_clouds'][..., 0:3].contiguous()  # [B, 50000, 3]
        super_features = []
        super_xyz_list = []
        for bs in range(source_xzy.shape[0]):
            super_xyz = deterministic_scatter_mean_dim0(source_xzy[bs], superpoint[bs]).unsqueeze(0)  # [1, super_num, 3]  计算每个超点的平均坐标，即得到中心坐标
            super_xyz_list.append(super_xyz)
            grouped_feature,ball_idx = self.super_grouper(points_xyz[bs].unsqueeze(0), super_xyz, mask_feats[bs].unsqueeze(0))  # [1, 288, super_num, nsample]
            grouped_xyz =(points_xyz[bs])[ball_idx.long().squeeze(0)].unsqueeze(0)
            super_xyz_expand=super_xyz.unsqueeze(2)
            rel_coord=grouped_xyz-super_xyz_expand
            rel_feat=(self.rel_encoder(rel_coord)).permute(0,3,1,2)
            grouped_feature=grouped_feature+rel_feat     
            super_feature = F.max_pool2d(grouped_feature, kernel_size=[1, grouped_feature.size(3)]).squeeze(-1).squeeze(0)  # [288, super_num]
            super_features.append(super_feature)

        # STEP 5. Query Points Generation
        end_points = self._generate_queries(
            points_xyz, points_features, end_points
        )
        cluster_feature = end_points['query_points_feature']    # (B, F=288, V=256)
        cluster_xyz = end_points['query_points_xyz']            # (B, V=256, 3)
        query = self.decoder_query_proj(cluster_feature)        
        query = query.transpose(1, 2).contiguous()              # (B, V=256, F=288)
        # projection 288 --> 64
        if self.contrastive_align_loss: 
            end_points['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )  # [B, 256, 64]

        # STEP 6.Proposals
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature,
            base_xyz=cluster_xyz,
            end_points=end_points,
            prefix='proposal_'
        )
        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()
        query_mask = None
        query_last = None

        # STEP 7. Decoder
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if i == self.num_decoder_layers-1 else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            # step Transformer Decoder Layer
            query = self.decoder[i](
                query, points_features.transpose(1, 2).contiguous(),
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(
                    detected_feats if self.butd
                    else None
                ),
                detected_mask=detected_mask if self.butd else None
            )  # (B, V, F)
            if (i == self.num_decoder_layers - 1
                    and self.decoder_query_adapter is not None):
                query, adapter_residual = self.decoder_query_adapter(
                    query,
                    text_feats,
                    text_padding_mask,
                    base_xyz,
                    base_size,
                )
                end_points['decoder_query_adapter_abs_residual_mean'] = (
                    adapter_residual.detach().abs().mean()
                )
                end_points['decoder_query_adapter_abs_residual_max'] = (
                    adapter_residual.detach().abs().amax()
                )
            # step project
            if self.contrastive_align_loss:
                proj_query = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )
                end_points[f'{prefix}proj_queries'] = proj_query
                if prefix == 'last_':
                    end_points['source_choice_candidate_feats'] = proj_query

            # step box Prediction head
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),     # ([B, F=288, V=256])
                base_xyz=cluster_xyz,                   # ([B, 256, 3])
                end_points=end_points,  # 
                prefix=prefix
            )
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

            query_last = query

        if self.source_moe_gate_use_evidence_features:
            end_points["source_moe_gate_candidate_feats"] = query_last

        decoder_query_last = query_last

        if self.use_sacr_source:
            slots = end_points.get('structured_slots')
            if slots is None:
                raise ValueError(
                    "SACR source requires structured slots from the text path"
                )
            candidate_boxes = torch.cat((
                end_points['last_center'],
                end_points['last_pred_size'].clamp(min=1e-6),
            ), dim=-1)
            default_scores = compute_default_source_scores(
                end_points, inputs
            )
            global_only, weak_generic = build_decomposition_masks(
                inputs,
                slots,
                min_parse_confidence=self.sacr_min_parse_confidence,
            )
            sacr_out = self.sacr_head(
                query_feats=query_last,
                pred_boxes=candidate_boxes,
                base_scores=default_scores,
                slot_dict=slots,
                global_only_mask=global_only,
                weak_generic_target_mask=weak_generic,
            )
            confidence = slots['parse_confidence'].float().unsqueeze(1)
            scale = self.sacr_residual_scale.tanh()
            end_points['sacr_structured_residual'] = (
                scale * confidence * sacr_out['structured_scores']
            )
            end_points['sacr_structured_valid_mask'] = sacr_out[
                'structured_valid_mask'
            ]
            end_points['sacr_target_attr_scores'] = sacr_out[
                'target_attr_scores'
            ]
            end_points['sacr_relation_anchor_scores'] = sacr_out[
                'relation_anchor_scores'
            ]
            end_points['sacr_parse_confidence'] = slots[
                'parse_confidence'
            ]
            end_points['sacr_residual_scale_value'] = scale.detach().reshape(())
            end_points['sacr_valid_ratio'] = sacr_out[
                'structured_valid_mask'
            ].float().mean().detach()
            end_points['sacr_relation_active_ratio'] = sacr_out[
                'relation_active_ratio'
            ].detach()

        # step Seg Prediction head
        query_last = self.x_query(query_last.transpose(1, 2)).transpose(1, 2)
#---------------------------text decoder-----------------------------------

        text_query=text_feats
        prediction_masks = []
        sp_pred_masks = []
        adaptive_weight_lists=[]

        for bs in range(query.shape[0]):
            bs_text_query=text_query[bs].unsqueeze(0)
            _, _, attn_masks = self.prediction_head(bs_text_query, super_features[bs].unsqueeze(0).transpose(1,2))
            for i in range(3): 
                bs_text_query, _,src_weight = self.swa_layers[i]( super_features[bs].unsqueeze(0).transpose(1,2).transpose(0,1),bs_text_query.transpose(0,1), attn_mask=attn_masks)#SWA模块
                bs_text_query = self.swa_ffn_layers[i](bs_text_query)
                _, _, attn_masks = self.prediction_head(bs_text_query, super_features[bs].unsqueeze(0).transpose(1,2))
            src_weight = src_weight.softmax(1)
            src_weight = torch.where(torch.isnan(src_weight), torch.zeros_like(src_weight), src_weight)#将src_weight中的NaN值替换为0
            q_score = (src_weight*~text_padding_mask[bs].unsqueeze(-1)).sum(-1) # [B, N_q]
            q_idx = q_score[0].argmax(dim=-1, keepdim=True) 
            pred_scores, pred_masks, _ = self.prediction_head(bs_text_query[:,q_idx,:], super_features[bs].unsqueeze(0).transpose(1,2))                
            prediction_masks.append(pred_masks.expand(1,256,pred_masks.shape[-1]))    # 将批次列表添加到总列表 
            adaptive_weight=torch.sigmoid(pred_scores.squeeze())
            adaptive_weight_lists.append(adaptive_weight)
            sp_pred_mask = self._seg_seeds_prediction(
                query_last[bs].unsqueeze(0),                                  # ([1, F=256, V=288])
                super_features[bs].unsqueeze(0),                             # ([1, F=288, V=super_num])
                end_points=end_points,  # 
                prefix=prefix
            ).squeeze(0)  
            sp_pred_masks.append(sp_pred_mask)
        if self.query_mask_fusion_calibrator is not None:
            base_alpha = torch.stack(
                [weight.reshape(()) for weight in adaptive_weight_lists], dim=0
            )
            mask_boxes = torch.cat((
                end_points['last_center'],
                end_points['last_pred_size'].clamp(min=1e-4),
            ), dim=-1)
            calibration = self.query_mask_fusion_calibrator(
                query_last,
                text_feats,
                text_padding_mask,
                mask_boxes,
                base_alpha,
            )
            end_points['query_mask_fusion_base_alpha'] = calibration[
                'base_alpha'
            ]
            end_points['query_mask_fusion_residual'] = calibration['residual']
            end_points['query_mask_fusion_abs_residual_mean'] = (
                calibration['residual'].detach().abs().mean()
            )
            end_points['query_mask_fusion_abs_residual_max'] = (
                calibration['residual'].detach().abs().amax()
            )
            end_points['query_mask_fusion_weight_std_mean'] = (
                calibration['weights'].detach().std(dim=1).mean()
            )
            adaptive_weight_lists = [
                row for row in calibration['weights'].unbind(dim=0)
            ]

        end_points['sp_last_pred_masks'] = sp_pred_masks  # list  BS* [256, super_num]
        end_points['last_pred_masks'] = prediction_masks  # bs*[ 256, super_num]
        end_points['adaptive_weights']=adaptive_weight_lists
        end_points['super_xyz_list'] = super_xyz_list 

        if self.source_choice_selector is not None or self.source_moe is not None:
            source_choice_batch = build_mcln_source_choice_batch(
                end_points,
                inputs,
                source_names=self.source_choice_selector_sources,
                include_rich_candidate_feats=(
                    self.source_moe_gate_use_rich_features
                    or self.use_joint_query_quality_reranker
                ),
            )
            arbiter = (
                self.source_choice_selector
                if self.source_choice_selector is not None
                else self.source_moe
            )
            arbiter_kwargs = dict(
                candidate_feats=source_choice_batch["candidate_feats"],
                candidate_boxes=source_choice_batch["candidate_boxes"],
                source_scores=source_choice_batch["source_scores"],
                valid_mask=source_choice_batch["valid_mask"],
                text_feats=source_choice_batch["text_feats"],
                text_mask=source_choice_batch["text_mask"],
            )
            if self.source_moe is not None:
                arbiter_kwargs["gate_candidate_feats"] = source_choice_batch[
                    "gate_candidate_feats"
                ]
                arbiter_kwargs["gate_rich_candidate_feats"] = (
                    source_choice_batch["rich_candidate_feats"]
                )
                arbiter_kwargs["source_validity"] = source_choice_batch[
                    "source_validity"
                ]
            selector_out = arbiter(**arbiter_kwargs)
            if (self.source_choice_selector is not None
                    and self.joint_query_quality_reranker is not None):
                valid_mask = source_choice_batch["valid_mask"]
                shared_source = self.source_moe_shared_source
                source_scores = source_choice_batch["source_scores"]
                if shared_source not in source_scores:
                    raise ValueError(
                        "selector shared source is unavailable: {}".format(
                            shared_source
                        )
                    )
                shared_scores = source_scores[shared_source]
                if (not isinstance(valid_mask, torch.Tensor)
                        or valid_mask.dtype != torch.bool
                        or not isinstance(shared_scores, torch.Tensor)
                        or shared_scores.shape != valid_mask.shape
                        or shared_scores.device != valid_mask.device
                        or not bool(valid_mask.any(dim=1).all().item())
                        or not bool(torch.isfinite(
                            shared_scores.masked_fill(~valid_mask, 0.0)
                        ).all().item())):
                    raise ValueError(
                        "selector shared query contract is invalid"
                    )
                selector_out.update({
                    "moe_shared_source": shared_source,
                    "moe_shared_query": shared_scores.masked_fill(
                        ~valid_mask, -float("inf")
                    ).argmax(dim=1),
                    "moe_valid_mask": valid_mask,
                })
            if self.joint_query_quality_reranker is not None:
                parent_scores = selector_out["selected_source_scores"]
                joint_source_batch = source_choice_batch
                joint_source_names = getattr(
                    self,
                    'joint_query_quality_source_names',
                    self.source_choice_selector_sources,
                )
                if (
                    tuple(joint_source_names)
                    != tuple(self.source_choice_selector_sources)
                ):
                    joint_source_batch = build_mcln_source_choice_batch(
                        end_points,
                        inputs,
                        source_names=joint_source_names,
                        include_rich_candidate_feats=False,
                    )
                base_mask_weights = None
                source_mask_evidence = None
                gate_evidence = None
                source_score_stack = None
                source_validity = None
                if getattr(
                        self,
                        "joint_query_quality_use_adaptive_source_mixing",
                        False):
                    source_score_stack = torch.stack([
                        joint_source_batch["source_scores"][name]
                        for name in joint_source_names
                    ], dim=-1)
                    source_validity = joint_source_batch["source_validity"]
                if self.joint_query_quality_use_mask_calibration:
                    query_count = parent_scores.shape[1]
                    normalized_weights = []
                    for batch_idx, weight in enumerate(adaptive_weight_lists):
                        normalized = query_fusion_weight(
                            weight, query_count, parent_scores[batch_idx]
                        )
                        if normalized.dim() == 0:
                            normalized = normalized.expand(query_count, 1)
                        normalized_weights.append(normalized.squeeze(-1))
                    base_mask_weights = torch.stack(
                        normalized_weights, dim=0
                    )
                    if self.joint_query_quality_use_source_mask_evidence:
                        source_mask_evidence = (
                            build_query_mask_source_evidence(
                                prediction_masks, sp_pred_masks
                            )
                        )
                        detached_evidence = source_mask_evidence.detach()
                        selector_out[
                            "joint_query_quality_source_mask_evidence_query_std"
                        ] = detached_evidence.std(
                            dim=1, unbiased=False
                        ).mean()
                        selector_out[
                            "joint_query_quality_source_mask_disagreement_mean"
                        ] = detached_evidence[..., -2:].mean()
                if self.joint_query_quality_use_gate_evidence:
                    gate_evidence = build_joint_query_gate_evidence(
                        selector_out, source_choice_batch["valid_mask"]
                    )
                    detached_gate_evidence = gate_evidence.detach()
                    selector_out[
                        "joint_query_quality_gate_evidence_query_std"
                    ] = detached_gate_evidence.std(
                        dim=1, unbiased=False
                    ).mean()
                    selector_out[
                        "joint_query_quality_gate_candidate_ratio"
                    ] = detached_gate_evidence[..., 0].mean()
                joint_out = self.joint_query_quality_reranker(
                    source_choice_batch["rich_candidate_feats"],
                    parent_scores,
                    source_choice_batch["valid_mask"],
                    base_mask_weights=base_mask_weights,
                    source_mask_evidence=source_mask_evidence,
                    gate_evidence=gate_evidence,
                    spatial_query_features=(
                        query_last
                        if self.joint_query_quality_use_spatial_mask_refiner
                        else None
                    ),
                    spatial_superpoint_features=(
                        super_features
                        if self.joint_query_quality_use_spatial_mask_refiner
                        else None
                    ),
                    source_score_stack=source_score_stack,
                    source_validity=source_validity,
                )
                if getattr(
                        self,
                        "joint_query_quality_use_adaptive_source_mixing",
                        False):
                    detached_weights = joint_out[
                        "source_mix_weights"
                    ].detach()
                    valid_rows = joint_out["valid_mask"]
                    for source_index, source_name in enumerate(
                            joint_source_names):
                        selector_out[
                            "joint_query_quality_source_mix_weight_{}".format(
                                source_name
                            )
                        ] = detached_weights[..., source_index][
                            valid_rows
                        ].mean()
                    selector_out[
                        "joint_query_quality_source_mix_residual_abs_mean"
                    ] = joint_out[
                        "source_mix_residual_logit"
                    ].detach()[valid_rows].abs().mean()
                    valid_sources = joint_out[
                        "source_mix_validity"
                    ]
                    selector_out[
                        "joint_query_quality_source_mix_router_residual_abs_mean"
                    ] = joint_out[
                        "source_mix_router_residual"
                    ].detach()[valid_sources].abs().mean()
                    selector_out[
                        "joint_query_quality_source_mix_weight_query_std_mean"
                    ] = detached_weights.std(
                        dim=1, unbiased=False
                    ).mean()
                    selector_out[
                        "joint_query_quality_source_mix_effective_count_mean"
                    ] = joint_out[
                        "source_mix_effective_source_count"
                    ].detach()[valid_rows].mean()
                if self.joint_query_quality_use_mask_calibration:
                    prediction_masks, sp_pred_masks, adaptive_weight_lists = (
                        apply_query_mask_calibration(
                            prediction_masks,
                            sp_pred_masks,
                            joint_out["mask_fusion_weights"],
                            joint_out["mask_logit_bias"],
                        )
                    )
                    end_points["last_pred_masks"] = prediction_masks
                    end_points["sp_last_pred_masks"] = sp_pred_masks
                    end_points["adaptive_weights"] = adaptive_weight_lists
                    selector_out[
                        "joint_query_quality_mask_alpha_residual_abs_mean"
                    ] = joint_out["mask_alpha_residual"].detach().abs().mean()
                    selector_out[
                        "joint_query_quality_mask_logit_bias_abs_mean"
                    ] = joint_out["mask_logit_bias"].detach().abs().mean()
                    selector_out[
                        "joint_query_quality_mask_logit_bias_abs_max"
                    ] = joint_out["mask_logit_bias"].detach().abs().amax()
                    selector_out[
                        "joint_query_quality_mask_weight_std_mean"
                    ] = joint_out["mask_fusion_weights"].detach().std(
                        dim=1
                    ).mean()
                if self.joint_query_quality_use_spatial_mask_refiner:
                    prediction_masks, sp_pred_masks = (
                        apply_query_superpoint_mask_residual(
                            prediction_masks,
                            sp_pred_masks,
                            joint_out["mask_spatial_residuals"],
                        )
                    )
                    end_points["last_pred_masks"] = prediction_masks
                    end_points["sp_last_pred_masks"] = sp_pred_masks
                    detached_spatial_rows = [
                        row.detach()[joint_out["valid_mask"][batch_idx]]
                        for batch_idx, row in enumerate(
                            joint_out["mask_spatial_residuals"]
                        )
                    ]
                    spatial_values = torch.cat([
                        row.reshape(-1) for row in detached_spatial_rows
                    ])
                    selector_out[
                        "joint_query_quality_mask_spatial_residual_abs_mean"
                    ] = spatial_values.abs().mean()
                    selector_out[
                        "joint_query_quality_mask_spatial_residual_abs_max"
                    ] = spatial_values.abs().amax()
                    selector_out[
                        "joint_query_quality_mask_spatial_superpoint_std_mean"
                    ] = torch.stack([
                        row.std(dim=1, unbiased=False).mean()
                        for row in detached_spatial_rows
                    ]).mean()
                    selector_out[
                        "joint_query_quality_mask_spatial_query_std_mean"
                    ] = torch.stack([
                        row.std(dim=0, unbiased=False).mean()
                        for row in detached_spatial_rows
                    ]).mean()
                selector_out["joint_query_quality_parent_scores"] = (
                    parent_scores
                )
                for key, value in joint_out.items():
                    selector_out[
                        "joint_query_quality_{}".format(key)
                    ] = value
                for key, value in summarize_joint_query_residual(
                        joint_out["residual"], joint_out["valid_mask"]
                ).items():
                    selector_out[
                        "joint_query_quality_{}".format(key)
                    ] = value
                for key, value in summarize_joint_query_residual(
                        joint_out["learned_residual"],
                        joint_out["valid_mask"],
                ).items():
                    selector_out[
                        "joint_query_quality_learned_{}".format(key)
                    ] = value
                if getattr(
                        self,
                        "joint_query_quality_preserve_parent_score",
                        False,
                ):
                    row = torch.arange(
                        parent_scores.shape[0], device=parent_scores.device
                    )
                    parent = joint_out["baseline_indices"]
                    selector_out[
                        "joint_query_quality_parent_score_drift_abs_max"
                    ] = (
                        joint_out["scores"][row, parent]
                        - parent_scores[row, parent]
                    ).detach().abs().amax()
                    selector_out[
                        "joint_query_quality_candidate_promotion_margin"
                    ] = parent_scores.new_tensor(
                        getattr(
                            self,
                            "joint_query_quality_candidate_promotion_margin",
                            0.0,
                        )
                    )
                if "parent_transition_advantage" in joint_out:
                    transition_values = joint_out[
                        "parent_transition_advantage"
                    ].detach()[joint_out["valid_mask"]]
                    selector_out[
                        "joint_query_quality_transition_advantage_abs_mean"
                    ] = transition_values.abs().mean()
                    selector_out[
                        "joint_query_quality_transition_advantage_abs_max"
                    ] = transition_values.abs().amax()
                selector_out["selected_source_scores"] = joint_out["scores"]
            end_points.update(selector_out)
            end_points["source_choice_source_scores"] = source_choice_batch[
                "source_scores"
            ]

        if self.use_sacr_score_refiner:
            contract_keys = (
                "last_center",
                "last_pred_size",
                "last_pred_masks",
                "sp_last_pred_masks",
                "adaptive_weights",
            )
            contract_snapshot = None
            if self.sacr_score_contract_audit:
                missing_contract = [
                    key for key in contract_keys if key not in end_points
                ]
                if missing_contract:
                    raise ValueError(
                        "SACR identity audit is missing frozen outputs: "
                        + ", ".join(missing_contract)
                    )

                def clone_tensor_tree(value):
                    if torch.is_tensor(value):
                        return value.detach().clone()
                    if isinstance(value, list):
                        return [clone_tensor_tree(item) for item in value]
                    if isinstance(value, tuple):
                        return tuple(clone_tensor_tree(item) for item in value)
                    if isinstance(value, dict):
                        return {
                            key: clone_tensor_tree(item)
                            for key, item in value.items()
                        }
                    return value

                contract_snapshot = {
                    key: clone_tensor_tree(end_points[key])
                    for key in contract_keys
                }
            slots = end_points.get("structured_slots")
            parent_scores = end_points.get("selected_source_scores")
            if slots is None or parent_scores is None:
                raise ValueError(
                    "SACR score refinement requires structured slots and "
                    "parent scores"
                )
            if self.sacr_score_contract_audit:
                contract_snapshot.update({
                    "decoder_query_last": clone_tensor_tree(
                        decoder_query_last
                    ),
                    "selected_source_scores": clone_tensor_tree(
                        parent_scores
                    ),
                })
            candidate_valid = source_choice_batch["valid_mask"]
            if (
                not isinstance(parent_scores, torch.Tensor)
                or parent_scores.shape != candidate_valid.shape
                or candidate_valid.dtype != torch.bool
            ):
                raise ValueError(
                    "SACR parent scores and candidate validity must align"
                )
            candidate_boxes = torch.cat((
                end_points["last_center"],
                end_points["last_pred_size"].clamp(min=1e-6),
            ), dim=-1)
            global_only, weak_generic = build_decomposition_masks(
                inputs,
                slots,
                min_parse_confidence=self.sacr_min_parse_confidence,
            )
            sacr_out = self.sacr_head(
                query_feats=decoder_query_last,
                pred_boxes=candidate_boxes,
                base_scores=parent_scores,
                slot_dict=slots,
                global_only_mask=global_only,
                weak_generic_target_mask=weak_generic,
            )
            raw_scores = sacr_out["structured_scores"]
            structured_valid = sacr_out[
                "structured_valid_mask"
            ]
            apply_mask = (
                structured_valid.unsqueeze(1) & candidate_valid
            )
            gate = self.sacr_score_gate.tanh()
            residual = (
                self.sacr_score_max_delta
                * gate
                * raw_scores.tanh()
            )
            refined_scores = torch.where(
                apply_mask,
                parent_scores + residual,
                parent_scores,
            )
            if self.sacr_score_contract_audit:
                if not torch.equal(gate, torch.zeros_like(gate)):
                    raise ValueError(
                        "SACR identity audit requires an exact zero gate"
                    )
                if not torch.equal(refined_scores, parent_scores):
                    raise ValueError(
                        "zero-gate SACR scores are not bitwise parent-identical"
                    )

                def tensor_tree_equal(left, right):
                    if torch.is_tensor(left) or torch.is_tensor(right):
                        return (
                            torch.is_tensor(left)
                            and torch.is_tensor(right)
                            and torch.equal(left, right)
                        )
                    if isinstance(left, (list, tuple)):
                        return (
                            isinstance(right, type(left))
                            and len(left) == len(right)
                            and all(
                                tensor_tree_equal(a, b)
                                for a, b in zip(left, right)
                            )
                        )
                    if isinstance(left, dict):
                        return (
                            isinstance(right, dict)
                            and set(left) == set(right)
                            and all(
                                tensor_tree_equal(left[key], right[key])
                                for key in left
                            )
                        )
                    return left == right

                changed_contract = [
                    key for key in contract_keys
                    if not tensor_tree_equal(
                        contract_snapshot[key], end_points[key]
                    )
                ]
                if not tensor_tree_equal(
                        contract_snapshot["decoder_query_last"],
                        decoder_query_last):
                    changed_contract.append("decoder_query_last")
                if not tensor_tree_equal(
                        contract_snapshot["selected_source_scores"],
                        parent_scores):
                    changed_contract.append("selected_source_scores")
                if changed_contract:
                    raise ValueError(
                        "SACR score branch mutated frozen box/mask tensors: "
                        + ", ".join(changed_contract)
                    )
                end_points["sacr_score_contract_audit_pass"] = (
                    parent_scores.new_tensor(1.0)
                )
                end_points["sacr_score_contract_audit_tensor_count"] = (
                    parent_scores.new_tensor(float(len(contract_keys) + 2))
                )
            active_count = apply_mask.float().sum().clamp(min=1.0)
            active_residual = residual.masked_fill(~apply_mask, 0.0)
            end_points["sacr_score_parent_scores"] = parent_scores
            end_points["sacr_score_refiner_scores"] = refined_scores
            end_points["sacr_score_valid_mask"] = apply_mask
            end_points["sacr_score_structured_valid_mask"] = (
                structured_valid
            )
            end_points["sacr_score_raw_scores"] = raw_scores
            end_points["sacr_score_residual"] = active_residual
            end_points["sacr_score_gate_value"] = gate.detach().reshape(())
            end_points["sacr_score_residual_abs_mean"] = (
                active_residual.detach().abs().sum() / active_count
            )
            end_points["sacr_score_residual_abs_max"] = (
                active_residual.detach().abs().amax()
            )
            end_points["sacr_score_structured_valid_ratio"] = (
                structured_valid.float().mean().detach()
            )
            end_points["sacr_score_relation_active_ratio"] = sacr_out[
                "relation_active_ratio"
            ].detach()
            end_points["selected_source_scores"] = refined_scores

        # The V105/V106 refiners are intentionally the final mask-only operation.  They are
        # placed after every REC arbiter/reranker score has been finalized and
        # cannot modify a REC score, rank, source choice, or query index.
        if self.egqs_mask_refiner is not None:
            mask_boxes = torch.cat((
                end_points['last_center'],
                end_points['last_pred_size'].clamp(min=1e-4),
            ), dim=-1)
            egqs_out = self.egqs_mask_refiner(
                query_last,
                super_features,
                super_xyz_list,
                mask_boxes,
                end_points['last_pred_masks'],
                end_points['sp_last_pred_masks'],
                end_points['adaptive_weights'],
            )
            prediction_masks, sp_pred_masks = (
                apply_query_superpoint_mask_residual(
                    end_points['last_pred_masks'],
                    end_points['sp_last_pred_masks'],
                    egqs_out['residuals'],
                )
            )
            end_points['last_pred_masks'] = prediction_masks
            end_points['sp_last_pred_masks'] = sp_pred_masks
            for name, value in egqs_out.items():
                if name != 'residuals':
                    end_points['egqs_mask_refiner_' + name] = value

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
