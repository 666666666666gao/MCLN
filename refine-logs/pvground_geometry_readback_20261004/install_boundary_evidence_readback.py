"""Explicit installation for the isolated delayed-semantic source port only."""
from pvground_boundary_evidence_readback import BoundaryEvidenceReadback


def install_boundary_evidence_readback(model):
    assert model.boundary_evidence_readback is None
    assert model.candidate_box_refiner is not None
    assert model.num_decoder_layers == 6
    assert model.prediction_heads[-1].compute_sem_scores
    model.boundary_evidence_readback = BoundaryEvidenceReadback()
