"""Use the existing REC candidate adapter's box extent policy for evaluation."""


def rec_evaluation_view(end_points):
    # The unconstrained size head produced negative off-target extents in the
    # native ScanRefer train-row probe. The existing rec_candidate_adapter
    # already floors these at 1e-6 when constructing deployable REC boxes.
    # Use that same representation for the evaluator's preliminary raw-box
    # checks, without changing the model outputs consumed by the training loss.
    view = dict(end_points)
    view['last_pred_size'] = end_points['last_pred_size'].clamp(min=1e-6)
    return view
