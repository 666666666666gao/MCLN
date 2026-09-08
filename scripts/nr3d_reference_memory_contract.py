"""Fixed four-arm comparison of relation readout and reference memory."""

from scripts.nr3d_pair_readout_contract import (
    CONTRACT as ORIGINAL_CONTRACT, compare_rows, covered_ranking_loss, split_rows,
)


ARMS = {
    "query_global": {"memory": "query", "readout": "global"},
    "query_pair": {"memory": "query", "readout": "pair"},
    "object_global": {"memory": "object", "readout": "global"},
    "object_pair": {"memory": "object", "readout": "pair"},
}
CONTRACT = dict(ORIGINAL_CONTRACT)
CONTRACT.update({
    "schema": "mcln-nr3d-reference-memory-train-v1",
    "arms": ARMS,
    "empty_supervision_batch": "skip all four optimizer updates",
    "memory_comparison": "existing legal Queries versus existing butd_cls object inputs",
    "parameter_matching": "exact within each readout across memories; global/pair counts differ",
    "primary_candidate": "object_pair",
    "previous_P2_variant_status": "failed and sealed; no inherited performance pass",
})


def improvement_screen(comparison):
    return (comparison["overall"]["025"]["delta_hits"] > 0
            and comparison["overall"]["050"]["delta_hits"] >= 0
            and comparison["long_13plus"]["025"]["delta_hits"] > 0
            and comparison["hard_2plus_distractors"]["025"]["delta_hits"] > 0)


def decide(rows):
    memory = compare_rows(rows, "query_pair", mode="object_pair")
    readout = compare_rows(rows, "object_global", mode="object_pair")
    parent = compare_rows(rows, "protected", mode="object_pair")
    practical = (parent["overall"]["025"]["delta_hits"] > 0
                 and parent["overall"]["050"]["delta_hits"] >= 0
                 and sum(row["scores"]["object_pair"]["mask_iou"] for row in rows)
                 >= sum(row["protected_mask_iou"] for row in rows))
    memory_pass, readout_pass = improvement_screen(memory), improvement_screen(readout)
    return {"object_pair_vs_query_pair": memory, "object_pair_vs_object_global": readout,
            "object_pair_vs_protected": parent,
            "memory_screen_pass": memory_pass, "readout_screen_pass": readout_pass,
            "practical_screen_pass": practical,
            "eligible_for_decoder_experiment": memory_pass and readout_pass and practical,
            "formal_promotion": False,
            "control_substituted_for_failed_primary_candidate": False}
