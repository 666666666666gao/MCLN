"""Utilities for fail-closed structured score sources."""

import torch


def _find_unique_text_span(text, phrase):
    if not phrase:
        return None
    first = text.lower().find(phrase.lower())
    if first < 0 or text.lower().find(phrase.lower(), first + 1) >= 0:
        return None
    return first, first + len(phrase)


def build_token_span_tensors(tokenized, span_batches, texts, device,
                             min_slots=1):
    """Convert character spans to padded RoBERTa token spans.

    Annotation offsets are checked against the exact text sent to RoBERTa.
    A unique span-text match repairs harmless whitespace normalization; all
    ambiguous or truncated spans remain invalid.
    """
    batch_size = tokenized["input_ids"].shape[0]
    if not isinstance(span_batches, (list, tuple)):
        span_batches = [[] for _ in range(batch_size)]
    if not isinstance(texts, (list, tuple)) or len(texts) != batch_size:
        raise ValueError("texts must be a batch-aligned sequence")
    max_spans = max(
        [
            len(spans)
            for spans in span_batches
            if isinstance(spans, list)
        ] + [0]
    )
    max_spans = max(max_spans, int(min_slots))
    output = torch.full(
        (batch_size, max_spans, 2), -1, dtype=torch.long, device=device
    )
    for batch_index in range(batch_size):
        spans = (
            span_batches[batch_index]
            if batch_index < len(span_batches)
            and isinstance(span_batches[batch_index], list)
            else []
        )
        text = str(texts[batch_index])
        for span_index, span in enumerate(spans[:max_spans]):
            if not isinstance(span, dict):
                continue
            try:
                start = int(span.get("start", -1))
                end = int(span.get("end", -1))
            except (TypeError, ValueError):
                continue
            phrase = str(span.get("text", ""))
            offset_matches = (
                0 <= start < end <= len(text)
                and (not phrase or text[start:end].lower() == phrase.lower())
            )
            if not offset_matches:
                repaired = _find_unique_text_span(text, phrase)
                if repaired is None:
                    continue
                start, end = repaired

            begin_token = tokenized.char_to_token(batch_index, start)
            end_token = tokenized.char_to_token(batch_index, end - 1)
            if begin_token is None:
                for probe in range(start + 1, min(end, start + 3)):
                    begin_token = tokenized.char_to_token(batch_index, probe)
                    if begin_token is not None:
                        break
            if end_token is None:
                for probe in range(end - 2, max(start - 1, end - 4), -1):
                    end_token = tokenized.char_to_token(batch_index, probe)
                    if end_token is not None:
                        break
            if (
                begin_token is None
                or end_token is None
                or end_token < begin_token
            ):
                continue
            output[batch_index, span_index, 0] = begin_token
            output[batch_index, span_index, 1] = end_token + 1
    return output


def _batch_numeric(value, batch_size, device, default=0.0):
    if torch.is_tensor(value):
        values = value.to(device=device).float().reshape(-1)
        if values.numel() >= batch_size:
            return values[:batch_size]
    elif isinstance(value, (list, tuple)):
        parsed = []
        for item in list(value)[:batch_size]:
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                parsed.append(float(default))
        parsed.extend([float(default)] * (batch_size - len(parsed)))
        return torch.tensor(parsed, device=device, dtype=torch.float32)
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        scalar = float(default)
    return torch.full((batch_size,), scalar, device=device)


def _coverage_numeric(inputs, key, batch_size, device, default=0.0):
    coverage = inputs.get("coverage_stats")
    if isinstance(coverage, dict):
        return _batch_numeric(
            coverage.get(key, default), batch_size, device, default
        )
    if isinstance(coverage, (list, tuple)):
        values = [
            item.get(key, default) if isinstance(item, dict) else default
            for item in list(coverage)[:batch_size]
        ]
        return _batch_numeric(values, batch_size, device, default)
    return torch.full((batch_size,), float(default), device=device)


def apply_authoritative_coverage(inputs, slot_dict):
    """Merge metadata without making failed token alignment valid."""
    batch_size = slot_dict["global_slot"].shape[0]
    device = slot_dict["global_slot"].device
    observed = dict(slot_dict.get("coverage_stats", {}))
    observed_target = observed.get(
        "has_target", torch.zeros(batch_size, device=device)
    ).to(device=device).bool()
    annotated_target = _coverage_numeric(
        inputs, "has_target", batch_size, device, default=1.0
    ).bool()
    global_only = (
        _coverage_numeric(
            inputs,
            "global_only_due_to_parse_error",
            batch_size,
            device,
        ).bool()
        | _coverage_numeric(
            inputs, "missing_target", batch_size, device
        ).bool()
    )
    available = _batch_numeric(
        inputs.get("structured_annotation_available", False),
        batch_size,
        device,
    ).bool()
    has_target = observed_target & annotated_target & available & ~global_only
    observed["has_target"] = has_target
    observed["annotated_has_target"] = annotated_target
    observed["global_only_due_to_parse_error"] = global_only
    observed["num_attrs_annotated"] = _coverage_numeric(
        inputs, "num_attrs", batch_size, device
    ).long()
    observed["num_pairs_annotated"] = _coverage_numeric(
        inputs, "valid_tuple_count", batch_size, device
    ).long()
    slot_dict["coverage_stats"] = observed
    confidence = _coverage_numeric(
        inputs, "parse_confidence", batch_size, device, default=0.0
    ).clamp(0.0, 1.0)
    slot_dict["parse_confidence"] = confidence * has_target.float()
    return slot_dict


def build_decomposition_masks(inputs, slot_dict, min_parse_confidence=0.0):
    batch_size = slot_dict["global_slot"].shape[0]
    device = slot_dict["global_slot"].device
    coverage = slot_dict.get("coverage_stats", {})
    has_target = coverage.get(
        "has_target", torch.zeros(batch_size, device=device)
    ).to(device=device).bool()
    confidence = slot_dict.get(
        "parse_confidence", torch.zeros(batch_size, device=device)
    ).to(device=device).float()
    global_only = ~has_target | (confidence <= float(min_parse_confidence))
    weak_generic = torch.zeros(batch_size, device=device, dtype=torch.bool)
    for key in (
        "target_generic_reference",
        "overgeneric_target_remaining",
        "target_overgeneric_canonical",
        "generic_target",
    ):
        weak_generic |= _coverage_numeric(
            inputs, key, batch_size, device
        ).bool()
    return global_only, weak_generic
