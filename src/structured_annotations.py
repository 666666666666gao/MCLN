"""Dataset-independent structured language annotation helpers."""

import difflib
import json


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y")
    return bool(value)


def _json_value(record, key, default):
    value = record.get(key, default)
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError as exc:
            raise ValueError(
                "invalid structured JSON field {}: {}".format(
                    key, value[:120]
                )
            ) from exc
    return value


def _span(span, utterance):
    if not isinstance(span, dict):
        return None
    try:
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
    except (TypeError, ValueError):
        return None
    text = str(span.get("text", ""))
    if start < 0 or end <= start or end > len(utterance):
        start = -1
    elif text and utterance[start:end].lower() != text.lower():
        start = -1
    if start < 0 and text:
        lowered = utterance.lower()
        first = lowered.find(text.lower())
        if first >= 0 and lowered.find(text.lower(), first + 1) < 0:
            start = first
            end = first + len(text)
    if start < 0:
        return None
    result = {"start": start, "end": end, "text": text or utterance[start:end]}
    for key in ("head", "tail", "type", "surface_text"):
        if key in span:
            result[key] = span[key]
    return result


def _metadata(coverage, has_target, structured_available):
    coverage = coverage if isinstance(coverage, dict) else {}
    coverage = dict(coverage)
    annotated_target = _as_bool(coverage.get("has_target"), has_target)
    missing_target = _as_bool(coverage.get("missing_target"), False)
    global_only = (
        not has_target
        or not annotated_target
        or missing_target
        or _as_bool(
            coverage.get("global_only_due_to_parse_error"), False
        )
    )
    weak_generic = any(
        _as_bool(coverage.get(key), False)
        for key in (
            "target_generic_reference",
            "overgeneric_target_remaining",
            "target_overgeneric_canonical",
            "generic_target",
        )
    )
    try:
        confidence = float(coverage.get("parse_confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not structured_available or global_only:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    coverage["has_target"] = int(has_target and annotated_target)
    coverage["global_only_due_to_parse_error"] = int(global_only)
    coverage["target_generic_reference"] = int(weak_generic)
    coverage["parse_confidence"] = confidence
    status = str(coverage.get("decomposition_status", "ok"))
    if global_only:
        status = "global_only_target_unresolved"
    elif weak_generic:
        status = "weak_generic_target_recovered"
    return {
        "coverage_stats": coverage,
        "parse_confidence": confidence,
        "decomposition_status": status,
        "decomp_global_only_mask": global_only,
        "decomp_weak_generic_mask": weak_generic,
        "structured_annotation_available": bool(structured_available),
    }


def build_structured_annotation(record, utterance):
    """Normalize ScanRefer/ReferIt3D structured fields to one contract."""
    if not isinstance(record, dict):
        record = {}
    utterance = str(utterance)
    target_slot = _json_value(record, "target_slot", {})
    entities = _json_value(record, "entities", [])
    attributes = _json_value(record, "attributes", [])
    relations = _json_value(record, "relations", [])
    attr_slot = _json_value(record, "attr_slot", {})
    rel_slots = _json_value(record, "rel_slots", [])
    anchor_slots = _json_value(record, "anchor_slots", [])
    coverage = _json_value(record, "coverage_stats", {})
    if not isinstance(coverage, dict):
        coverage = {}
    if "parse_confidence" not in coverage and record.get(
            "parse_confidence") not in (None, ""):
        coverage["parse_confidence"] = record.get("parse_confidence")

    target = _span(target_slot, utterance)
    target_spans = [target] if target is not None else []
    entity_spans = []
    entity_index = {}

    def add_entity(raw_span):
        normalized = _span(raw_span, utterance)
        if normalized is None:
            return -1
        key = (normalized["start"], normalized["end"])
        if key not in entity_index:
            entity_index[key] = len(entity_spans)
            entity_spans.append(normalized)
        return entity_index[key]

    if target is not None:
        add_entity(target)
    if isinstance(entities, list):
        for item in entities:
            add_entity(item)

    if isinstance(attr_slot, dict) and isinstance(
            attr_slot.get("items"), list):
        attributes = attr_slot["items"]
    attr_spans = []
    if isinstance(attributes, list):
        for item in attributes:
            normalized = _span(item, utterance)
            if normalized is not None:
                attr_spans.append(normalized)

    if isinstance(rel_slots, list) and rel_slots:
        relations = rel_slots
    rel_spans = []
    if isinstance(relations, list):
        for item in relations:
            normalized = _span(item, utterance)
            if normalized is not None:
                rel_spans.append(normalized)

    structured_anchor_ids = []
    if isinstance(anchor_slots, list):
        for item in anchor_slots[:len(rel_spans)]:
            structured_anchor_ids.append(add_entity(item))
    structured_anchor_ids.extend(
        [-1] * (len(rel_spans) - len(structured_anchor_ids))
    )

    structured_available = bool(
        target_spans
        and record
        and (coverage or entities or attributes or relations or anchor_slots)
    )
    result = {
        "target_spans": target_spans,
        "entity_spans": entity_spans,
        "attr_spans": attr_spans,
        "rel_spans": rel_spans,
        "structured_anchor_ids": structured_anchor_ids,
        "target_slot": target_slot if isinstance(target_slot, dict) else {},
        "attr_slot": attr_slot if isinstance(attr_slot, dict) else {},
        "rel_slots": rel_slots if isinstance(rel_slots, list) else [],
        "anchor_slots": anchor_slots if isinstance(anchor_slots, list) else [],
    }
    result.update(_metadata(coverage, bool(target_spans), structured_available))
    return result


def realign_structured_annotation(annotation, source_text, target_text):
    """Map structured spans through the legacy MCLN text normalization."""
    source_text = str(source_text)
    target_text = str(target_text)
    if source_text == target_text:
        return annotation
    boundary = [None] * (len(source_text) + 1)
    matcher = difflib.SequenceMatcher(None, source_text, target_text)
    for tag, source_start, source_end, target_start, target_end in (
            matcher.get_opcodes()):
        source_width = source_end - source_start
        target_width = target_end - target_start
        if source_width == 0:
            continue
        for offset in range(source_width + 1):
            ratio = float(offset) / float(source_width)
            mapped = target_start + int(round(ratio * target_width))
            boundary[source_start + offset] = mapped
    last = 0
    for index in range(len(boundary)):
        if boundary[index] is None:
            boundary[index] = last
        else:
            last = boundary[index]

    def remap(span):
        if not isinstance(span, dict):
            return None
        start = int(span.get("start", -1))
        end = int(span.get("end", -1))
        if start < 0 or end <= start or end > len(source_text):
            return None
        mapped_start = max(0, min(len(target_text), boundary[start]))
        mapped_end = max(mapped_start, min(len(target_text), boundary[end]))
        while mapped_start < mapped_end and target_text[mapped_start].isspace():
            mapped_start += 1
        while mapped_end > mapped_start and target_text[mapped_end - 1].isspace():
            mapped_end -= 1
        if mapped_end <= mapped_start:
            return None
        result = dict(span)
        result["start"] = mapped_start
        result["end"] = mapped_end
        result["text"] = target_text[mapped_start:mapped_end]
        return result

    for key in ("target_spans", "entity_spans", "attr_spans", "rel_spans"):
        remapped = [remap(span) for span in annotation.get(key, [])]
        if key == "target_spans":
            annotation[key] = [span for span in remapped if span is not None]
        else:
            # Preserve indices so relation-to-anchor links remain aligned.
            annotation[key] = [
                span if span is not None
                else {"start": -1, "end": -1, "text": ""}
                for span in remapped
            ]
    if not annotation.get("target_spans"):
        annotation["structured_annotation_available"] = False
        annotation["parse_confidence"] = 0.0
        annotation["decomp_global_only_mask"] = True
        coverage = dict(annotation.get("coverage_stats", {}))
        coverage["has_target"] = 0
        coverage["parse_confidence"] = 0.0
        annotation["coverage_stats"] = coverage
    return annotation


def scanrefer_annotation_key(record, fallback=""):
    return (
        str(record.get("scene_id", "")),
        str(record.get("object_id", "")),
        str(record.get("ann_id", record.get("ann_id_key", fallback))),
    )


def build_scanrefer_structured_lookup(records):
    lookup = {}
    for index, record in enumerate(records):
        lookup[scanrefer_annotation_key(record, str(index))] = record
    return lookup
