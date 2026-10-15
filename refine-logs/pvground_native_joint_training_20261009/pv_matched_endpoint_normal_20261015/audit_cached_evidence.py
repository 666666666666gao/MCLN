"""Cached artifact audit only. Standard library; no project imports or remote actions."""
import ast
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import sys

EXPECTED = {
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\052___init__.py": "2ca09339e49e64e99d7fae6da86eb56eb5fe9ebcf3edd4ffe7ecf2b8fb931649",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\108___init__.py": "d0f1d31d6e0207a37dcb8a33315a409226d651670a8d0a238223009a4118f57b",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\132_pvground_task_observation_query.py": "39349bf1fe8f3f51149f0872911020ecd0ede98fc9a8278cecb5ae466ed8643d",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\EXPERIMENT_PLAN.md": "0030e1626665019c82239dd68f878af9ff57a7f9f7d0be1708613b6ea243246f",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\EXPERIMENT_TRACKER.md": "84996b144edbd9a0115ed4add8b684f5c6df92fd0cc4c747c780471789cd1bcd",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\source\\matched_endpoint_geometry_objective.py": "a036560ceab0968ea889d13e1ab02ba6f0ccf95207d42c246a942b04478b58b0",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\source\\matched_endpoint_config.json": "f2afab85ca0565f5febd3a7f69da5a1f53c010f7aba304fd99c97fabca36cf6e",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\final_duplicate_config.json": "4d5aca63e634f1d634905c32540ce331042073b19f131c052f999b91577abe75",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\runtime_preparation\\native_matched_endpoint_preflight.py": "055182c25987e379a7a5f9cbb30a8c5afd47f537252f4fb0f4bb900182c976e3",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\runtime_preparation\\initialization_reference.json": "cbb56a015f3e58eacc710a51417130536fba978393d06973b01b96baa4ede018",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\runtime_preparation\\final_duplicate_protocol.json": "9fad39d365a4e4db6fab2e530b39a9f365f4aea0aca3056f2783eff9e69822c3",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\runtime_preparation\\coarse_endpoint_protocol.json": "ef873e632aa9711d08202e3aa4ec138b8e325f58ae0118c28ba563bd669ef70f",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\runtime_preparation\\coarse_endpoint_matched_endpoint_config.json": "f2afab85ca0565f5febd3a7f69da5a1f53c010f7aba304fd99c97fabca36cf6e",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\runtime_preparation\\final_duplicate_matched_endpoint_config.json": "4d5aca63e634f1d634905c32540ce331042073b19f131c052f999b91577abe75",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_INTAKE.json": "3b426779b8883cb5c6aa6bc2715cb15c673cf01191c3574de4b066bfd9ba873b",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_EXIT.json": "e89df0eae5ec7db81135aed16463968955c578bcc4f2ec43172eff351f080c17",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_INTAKE.json": "71a99210115359f3290ac01b2047c91bc597cc5e7441adc6d3385cd12f79f4ba",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\PREFLIGHT_LAUNCH.json": "c8f88ab1701b33744084edc93ec25f4178c671292e40c037654c729d9442eed3",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\PREFLIGHT_TRANSPORT_EXIT.json": "e89df0eae5ec7db81135aed16463968955c578bcc4f2ec43172eff351f080c17",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\admission.json": "4dfdfe25730bfe275325f307001451a9eba351425cb8248235e27b18a64e8308",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\controller.log.tail.txt": "6784a073b286750183dc95b83ccd52db1a6fb8e2491424596a85078edede96bc",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\status.json": "d38245b1d3c86a27906cddb74251ac1bf51fd177d034ca48359408954cfc5349",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\coarse_endpoint\\preflight.exit": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\coarse_endpoint\\preflight.log.tail.txt": "96374892fe448f889d2133c003bed55417332ee54392b5308a9880678a6b4dfb",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\coarse_endpoint\\witness\\NATIVE_M0_RECEIPT.json": "3958bfec4ea1656a9c0ba3967f2d5c05d892b2b361246e1b6a978b22132a9ba7",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\coarse_endpoint\\witness\\NATIVE_M0_TRAINING_DIAGNOSTIC.json": "026997d4a8e17ca5e5ae7fd80793374409b49f715082a6a4e724143bb7f6cb20",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\final_duplicate\\preflight.exit": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\final_duplicate\\preflight.log.tail.txt": "e0d2599221f96e78a1f4afaadca58ba9b472da3d11b25277dccaeeb837833ec3",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\final_duplicate\\witness\\NATIVE_M0_RECEIPT.json": "4ca6421f06097e40cd81342a65f3cd2ac8ad9f72a2c191e228edbc7c77667264",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_terminal_expected_files\\final_duplicate\\witness\\NATIVE_M0_TRAINING_DIAGNOSTIC.json": "7507d2194d33e20d9dae7a9fe9de14d5812604e7763f6be389b47da7ab45a2a3",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_files\\admission.json": "4dfdfe25730bfe275325f307001451a9eba351425cb8248235e27b18a64e8308",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_files\\controller.log.tail.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_files\\status.json": "99ddbbf6b9da9dd995013151865f2e0af890017e2d28812a529ea9b20f04fbd6",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_files\\coarse_endpoint\\preflight.log.tail.txt": "45c527da04dafb78037c9d14ed5f35789de18c7230ec4286deee7ef8055b4aa0",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_files\\final_duplicate\\preflight.exit": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_files\\final_duplicate\\preflight.log.tail.txt": "e0d2599221f96e78a1f4afaadca58ba9b472da3d11b25277dccaeeb837833ec3",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_files\\final_duplicate\\witness\\NATIVE_M0_RECEIPT.json": "4ca6421f06097e40cd81342a65f3cd2ac8ad9f72a2c191e228edbc7c77667264",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_files\\final_duplicate\\witness\\NATIVE_M0_TRAINING_DIAGNOSTIC.json": "7507d2194d33e20d9dae7a9fe9de14d5812604e7763f6be389b47da7ab45a2a3",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\source\\models\\losses.py": "28f1bbc92563e5dd5a11c980f2cb8b3068862db12e5939e8aa5653b00cb6b0c4",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\launch_preparation\\matched_endpoint_preflight_pair_controller.py": "886e8fa8bf08b04a0f9037871ba98dc39fff38a89a6fc3411bc38947ef76ac70",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\launch_preparation\\normal_matched_endpoint_controller.py": "e0edb3c5f9495016de895f4ad348da30f2d4b9355337264d624819d57daef4a9",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\launch_preflight_after_terminal.py": "45c1d6f3dd3c3283961a82b3fb860b449e0b6f702986112dfb877c6b08cc173a",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\launch_normal_after_preflight.py": "cc4b4d5b3fb79f941e619d3fb8f341407a7c901d07ceb20f2a471599faf0d8dd",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\isolated_source_stage_20261014\\SOURCE_STAGE_RECEIPT.json": "5aebf34cf9dd21ef8b739bc281645ea87f555ceb67bc9f8de9b3a82f0ebad2eb",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_runtime_staging_20261013_v2\\SOURCE_STAGE_RECEIPT.json": "a3b8d98a81a9775bf8aef2675721dd657a95453c0fcf3e8eea4c914f64d2686f",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\055_losses.py": "b960fd486e4b325f346f3925d98bcda84be59020310d123832a789a525ceefee",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\058_pv_ground.py": "7a2db438811cb2fccf77a85f8f45a6a572cb0a1bbbd60d337e98afcbfa5e5b84",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\050_main_utils.py": "4254657bb68420db325fb42746d9f94a8e03fb677d5af9de6e8399e0f33d4733",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\103_joint_det_dataset.py": "3afebfc69232f9547e6810563ba6dd4f34a0f770f929df2f4203b21493de0e3d",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\124_extremal_span_mixer.py": "5ab81a56ffcd635eb61ff245f6771c40d20027b7708656b09e8d92bf3f5ac087",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\125_mask_support_corrector.py": "5b1c2d7bffa2332c86102e8003272ab8caed0e361034ad97a875198ebad51351",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\126_native_mask_geometry.py": "8386111df445bdbf0a867b7c7f6e478783f349e4221f4526c391715c83b3e4c8",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\127_native_model_initialization.py": "27169c76bda9cf1a0247fc613b16017d74f92c7662a5f5b5dcaa9d827f3288db",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\128_native_root_bbs.py": "b0357bd54e8753672ee31d3056e50c1d53a854d7f0c71cd01a7bc8cecf422c8b",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\140_039_env_spec.json": "73116c8d67e6cad3b35065f6c4da8328e7167f7db346b2834fb3c71e07e62f15",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\107_train_dist_mod.py": "91ec636c1235c7eb2bca758ddc00762d8ffa74a3ccec810af022bb89ef5e3bfa",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\130_pvground_semantic_assignment.py": "bec33ac6f24b3b83ab4bdfdeaf19cd7d400a372539a023a9929dadccd3ecad6d",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\135_selected_support_geometry_objective.py": "74a1d38018b3f8cf38d2b9f133b2df44f90cc2d17e13775b5812c0b1d0d12d30",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\113_lr_scheduler.py": "6ef012a837653b970ecd46854b96a6155283db9831bf1d2b2465f25c506034a7",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\span_lr_actual_pair_review_20261014\\snapshots\\123_scanrefer_data_contract.py": "9ea136508e02cfc7dbd8065eafe5cf7f5e8959ac25b143e7e777ec738bb96acd",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\review\\REVIEW_COMPLETE.json": "e8e8cef03f8d01c8199a374cad74ddca718965d0b9fdd87ed04a0dd80304f88b",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\review_execution\\EXECUTION_REVIEW_COMPLETE.json": "44ad40bf300dbf34430caa3589d00bec22bfe46330ef01e034342b58ca7af163",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\review_launch\\LAUNCHERS_REVIEW_COMPLETE.json": "93a39bac9ac143c307f019feca851271bb9bdb9ac9cdaf72d56fbc5deb5d28a8",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\review_runtime\\RUNTIME_REVIEW_COMPLETE.json": "3215c18ecb33ae85cba464ed2403cf47687abfd819856d8d0bc380626cb137c4",
  "C:\\Users\\gb\\.codex\\skills\\experiment-audit\\SKILL.md": "0ab77f219f554cbe42f22b79d780528e61c5e58169af4c2f56acf59596ab8db7",
  "C:\\Users\\gb\\.codex\\skills\\shared-references\\local-codex-policy.md": "b0249bfccd48ff79a5976afa7a54c64a3cfb65a5496a2105c5bb21069d197783",
  "C:\\Users\\gb\\.codex\\skills\\shared-references\\reviewer-independence.md": "1aa1756a99dc47009072f0ee7bf570d797bf7e2cc84439401e2a9312d73659b0",
  "C:\\Users\\gb\\.codex\\skills\\shared-references\\experiment-integrity.md": "22150fbdc5867e7790c88f968bccb367079c2db4c190ddce980709aea7e76f60",
  "C:\\Users\\gb\\.codex\\skills\\shared-references\\review-tracing.md": "7f556fb9ef8922cfd0d047e5a2d857b0c7f9408492b8c8dcec00207e76405395",
  "D:\\Program Files\\UserCache\\gb\\codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014\\OBS_preflight_actual_pair_first_EXIT.json": "e89df0eae5ec7db81135aed16463968955c578bcc4f2ec43172eff351f080c17"
}
K = Path("C:\\Users\\gb\\.codex\\tmp\\pvground_native_joint_training_20261009\\native_joint_v2\\deployment_geometry_training_20261012\\coarse_endpoint_training_20261014").resolve()
D = K.parent
OUT = K / 'actual_pair_review'
def disk(path):
    return Path(chr(92)*2+'?'+chr(92)+str(Path(path).resolve()))
def raw(path):
    return disk(path).read_bytes()
def parsed(path):
    return json.loads(raw(path))
def h(path):
    return hashlib.sha256(raw(path)).hexdigest()
def kp(name):
    return K / name
def snap(name):
    return D / 'span_lr_actual_pair_review_20261014' / 'snapshots' / name
tests = []
def passed(name, details):
    tests.append(dict(name=name, status='PASS', details=details))
for path, digest in EXPECTED.items():
    assert h(path) == digest, path
passed('all_audited_input_raw_hashes_stable', dict(inputs=len(EXPECTED)))
python_paths = [path for path in EXPECTED if path.endswith('.py')]
for path in python_paths:
    ast.parse(raw(path).decode('utf-8-sig'), filename=path)
passed('all_read_python_sources_parse_without_imports', dict(python_sources=len(python_paths), project_imports=0))

prefixes = ['OBS_preflight_actual_pair_first', 'OBS_preflight_actual_pair_terminal_expected']
intakes = {}
captures = 0
for prefix in prefixes:
    intake = parsed(kp(prefix + '_INTAKE.json'))
    intakes[prefix] = intake
    folder = kp(prefix + '_files')
    assert parsed(kp(prefix + '_EXIT.json')) == dict(exit_code=0)
    for row in intake['files']:
        data = raw(folder / row['name'])
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
        captures += 1
    assert parsed(folder/'status.json') == intake['remote_status']
    for arm, receipt in intake['receipts'].items():
        assert receipt == parsed(folder/arm/'witness/NATIVE_M0_RECEIPT.json')
    for tail in intake['tails']:
        # The captured tails here contain each complete log (all under the cap).
        assert len(raw(folder/(tail['name']+'.tail.txt'))) == tail['total_bytes']
passed('intake_exact_captured_files_receipts_status_exits_and_log_lengths',
       dict(file_rows=captures, tail_files=6, intake_exits=[0,0]))
first = intakes[prefixes[0]]
last = intakes[prefixes[1]]
assert first['remote_status']['status']=='running' and first['controller_alive'] and first['child_alive']
assert first['remote_status']['completed_arms']==['final_duplicate']
assert set(first['receipts']) == {'final_duplicate'}
assert not last['controller_alive'] and not last['child_alive']
status = last['remote_status']
assert status['status']=='complete' and status['exit_code']==0 and status['actual_total_native_updates']==4
assert status['completed_arms']==['final_duplicate','coarse_endpoint']
assert all(status[key] for key in ('same_process_recovery_verified','same_augmented_inputs_verified','same_initial_model_state_verified'))
assert not status['normal_training_started'] and not status['formal_training_admitted'] and status['formal_accuracy'] is None
for suffix in ['admission.json','final_duplicate/preflight.exit','final_duplicate/preflight.log.tail.txt',
               'final_duplicate/witness/NATIVE_M0_RECEIPT.json','final_duplicate/witness/NATIVE_M0_TRAINING_DIAGNOSTIC.json']:
    assert raw(kp(prefixes[0]+'_files')/suffix) == raw(kp(prefixes[1]+'_files')/suffix)
assert json.loads(raw(kp(prefixes[1]+'_files/controller.log.tail.txt'))) == status
passed('first_to_terminal_history_and_duplicate_stability',
       dict(first_observed=first['observed_cst'], terminal_observed=last['observed_cst'],
            completed=status['completed_cst'], actual_updates=4, formal_accuracy=None))

stage = parsed(kp('isolated_source_stage_20261014/SOURCE_STAGE_RECEIPT.json'))
parent_path = D/'span_lr_runtime_staging_20261013_v2/SOURCE_STAGE_RECEIPT.json'
parent = parsed(parent_path)
admission = parsed(kp(prefixes[1]+'_files/admission.json'))
launch = parsed(kp('PREFLIGHT_LAUNCH.json'))
assert parsed(kp('PREFLIGHT_TRANSPORT_EXIT.json')) == dict(exit_code=0)
assert admission == launch['admission'] and stage == admission['source_stage']
assert stage['parent_source_receipt_sha256'] == h(parent_path)
assert stage['source_file_count']==len(stage['source_files'])==240 and len(parent['source_files'])==118
lineage = {}
for arm in ('final_duplicate','coarse_endpoint'):
    changed=[];added=[];same=[]
    rows = {key[len(arm)+1:]:value for key,value in stage['source_files'].items() if key.startswith(arm+'/')}
    assert len(rows)==120 and set(parent['source_files']) <= set(rows)
    for name,row in rows.items():
        assert row['path']==stage['model_sources'][arm]+'/'+name
        if name not in parent['source_files']: added.append(name)
        elif row['sha256']==parent['source_files'][name]['sha256'] and row['bytes']==parent['source_files'][name]['bytes']: same.append(name)
        else: changed.append(name)
    assert changed==['models/losses.py'] and added==['matched_endpoint_geometry_objective.py','matched_endpoint_config.json'] and len(same)==117
    lineage[arm]=dict(unchanged=len(same),modified=changed,added=added)
passed('source_manifest_lineage_and_admission_bindings', lineage)
snapshot_sources = {
'052___init__.py':'models/__init__.py','108___init__.py':'utils/__init__.py',
'132_pvground_task_observation_query.py':'pvground_task_observation_query.py',
'055_losses.py':'models/losses.py','058_pv_ground.py':'models/pv_ground.py',
'050_main_utils.py':'main_utils.py','103_joint_det_dataset.py':'src/joint_det_dataset.py',
'107_train_dist_mod.py':'train_dist_mod.py','124_extremal_span_mixer.py':'extremal_span_mixer.py',
'125_mask_support_corrector.py':'mask_support_corrector.py','126_native_mask_geometry.py':'native_mask_geometry.py',
'127_native_model_initialization.py':'native_model_initialization.py','128_native_root_bbs.py':'native_root_bbs.py',
'130_pvground_semantic_assignment.py':'pvground_semantic_assignment.py',
'135_selected_support_geometry_objective.py':'selected_support_geometry_objective.py',
'113_lr_scheduler.py':'utils/lr_scheduler.py'}
assert 'scanrefer_data_contract.py' not in parent['source_files']
for filename,relative in snapshot_sources.items():
    data=raw(snap(filename)); digest=hashlib.sha256(data).hexdigest()
    assert parent['source_files'][relative]['sha256']==digest and parent['source_files'][relative]['bytes']==len(data), filename
    if filename!='055_losses.py':
        for arm in ('final_duplicate','coarse_endpoint'):
            row=stage['source_files'][arm+'/'+relative]
            assert row['sha256']==digest and row['bytes']==len(data)
for arm,config in (('final_duplicate','final_duplicate_config.json'),('coarse_endpoint','source/matched_endpoint_config.json')):
    for relative,local in [('models/losses.py','source/models/losses.py'),
                           ('matched_endpoint_geometry_objective.py','source/matched_endpoint_geometry_objective.py'),
                           ('matched_endpoint_config.json',config)]:
        row=stage['source_files'][arm+'/'+relative]
        assert h(kp(local))==row['sha256'] and len(raw(kp(local)))==row['bytes']
passed('actual_read_source_snapshots_bound_to_both_arms',dict(parent_snapshots=len(snapshot_sources),new_or_modified_files_per_arm=3))
old=raw(snap('055_losses.py')).decode('utf-8').splitlines(keepends=True)
new=raw(kp('source/models/losses.py')).decode('utf-8').splitlines(keepends=True)
ops=[op for op in difflib.SequenceMatcher(a=old,b=new,autojunk=False).get_opcodes() if op[0]!='equal']
assert len(ops)==1 and ops[0][0]=='insert' and ops[0][4]-ops[0][3]==7
insert=''.join(new[ops[0][3]:ops[0][4]])
assert 'end_points, target, last_indices, set_criterion, num_decoder_layers, SUPERVISION_ENDPOINT)' in insert
assert 'loss = loss + endpoint_extra' in insert
loss_tree=ast.parse(''.join(new))
assert not [n for n in ast.walk(loss_tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calculate_diou_3d']
passed('criterion_exact_seven_line_insertion_and_unused_legacy_diou',
       dict(inserted_lines=7,other_edits=0,last_indices_preserved=True,legacy_diou_called=False))

helper_locals={'initialization_reference.json':'runtime_preparation/initialization_reference.json',
'native_matched_endpoint_preflight.py':'runtime_preparation/native_matched_endpoint_preflight.py',
'matched_endpoint_preflight_pair_controller.py':'launch_preparation/matched_endpoint_preflight_pair_controller.py',
'normal_matched_endpoint_controller.py':'launch_preparation/normal_matched_endpoint_controller.py'}
for name,local in helper_locals.items():
    assert h(kp(local))==admission['helper_sha256'][name]==stage['assets'][name]['sha256']
    assert len(raw(kp(local)))==stage['assets'][name]['bytes']
protocols={}
for arm in ('final_duplicate','coarse_endpoint'):
    protocol=parsed(kp('runtime_preparation/'+arm+'_protocol.json'))
    protocols[arm]=protocol
    staged=dict(protocol)
    staged.update(status='SOURCE_STAGED_NOT_GPU_ADMITTED_OR_QUEUED',source_local_preparation_only=False,source_stage_only=True)
    b=(json.dumps(staged,indent=2)+'\n').encode()
    assert hashlib.sha256(b).hexdigest()==admission['helper_sha256'][arm+'_protocol.json']==stage['assets'][arm+'_protocol.json']['sha256']
    assert len(b)==stage['assets'][arm+'_protocol.json']['bytes']
    assert '--checkpoint_path' not in protocol['common_arguments'] and '--frozen' not in protocol['common_arguments']
diff=[key for key in protocols['final_duplicate'] if protocols['final_duplicate'][key]!=protocols['coarse_endpoint'][key]]
assert set(diff)=={'arm','matched_endpoint','model_source'}
env=parsed(snap('140_039_env_spec.json'))
env_hash=hashlib.sha256(json.dumps(env,sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert env_hash==admission['env_spec_sha256']==stage['env_spec_sha256']==parent['env_spec_sha256']
assert launch['controller_argv'][:4]==['flock','--no-fork','-n',env['resource_limits']['gpu_lock']]
assert launch['controller_pid']==last['controller_pid']==739470
assert admission['runtime']=='/root/autodl-tmp/mcln_pvground_runtime_20260908_v1/venv/bin/python'
review=parsed(kp('review_execution/EXECUTION_REVIEW_COMPLETE.json'))
assert h(kp('review_execution/EXECUTION_REVIEW_COMPLETE.json'))==admission['source_review_sha256']
for filename in ('launch_preflight_after_terminal.py','launch_normal_after_preflight.py'):
    assert h(kp(filename))==review['verified_source_sha256'][filename]
assert h(kp('review_launch/LAUNCHERS_REVIEW_COMPLETE.json'))==stage['reviewed_local_source_marker_sha256']
passed('helper_protocol_environment_and_launch_source_bindings',
       dict(protocol_difference_keys=diff,environment_canonical_sha256=env_hash,helper_assets=6,flock=env['resource_limits']['gpu_lock']))
receipts=last['receipts']
fields=('initial_model_state_sha256','input_batch_sha256','optimizer_group_lrs','optimizer_group_elements')
for field in fields:
    assert receipts['final_duplicate'][field]==receipts['coarse_endpoint'][field]
assert receipts['final_duplicate']['optimizer_group_lrs']==[1e-6,1e-6,1e-5,1e-5,1e-6]
assert receipts['final_duplicate']['optimizer_group_elements']==[25397851,3485376,0,27841,29793]
assert len(receipts['final_duplicate']['input_batch_sha256'])==2
passed('common_initial_state_inputs_and_all_five_optimizer_groups',
       {key:receipts['final_duplicate'][key] for key in fields})
groups=('backbone','original_core_and_G_reader','support_corrector','span_internal','span_output')
arms={}
for arm,endpoint in (('final_duplicate','final'),('coarse_endpoint','coarse')):
    receipt=receipts[arm]
    folder=kp(prefixes[1]+'_files')/arm
    diagnostic=parsed(folder/'witness/NATIVE_M0_TRAINING_DIAGNOSTIC.json')
    for key in ('source_mode','gradient_norms','updates','losses','gradient_acceptance','actual_updates','formal_accuracy'):
        assert diagnostic[key]==receipt[key]
    assert diagnostic['status']=='TWO_ACTUAL_NATIVE_UPDATES_BEFORE_ACCEPTANCE'
    assert not diagnostic['normal_epoch_training_started']
    assert receipt['status']=='ACTUAL_MATCHED_ENDPOINT_NORMAL_TRAINER_TWO_STEP_AND_SAME_PROCESS_RECOVERY_COMPLETE'
    assert receipt['source_mode']==arm and receipt['matched_endpoint']==endpoint
    assert receipt['actual_updates']==2 and receipt['real_training_batches']==2 and receipt['actual_input_rows']==16
    assert receipt['real_train_dataset_rows']==36665 and receipt['real_formal_dataset_rows']==9508
    assert receipt['normal_epoch_updates']==4583 and receipt['full_state_tensors']==1295
    assert receipt['native_train_one_epoch_called'] and receipt['original_trainable_core_jointly_updated']
    assert receipt['original_roberta_frozen'] and receipt['selected_geometry_supervision'] and receipt['selected_mask_supervision_disabled']
    assert receipt['full_recovery_exact'] and receipt['restored_optimizer_states']==820
    assert receipt['pretrained_span_initial_mix_formula_verified']
    assert receipt['span_initialization_identity']==parsed(kp('runtime_preparation/initialization_reference.json'))['span_checkpoint_sha256']
    assert receipt['formal_accuracy'] is None and receipt['saved_retained_metrics'] is None
    assert not receipt['preflight_weights_used_for_formal_training'] and not receipt['actual_normal_epoch_training_started']
    assert [row['step'] for row in receipt['losses']]==[1,2]
    assert len(receipt['gradient_norms'])==2
    for gradient in receipt['gradient_norms']:
        assert set(gradient)==set(groups)
        assert all(math.isfinite(value) and value>0 for value in gradient.values())
    for group in groups:
        u=receipt['updates'][group]
        assert u['changed_elements']>0 and u['changed_parameters']>0 and math.isfinite(u['maximum_change']) and u['maximum_change']>0
    for witness,loss in zip(receipt['endpoint_witness'],receipt['losses']):
        assert loss['matched_endpoint']==witness and witness['endpoint']==endpoint
        assert witness['matched_pairs']==8 and witness['nonzero_direct_gradient_queries']==8
        assert witness['direct_gradient_l1']>0 and witness['unmatched_direct_gradient_l1']==0
        assert witness['other_GT_slots_matched']==0 and len(witness['original_indices'])==8
        assert all(len(row['query'])==1 and 0<=row['query'][0]<256 and row['target']==[0] for row in witness['original_indices'])
        assert math.isfinite(loss['total_loss']) and math.isfinite(witness['auxiliary_loss'])
        geom=loss['extra_geometry']
        assert geom['extra_rows']==loss['extra_direct_gradient_nonzero_queries']>0
        assert loss['extra_direct_gradient_l1']>0 and geom['original_matching_preserved']
    for module,path in receipt['actual_imports'].items():
        relative={'models.pv_ground':'models/pv_ground.py','models.losses':'models/losses.py'}.get(module,module+'.py')
        assert path==stage['source_files'][arm+'/'+relative]['path']
    weight=[w for w in last['weights'] if w['arm']==arm][0]
    assert weight['bytes']==receipt['checkpoint_bytes']==841676128
    assert weight['sha256']==receipt['checkpoint_sha256'] and weight['unscored_engineering_only']
    assert '/preflight/'+arm+'/witness/' in weight['path'] and weight['path'].endswith('/best.pth')
    log=raw(folder/'preflight.log.tail.txt').decode('utf-8')
    lines=[line for line in log.splitlines() if 'adaptive_weight_loss_dice' in line]
    assert len(lines)==2
    for index,line in enumerate(lines):
        for key,values in [('loss',[row['total_loss'] for row in receipt['losses']]),
                           ('matched_endpoint_aux_loss',[row['auxiliary_loss'] for row in receipt['endpoint_witness']]),
                           ('selected_geometry_extra_loss',[row['extra_geometry']['loss'] for row in receipt['losses']])]:
            actual=float(re.search(r'(?:^|\s)'+key+r' ([0-9.]+)',line).group(1))
            mean=sum(values[:index+1])/(index+1)
            assert abs(actual-mean)<=0.00005001
    final=json.loads(log.splitlines()[-1])
    assert final['status']==receipt['status'] and final['actual_updates']==2 and final['checkpoint_bytes']==receipt['checkpoint_bytes']
    assert final['step_seconds']==receipt['step_seconds'] and final['formal_accuracy'] is None
    arms[arm]=dict(actual_updates=2,checkpoint_sha256=weight['sha256'],checkpoint_bytes=weight['bytes'],
                   restored_optimizer_states=820,matched_queries_per_step=[8,8],other_gt_slots_per_step=[0,0],
                   auxiliary_losses=[w['auxiliary_loss'] for w in receipt['endpoint_witness']],
                   step_total_losses=[x['total_loss'] for x in receipt['losses']],
                   auxiliary_direct_gradient_l1=[w['direct_gradient_l1'] for w in receipt['endpoint_witness']],
                   training_seconds=receipt['step_seconds'],peak_memory_bytes=receipt['peak_memory_bytes'])
assert receipts['final_duplicate']['checkpoint_sha256']!=receipts['coarse_endpoint']['checkpoint_sha256']
assert [w['original_indices'] for w in receipts['final_duplicate']['endpoint_witness']]==[w['original_indices'] for w in receipts['coarse_endpoint']['endpoint_witness']]
passed('actual_receipts_diagnostics_gradients_updates_imports_and_checkpoint_intake',arms)
passed('native_cumulative_log_means_match_per_step_diagnostics',dict(loss_keys=3,steps_per_arm=2,decimal_tolerance=0.00005001))
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
result=dict(status='PASS',tests_passed=len(tests),tests=tests,
            execution=dict(stdlib_only=True,project_modules_imported=0,model_factory_calls=0,
                           raw_dataset_loads=0,checkpoint_payload_loads=0,SSH_calls=0,
                           live_GPU_queries=0,live_process_queries=0,new_neural_forwards=0,
                           new_optimizer_updates=0),
            input_hashes=EXPECTED)
disk(OUT/'DETERMINISTIC_CHECKS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],tests_passed=len(tests),inputs=len(EXPECTED),
                     output=str(OUT/'DETERMINISTIC_CHECKS.json'))))

