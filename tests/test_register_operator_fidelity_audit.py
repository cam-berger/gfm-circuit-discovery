"""Fidelity authorization must fail on missing data, wrong masks and changed scores."""
from __future__ import annotations

import copy
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "register_operator_fidelity_audit_tested",
    ROOT / "scripts/splicebert_register_operator_replication/audit_fidelity.py",
)
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

LOGITS = [-4.0, 2.5, -2.0, -1.0, -5.0, -2.0, .5, -.1, 2.0, 1.0]


def scalar_scores(base: str) -> dict:
    """Independent fixture oracle; alternatives differ and a special token wins."""
    values = np.asarray(LOGITS, dtype=np.float32)
    bases = values[[6,7,8,9]]
    target = values[audit.IDS[base]]
    other = [values[audit.IDS[b]] for b in "ACGT" if b != base]
    return {
        "linear_margin": float(np.float32(target - np.float32(np.float32(bases.sum()-target)/np.float32(3)))),
        "max_margin": float(np.float32(target-max(other))),
        "true_base_log_probability": float(np.float32(float(target)-np.log(np.exp(np.asarray(LOGITS,dtype=float)).sum()))),
        "correct": False,
    }


def emitted(request) -> dict:
    token_ids = [2, *[audit.IDS[b] for b in request.sequence], 3]
    token_ids[request.position+1] = 4
    return {
        "request_id":request.request_id,"sequence_sha256":audit.sequence_digest(request.sequence),
        "position":request.position,"token_position":request.position+1,"true_base":request.true_base,
        "input_ids_sha256":audit.digest(token_ids),"attention_mask_sha256":audit.digest([1]*1024),
        "token_count":1024,"masked_logits":list(LOGITS),"base_logits":[LOGITS[audit.IDS[b]] for b in "ACGT"],
        "all_forward_logits_finite":True,
        **scalar_scores(request.true_base),
    }


@pytest.fixture
def old_shaped():
    genome = list("C"*18000)
    sources = {}
    for i in range(6):
        start = 1000 + 2000*i
        sequence = list("C"*1022)
        for j,position in enumerate((*audit.PROBES["donor"],*audit.PROBES["acceptor"])):
            sequence[position] = "ACGT"[(j+i)%4]
        sequence[392:394] = "GT"
        sequence[510:512] = "AG"
        sequence = "".join(sequence)
        genome[start:start+1022] = sequence
        source_id = f"source-{i}"
        def frag(left,right,kind):
            return {"chrom":"chr1","genomic_start":start+left,"genomic_end":start+right,
                    "strand":"+","sequence_sha256":audit.sequence_digest(sequence[left:right]),
                    "kind":kind,"source_id":source_id}
        sources[i] = {
            "source_id":source_id,"gene_id":f"gene-{i}","boundary_phase":i%3,"chrom":"chr1","strand":"+",
            "donor_boundary":start+392,"acceptor_boundary":start+512,"intron_length":120,"native_donor_pos":392,
            "donor_fragment":frag(0,424,"donor"),"acceptor_fragment":frag(448,1022,"acceptor"),
            "native_window":{"chrom":"chr1","window_start":start,"window_end":start+1022,
                             "strand":"+","sequence_sha256":audit.sequence_digest(sequence)},
            "_sequence":sequence,
        }
    bridge = "A" + "C"*10 + "AG" + "C"*10 + "T"
    genome[16000:16024] = bridge
    genome = "".join(genome)
    block = {"block_id":"replication:chr1:+:b000","chrom":"chr1","strand":"+",
             "donors":{str(p):sources[p] for p in range(3)},
             "acceptors":{str(q):sources[3+q] for q in range(3)},
             "bridge":{"chrom":"chr1","strand":"+","genomic_start":16000,"genomic_end":16024,
                       "sequence_sha256":audit.sequence_digest(bridge)},"cells":[]}
    cells, native = [], []
    margin = scalar_scores("G")["linear_margin"]
    for role,offset in (("donor",0),("acceptor",3)):
        for phase in range(3):
            native.append({"block_id":block["block_id"],"source_role":role,"phase":phase,
                           "source_id":sources[offset+phase]["source_id"],"chrom":"chr1","strand":"+",
                           "donor_margin_intact":margin,"donor_margin_true_acceptor_ac":margin,
                           "acceptor_margin_intact":margin,"c_ac":0.0})
    for p in range(3):
        for q in range(3):
            sequence = sources[p]["_sequence"][:424] + bridge + sources[q+3]["_sequence"][448:]
            block["cells"].append({"p":p,"q":q,"sequence_sha256":audit.sequence_digest(sequence)})
            key = f"{block['block_id']}:p{p}:q{q}"
            row = {"block_id":block["block_id"],"p":p,"q":q,"donor_phase":p,"acceptor_phase":q,
                   "phase_gap":(q-p)%3,"compatible":p==q,"chrom":"chr1","strand":"+",
                   "donor_source_id":sources[p]["source_id"],"acceptor_source_id":sources[q+3]["source_id"],
                   "donor_margins":{"intact":margin,"true_acceptor_ac":margin,"true_acceptor_at":margin,"bridge_ac":margin},
                   "acceptor_margin_intact":margin,"effects":{"c_ac":0.0,"c_at":0.0,"n_bridge":0.0},"protected_probes":[]}
            for module,positions in audit.PROBES.items():
                source_id = row[f"{module}_source_id"]
                for position in positions:
                    score = scalar_scores(sequence[position])
                    row["protected_probes"].append({
                        "module":module,"position":position,"true_base":sequence[position],
                        "construct_request_id":f"{key}:health:{module}:{position}:construct",
                        "source_request_id":f"source:{module}:{source_id}:health:{position}",
                        "construct_true_base_log_probability":score["true_base_log_probability"],
                        "source_true_base_log_probability":score["true_base_log_probability"],
                        "construct_correct":False,"source_correct":False,"true_base_log_probability_delta":0.0})
            cells.append(row)
    fetch = lambda chrom,start,end,strand: genome[start:end]
    return {"cohort":{"blocks":[block]},"cells":{"cell_rows":cells},"native":{"native_rows":native},"fetch":fetch}


@pytest.fixture
def valid(old_shaped):
    reference = audit.expected_requests(old_shaped["cohort"],old_shaped["cells"],old_shaped["native"],
                                        old_shaped["fetch"],expected_blocks=1)
    payload = {"schema":audit.SCHEMA,"stage":"fidelity","status":"COMPLETE","batch_size":16,
               "cohort_sha256":"cohort","freeze_sha256":"freeze",
               "tokenizer_ids":dict(audit.IDS)}
    for stage,requests in reference.items():
        payload[f"{stage}_requests"] = [emitted(r) for r in requests]
        payload[f"{stage}_repeats"] = copy.deepcopy(payload[f"{stage}_requests"][:16])
    return payload,reference


def run(valid):
    return audit.audit_records(*valid,cohort_sha256="cohort",freeze_sha256="freeze")


def test_full_old_shaped_request_family_passes(valid):
    result = run(valid)
    assert result["passed"], result
    assert result["request_counts"] == {"cell":{"expected":333,"observed":333},"native":{"expected":18,"observed":18}}


@pytest.mark.parametrize("stage",["cell","native"])
@pytest.mark.parametrize("fault",["drop","duplicate","swap","extra_cc"])
def test_missing_duplicate_reordered_or_extra_requests_cannot_authorize(valid,stage,fault):
    rows = valid[0][f"{stage}_requests"]
    if fault == "drop":
        rows.pop()
    elif fault == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif fault == "swap":
        rows[0],rows[1] = rows[1],rows[0]
    else:
        extra = copy.deepcopy(rows[0]); extra["request_id"] += ":CC"; rows.append(extra)
    assert not run(valid)["passed"]


@pytest.mark.parametrize("field,value",[("position",393),("token_position",392),("token_count",1023),
    ("input_ids_sha256","wrong"),("sequence_sha256","wrong"),("attention_mask_sha256","wrong"),
    ("true_base","A"),("all_forward_logits_finite",False)])
def test_mask_source_or_forward_health_mismatch_fails(valid,field,value):
    valid[0]["cell_requests"][20][field] = value
    assert not run(valid)["passed"]


def test_discrete_protected_accuracy_has_no_tolerance(valid):
    row = next(r for r in valid[0]["cell_requests"][16:] if ":health:" in r["request_id"])
    row["correct"] = True
    assert not run(valid)["passed"]


@pytest.mark.parametrize("field",["linear_margin","max_margin","true_base_log_probability"])
def test_a_raw_discrepancy_above_frozen_tolerance_fails(valid,field):
    valid[0]["cell_requests"][20][field] += 1.01e-4
    assert not run(valid)["passed"]


def test_protected_log_probability_uses_full_vocabulary(valid):
    row = valid[0]["cell_requests"][20]
    nt = np.asarray(LOGITS)[[6,7,8,9]]
    row["true_base_log_probability"] = float(LOGITS[audit.IDS[row["true_base"]]] - np.log(np.exp(nt).sum()))
    assert not run(valid)["passed"]


@pytest.mark.parametrize("base_logits",[None,[0.0]*3,[0.0]*4,[float("nan")]*4])
def test_four_base_copy_must_match_complete_logits(valid,base_logits):
    valid[0]["cell_requests"][20]["base_logits"] = base_logits
    assert not run(valid)["passed"]


def test_max_margin_excludes_special_tokens(valid):
    row = valid[0]["cell_requests"][53]
    row["max_margin"] = float(LOGITS[8] - max(LOGITS[:8]+LOGITS[9:]))
    assert not run(valid)["passed"]


@pytest.mark.parametrize("field",["linear_margin","max_margin","true_base_log_probability"])
def test_repeat_gate_retains_one_e_minus_eight(valid,field):
    valid[0]["cell_repeats"][0][field] += 1.1e-8
    assert not run(valid)["passed"]


def test_repeat_logit_change_fails_even_when_rounded_scores_match(valid):
    valid[0]["native_repeats"][0]["masked_logits"][6] += 1.1e-8
    assert not run(valid)["passed"]


@pytest.mark.parametrize("field,value",[("schema","other"),("stage","confirmatory"),
    ("status","INSTRUMENT_UNRESOLVED"),("batch_size",8),
    ("cohort_sha256","subset"),("freeze_sha256","stale"),("tokenizer_ids",{})])
def test_fidelity_binding_or_stage_mismatch_fails(valid,field,value):
    valid[0][field] = value
    assert not run(valid)["passed"]


@pytest.mark.parametrize("fault",["missing_probe","wrong_source","wrong_cell_hash","native_position","duplicate_cell"])
def test_reference_reconstruction_rejects_corrupt_old_shaped_inputs(old_shaped,fault):
    row = old_shaped["cells"]["cell_rows"][0]
    if fault == "missing_probe": row["protected_probes"].pop()
    elif fault == "wrong_source": row["protected_probes"][0]["source_request_id"] = "other"
    elif fault == "wrong_cell_hash": old_shaped["cohort"]["blocks"][0]["cells"][0]["sequence_sha256"] = "wrong"
    elif fault == "native_position": old_shaped["cohort"]["blocks"][0]["donors"]["0"]["native_donor_pos"] += 1
    else: old_shaped["cells"]["cell_rows"][-1] = copy.deepcopy(row)
    with pytest.raises(ValueError):
        audit.expected_requests(old_shaped["cohort"],old_shaped["cells"],old_shaped["native"],old_shaped["fetch"],expected_blocks=1)


def test_inference_comparison_checks_shapes_flags_and_numerics():
    expected = {"a":[.1,.2],"flag":True,"n":384}
    assert audit.compare_inference(copy.deepcopy(expected),expected)["bitwise_identical"]
    changed = copy.deepcopy(expected); changed["a"][0] += 5e-11
    result = audit.compare_inference(changed,expected)
    assert result["passed"] and not result["bitwise_identical"]
    for altered in ({"a":[.1],"flag":True,"n":384}, {"a":[.1,.2],"flag":1,"n":384},
                    {"a":[.1,.2001],"flag":True,"n":384}):
        result = audit.compare_inference(altered,expected)
        assert not result["passed"] and not result["bitwise_identical"]
    result = audit.compare_inference({"zero":-0.0},{"zero":0.0})
    assert result["passed"] and not result["bitwise_identical"]


def test_independent_inference_detects_known_lattice_and_regenerates_all_pseudos():
    cells, ids, stored = [], [], {}
    for i in range(384):
        chrom, strand = f"chr{1 + i % 2}", "+" if i % 4 < 2 else "-"
        bid = f"replication:{chrom}:{strand}:b{i:03d}"; ids.append(bid)
        for p in range(3):
            for q in range(3):
                effect = .12 if p == q else 0.0
                cells.append({"block_id":bid,"p":p,"q":q,"chrom":chrom,"strand":strand,
                              "donor_margins":{"intact":1.0,"true_acceptor_ac":1.0-effect,"true_acceptor_at":1.0-effect}})
    import hashlib
    choices = list(permutations(range(3)))
    for seed in range(2026080401,2026080500):
        order = sorted(ids,key=lambda bid:hashlib.sha256(f"crossed-phase-null-v1\0{seed}\0{bid}".encode()).hexdigest())
        assignment = {bid:list(choices[rank//64]) for rank,bid in enumerate(order)}
        stored[str(seed)] = {"assignments":[{"block_id":bid,"acceptor_label_permutation":assignment[bid]} for bid in ids]}
    stop_blocks = [{"block_id":bid,"stop_diagnostics":{
        "stop_count_matrix":((np.ones((3,3))-np.eye(3))*(i%2)).tolist(),
        "omega_mismatch_minus_diagonal":float(i%2)}} for i,bid in enumerate(ids)]
    cohort = {"pseudo_phase_family":{"roles":{"replication":{"assignments_by_seed":stored}}},
              "blocks":stop_blocks,"stop_diagnostics":{"replication_median_omega_cutpoint":.5}}
    result = audit.historical_inference({"cell_rows":cells},cohort)
    assert result["primary_psi"]["mean"] == pytest.approx(.12,abs=1e-14)
    assert result["primary_psi"]["mde80"] < 1e-12
    assert len(result["pseudo_phase"]["control_means"]) == 99
    assert max(abs(v) for v in result["pseudo_phase"]["control_means"]) < 1e-14
    assert result["pseudo_phase"]["exact_add_one_one_sided_p"] == .01
    assert result["stop_sensitivity"]["cutpoint"] == .5
    assert result["stop_sensitivity"]["spearman_rho_omega_psi"] is None
    assert result["stop_sensitivity"]["low_or_equal"]["n"] == 192
    assert result["stop_count_mean_3x3"] == ((np.ones((3,3))-np.eye(3))*.5).tolist()
    cohort["stop_diagnostics"]["replication_median_omega_cutpoint"] = .1
    with pytest.raises(ValueError,match="stop cutpoint"):
        audit.historical_inference({"cell_rows":cells},cohort)
    cohort["stop_diagnostics"]["replication_median_omega_cutpoint"] = .5
    stop_blocks[0]["stop_diagnostics"]["omega_mismatch_minus_diagonal"] = 1.0
    with pytest.raises(ValueError,match="stop-load"):
        audit.historical_inference({"cell_rows":cells},cohort)
    stop_blocks[0]["stop_diagnostics"]["omega_mismatch_minus_diagonal"] = 0.0
    stored["2026080401"]["assignments"][0]["acceptor_label_permutation"] = [0,0,0]
    with pytest.raises(ValueError,match="pseudo assignment"):
        audit.historical_inference({"cell_rows":cells},cohort)


def test_cli_writes_immutable_failure_and_manifest_for_invalid_reference(tmp_path):
    args = []
    for name in ("fidelity","cohort","fasta","cells","native","historical-audit","freeze"):
        path = tmp_path/(name+".json"); path.write_text("{}\n")
        args += ["--"+name,str(path)]
    out = tmp_path/"audit.json"; args += ["--out",str(out)]
    assert audit.main(args) == 1
    value = json.loads(out.read_text())
    assert value["status"] == "FAIL" and value["authorized_for_cc"] is False
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["status"] == "FAIL" and manifest["outputs"][0]["sha256"] == audit.sha256_file(out)
    before = out.read_bytes()
    with pytest.raises(FileExistsError): audit.main(args)
    assert out.read_bytes() == before


@pytest.mark.parametrize("destination",["artifact","manifest"])
def test_cli_rejects_dangling_output_symlinks(tmp_path,destination):
    out = tmp_path/"audit.json"
    link = out if destination == "artifact" else out.with_suffix(".manifest.json")
    absent = tmp_path/"absent-target.json"
    link.symlink_to(absent)
    args = ["--out",str(out)]
    for name in ("fidelity","cohort","fasta","cells","native","historical-audit","freeze"):
        args += ["--"+name,str(tmp_path/"unused.json")]
    with pytest.raises(FileExistsError): audit.main(args)
    assert link.is_symlink() and not absent.exists()
