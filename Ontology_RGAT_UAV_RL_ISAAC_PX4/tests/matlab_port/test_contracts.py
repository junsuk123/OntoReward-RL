import copy

import pytest

from ontology_rgat.direct_policy.contracts import make_contract, validate_checkpoint_metadata
from ontology_rgat.direct_policy.models import checkpoint_metadata


def test_hash_domains_are_separate_and_stable():
    local = make_contract(2, backend="local", safety_profile="direct")
    isaac = make_contract(2, backend="isaac", safety_profile="direct")
    shielded = make_contract(2, backend="local", safety_profile="shielded")
    assert local.algorithm_hash == isaac.algorithm_hash == shielded.algorithm_hash
    assert local.task_contract_hash == isaac.task_contract_hash == shielded.task_contract_hash
    assert local.execution_hash != isaac.execution_hash
    assert local.execution_hash != shielded.execution_hash


def test_checkpoint_metadata_rejects_every_contract_mismatch():
    contract = make_contract(2)
    metadata = checkpoint_metadata(contract, training_step=3, optimizer_state={}, rng_state=[],
                                   method_hash="fixture")
    validate_checkpoint_metadata(metadata, contract)
    for key in ("source_sha", "algorithm_hash", "task_contract_hash", "execution_hash"):
        changed = copy.deepcopy(metadata)
        changed[key] = "0" * 64
        with pytest.raises(ValueError, match=key):
            validate_checkpoint_metadata(changed, contract)
