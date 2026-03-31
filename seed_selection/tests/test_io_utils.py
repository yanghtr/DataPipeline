from seed_selection.io_utils import infer_domain, infer_source, make_id


def test_infer_domain_stage1_icon():
    assert infer_domain("stage1/icon/generation/img2svg/data.jsonl") == "stage1_icon"


def test_infer_domain_stage2_icon():
    assert infer_domain("stage2/icon/generation/img2svg/data.jsonl") == "stage2_icon"


def test_infer_domain_illustration():
    assert infer_domain("stage2/illustration/img2svg/data.jsonl") == "stage2_illustration"


def test_infer_source_img2svg():
    assert infer_source("stage1/icon/generation/img2svg/data.jsonl") == "img2svg"


def test_infer_source_text2svg():
    assert infer_source("stage1/icon/generation/text2svg/data.jsonl") == "text2svg"


def test_make_id():
    assert make_id("data_000000", 42) == "data_000000:42"
