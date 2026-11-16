from collections import Counter

from sourcelens.data import load_agent_requests, load_candidates, load_jds


def test_counts():
    cands, jds = load_candidates(), load_jds()
    assert len(cands) == 30
    assert len(jds) == 3
    assert len({c.id for c in cands}) == 30
    assert len({j.id for j in jds}) == 3


def test_every_candidate_labelled():
    ids = {c.id for c in load_candidates()}
    for jd in load_jds():
        assert set(jd.labels) == ids, jd.id
        assert set(jd.labels.values()) <= {0, 1, 2}
        assert sum(v == 2 for v in jd.labels.values()) >= 4, jd.id


def test_traps():
    cands = load_candidates()
    tags = Counter(c.trap for c in cands if c.trap)
    for tag in ("stuffer", "synonym", "location", "seniority"):
        assert tags[tag] >= 2, tag
    assert tags["injection"] == 3
    zero_tags = {c.id for c in cands if c.trap in ("injection", "stuffer")}
    for jd in load_jds():
        assert all(jd.labels[cid] == 0 for cid in zero_tags), jd.id


def test_aliases_cover_gold():
    for jd in load_jds():
        for name in jd.gold.must_have + jd.gold.nice_to_have:
            assert name in jd.aliases, (jd.id, name)


def test_text_contains_bullets():
    c = load_candidates()[0]
    text = c.text()
    for exp in c.experience:
        for b in exp.bullets:
            assert b in text


def test_agent_requests_point_at_jds():
    jd_ids = {j.id for j in load_jds()}
    reqs = load_agent_requests()
    assert len(reqs) == 3
    assert {r.jd_id for r in reqs} == jd_ids
