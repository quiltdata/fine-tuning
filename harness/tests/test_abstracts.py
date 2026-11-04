from abstracts import LIMIT_BYTES, article_fields, main_abstract, protocol_flags, truncate_utf8, utf8_bytes

STRUCTURED = """<article article-type="research-article"><front><journal-meta><journal-title>Plant Methods</journal-title>
</journal-meta><article-meta><title-group><article-title>Leaves in <italic>3D</italic></article-title></title-group>
<abstract abstract-type="graphical"><p>A picture.</p></abstract>
<abstract abstract-type="structured"><sec><title>Background</title><p>Leaves &amp; air.</p></sec>
<sec><title>Results</title><p>Porosity rose 12%   in   wet
 leaves.</p></sec></abstract></article-meta></front><body><abstract><p>Not this one.</p></abstract></body></article>"""


def test_main_abstract_skips_secondary_and_keeps_headings():
    a = main_abstract(STRUCTURED)
    assert a["text"] == "Background. Leaves & air. Results. Porosity rose 12% in wet leaves."
    assert a["headings"] == ["Background", "Results"]
    assert a["structured"] and a["abstract_type"] == "structured"
    assert a["bytes"] == utf8_bytes(a["text"])


def test_main_abstract_untagged_and_missing():
    assert main_abstract("<article-meta><abstract><p>Plain.</p></abstract></article-meta>")["abstract_type"] == "main"
    assert main_abstract("<article-meta><abstract abstract-type=\"toc\"><p>x</p></abstract></article-meta>") is None
    assert main_abstract("<article-meta></article-meta>") is None


def test_article_fields():
    f = article_fields(STRUCTURED)
    assert f == {"title": "Leaves in 3D", "journal": "Plant Methods", "jats_article_type": "research-article"}


def test_truncate_fits_unchanged():
    assert truncate_utf8("short.") == ("short.", False)
    exact = "a" * LIMIT_BYTES
    assert truncate_utf8(exact) == (exact, False)


def test_truncate_never_splits_a_character():
    text = "µ" * 3000  # 2 bytes each: 6,000 bytes
    out, cut = truncate_utf8(text)
    assert cut and utf8_bytes(out) <= LIMIT_BYTES and set(out) == {"µ"}
    emoji = "a" + "😀" * 2000  # 4-byte characters, odd offset
    out, _ = truncate_utf8(emoji)
    assert utf8_bytes(out) <= LIMIT_BYTES and out.encode("utf-8").decode("utf-8") == out


def test_truncate_prefers_sentence_then_space():
    sentence = ("x" * 3500) + ". Results came next " + "y" * 1000
    out, cut = truncate_utf8(sentence)
    assert cut and out.endswith(".") and utf8_bytes(out) == 3501
    words = "word " * 1000  # 5,000 bytes, no sentence end
    out, _ = truncate_utf8(words)
    assert out.endswith("word") and utf8_bytes(out) <= LIMIT_BYTES
    early = "Short. " + "z" * 5000  # only sentence end is far before three quarters: fall back
    out, _ = truncate_utf8(early)
    assert utf8_bytes(out) == LIMIT_BYTES


def test_protocol_flags_on_34c_cases():
    yes = [
        ([], "Dietary Approaches (DIAMOND): protocol for a randomised feasibility trial", []),
        ([], "Differential methylation in rare ophthalmic disorders: a systematic review protocol", []),
        (["Journal Article", "Clinical Trial Protocol"], "Effects of training on injury risk", []),
        ([], "A blended knowledge translation initiative", ["Background", "Methods/Design", "Discussion"]),
        ([], "Effects of therapy", ["Introduction", "Methods and analysis", "Ethics and dissemination"]),
    ]
    for pub, title, heads in yes:
        assert protocol_flags(pub, title, heads)["protocol"], title
    no = [
        ([], "A membrane-free spot-plating protocol for Agrobacterium-mediated transformation of fungi", []),
        ([], "Protocol for detecting cancer in blood using targeted DNA methylation", []),
        ([], "RodZ activates the elongasome", ["IMPORTANCE"]),
        ([], "Study of X", ["Methods and analysis", "Results"]),
    ]
    for pub, title, heads in no:
        assert not protocol_flags(pub, title, heads)["protocol"], title
