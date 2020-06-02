import re
import spacy
from spacy.attrs import ORTH


def _add_special_cases(tokenizer):
    # special case ##ed:

    case = [{ORTH: "burn"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("burned", case)

    case = [{ORTH: "dice"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("diced", case)

    case = [{ORTH: "slice"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("sliced", case)

    case = [{ORTH: "chop"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("chopped", case)

    case = [{ORTH: "close"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("closed", case)

    case = [{ORTH: "open"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("opened", case)

    case = [{ORTH: "fry"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("fried", case)

    case = [{ORTH: "grill"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("grilled", case)

    case = [{ORTH: "roast"}, {ORTH: "##ed"}]
    tokenizer.add_special_case("roasted", case)

    # special case ##ing

    case = [{ORTH: "fry"}, {ORTH: "##ing"}]
    tokenizer.add_special_case("frying", case)

    case = [{ORTH: "grill"}, {ORTH: "##ing"}]
    tokenizer.add_special_case("grilling", case)

    case = [{ORTH: "roast"}, {ORTH: "##ing"}]
    tokenizer.add_special_case("roasting", case)

    return tokenizer


def get_tokenizer():
    """
    get spacy nlp and modify its tokenizer
    """
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
    # remove dash (-) from infixes
    infixes = list(nlp.Defaults.infixes).copy()
    del infixes[6]
    nlp.tokenizer.infix_finditer = spacy.util.compile_infix_regex(infixes).finditer
    # add special cases
    tokenizer = _add_special_cases(tokenizer=nlp.tokenizer)
    return tokenizer


def preproc(string, tokenizer):
    """
    basically to tokenize
    """
    if string is None:
        return [None]

    pattern = r"[_\\|/$>,>]"
    s = re.sub(pattern, '', string).replace("\n", ' ').strip().lower()

    if '-=' in s:
        pattern = r"""-= [\s\S]* =-"""
        m = re.search(pattern, s)
        span = m.span()
        s = s[:span[0]] + m.group(0).replace(' ', '') + s[span[1]:]
    if '***' in s:
        pattern = r"""\*\*\*[\s\S]*\*\*\*"""
        m = re.search(pattern, s)
        span = m.span()
        s = s[:span[0]] + m.group(0).replace('*', '-').replace(' ', '-') + s[span[1]:]

    return [t.text for t in tokenizer(s) if not t.is_space]
