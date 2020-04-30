import re
import spacy


def get_nlp():
    """
    get spacy nlp and modify its tokenizer
    """
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger'])
    # remove dash (-) from infixes
    infixes = list(nlp.Defaults.infixes).copy()
    del infixes[6]
    nlp.tokenizer.infix_finditer = spacy.util.compile_infix_regex(infixes).finditer
    return nlp


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
