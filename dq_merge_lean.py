"""
HHS-ORR to affirmative asylum merge: lean version
=================================================

Same matching as dq_merge.py, trimmed to what produces the merged table:

    res = run_dq_merge(orr_df, asylum_df)
    res["merged"]   # ORR rows + asylum columns + DQ columns + matched_a_number
    res["passed"]   # True / False
    res["checks"]   # Which of the three pass/fail checks failed, with counts

STEPS
    1. normalize()          Clean each comparison field into a helper column (o__* / a__*).
    2. _orr_row_issues()    Single-table problems of each ORR row -> dq_row_issues.
    3. LEFT join            ORR to asylum on the canonical a_id. No ORR row is dropped.
    4. _score_matches()     One vectorized rapidfuzz pass per matched row (names + notes), then dq_status.
    5. _other_keys()        Asylum cases with the same person under a different a_id.
    6. matched_a_number     The a_id to use for the later full left join.
       dq_all_notes         Everything that happened to the row, in words (see _all_notes).
    7. _passed()            Three pass/fail checks in one aggregation pass.

DQ_STATUS
    VERIFIED            Name, DOB, sex and country of birth all agree.
    NEEDS_REVIEW        Minor differences only, or differences the notes explain.
    CONFLICT            Same a_id, but probably a different person.
    NO_MATCH            No asylum case has this a_id, and none was found under another a_id.
    POSSIBLE_KEY_ERROR  No a_id match, but an asylum case for the same person exists under another a_id.

MATCHED_A_NUMBER (the join key for the full left join)
    POSSIBLE_KEY_ERROR  The other asylum case's a_id, as the asylum table stores it. If several were
                        found: the strongest source (fields > notes_dob > notes_name), then the lowest
                        a_id. All of them stay in dq_possible_asylum_keys; dq_multiple_possible_keys flags it.
                        dq_possible_receipt_numbers / dq_possible_case_ids line up with the keys (same position
                        = same case), and matched_receipt_number / matched_case_id belong to the chosen case.
    Matched rows        Their own asylum case's a_id, as the asylum table stores it.
    NO_MATCH            The ORR a_id.

REQUIREMENTS
    Spark 3.5+ / DBR 13.3+, and rapidfuzz on the cluster (%pip install rapidfuzz).
"""
import functools  # functools.reduce: a notebook "from pyspark.sql.functions import *" shadows bare reduce
import re
import sys
import unicodedata
from operator import add, and_, or_

from pyspark.sql import DataFrame, functions as F


# =============================================================================
# CONFIGURATION (same names and defaults as dq_merge.py)
# =============================================================================

ORR_COLS = {"key": "a_id", "first": "first_name", "last": "last_name",
            "dob": "dob", "sex": "sex", "cob": "country_of_birth"}
# Only the asylum table has a middle name. receipt / case_id are not compared, only returned with
# possible matches (dq_possible_receipt_numbers, matched_case_id, ...).
ASY_COLS = {**ORR_COLS, "middle": "middle_name", "receipt": "receipt_number", "case_id": "case_id"}
NOTES_COL = "additional_notes"                     # Optional ORR free-text column
ASY_PREFIX = "asy_"                                # Prefix for asylum columns in the output
MATCH_FLAG_COL = "asylum_case_exists"
CACHE = True                                       # Set False on serverless, where persist() is unsupported

# Cleaning
KEY_PATTERN = r"^A?[0-9]{8,9}$"
DOB_FORMATS = ["yyyy-MM-dd", "M/d/yyyy", "yyyyMMdd", "yyyy-MM-dd HH:mm:ss"]
MIN_DOB = "1900-01-01"
NULL_TOKENS = ["", "NULL", "NONE", "N/A", "NA", "NAN", "?", "-"]
SEX_MAP = {"M": "M", "MALE": "M", "F": "F", "FEMALE": "F", "X": "X", "U": "U", "UNKNOWN": "U"}
COB_MAP = {}
PLACEHOLDER_NAMES = ["TEST", "UNKNOWN", "FNU", "LNU", "NFN", "NLN", "XXX", "DUMMY"]
NO_MIDDLE_TOKENS = ["NMN", "NOMIDDLENAME"]
ACCENTS, PLAIN = "ÁÀÂÄÃÅÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÑÇÝ", "AAAAAAEEEEIIIIOOOOOUUUUNCY"

# Notes
NOTE_DATE_REGEX = r"(\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\b\d{8}\b)"
NOTE_DATE_FORMATS = ["M/d/yyyy", "d/M/yyyy", "yyyy/M/d", "M/d/yy", "d/M/yy", "yyyyMMdd"]  # Both day orders
NOTE_DOB_LABEL = r"(?i)\b(d\.?o\.?b|birth|born|fecha de nac\w*)\b"
NOTE_STOPWORDS = [  # Never names; they also end a name candidate
    "NTA", "AS", "PER", "PERSON", "ACCORDING", "TO", "SPONSOR", "DOCS", "DOCUMENTS", "BIRTH", "CERTIFICATE",
    "PASSPORT", "ORR", "UAC", "UC", "CBP", "ICE", "FILE", "RECORD", "RECORDS",
    "CORRECT", "CORRECTED", "CORRECTION", "ACTUAL", "UPDATED", "CHANGED", "WRONG", "INCORRECT", "SHOULD", "BE",
    "IS", "WAS",
    "NAME", "NAMES", "FIRST", "LAST", "MIDDLE", "FULL", "AKA", "ALIAS", "KNOWN", "ALSO", "DOB", "DATE", "BORN",
    "NACIMIENTO", "FECHA", "NOMBRE",
    "CHILD", "MINOR", "STATES", "STATED", "SAYS", "REPORTS", "REPORTED", "THE", "AND", "OR", "OF", "IN", "ON",
    "AT", "BY", "FOR", "WITH", "HIS", "HER", "THEIR", "MOTHER", "FATHER", "SIBLING",
]
NAME_PARTICLES = ["DE", "LA", "DEL", "LOS", "LAS", "Y", "DA", "DOS", "VAN"]
NOTE_MAX_NAME_WORDS = 5

# Matching (rapidfuzz scores are 0-100)
NAME_SIM_MIN = 90   # Jaro-Winkler, single names
FULL_SIM_MIN = 88   # token_sort_ratio, full names
WEIGHTS = {"first": 0.20, "last": 0.25, "dob": 0.30, "sex": 0.10, "cob": 0.15}
CREDIT = {"EXACT": 1.0, "MIDDLE_IN_LAST": 1.0, "MIDDLE_INITIAL_IN_LAST": 1.0, "MIDDLE_IN_FIRST": 1.0,
          "MIDDLE_INITIAL_IN_FIRST": 1.0, "MIDDLE_OMITTED": 1.0, "MIDDLE_INITIAL": 1.0,
          "NOTES_MATCH": 0.9, "REORDERED": 0.8, "SUBSET": 0.6, "PARTIAL": 0.6, "FUZZY": 0.6, "SWAPPED": 0.6,
          "DAY_MONTH_SWAP": 0.5, "OFF_BY_ONE_DAY": 0.5, "YEAR_DIFF": 0.25, "MISSING": 0.0, "MISMATCH": 0.0}
EQUIVALENT = [s for s, c in CREDIT.items() if c == 1.0]
REVIEW_THRESHOLD = 0.70
MAX_CONFLICT_RATE = 0.05
MIN_MATCHES_FOR_RATE = 20

STATUS = {"full": "dq_full_name_status", "first": "dq_first_name_status", "last": "dq_last_name_status",
          "dob": "dq_dob_status", "sex": "dq_sex_status", "cob": "dq_cob_status"}
SIM_COLS = {"first": "dq_first_name_similarity", "last": "dq_last_name_similarity",
            "full": "dq_full_name_similarity"}
SOURCE_RANK = {"fields": 1, "notes_dob": 2, "notes_name": 3}  # Which other-key search to trust most


# =============================================================================
# 1. CLEANING (Spark expressions; identical to dq_merge.py)
# =============================================================================

def _col(name):
    return F.col(f"`{name}`")  # Backticks so names with dots or spaces work


def _lookup(c, mapping):
    expr = F.lit(None).cast("string")  # CASE WHEN chain: null for unmapped values, safe under ANSI mode
    for k, v in mapping.items():
        expr = F.when(c == k, F.lit(v)).otherwise(expr)
    return expr


def _clean(c):
    v = F.upper(F.trim(F.regexp_replace(c.cast("string"), r"\s+", " ")))
    return F.when(~v.isin(NULL_TOKENS), v)


def _tokens(*cols):
    """Name columns -> array of words: accents folded, apostrophes dropped, other separators split."""
    s = F.concat_ws(" ", *[F.translate(_clean(c), ACCENTS, PLAIN) for c in cols])
    s = F.regexp_replace(F.regexp_replace(s, r"['.`]", ""), r"[^A-Z]+", " ")
    toks = F.filter(F.split(F.trim(s), " "), lambda x: x != "")
    return F.when(F.size(toks) > 0, toks)


def _compact(c):
    return F.array_join(_tokens(c), "")  # "José-María O'Neil" -> JOSEMARIAONEIL


def _key(c, dtype):
    """Canonical a_id for the join: 'A-12345678', '012345678' and 12345678 -> '012345678'."""
    if dtype in ("tinyint", "smallint", "int", "bigint", "float", "double") or dtype.startswith("decimal"):
        c = c.cast("decimal(38,0)")
    v = F.regexp_replace(_clean(c), r"[\s\-]", "")
    digits = F.regexp_extract(v, r"^A?([0-9]{1,9})$", 1)
    return F.when(digits != "", F.lpad(digits, 9, "0")).otherwise(v)


def _parse_dob(name, dtype):
    if dtype == "date":
        return _col(name)
    if dtype.startswith("timestamp"):
        return F.to_date(_col(name))
    return F.coalesce(*[F.to_date(F.expr(f"try_to_timestamp(trim(`{name}`), '{fmt}')")) for fmt in DOB_FORMATS])


def _note_dates(strings):
    """Date-like strings -> plausible dates, every format tried ('01/02/2011' -> Jan 2 and Feb 1)."""
    parsed = F.flatten(F.transform(strings, lambda s: F.array(*[
        F.to_date(F.try_to_timestamp(F.regexp_replace(s, r"[.\-]", "/"), F.lit(fmt))) for fmt in NOTE_DATE_FORMATS])))
    plausible = F.filter(parsed, lambda d: d.isNotNull() & (d >= F.lit(MIN_DOB).cast("date")) & (d <= F.current_date()))
    return F.coalesce(F.array_distinct(plausible), F.array().cast("array<date>"))


def normalize(df, cols, p):
    """Add one cleaned helper column per field (<p><field>) and a row id; the originals stay untouched.

    The row id is a hash of the whole row. It needs no window or shuffle, and it is the same on every
    recompute. Two exactly identical rows share an id, which is harmless: everything joined back on
    it is grouped to one row per id first, so nothing fans out.
    """
    dt = dict(df.dtypes)
    raw = lambda f: _col(cols[f])
    cob = _clean(raw("cob"))
    rules = {
        "key": _key(raw("key"), dt[cols["key"]]),
        "first": _compact(raw("first")), "last": _compact(raw("last")),
        "dob": _parse_dob(cols["dob"], dt[cols["dob"]]),
        "sex": _lookup(_clean(raw("sex")), SEX_MAP),
        "cob": F.coalesce(_lookup(cob, COB_MAP), cob),
        "full_tok": _tokens(raw("first"), raw("last")),
        "first_tok": _tokens(raw("first")), "last_tok": _tokens(raw("last")),
    }
    if "notes" in cols:
        notes = raw("notes").cast("string")
        labelled = F.regexp_extract(notes, NOTE_DOB_LABEL + r"[^0-9]{0,20}" + NOTE_DATE_REGEX, 2)
        rules.update(notes=notes,
                     note_dobs=_note_dates(F.regexp_extract_all(notes, F.lit(NOTE_DATE_REGEX), 1)),
                     note_label_dobs=_note_dates(F.when(labelled != "", F.array(labelled))))
    # The ORR name may include, drop or abbreviate the asylum middle name, so keep three versions
    rules["nomid_tok"] = rules["init_tok"] = rules["full_tok"]
    if "middle" in cols:
        mid = F.when(~_compact(raw("middle")).isin(NO_MIDDLE_TOKENS), raw("middle"))
        rules["middle"] = _compact(mid)
        rules["full_tok"] = _tokens(raw("first"), mid, raw("last"))
        rules["init_tok"] = _tokens(raw("first"), F.substring(_compact(mid), 1, 1), raw("last"))
    row_id = F.xxhash64(*[_col(c) for c in df.columns])
    return df.select("*", row_id.alias(p + "row_id"), *[e.alias(p + f) for f, e in rules.items()])


# =============================================================================
# 2. NOTES NAME CANDIDATES + RAPIDFUZZ (plain Python, called inside vectorized UDFs)
# =============================================================================

def _py_words(text):
    s = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().upper()
    return [w for w in re.split(r"[^A-Z]+", re.sub(r"['.`]", "", s)) if len(w) > 1]


def _splits(seg):
    """Segment words -> (text, first, last, layout). A comma means LAST, FIRST (each comma part is also
    tried alone); otherwise every split point in both orders. Particles never end a part."""
    words = [w for w in seg if w != ","]
    if not words or len(words) > NOTE_MAX_NAME_WORDS or all(w in NAME_PARTICLES for w in words):
        return []
    if "," in seg:
        i = seg.index(",")
        last, first = [w for w in seg[:i] if w != ","], [w for w in seg[i + 1:] if w != ","]
        out = [(" ".join(last) + ", " + " ".join(first), " ".join(first), " ".join(last), "LAST, FIRST")] \
            if first and last else []
        part = []
        for w in seg + [","]:
            if w == ",":
                out, part = out + _splits(part), []
            else:
                part.append(w)
        return out
    text = " ".join(words)
    if len(words) == 1:
        return [(text, None, None, "SINGLE")]
    out = []
    for k in range(1, len(words)):
        if words[k - 1] not in NAME_PARTICLES:
            a, b = " ".join(words[:k]), " ".join(words[k:])
            out += [(text, a, b, "FIRST LAST"), (text, b, a, "LAST FIRST")]
    return out


def _note_name_candidates(note):
    """Cut a note at punctuation, digits and stop words into segments, and split each into names."""
    if note is None:
        return []
    stop, out, seen = set(NOTE_STOPWORDS), [], set()
    for chunk in re.split(r"[;/|()\[\]{}\":.!?\n\r\t&+=]|\d+", str(note)):
        seg = []
        for piece in re.split(r"(,)", chunk):
            for w in ([","] if piece == "," else _py_words(piece)):
                if w in stop:
                    out, seg = out + _splits(seg), []
                else:
                    seg.append(w)
        out += _splits(seg)
    return [c for c in out if not (c in seen or seen.add(c))]


def _jw(x, y):
    from rapidfuzz.distance import JaroWinkler
    return round(JaroWinkler.normalized_similarity(x.replace(" ", ""), y.replace(" ", "")) * 100, 1) if x and y else 0.0


def _token_sort(x, y):
    from rapidfuzz import fuzz
    return round(fuzz.token_sort_ratio(x, y), 1) if x and y else 0.0


def _best_note_name(note, a_first, a_last, a_full, a_nomid):
    """Best note candidate for the asylum name: (text, first, last, layout, first, last, full scores).
    All scores come from one candidate, so words are never mixed across name versions."""
    best, best_rank = None, (-1.0, -1.0)
    for text, first, last, layout in _note_name_candidates(note):
        if layout == "SINGLE":
            f, l, full = _jw(text, a_first), _jw(text, a_last), 0.0
            rank = (max(f, l) / 2,) * 2
        else:
            f, l = _jw(first, a_first), _jw(last, a_last)
            full = max(_token_sort(f"{first} {last}", a_full), _token_sort(f"{first} {last}", a_nomid))
            rank = (max((f + l) / 2, full), (f + l) / 2)
        if rank > best_rank:
            best, best_rank = (text, first, last, layout, f, l, full), rank
    return best


def _ship_module_with_udfs():
    """Send this module's code to the Spark workers together with the UDFs.

    By default Spark only sends a reference to the module, and each worker must be able to import
    dq_merge_lean itself. In a scheduled job (repo files, job clusters, shared or serverless compute)
    that is not guaranteed and fails with ModuleNotFoundError on the workers. Pickling the module
    by value removes that dependency. Harmless if it is already pasted into a notebook.
    """
    module = sys.modules.get(__name__)
    if module is None or __name__ == "__main__":
        return
    try:
        from pyspark import cloudpickle
        cloudpickle.register_pickle_by_value(module)
    except Exception:  # Older Spark: fall back to the default (module must be importable on workers)
        pass


def _udfs():
    """The three vectorized (pandas) UDFs. Built at call time: importing this file needs no Spark session.

    pair_scores   One call per matched row: field similarities + the best note name, all at once
                  (dq_merge.py made five separate Python calls per row for the same result).
    candidates    One call per ORR row with notes: its name splits, as 'first<TAB>last<TAB>text<TAB>layout'.
    split_scores  One call per (note split, asylum case) pair that survived the DOB + soundex blocking,
                  scored and ranked exactly like _best_note_name (so the best split can be picked in Spark).
    """
    import pandas as pd
    _ship_module_with_udfs()

    def s(v):  # Arrow hands over None / NaN for nulls
        return v if isinstance(v, str) and v else None

    @F.pandas_udf("struct<first:double,last:double,full:double,note_text:string,note_first:string,"
                  "note_last:string,note_layout:string,note_first_score:double,note_last_score:double,"
                  "note_full_score:double>")
    def pair_scores(o_first, a_first, o_last, a_last, o_full, a_full, a_nomid, notes, a_first_w, a_last_w):
        rows = []
        for vals in zip(o_first, a_first, o_last, a_last, o_full, a_full, a_nomid, notes, a_first_w, a_last_w):
            of, af, ol, al, ofu, afu, anm, note, afw, alw = map(s, vals)
            sim = lambda fn, x, y: fn(x, y) if x and y else None
            best = _best_note_name(note, afw, alw, afu, anm) if note and afu else None
            rows.append((sim(_jw, of, af), sim(_jw, ol, al),
                         max(_token_sort(ofu, afu), _token_sort(ofu, anm)) if ofu and afu else None,
                         *(best or (None,) * 7)))
        return pd.DataFrame(rows, columns=["first", "last", "full", "note_text", "note_first", "note_last",
                                           "note_layout", "note_first_score", "note_last_score",
                                           "note_full_score"])

    @F.pandas_udf("array<string>")
    def candidates(notes):
        return pd.Series([[f"{f or ''}\t{l or ''}\t{t}\t{lay}" for t, f, l, lay in _note_name_candidates(s(n))]
                          for n in notes])

    @F.pandas_udf("struct<rank:double,rank2:double,first:double,last:double,full:double>")
    def split_scores(first, last, text, layout, a_first, a_last, a_full, a_nomid):
        rows = []
        for f, l, t, lay, af, al, afu, anm in zip(first, last, text, layout, a_first, a_last, a_full, a_nomid):
            af, al, afu, anm = s(af), s(al), s(afu), s(anm)
            if lay == "SINGLE":  # One word, compared against both first and last, ranked at half
                fs, ls, full = _jw(t, af), _jw(t, al), 0.0
                rank = (max(fs, ls) / 2,) * 2
            else:
                fs, ls = _jw(f, af), _jw(l, al)
                full = max(_token_sort(f"{f} {l}", afu), _token_sort(f"{f} {l}", anm))
                rank = (max((fs + ls) / 2, full), (fs + ls) / 2)
            rows.append((*rank, fs, ls, full))
        return pd.DataFrame(rows, columns=["rank", "rank2", "first", "last", "full"])

    return pair_scores, candidates, split_scores


# =============================================================================
# 3. ORR SINGLE-TABLE ISSUES -> dq_row_issues
# =============================================================================

def _orr_row_issues(o, cols):
    """One row per ORR row id with its issues, e.g. ['HIGH:MISSING_DOB', 'CRITICAL:DUPLICATE_KEY']."""
    c = lambda f: F.col("o__" + f)
    raw = lambda f: _col(cols[f])
    person = F.when(c("full_tok").isNotNull() & c("dob").isNotNull(),
                    F.concat_ws("|", F.array_join(F.array_sort(c("full_tok")), " "), c("dob")))
    df = o.withColumn("_person", person)
    key_cnt = df.where(c("key").isNotNull()).groupBy("o__key").agg(F.count("*").alias("_key_cnt"))
    person_keys = df.where(F.col("_person").isNotNull()).groupBy("_person").agg(
        F.countDistinct("o__key").alias("_person_keys"))
    df = df.join(key_cnt, "o__key", "left").join(person_keys, "_person", "left")

    raw_key = F.regexp_replace(_clean(raw("key")), r"[\s\-]", "")
    names = ("first", "last")
    checks = [
        ("NULL_KEY", "CRITICAL", c("key").isNull()),
        ("DUPLICATE_KEY", "CRITICAL", F.col("_key_cnt") > 1),
        ("KEY_FORMAT", "MEDIUM", raw_key.isNotNull() & ~raw_key.rlike(KEY_PATTERN)),
        ("MISSING_FIRST_NAME", "HIGH", c("first").isNull()),
        ("MISSING_LAST_NAME", "HIGH", c("last").isNull()),
        ("NAME_HAS_DIGITS", "MEDIUM", functools.reduce(or_, [raw(f).cast("string").rlike("[0-9]") for f in names])),
        ("PLACEHOLDER_NAME", "MEDIUM", functools.reduce(or_, [c(f).isin(PLACEHOLDER_NAMES) for f in names])),
        ("MISSING_DOB", "HIGH", _clean(raw("dob")).isNull()),
        ("UNPARSEABLE_DOB", "HIGH", _clean(raw("dob")).isNotNull() & c("dob").isNull()),
        ("FUTURE_DOB", "HIGH", c("dob") > F.current_date()),
        ("IMPLAUSIBLE_DOB", "MEDIUM", c("dob") < F.lit(MIN_DOB).cast("date")),
        ("MISSING_SEX", "LOW", _clean(raw("sex")).isNull()),
        ("INVALID_SEX", "MEDIUM", _clean(raw("sex")).isNotNull() & c("sex").isNull()),
        ("MISSING_COB", "LOW", c("cob").isNull()),
        ("SAME_PERSON_MULTIPLE_KEYS", "HIGH", F.col("_person_keys") > 1),
    ]
    if "notes" in cols:
        checks.append(("NOTES_DOB_DIFFERS", "MEDIUM", (F.size(c("note_label_dobs")) > 0)
                       & (c("dob").isNull() | ~F.array_contains(c("note_label_dobs"), c("dob")))))
    fired = F.filter(F.array(*[F.when(cond, F.lit(f"{sev}:{name}")) for name, sev, cond in checks]),
                     lambda x: x.isNotNull())
    return (df.select("o__row_id", F.explode(fired).alias("issue"))
              .groupBy("o__row_id").agg(F.array_sort(F.collect_set("issue")).alias("dq_row_issues")))


# =============================================================================
# 4. MATCHED PAIRS: STATUSES, SCORE, dq_status
# =============================================================================

def _cmp_name(o, a, sim, variants=()):
    affix = lambda x, y: x.startswith(y) | x.endswith(y)
    expr = F.when(o.isNull() | a.isNull(), "MISSING").when(o == a, "EXACT")
    for expected, status in variants:
        expr = expr.when(o == expected, status)
    return (expr.when((F.least(F.length(o), F.length(a)) >= 3) & (affix(o, a) | affix(a, o)), "PARTIAL")
                .when(sim >= NAME_SIM_MIN, "FUZZY").otherwise("MISMATCH"))


def _cmp_dob(o, a):
    return (F.when(o.isNull() | a.isNull(), "MISSING").when(o == a, "EXACT")
             .when((F.year(o) == F.year(a)) & (F.month(o) == F.dayofmonth(a)) & (F.dayofmonth(o) == F.month(a)),
                   "DAY_MONTH_SWAP")
             .when(F.abs(F.datediff(o, a)) == 1, "OFF_BY_ONE_DAY")
             .when((F.month(o) == F.month(a)) & (F.dayofmonth(o) == F.dayofmonth(a)), "YEAR_DIFF")
             .otherwise("MISMATCH"))


def _cmp_exact(o, a):
    return F.when(o.isNull() | a.isNull(), "MISSING").when(o == a, "EXACT").otherwise("MISMATCH")


def _cmp_full(o, a_full, a_nomid, a_init, sim):
    joined = lambda x: F.array_join(x, "")
    same_words = lambda x, y: F.array_sort(x) == F.array_sort(y)
    subset = lambda x, y: F.size(F.array_except(x, y)) == 0
    return (F.when(o.isNull() | a_full.isNull(), "MISSING")
             .when(joined(o) == joined(a_full), "EXACT")
             .when(joined(o) == joined(a_nomid), "MIDDLE_OMITTED")
             .when(joined(o) == joined(a_init), "MIDDLE_INITIAL")
             .when(same_words(o, a_full) | same_words(o, a_nomid) | same_words(o, a_init), "REORDERED")
             .when((F.size(o) >= 2) & (subset(o, a_full) | subset(a_nomid, o)), "SUBSET")
             .when(sim >= FULL_SIM_MIN, "FUZZY")
             .otherwise("MISMATCH"))


def _score_matches(j, matched, pair_scores, has_notes):
    """Per-field statuses, notes evidence, weighted score and dq_status for each joined row."""
    o = lambda f: F.col("o__" + f)
    a = lambda f: F.col("a__" + f)
    words = lambda c: F.array_join(c, " ")

    # The one Python pass: field similarities + best note name. Unmatched rows return at once.
    notes = o("notes") if has_notes else F.lit(None).cast("string")
    j = j.withColumn("_ps", F.when(matched, pair_scores(
        o("first"), a("first"), o("last"), a("last"), words(o("full_tok")), words(a("full_tok")),
        words(a("nomid_tok")), notes, words(a("first_tok")), words(a("last_tok")))))
    ps = F.col("_ps")
    j = j.withColumns({SIM_COLS[f]: ps[f] for f in SIM_COLS})

    variants = {"first": [], "last": []}
    if "middle" in ASY_COLS:  # The ORR middle name usually sits inside its first or last name
        mid, ini = a("middle"), F.substring(a("middle"), 1, 1)
        variants = {"first": [(F.concat(a("first"), mid), "MIDDLE_IN_FIRST"),
                              (F.concat(a("first"), ini), "MIDDLE_INITIAL_IN_FIRST")],
                    "last": [(F.concat(mid, a("last")), "MIDDLE_IN_LAST"),
                             (F.concat(ini, a("last")), "MIDDLE_INITIAL_IN_LAST")]}
    swapped = (o("first") == a("last")) & (o("last") == a("first")) & (o("first") != o("last"))
    st = {f: F.when(swapped, "SWAPPED").otherwise(_cmp_name(o(f), a(f), ps[f], variants[f])) for f in ("first", "last")}
    st.update(dob=_cmp_dob(o("dob"), a("dob")), sex=_cmp_exact(o("sex"), a("sex")), cob=_cmp_exact(o("cob"), a("cob")),
              full=_cmp_full(o("full_tok"), a("full_tok"), a("nomid_tok"), a("init_tok"), ps["full"]))
    j = j.withColumns({STATUS[f]: F.when(matched, e) for f, e in st.items()})
    s = {f: F.col(STATUS[f]) for f in st}

    names_ok = (s["first"].isin(EQUIVALENT) & s["last"].isin(EQUIVALENT)) | s["full"].isin(EQUIVALENT)
    # Notes evidence: a field that disagrees but whose asylum value is in the notes -> NOTES_MATCH.
    # Scoring uses these effective statuses; VERIFIED still needs the ORR fields themselves to agree.
    eff = dict(s)
    if has_notes:
        full_hit = ps["note_full_score"] >= FULL_SIM_MIN
        hits = {"dob": F.exists(o("note_dobs"), lambda d: d == a("dob")),
                "first": (ps["note_first_score"] >= NAME_SIM_MIN) | full_hit,
                "last": (ps["note_last_score"] >= NAME_SIM_MIN) | full_hit}
        for f, hit in hits.items():
            unresolved = ~s[f].isin(EQUIVALENT) & ~(names_ok if f in ("first", "last") else F.lit(False))
            eff[f] = F.when(unresolved & F.coalesce(hit, F.lit(False)), "NOTES_MATCH").otherwise(s[f])

    credit = lambda x: _lookup(x, CREDIT).cast("double")
    others = ["dob", "sex", "cob"]
    w_name = WEIGHTS["first"] + WEIGHTS["last"]
    field_credit = (WEIGHTS["first"] * credit(eff["first"]) + WEIGHTS["last"] * credit(eff["last"])) / w_name
    score = (w_name * F.greatest(field_credit, credit(s["full"]))
             + functools.reduce(add, [WEIGHTS[f] * credit(eff[f]) for f in others]))
    full_close = s["full"].isin(*EQUIVALENT, "REORDERED", "SUBSET", "FUZZY")
    all_ok = names_ok & functools.reduce(and_, [s[f].isin(EQUIVALENT) for f in others])
    hard_conflict = ((eff["dob"] == "MISMATCH")
                     | (~full_close & (((eff["first"] == "MISMATCH") & (eff["last"] == "MISMATCH"))
                                       | ((eff["last"] == "MISMATCH") & (eff["dob"] != "EXACT")))))
    present = lambda cols: F.filter(F.array(*cols), lambda x: x.isNotNull())
    nonexact = ([F.when(~names_ok & ~s[f].isin(EQUIVALENT), F.lit(f)) for f in ("first", "last")]
                + [F.when(~s[f].isin(EQUIVALENT), F.lit(f)) for f in others])
    by_notes = [F.when(eff[f] == "NOTES_MATCH", F.lit(f)) for f in ("first", "last", "dob")]

    # Each CONFLICT rule kept separately, so dq_all_notes can say exactly which one fired
    j = j.withColumns({
        "_why_dob": F.coalesce(eff["dob"] == "MISMATCH", F.lit(False)),
        "_why_names": F.coalesce(~full_close & (eff["first"] == "MISMATCH") & (eff["last"] == "MISMATCH"), F.lit(False)),
        "_why_last": F.coalesce(~full_close & (eff["last"] == "MISMATCH") & (eff["dob"] != "EXACT"), F.lit(False)),
        "_why_score": F.coalesce(score < REVIEW_THRESHOLD, F.lit(False)),
        "_eff_dob": eff["dob"],
    })
    return j.withColumns({
        "dq_match_score": F.when(matched, F.round(score, 3)),
        "dq_status": (F.when(~matched, "NO_MATCH").when(all_ok, "VERIFIED")
                       .when(hard_conflict | (score < REVIEW_THRESHOLD), "CONFLICT").otherwise("NEEDS_REVIEW")),
        "dq_nonexact_fields": F.when(matched, present(nonexact)),
        "dq_resolved_by_notes": F.when(matched, present(by_notes)),
        "dq_note_best_name": F.when(ps["note_text"].isNotNull(), F.format_string("%s -> first=%s last=%s [%s] %.0f/%.0f/%.0f",
                                             ps["note_text"], ps["note_first"], ps["note_last"], ps["note_layout"],
                                             ps["note_first_score"], ps["note_last_score"], ps["note_full_score"])),
    })


# =============================================================================
# 5. THE SAME PERSON UNDER ANOTHER a_id
# =============================================================================

def _other_keys(o, a, carry, candidates, split_scores, has_notes):
    """Asylum cases (other than the row's own a_id) that look like the same person, from three searches:

        fields      ORR name + ORR DOB (exact, in Spark)
        notes_dob   ORR name + a date from the notes (exact, in Spark)
        notes_name  a name from the notes + the ORR DOB or a note date, scored with rapidfuzz exactly
                    like dq_merge.py: every split of every name version is scored, the best split per
                    (ORR row, asylum case) is kept, and it must reach NAME_SIM_MIN on first and last
                    (or FULL_SIM_MIN on the full name). To keep this fast, a pair is only scored when
                    the DOB matches and the first OR last name sounds alike (soundex).

    `carry` = the asylum columns returned with each hit: [a_id, receipt_number, case_id], as stored.
    Returns one row per ORR row id: the possible cases (three aligned arrays), the sources, the best
    case (strongest source, then lowest a_id), whether there was more than one, and the evidence per hit.
    Also returns the note name versions per ORR row (for dq_all_notes), or None without notes.
    """
    asy_key_raw, asy_receipt, asy_case = carry
    keys = lambda tok: [F.array_join(tok, ""), F.array_join(F.array_sort(tok), "")]
    other_key = ~F.col("a__key").eqNullSafe(F.col("o__key"))
    a_ok = a.where(F.col("a__key").isNotNull() & F.col("a__full_tok").isNotNull() & F.col("a__dob").isNotNull())
    a_keys = (a_ok.select("a__key", "a__dob", *carry, F.explode(F.array_distinct(F.array(
                  *[k for t in ("full_tok", "nomid_tok", "init_tok") for k in keys(F.col("a__" + t))])))
                  .alias("a__name_key")).distinct())

    with_name = o.where(F.col("o__full_tok").isNotNull())
    searches = [with_name.where(F.col("o__dob").isNotNull())
                         .select("o__row_id", "o__key", "o__full_tok", F.col("o__dob").alias("cand_dob"),
                                 F.lit("fields").alias("source"))]
    if has_notes:
        searches.append(with_name.select("o__row_id", "o__key", "o__full_tok",
                                         F.explode("o__note_dobs").alias("cand_dob"), F.lit("notes_dob").alias("source")))
    by_name = (functools.reduce(lambda x, y: x.unionByName(y), searches)
               .select("*", F.explode(F.array_distinct(F.array(*keys(F.col("o__full_tok"))))).alias("o__name_key")))
    evidence = F.concat(F.when(F.col("source") == "fields", F.lit("same name + DOB "))
                         .otherwise(F.lit("ORR name + date from the notes ")), F.col("a__dob").cast("string"))
    hits = [by_name.join(a_keys, (F.col("o__name_key") == F.col("a__name_key"))
                                 & (F.col("cand_dob") == F.col("a__dob")) & other_key)
                   .select("o__row_id", "a__key", *carry, "source", evidence.alias("evidence"))]
    note_names = None

    if has_notes:
        cand_dobs = F.array_distinct(F.filter(F.concat(F.array(F.col("o__dob")), F.col("o__note_dobs")),
                                              lambda d: d.isNotNull()))
        part = lambda i: F.split("_c", "\t")[i]
        # Every name split of every note, once per ORR row (cached: also used for dq_all_notes)
        splits = _cache(o.where(F.col("o__notes").isNotNull())
                         .select("o__row_id", "o__key", "o__dob", "o__note_dobs",
                                 F.explode(candidates("o__notes")).alias("_c"))
                         .select("o__row_id", "o__key", "o__dob", "o__note_dobs",
                                 part(0).alias("n_first"), part(1).alias("n_last"),
                                 part(2).alias("n_text"), part(3).alias("n_layout")))
        note_names = splits.groupBy("o__row_id").agg(
            F.array_sort(F.collect_set("n_text")).alias("_note_names"))

        # Block: DOB + soundex of the first name OR the last name (a single word blocks on both)
        sx = lambda c: F.soundex(F.regexp_replace(c, " ", ""))
        blocked = (splits.select("*", F.explode(cand_dobs).alias("cand_dob"))
                         .withColumn("_sx_first", F.when(F.col("n_layout") == "SINGLE", sx("n_text"))
                                                   .otherwise(sx("n_first")))
                         .withColumn("_sx_last", F.when(F.col("n_layout") == "SINGLE", sx("n_text"))
                                                  .otherwise(sx("n_last"))))
        a_sx = a_ok.withColumn("_asx_first", F.soundex("a__first")).withColumn("_asx_last", F.soundex("a__last"))
        on = lambda which: ((F.col("cand_dob") == F.col("a__dob")) & other_key
                            & (F.col(f"_sx_{which}") == F.col(f"_asx_{which}")))
        split_cols = ["o__row_id", "a__key", "a__dob", "n_first", "n_last", "n_text", "n_layout"]
        a_words = ["a__first_tok", "a__last_tok", "a__full_tok", "a__nomid_tok"]
        pairs = (blocked.join(a_sx, on("first")).select(*split_cols, *carry, *a_words)
                 .unionByName(blocked.join(a_sx, on("last")).select(*split_cols, *carry, *a_words))
                 .dropDuplicates(["o__row_id", "a__key", "n_text", "n_first", "n_last", "n_layout"]))
        words = lambda c: F.array_join(F.col(c), " ")
        scored = pairs.withColumn("_sc", split_scores("n_first", "n_last", "n_text", "n_layout",
                                                      *[words(c) for c in a_words]))
        # Best split per (ORR row, asylum case), ranked like _best_note_name, then the thresholds
        best = (scored.groupBy("o__row_id", "a__key", "a__dob", *carry)
                      .agg(F.max(F.struct(F.col("_sc.rank").alias("rank"), F.col("_sc.rank2").alias("rank2"),
                                          F.col("_sc.first").alias("f"), F.col("_sc.last").alias("l"),
                                          F.col("_sc.full").alias("full"), "n_text", "n_first", "n_last",
                                          "n_layout")).alias("b")))
        b = lambda k: F.col("b." + k)
        full_hit = b("full") >= FULL_SIM_MIN
        read_as = F.when(b("n_layout") == "SINGLE", F.lit("(one word)")).otherwise(
            F.format_string("read as first=%s last=%s [%s]", b("n_first"), b("n_last"), b("n_layout")))
        note_ev = F.format_string("note name '%s' %s + DOB %s (scores %.0f/%.0f/%.0f)", b("n_text"), read_as,
                                  F.col("a__dob").cast("string"), b("f"), b("l"), b("full"))
        hits.append(best.where(((b("f") >= NAME_SIM_MIN) | full_hit) & ((b("l") >= NAME_SIM_MIN) | full_hit))
                        .select("o__row_id", "a__key", *carry, F.lit("notes_name").alias("source"),
                                note_ev.alias("evidence")))

    rank = _lookup(F.col("source"), {k: str(v) for k, v in SOURCE_RANK.items()}).cast("int")
    case = lambda: [F.col(asy_key_raw).cast("string").alias("a_id"), F.col(asy_receipt).cast("string").alias("receipt"),
                    F.col(asy_case).cast("string").alias("case_id")]
    # One struct per possible case, sorted by a_id; a case found by several searches appears once
    cases = F.array_sort(F.collect_set(F.struct(*case())))
    field = lambda name: F.transform(cases, lambda c: c[name])
    return ((functools.reduce(lambda x, y: x.unionByName(y), hits)
            .groupBy("o__row_id")
            .agg(field("a_id").alias("dq_possible_asylum_keys"),
                 field("receipt").alias("dq_possible_receipt_numbers"),
                 field("case_id").alias("dq_possible_case_ids"),
                 F.array_sort(F.collect_set("source")).alias("dq_possible_key_source"),
                 F.min(F.struct(rank.alias("rank"), F.col("a__key"), *case())).alias("_best_other"),
                 (F.countDistinct("a__key") > 1).alias("dq_multiple_possible_keys"),
                 F.array_sort(F.collect_set(F.concat(F.col(asy_key_raw).cast("string"), F.lit(" via "),
                                                     F.col("source"), F.lit(": "), F.col("evidence"))))
                  .alias("_key_evidence")))), note_names


# =============================================================================
# 6b. dq_all_notes: EVERYTHING THAT HAPPENED TO A ROW, IN WORDS
# =============================================================================

def _all_notes(matched, has_notes, carry):
    """One readable string per row, sections separated by ' | ' (split on ' \\| ' to get a list):

        STATUS    the status and exactly why (for CONFLICT: every rule that fired)
        KEY       which asylum case the a_id matched, and if the a_id formats differed
        FIRST / LAST / FULL / DOB / SEX / COB   each comparison with both values and the similarity
        NOTES     dates read, whether they back the asylum DOB, the best name version and what it backs,
                  and every name version found in the note
        CHOSEN / ALSO MATCHES / EVIDENCE / MULTIPLE   other asylum a_ids for this person and how found
        ISSUES    the row's own data problems
        SCORE     the weighted match score
    """
    o = lambda f: F.col("o__" + f)
    a = lambda f: F.col("a__" + f)
    v = lambda c: F.coalesce(c.cast("string"), F.lit("(blank)"))           # Never let a null hide a section
    raw_o = lambda f: v(_col(ORR_COLS[f]))
    raw_a = lambda f: v(_col(ASY_PREFIX + ASY_COLS[f]))
    st = lambda f: F.col(STATUS[f])
    sim = lambda f: F.coalesce(F.format_number(F.col(SIM_COLS[f]), 0), F.lit("n/a"))
    join = lambda sep, *parts: F.concat_ws(sep, *parts)
    status = F.col("dq_status")
    ps = F.col("_ps")
    has_note = o("notes").isNotNull() if has_notes else F.lit(False)
    score_txt = F.format_number(F.col("dq_match_score"), 3)
    listed = lambda arr: F.array_join(arr, ", ")

    # --- STATUS with the reason
    layouts = F.filter(F.array(*[F.when(st(f).isin(EQUIVALENT) & (st(f) != "EXACT"),
                                        F.concat(F.lit(f + "="), st(f))) for f in ("first", "last", "full")]),
                       lambda x: x.isNotNull())
    reasons = join("; ",
        F.when(F.col("_why_dob"), F.concat(F.lit("DOB disagrees outright (ORR "), v(o("dob")), F.lit(" vs asylum "),
                                           v(a("dob")), F.lit(")"),
                                           F.lit(" and the notes do not contain the asylum DOB" if has_notes else ""))),
        F.when(F.col("_why_names"), F.concat(F.lit("first and last name both disagree and the full names are not close "
                                                   "(full similarity "), sim("full"), F.lit(")"))),
        F.when(F.col("_why_last") & ~F.col("_why_names"),
               F.concat(F.lit("last name disagrees and the DOB is not an exact match (DOB "), st("dob"), F.lit(")"))),
        F.when(F.col("_why_score"), F.concat(F.lit("match score "), score_txt,
                                             F.lit(f" is below the review threshold {REVIEW_THRESHOLD:.2f}"))))
    n_keys = F.size(F.col("dq_possible_asylum_keys"))
    status_txt = (
        F.when(status == "VERIFIED", F.concat(F.lit("STATUS VERIFIED: all fields agree"),
               F.when(F.size(layouts) > 0, F.concat(F.lit(" (layout differences only: "), listed(layouts), F.lit(")")))
                .otherwise(F.lit(""))))
         .when(status == "NEEDS_REVIEW", F.concat(F.lit("STATUS NEEDS_REVIEW: differs in "),
               listed(F.col("dq_nonexact_fields")),
               F.when(F.size("dq_resolved_by_notes") > 0,
                      F.concat(F.lit("; resolved by the notes: "), listed(F.col("dq_resolved_by_notes"))))
                .otherwise(F.lit("")),
               F.lit("; minor differences only, score "), score_txt))
         .when(status == "CONFLICT", F.concat(F.lit("STATUS CONFLICT: "), reasons))
         .when(status == "POSSIBLE_KEY_ERROR", F.concat(F.lit("STATUS POSSIBLE_KEY_ERROR: a_id "), raw_o("key"),
               F.lit(" is not in the asylum table, but "), n_keys.cast("string"),
               F.when(n_keys > 1, F.lit(" possible cases were found")).otherwise(F.lit(" possible case was found"))))
         .otherwise(F.concat(F.lit("STATUS NO_MATCH: a_id "), raw_o("key"),
                             F.lit(" is not in the asylum table and no case was found by name + DOB"
                                   + (", note dates or note names" if has_notes else "")))))

    # --- KEY and the field comparisons (matched rows only)
    key_txt = F.when(matched, F.concat(
        F.lit("KEY ORR a_id "), raw_o("key"), F.lit(" matched asylum case "), v(_col(carry[0])),
        F.lit(" (receipt "), v(_col(carry[1])), F.lit(", case "), v(_col(carry[2])), F.lit(")"),
        F.when(_col(ORR_COLS["key"]).cast("string") != _col(carry[0]).cast("string"),
               F.concat(F.lit("; formats differ, matched on the standardized a_id "), o("key"))).otherwise(F.lit(""))))
    cmp_name = lambda f, label: F.when(matched, F.concat(
        F.lit(label + " "), st(f), F.lit(": ORR '"), raw_o(f), F.lit("' vs asylum '"), raw_a(f),
        F.lit("' (similarity "), sim(f), F.lit(")")))
    full_txt = F.when(matched, F.concat(
        F.lit("FULL "), st("full"), F.lit(": ORR '"), join(" ", _col(ORR_COLS["first"]), _col(ORR_COLS["last"])),
        F.lit("' vs asylum '"), join(" ", *[_col(ASY_PREFIX + ASY_COLS[f]) for f in ("first", "middle", "last")
                                           if f in ASY_COLS]),
        F.lit("' (similarity "), sim("full"), F.lit(")")))
    cmp_val = lambda f, label, ov, av: F.when(matched, F.when(st(f) == "EXACT", F.concat(F.lit(label + " EXACT: "), ov))
        .otherwise(F.concat(F.lit(label + " "), st(f), F.lit(": ORR "), ov, F.lit(" vs asylum "), av)))

    # --- NOTES
    notes_txt = F.lit(None).cast("string")
    if has_notes:
        dates = F.when(F.size(o("note_dobs")) > 0,
                       F.concat(F.lit("dates read: "), listed(F.transform(o("note_dobs"), lambda d: d.cast("string")))))\
                 .otherwise(F.lit("no dates in the notes"))
        dob_txt = F.when(matched & (F.col("_eff_dob") == "NOTES_MATCH"),
                         F.concat(F.lit("DOB backed: the notes contain the asylum DOB "), v(a("dob"))))\
                   .when(matched & (st("dob") != "EXACT") & (F.size(o("note_dobs")) > 0),
                         F.concat(F.lit("no date in the notes equals the asylum DOB "), v(a("dob"))))
        full_hit = ps["note_full_score"] >= FULL_SIM_MIN
        backs = F.filter(F.array(F.when((ps["note_first_score"] >= NAME_SIM_MIN) | full_hit, F.lit("first")),
                                 F.when((ps["note_last_score"] >= NAME_SIM_MIN) | full_hit, F.lit("last"))),
                         lambda x: x.isNotNull())
        read_as = F.when(ps["note_layout"] == "SINGLE", F.lit("' (one word), scores ")).otherwise(F.concat(
            F.lit("' read as first="), v(ps["note_first"]), F.lit(" last="), v(ps["note_last"]),
            F.lit(" ["), ps["note_layout"], F.lit("], scores ")))
        best_txt = F.when(matched & ps["note_text"].isNotNull(), F.concat(
            F.lit("best name version '"), ps["note_text"], read_as,
            F.format_string("%.0f/%.0f/%.0f", ps["note_first_score"], ps["note_last_score"], ps["note_full_score"]),
            F.when(F.size(backs) > 0, F.concat(F.lit(" -> backs "), listed(backs)))
             .otherwise(F.lit(f" -> backs nothing (needs {NAME_SIM_MIN} per name or {FULL_SIM_MIN} full)"))))
        names_txt = F.when(F.size(F.col("_note_names")) > 0,
                           F.concat(F.lit("name versions found: "), F.array_join(F.col("_note_names"), "; ")))\
                     .otherwise(F.lit("no name versions found"))
        notes_txt = F.when(has_note, F.concat(F.lit("NOTES "), join("; ", dates, dob_txt, best_txt, names_txt)))

    # --- Other asylum a_ids for this person
    best = F.col("_best_other")
    best_source = _lookup(best["rank"].cast("string"), {str(r): k for k, r in SOURCE_RANK.items()})
    has_other = F.col("dq_possible_asylum_keys").isNotNull()
    case_txt = F.concat(best["a_id"], F.lit(" (receipt "), v(best["receipt"]), F.lit(", case "), v(best["case_id"]),
                        F.lit("), source "), best_source)
    other_txt = F.when(has_other & (status == "POSSIBLE_KEY_ERROR"), F.concat(F.lit("CHOSEN "), case_txt))\
                 .when(has_other, F.concat(F.lit("ALSO MATCHES other asylum case(s) "),
                                           listed(F.col("dq_possible_asylum_keys")),
                                           F.lit(" (check for a duplicate case)")))
    evidence_txt = F.when(has_other, F.concat(F.lit("EVIDENCE "), F.array_join(F.col("_key_evidence"), "; ")))
    multi_txt = F.when(F.col("dq_multiple_possible_keys"), F.concat(
        F.lit("MULTIPLE "), n_keys.cast("string"),
        F.lit(" possible cases; chosen by source (fields > notes_dob > notes_name), then lowest a_id")))
    issues_txt = F.when(F.size("dq_row_issues") > 0, F.concat(F.lit("ISSUES "), listed(F.col("dq_row_issues"))))
    score_line = F.when(matched, F.concat(F.lit("SCORE "), score_txt,
                                          F.lit(f" (CONFLICT below {REVIEW_THRESHOLD:.2f})")))

    return join(" | ", status_txt, key_txt, cmp_name("first", "FIRST"), cmp_name("last", "LAST"), full_txt,
                cmp_val("dob", "DOB", v(o("dob")), v(a("dob"))), cmp_val("sex", "SEX", v(o("sex")), v(a("sex"))),
                cmp_val("cob", "COB", v(o("cob")), v(a("cob"))), notes_txt, other_txt, evidence_txt, multi_txt,
                issues_txt, score_line)


# =============================================================================
# 6. PASS / FAIL
# =============================================================================

def _passed(o, a, merged):
    """Three checks: the join kept every ORR row, no asylum a_id appears twice (that would duplicate
    ORR rows), and the conflict rate is within MAX_CONFLICT_RATE (only judged from MIN_MATCHES_FOR_RATE
    matches up). One aggregation over merged, which also materializes the cached result."""
    m = merged.agg(F.count("*").alias("rows"),
                   F.sum(_col(MATCH_FLAG_COL).cast("int")).alias("matched"),
                   F.sum((F.col("dq_status") == "CONFLICT").cast("int")).alias("conflicts")).first()
    rows_in = o.count()
    dup_asylum_keys = a.where(F.col("a__key").isNotNull()).groupBy("a__key").count().where("count > 1").count()
    matched, conflicts = m["matched"] or 0, m["conflicts"] or 0
    rate = conflicts / matched if matched else 0.0
    checks = {
        "ROW_COUNT_PRESERVED": m["rows"] == rows_in,
        "ASYLUM_KEYS_UNIQUE": dup_asylum_keys == 0,
        "CONFLICT_RATE": rate <= MAX_CONFLICT_RATE or matched < MIN_MATCHES_FOR_RATE,
        "detail": {"orr_rows": rows_in, "merged_rows": m["rows"], "duplicate_asylum_keys": dup_asylum_keys,
                   "matched": matched, "conflicts": conflicts, "conflict_rate": round(rate, 4)},
    }
    return all(v for k, v in checks.items() if k != "detail"), checks


# =============================================================================
# ENTRY POINT
# =============================================================================

def _cache(df):
    if not CACHE:
        return df
    try:
        return df.persist()
    except Exception:
        return df


def run_dq_merge(orr_df: DataFrame, asylum_df: DataFrame) -> dict:
    """Build the merged table. Returns {"merged": DataFrame, "passed": bool, "checks": dict}."""
    try:
        import rapidfuzz  # noqa: F401
    except ImportError as err:
        raise ImportError("rapidfuzz is required: run %pip install rapidfuzz") from err
    for name, df, cols in [("ORR", orr_df, ORR_COLS), ("asylum", asylum_df, ASY_COLS)]:
        missing = [c for c in cols.values() if c not in df.columns]
        if missing:
            raise ValueError(f"{name} data is missing required columns: {missing}")
    clash = [c for c in orr_df.columns if c in (MATCH_FLAG_COL, "matched_a_number", "matched_receipt_number",
                                             "matched_case_id", "_ps", "_person", "_best_other",
                                             "_key_evidence", "_note_names")
             or c.startswith(("dq_", ASY_PREFIX, "o__", "a__"))]
    if clash:
        raise ValueError(f"ORR columns collide with generated columns: {clash}")

    has_notes = NOTES_COL in orr_df.columns
    orr_cols = {**ORR_COLS, "notes": NOTES_COL} if has_notes else ORR_COLS
    asy_cols = {f: ASY_PREFIX + c for f, c in ASY_COLS.items()}
    asy_key_raw = asy_cols["key"]  # The asylum a_id exactly as stored, e.g. asy_a_id
    carry = [asy_key_raw, asy_cols["receipt"], asy_cols["case_id"]]
    pair_scores, candidates, split_scores = _udfs()

    # 1. Normalize both tables (asylum columns prefixed first so names never collide)
    o = _cache(normalize(orr_df, orr_cols, "o__"))
    a = _cache(normalize(asylum_df.select([_col(c).alias(ASY_PREFIX + c) for c in asylum_df.columns]),
                         asy_cols, "a__"))

    # 3-4. LEFT join on the canonical a_id, then score the matched pairs
    j = o.join(a.where(F.col("a__key").isNotNull()), F.col("o__key") == F.col("a__key"), "left")
    matched = F.col("a__key").isNotNull()
    j = _score_matches(j.withColumn(MATCH_FLAG_COL, matched), matched, pair_scores, has_notes)

    # 2 + 5. Attach each row's issues and any other asylum a_ids for the same person
    other_keys, note_names = _other_keys(o, a, carry, candidates, split_scores, has_notes)
    j = j.join(_orr_row_issues(o, orr_cols), "o__row_id", "left").join(other_keys, "o__row_id", "left")
    if note_names is not None:
        j = j.join(note_names, "o__row_id", "left")
    pke = (F.col("dq_status") == "NO_MATCH") & F.col("dq_possible_asylum_keys").isNotNull()
    j = j.withColumn("dq_status", F.when(pke, "POSSIBLE_KEY_ERROR").otherwise(F.col("dq_status")))

    # 6. The key for the full left join later
    # matched_receipt_number / matched_case_id follow the same rules (null for NO_MATCH)
    is_pke = F.col("dq_status") == "POSSIBLE_KEY_ERROR"
    best = F.col("_best_other")
    j = j.withColumns({
        "matched_a_number": (F.when(is_pke, best["a_id"]).when(matched, _col(asy_key_raw).cast("string"))
                              .otherwise(_col(ORR_COLS["key"]).cast("string"))),
        "matched_receipt_number": F.when(is_pke, best["receipt"]).when(matched, _col(carry[1]).cast("string")),
        "matched_case_id": F.when(is_pke, best["case_id"]).when(matched, _col(carry[2]).cast("string")),
    })

    j = j.withColumn("dq_all_notes", _all_notes(matched, has_notes, carry))

    merged = _cache(j.select(
        *[_col(c) for c in orr_df.columns], *[_col(ASY_PREFIX + c) for c in asylum_df.columns],
        _col(MATCH_FLAG_COL), "matched_a_number", "matched_receipt_number", "matched_case_id", "dq_status", "dq_all_notes", "dq_match_score", "dq_nonexact_fields",
        "dq_resolved_by_notes", "dq_note_best_name", *STATUS.values(), *SIM_COLS.values(),
        "dq_possible_asylum_keys", "dq_possible_receipt_numbers", "dq_possible_case_ids", "dq_possible_key_source",
        F.coalesce("dq_multiple_possible_keys", F.lit(False)).alias("dq_multiple_possible_keys"),
        "dq_row_issues"))

    # 7. Pass / fail
    passed, checks = _passed(o, a, merged)
    return {"merged": merged, "passed": passed, "checks": checks}


# =============================================================================
# USAGE
# =============================================================================
# %pip install rapidfuzz
# from dq_merge_lean import run_dq_merge
#
# res = run_dq_merge(spark.table("hhs_orr_subset"), spark.table("affirmative_asylum_subset"))
# res["passed"]                         # True / False
# res["checks"]                         # Which check failed, and the counts behind it
# display(res["merged"])
#
# # Later, the full left join on the corrected key:
# full = res["merged"].select(...).join(spark.table("affirmative_asylum_cases"),
#                                       F.col("matched_a_number") == F.col("a_id"), "left")
