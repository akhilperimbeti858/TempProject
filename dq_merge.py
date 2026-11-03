"""
HHS-ORR to affirmative asylum merge, with a data quality gate
=============================================================

WHAT IT DOES
    Left-joins the HHS-ORR data to the affirmative asylum cases on a_id, flags whether each
    ORR record already has an asylum case, and checks that matched records really describe
    the same person (name, DOB, sex, country of birth) before anything is written.

PIPELINE
    1. normalize()        Clean every comparison field into a helper column.
    2. table_checks()     Profile each table on its own (nulls, duplicates, bad values).
    3. join               LEFT join on the canonical a_id. No ORR row is ever dropped.
    4. compare()          Score each matched pair by field and by full name, then set dq_status.
    5. find_other_keys()  Find asylum cases with the same name + DOB under a different a_id.
    6. gate()             Reconcile counts and conflict rate. publish() refuses to write on failure.

ADDITIONAL NOTES (optional ORR free-text column)
    Used as evidence only, never to overwrite a value. If a field disagrees but the asylum value
    appears in the notes (a date in any common format and either day order, or the name words in
    any order), the field counts as NOTES_MATCH. The row can then reach NEEDS_REVIEW but never
    VERIFIED, and dq_resolved_by_notes names the fields. Notes are also searched for names and
    DOBs that match asylum cases under another a_id (dq_possible_key_source).

DQ_STATUS VALUES
    VERIFIED            Name (fields or full name), DOB, sex and country of birth all agree.
    NEEDS_REVIEW        Minor differences only (typo, partial name, swapped fields).
    CONFLICT            Same a_id, but probably a different person.
    NO_MATCH            No affirmative asylum case has this a_id.
    POSSIBLE_KEY_ERROR  No a_id match, but an asylum case has the same name + DOB under another a_id.

REQUIREMENTS
    Spark 3.5+ / Databricks Runtime 13.3+ (try_to_timestamp, withColumns).
"""
import functools  # functools.reduce: a notebook "from pyspark.sql.functions import *" shadows bare reduce
from operator import add, and_, or_

from pyspark.sql import DataFrame, Window, functions as F


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Logical field -> physical column name. Edit if the schemas differ.
ORR_COLS = {"key": "a_id", "first": "first_name", "last": "last_name",
            "dob": "dob", "sex": "sex", "cob": "country_of_birth"}
# Only the asylum table has a middle name. In the ORR data the middle name usually ends up
# in the last-name column ("JAMES SMITH"), sometimes in the first-name column.
ASY_COLS = {**ORR_COLS, "middle": "middle_name"}

ORR, ASY = "hhs_orr", "affirmative_asylum"  # Table labels used in the reports
ASY_PREFIX = "asy_"                          # Prefix for asylum columns in the output
MATCH_FLAG_COL = "asylum_case_exists"        # A name with a space needs Delta column mapping (publish handles it)
CACHE = True                                 # Set False on Databricks serverless, where persist() is unsupported

# Normalization
KEY_PATTERN = r"^A?[0-9]{8,9}$"   # Expected raw a_id shape. Flagged only, never rejected.
DOB_FORMATS = ["yyyy-MM-dd", "M/d/yyyy", "yyyyMMdd", "yyyy-MM-dd HH:mm:ss"]
MIN_DOB = "1900-01-01"
NULL_TOKENS = ["", "NULL", "NONE", "N/A", "NA", "NAN", "?", "-"]
SEX_MAP = {"M": "M", "MALE": "M", "F": "F", "FEMALE": "F", "X": "X", "U": "U", "UNKNOWN": "U"}
COB_MAP = {}                       # e.g. {"GUAT": "GUATEMALA", "HOND": "HONDURAS"}
PLACEHOLDER_NAMES = ["TEST", "UNKNOWN", "FNU", "LNU", "NFN", "NLN", "XXX", "DUMMY"]
NO_MIDDLE_TOKENS = ["NMN", "NOMIDDLENAME"]
ACCENTS, PLAIN = "ÁÀÂÄÃÅÉÈÊËÍÌÎÏÓÒÔÖÕÚÙÛÜÑÇÝ", "AAAAAAEEEEIIIIOOOOOUUUUNCY"

# Free-text notes (ORR only). Used as evidence, never to overwrite a value. Skipped if the column is absent.
NOTES_COL = "additional_notes"
NOTE_DATE_REGEX = r"(\d{4}[/.\-]\d{1,2}[/.\-]\d{1,2}|\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}|\b\d{8}\b)"
NOTE_DATE_FORMATS = ["M/d/yyyy", "d/M/yyyy", "yyyy/M/d", "M/d/yy", "d/M/yy", "yyyyMMdd"]  # Both day orders
NOTE_DOB_LABEL = r"(?i)\b(d\.?o\.?b|birth|born|fecha de nac\w*)\b"
# Filler in the notes that is never a name ("as per nta:", "correct name", "child states"). Removed from the
# note words before name matching so it cannot fuzzy-match a real name (CORRECT vs CORREA). Extend as needed.
# Name particles (DE, LA, DEL) are listed too: they are dropped on both sides, so "DE LA CRUZ" matches on CRUZ.
NOTE_STOPWORDS = [
    # documents / sources
    "NTA", "AS", "PER", "PERSON", "ACCORDING", "TO", "SPONSOR", "DOCS", "DOCUMENTS", "BIRTH", "CERTIFICATE",
    "PASSPORT", "ORR", "UAC", "UC", "CBP", "ICE", "FILE", "RECORD", "RECORDS",
    # corrections
    "CORRECT", "CORRECTED", "CORRECTION", "ACTUAL", "UPDATED", "CHANGED", "WRONG", "INCORRECT", "SHOULD", "BE",
    "IS", "WAS",
    # field labels
    "NAME", "NAMES", "FIRST", "LAST", "MIDDLE", "FULL", "AKA", "ALIAS", "KNOWN", "ALSO", "DOB", "DATE", "BORN",
    "NACIMIENTO", "FECHA", "NOMBRE",
    # narrative and particles
    "CHILD", "MINOR", "STATES", "STATED", "SAYS", "REPORTS", "REPORTED", "THE", "AND", "OR", "OF", "IN", "ON",
    "AT", "BY", "FOR", "WITH", "HIS", "HER", "THEIR", "DE", "LA", "DEL", "LOS", "LAS", "Y",
]

# Matching. DOB and last name carry the most identity, so they weigh the most.
NAME_FUZZY_MAX_DIST = 2            # Edit distance allowed for a typo (1 for names under 5 letters)
WEIGHTS = {"first": 0.20, "last": 0.25, "dob": 0.30, "sex": 0.10, "cob": 0.15}
CREDIT = {"EXACT": 1.0,
          "MIDDLE_IN_LAST": 1.0, "MIDDLE_INITIAL_IN_LAST": 1.0,
          "MIDDLE_IN_FIRST": 1.0, "MIDDLE_INITIAL_IN_FIRST": 1.0,
          "MIDDLE_OMITTED": 1.0, "MIDDLE_INITIAL": 1.0,
          "REORDERED": 0.8, "SUBSET": 0.6, "PARTIAL": 0.6, "FUZZY": 0.6, "SWAPPED": 0.6,
          "NOTES_MATCH": 0.9,  # The ORR field differs, but additional_notes contains the asylum value
          "DAY_MONTH_SWAP": 0.5, "OFF_BY_ONE_DAY": 0.5, "YEAR_DIFF": 0.25,
          "MISSING": 0.0, "MISMATCH": 0.0}
EQUIVALENT = [s for s, c in CREDIT.items() if c == 1.0]  # Same value, different layout
REVIEW_THRESHOLD = 0.70        # Matched rows scoring below this become CONFLICT
MAX_CONFLICT_RATE = 0.05       # Gate blocks the write above 5% conflicts...
MIN_MATCHES_FOR_RATE = 20      # ...but only once there are enough matches for the rate to mean something

STATUS = {"full": "dq_full_name_status", "first": "dq_first_name_status", "last": "dq_last_name_status",
          "dob": "dq_dob_status", "sex": "dq_sex_status", "cob": "dq_cob_status"}
OTHER_KEYS_COL = "dq_possible_asylum_keys"
OTHER_KEYS_SOURCE_COL = "dq_possible_key_source"  # fields / notes_dob / notes_name
NOTES_RESOLVED_COL = "dq_resolved_by_notes"


# =============================================================================
# 2. CLEANING
# =============================================================================

def _col(name):
    """Column by exact name, so names containing dots or spaces work."""
    return F.col(f"`{name}`")


def _lookup(c, mapping):
    """Dictionary lookup as a CASE WHEN chain: returns null for unmapped values instead of
    raising, which map[] does under ANSI mode."""
    expr = F.lit(None).cast("string")
    for k, v in mapping.items():
        expr = F.when(c == k, F.lit(v)).otherwise(expr)
    return expr


def _clean(c):
    """Trim, collapse whitespace, uppercase, and turn null-like tokens ('N/A', '') into nulls."""
    v = F.upper(F.trim(F.regexp_replace(c.cast("string"), r"\s+", " ")))
    return F.when(~v.isin(NULL_TOKENS), v)


def _tokens(*cols):
    """Name columns -> array of words, accents folded: ("Mary Ann", "Smith-Jones") ->
    ['MARY', 'ANN', 'SMITH', 'JONES']. Apostrophes and periods are dropped ("O'Neil" -> ONEIL),
    other separators split words. Comparing words ignores how names were split across columns."""
    s = F.concat_ws(" ", *[F.translate(_clean(c), ACCENTS, PLAIN) for c in cols])
    s = F.regexp_replace(F.regexp_replace(s, r"['.`]", ""), r"[^A-Z]+", " ")
    toks = F.filter(F.split(F.trim(s), " "), lambda x: x != "")
    return F.when(F.size(toks) > 0, toks)


def _compact(c):
    """One name column as letters only: "José-María O'Neil" -> 'JOSEMARIAONEIL'."""
    return F.array_join(_tokens(c), "")


def _key(c, dtype):
    """Canonical a_id: 'A-12345678', '012345678' and 12345678 all become '012345678'.
    Numeric columns go through decimal so a double never renders as '1.2345678E7'.
    Values that are not A + digits are kept as cleaned, so they still fail KEY_FORMAT."""
    if dtype in ("tinyint", "smallint", "int", "bigint", "float", "double") or dtype.startswith("decimal"):
        c = c.cast("decimal(38,0)")
    v = F.regexp_replace(_clean(c), r"[\s\-]", "")
    digits = F.regexp_extract(v, r"^A?([0-9]{1,9})$", 1)
    return F.when(digits != "", F.lpad(digits, 9, "0")).otherwise(v)


def _parse_dob(name, dtype):
    """DOB as DateType whatever the source type. try_to_timestamp returns null on bad input."""
    if dtype == "date":
        return _col(name)
    if dtype.startswith("timestamp"):
        return F.to_date(_col(name))
    return F.coalesce(*[F.to_date(F.expr(f"try_to_timestamp(trim(`{name}`), '{fmt}')"))
                        for fmt in DOB_FORMATS])


def _note_dates(strings):
    """Array of date-like strings -> array of plausible dates. Every format is tried, so an
    ambiguous '01/02/2011' yields both Jan 2 and Feb 1."""
    parsed = F.flatten(F.transform(strings, lambda s: F.array(*[
        F.to_date(F.try_to_timestamp(F.regexp_replace(s, r"[.\-]", "/"), F.lit(fmt)))
        for fmt in NOTE_DATE_FORMATS])))
    plausible = F.filter(parsed, lambda d: d.isNotNull() & (d >= F.lit(MIN_DOB).cast("date"))
                         & (d <= F.current_date()))
    return F.coalesce(F.array_distinct(plausible), F.array().cast("array<date>"))


def _words_in(words, note_tok):
    """True if every word in `words` appears in the note words, in any order.
    Words of 6+ letters may differ by NAME_FUZZY_MAX_DIST, 4-5 letters by 1, shorter must be exact.
    Stop words in the name (DE, LA) are skipped; at least one real name word must remain."""
    def close(w, t):
        dist = F.when(F.length(w) < 4, 0).when(F.length(w) < 6, 1).otherwise(NAME_FUZZY_MAX_DIST)
        return F.levenshtein(w, t) <= dist
    words = F.array_except(words, _stop())
    return (F.size(words) > 0) & F.forall(words, lambda w: F.exists(note_tok, lambda t: close(w, t)))


def _stop():
    return F.array(*[F.lit(w) for w in NOTE_STOPWORDS])


def _note_words(notes):
    """Name-like words in a note: cleaned like names, with stop words and single letters removed."""
    toks = F.filter(F.array_except(_tokens(notes), _stop()), lambda t: F.length(t) > 1)
    return F.coalesce(toks, F.array().cast("array<string>"))


def normalize(df, cols, p):
    """Add a cleaned helper column per field (<p><field>) plus a deterministic row id.

    The original columns stay untouched for the output. The row id is a hash of the whole
    row (plus a counter for exact duplicates), so it stays the same if Spark recomputes.
    """
    dt = dict(df.dtypes)
    raw = lambda f: _col(cols[f])
    cob = _clean(raw("cob"))
    rules = {
        "key":      _key(raw("key"), dt[cols["key"]]),
        "first":    _compact(raw("first")),
        "last":     _compact(raw("last")),
        "dob":      _parse_dob(cols["dob"], dt[cols["dob"]]),
        "sex":      _lookup(_clean(raw("sex")), SEX_MAP),
        "cob":      F.coalesce(_lookup(cob, COB_MAP), cob),
        "full_tok": _tokens(raw("first"), raw("last")),
        "first_tok": _tokens(raw("first")),
        "last_tok": _tokens(raw("last")),
    }
    if "notes" in cols:
        notes = raw("notes").cast("string")
        rules["note_tok"] = _note_words(notes)
        rules["note_dobs"] = _note_dates(F.regexp_extract_all(notes, F.lit(NOTE_DATE_REGEX), 1))
        # Dates right after a DOB label ("dob:", "born", "fecha de nacimiento"), for NOTES_DOB_DIFFERS
        labelled = F.regexp_extract(notes, NOTE_DOB_LABEL + r"[^0-9]{0,20}" + NOTE_DATE_REGEX, 2)
        rules["note_label_dobs"] = _note_dates(F.when(labelled != "", F.array(labelled)))
    # Three versions of the full name, because the ORR side may include, drop or abbreviate the middle
    rules["nomid_tok"] = rules["init_tok"] = rules["full_tok"]
    if "middle" in cols:
        mid = F.when(~_compact(raw("middle")).isin(NO_MIDDLE_TOKENS), raw("middle"))
        rules["middle"] = _compact(mid)
        rules["full_tok"] = _tokens(raw("first"), mid, raw("last"))
        rules["init_tok"] = _tokens(raw("first"), F.substring(_compact(mid), 1, 1), raw("last"))

    h = F.xxhash64(*[_col(c) for c in df.columns])
    row_id = F.concat_ws("-", h, F.row_number().over(Window.partitionBy(h).orderBy(h)))
    return df.select("*", row_id.alias(p + "row_id"), *[e.alias(p + f) for f, e in rules.items()])


# =============================================================================
# 3. SINGLE-TABLE CHECKS
# =============================================================================

def table_checks(df, cols, p, table):
    """Profile one table: one row per issue (table, row_id, a_id_raw, check, severity).

    Duplicate counts use groupBy + join instead of windows, so rows with a null key or
    no name/DOB never pile into one skewed partition.
    """
    c = lambda f: F.col(p + f)
    raw = lambda f: _col(cols[f])
    names = [f for f in ("first", "middle", "last") if f in cols]

    # A person = full-name words in any order + DOB, so "JOHN JAMES / SMITH" and "JOHN / JAMES SMITH" group
    df = df.withColumn("_person", F.when(c("full_tok").isNotNull() & c("dob").isNotNull(), F.concat_ws(
        "|", F.array_join(F.array_sort(c("full_tok")), " "), c("dob"))))
    key_cnt = df.where(c("key").isNotNull()).groupBy(p + "key").agg(F.count("*").alias("_key_cnt"))
    person_keys = (df.where(F.col("_person").isNotNull()).groupBy("_person")
                     .agg(F.countDistinct(p + "key").alias("_person_keys")))
    df = df.join(key_cnt, p + "key", "left").join(person_keys, "_person", "left")

    raw_key = F.regexp_replace(_clean(raw("key")), r"[\s\-]", "")
    checks = [
        ("NULL_KEY",                  "CRITICAL", c("key").isNull()),
        ("DUPLICATE_KEY",             "CRITICAL", F.col("_key_cnt") > 1),
        ("KEY_FORMAT",                "MEDIUM",   raw_key.isNotNull() & ~raw_key.rlike(KEY_PATTERN)),
        ("MISSING_FIRST_NAME",        "HIGH",     c("first").isNull()),
        ("MISSING_LAST_NAME",         "HIGH",     c("last").isNull()),
        ("NAME_HAS_DIGITS",           "MEDIUM",   functools.reduce(or_, [raw(f).cast("string").rlike("[0-9]") for f in names])),
        ("PLACEHOLDER_NAME",          "MEDIUM",   functools.reduce(or_, [c(f).isin(PLACEHOLDER_NAMES) for f in names])),
        ("MISSING_DOB",               "HIGH",     _clean(raw("dob")).isNull()),
        ("UNPARSEABLE_DOB",           "HIGH",     _clean(raw("dob")).isNotNull() & c("dob").isNull()),
        ("FUTURE_DOB",                "HIGH",     c("dob") > F.current_date()),
        ("IMPLAUSIBLE_DOB",           "MEDIUM",   c("dob") < F.lit(MIN_DOB).cast("date")),
        ("MISSING_SEX",               "LOW",      _clean(raw("sex")).isNull()),
        ("INVALID_SEX",               "MEDIUM",   _clean(raw("sex")).isNotNull() & c("sex").isNull()),
        ("MISSING_COB",               "LOW",      c("cob").isNull()),
        ("SAME_PERSON_MULTIPLE_KEYS", "HIGH",     F.col("_person_keys") > 1),
    ]
    if "notes" in cols:  # A labelled DOB in the notes that the dob column does not agree with
        label_dobs = c("note_label_dobs")
        checks.append(("NOTES_DOB_DIFFERS", "MEDIUM", (F.size(label_dobs) > 0)
                       & (c("dob").isNull() | ~F.array_contains(label_dobs, c("dob")))))
    fired = F.filter(F.array(*[F.when(cond, F.struct(F.lit(n).alias("check"), F.lit(s).alias("severity")))
                               for n, s, cond in checks]), lambda x: x.isNotNull())
    return (df.select(F.lit(table).alias("table"), F.col(p + "row_id").alias("row_id"),
                      raw("key").cast("string").alias("a_id_raw"), F.explode(fired).alias("i"))
              .select("table", "row_id", "a_id_raw", "i.check", "i.severity"))


# =============================================================================
# 4. COMPARATORS (each returns a graded status string)
# =============================================================================

def _cmp_name(o, a, variants=()):
    """EXACT, then middle-name layouts, then PARTIAL (one name starts or ends with the other,
    'GARCIA' vs 'GARCIALOPEZ'), then FUZZY (typo, or same soundex AND same first two letters)."""
    lev = F.levenshtein(o, a)
    max_dist = F.when(F.least(F.length(o), F.length(a)) < 5, 1).otherwise(NAME_FUZZY_MAX_DIST)
    affix = lambda x, y: x.startswith(y) | x.endswith(y)
    expr = F.when(o.isNull() | a.isNull(), "MISSING").when(o == a, "EXACT")
    for expected, status in variants:
        expr = expr.when(o == expected, status)
    return (expr
            .when((F.least(F.length(o), F.length(a)) >= 3) & (affix(o, a) | affix(a, o)), "PARTIAL")
            .when((lev <= max_dist)
                  | ((F.soundex(o) == F.soundex(a)) & (F.substring(o, 1, 2) == F.substring(a, 1, 2))
                     & (lev <= NAME_FUZZY_MAX_DIST + 1)), "FUZZY")
            .otherwise("MISMATCH"))


def _cmp_dob(o, a):
    """Common DOB entry errors: day/month transposed, off by one day, wrong year only."""
    return (F.when(o.isNull() | a.isNull(), "MISSING")
             .when(o == a, "EXACT")
             .when((F.year(o) == F.year(a)) & (F.month(o) == F.dayofmonth(a))
                   & (F.dayofmonth(o) == F.month(a)), "DAY_MONTH_SWAP")
             .when(F.abs(F.datediff(o, a)) == 1, "OFF_BY_ONE_DAY")
             .when((F.month(o) == F.month(a)) & (F.dayofmonth(o) == F.dayofmonth(a)), "YEAR_DIFF")
             .otherwise("MISMATCH"))


def _cmp_exact(o, a):
    return F.when(o.isNull() | a.isNull(), "MISSING").when(o == a, "EXACT").otherwise("MISMATCH")


def _cmp_full(o, a_full, a_nomid, a_init):
    """Full names (word arrays), regardless of the column split.
    o = ORR first + last; a_full / a_nomid / a_init = asylum name with middle, without, with initial."""
    joined = lambda x: F.array_join(x, "")
    same_words = lambda x, y: F.array_sort(x) == F.array_sort(y)
    subset = lambda x, y: F.size(F.array_except(x, y)) == 0
    near = lambda x, y: F.levenshtein(joined(x), joined(y)) <= NAME_FUZZY_MAX_DIST
    return (F.when(o.isNull() | a_full.isNull(), "MISSING")
             .when(joined(o) == joined(a_full), "EXACT")
             .when(joined(o) == joined(a_nomid), "MIDDLE_OMITTED")
             .when(joined(o) == joined(a_init), "MIDDLE_INITIAL")
             .when(same_words(o, a_full) | same_words(o, a_nomid) | same_words(o, a_init), "REORDERED")
             .when((F.size(o) >= 2) & (subset(o, a_full) | subset(a_nomid, o)), "SUBSET")
             .when(near(o, a_full) | near(o, a_nomid), "FUZZY")
             .otherwise("MISMATCH"))


# =============================================================================
# 5. PIPELINE STEPS
# =============================================================================

def compare(j, matched):
    """Score matched ORR/asylum pairs and assign dq_status.

    A shared a_id does not prove it is the same person. The weighted score absorbs minor
    noise; hard_conflict overrides it when DOB or last name disagree outright.
    """
    o = lambda f: F.col("o__" + f)
    a = lambda f: F.col("a__" + f)

    variants = {}
    if "middle" in ASY_COLS:  # The ORR middle name usually sits inside its first or last name
        mid, ini = a("middle"), F.substring(a("middle"), 1, 1)
        variants = {"first": [(F.concat(a("first"), mid), "MIDDLE_IN_FIRST"),
                              (F.concat(a("first"), ini), "MIDDLE_INITIAL_IN_FIRST")],
                    "last":  [(F.concat(mid, a("last")), "MIDDLE_IN_LAST"),
                              (F.concat(ini, a("last")), "MIDDLE_INITIAL_IN_LAST")]}

    swapped = (o("first") == a("last")) & (o("last") == a("first")) & (o("first") != o("last"))
    st = {f: F.when(swapped, "SWAPPED").otherwise(_cmp_name(o(f), a(f), variants.get(f, [])))
          for f in ("first", "last")}
    st.update(dob=_cmp_dob(o("dob"), a("dob")), sex=_cmp_exact(o("sex"), a("sex")),
              cob=_cmp_exact(o("cob"), a("cob")),
              full=_cmp_full(o("full_tok"), a("full_tok"), a("nomid_tok"), a("init_tok")))
    j = j.withColumns({STATUS[f]: F.when(matched, e) for f, e in st.items()})

    s = {f: F.col(STATUS[f]) for f in st}
    credit = lambda x: _lookup(x, CREDIT).cast("double")
    others = ["dob", "sex", "cob"]
    names_ok = (s["first"].isin(EQUIVALENT) & s["last"].isin(EQUIVALENT)) | s["full"].isin(EQUIVALENT)

    # Notes as evidence: a field that disagrees but whose asylum value appears in additional_notes
    # counts as NOTES_MATCH for scoring and conflicts. The dq_*_status columns keep the ORR result,
    # and VERIFIED still needs the ORR fields themselves to agree, so these rows end up NEEDS_REVIEW.
    eff = dict(s)
    if "o__note_tok" in j.columns:
        hits = {"dob": F.exists(o("note_dobs"), lambda d: d == a("dob")),
                "first": _words_in(a("first_tok"), o("note_tok")),
                "last": _words_in(a("last_tok"), o("note_tok"))}
        for f, hit in hits.items():
            unresolved = ~s[f].isin(EQUIVALENT) & ~(names_ok if f in ("first", "last") else F.lit(False))
            eff[f] = F.when(unresolved & F.coalesce(hit, F.lit(False)), "NOTES_MATCH").otherwise(s[f])

    # Name credit = the better of the first/last fields and the full name
    w_name = WEIGHTS["first"] + WEIGHTS["last"]
    field_credit = (WEIGHTS["first"] * credit(eff["first"]) + WEIGHTS["last"] * credit(eff["last"])) / w_name
    score = (w_name * F.greatest(field_credit, credit(s["full"]))
             + functools.reduce(add, [WEIGHTS[f] * credit(eff[f]) for f in others]))

    full_close = s["full"].isin(*EQUIVALENT, "REORDERED", "SUBSET", "FUZZY")
    all_ok = names_ok & functools.reduce(and_, [s[f].isin(EQUIVALENT) for f in others])
    hard_conflict = ((eff["dob"] == "MISMATCH")
                     | (~full_close & (((eff["first"] == "MISMATCH") & (eff["last"] == "MISMATCH"))
                                       | ((eff["last"] == "MISMATCH") & (eff["dob"] != "EXACT")))))
    nonexact = ([F.when(~names_ok & ~s[f].isin(EQUIVALENT), F.lit(f)) for f in ("first", "last")]
                + [F.when(~s[f].isin(EQUIVALENT), F.lit(f)) for f in others])
    by_notes = [F.when(eff[f] == "NOTES_MATCH", F.lit(f)) for f in ("first", "last", "dob")]
    present = lambda cols: F.filter(F.array(*cols), lambda x: x.isNotNull())

    return j.withColumns({
        "dq_match_score": F.when(matched, F.round(score, 3)),
        "dq_status": (F.when(~matched, "NO_MATCH")
                       .when(all_ok, "VERIFIED")
                       .when(hard_conflict | (score < REVIEW_THRESHOLD), "CONFLICT")
                       .otherwise("NEEDS_REVIEW")),
        "dq_nonexact_fields": F.when(matched, present(nonexact)),
        NOTES_RESOLVED_COL: F.when(matched, present(by_notes)),
    })


def find_other_keys(o, a):
    """For each ORR row, the asylum a_ids (other than its own) that look like the same person,
    and which search found them:

        fields      ORR name + ORR DOB
        notes_dob   ORR name + a date found in additional_notes
        notes_name  ORR DOB or a note date, and the asylum first AND last name words appear in the notes

    On a NO_MATCH row a hit means a probable a_id error (-> POSSIBLE_KEY_ERROR). On a CONFLICT row it
    points to where the person may really be. Names are compared as joined words, in entered and
    sorted order, and against the asylum name with the middle name kept, dropped or abbreviated.
    """
    keys = lambda tok: [F.array_join(tok, ""), F.array_join(F.array_sort(tok), "")]
    other_key = ~F.col("a__key").eqNullSafe(F.col("o__key"))
    a_ok = a.where(F.col("a__key").isNotNull() & F.col("a__full_tok").isNotNull() & F.col("a__dob").isNotNull())
    a_keys = (a_ok.select("a__key", "a__dob", F.explode(F.array_distinct(F.array(
                  *[k for t in ("full_tok", "nomid_tok", "init_tok") for k in keys(F.col("a__" + t))])))
                  .alias("a__name_key"))
                  .distinct())

    # ORR name keys paired with the DOB to search on
    with_name = o.where(F.col("o__full_tok").isNotNull())
    name_keys = F.explode(F.array_distinct(F.array(*keys(F.col("o__full_tok"))))).alias("o__name_key")
    searches = [with_name.where(F.col("o__dob").isNotNull())
                         .select("o__row_id", "o__key", "o__full_tok", F.col("o__dob").alias("o__cand_dob"),
                                 F.lit("fields").alias("source"))]
    has_notes = "o__note_tok" in o.columns
    if has_notes:
        searches.append(with_name.select("o__row_id", "o__key", "o__full_tok",
                                         F.explode("o__note_dobs").alias("o__cand_dob"),
                                         F.lit("notes_dob").alias("source")))
    by_name = functools.reduce(lambda x, y: x.unionByName(y), searches).select("*", name_keys)
    hits = [by_name.join(a_keys, (F.col("o__name_key") == F.col("a__name_key"))
                                 & (F.col("o__cand_dob") == F.col("a__dob")) & other_key)
                   .select("o__row_id", "a__key", "source")]

    if has_notes:  # Names written in the notes, checked against asylum cases with a candidate DOB
        cand_dobs = F.array_distinct(F.filter(F.concat(F.array(F.col("o__dob")), F.col("o__note_dobs")),
                                              lambda d: d.isNotNull()))
        o_notes = (o.where(F.size("o__note_tok") > 0)
                    .select("o__row_id", "o__key", "o__note_tok", F.explode(cand_dobs).alias("o__cand_dob")))
        hits.append(o_notes.join(a_ok, (F.col("o__cand_dob") == F.col("a__dob")) & other_key)
                           .where(_words_in(F.col("a__first_tok"), F.col("o__note_tok"))
                                  & _words_in(F.col("a__last_tok"), F.col("o__note_tok")))
                           .select("o__row_id", "a__key", F.lit("notes_name").alias("source")))

    return (functools.reduce(lambda x, y: x.unionByName(y), hits)
            .groupBy("o__row_id")
            .agg(F.array_sort(F.collect_set("a__key")).alias(OTHER_KEYS_COL),
                 F.array_sort(F.collect_set("source")).alias(OTHER_KEYS_SOURCE_COL)))


def gate(orr_df, merged, issues):
    """Batch-level checks. Only failed CRITICAL checks block publish(); HIGH ones are for review."""
    n_in, n_out = orr_df.count(), merged.count()
    n_matched = merged.where(_col(MATCH_FLAG_COL)).count()
    status = {r["dq_status"]: r["count"] for r in merged.groupBy("dq_status").count().collect()}
    issue = {(r["table"], r["check"]): r["count"] for r in issues.groupBy("table", "check").count().collect()}
    ic = lambda t, c: issue.get((t, c), 0)
    conflict_rate = status.get("CONFLICT", 0) / n_matched if n_matched else 0.0

    rows = [
        ("ROW_COUNT_PRESERVED",  "CRITICAL", n_out == n_in, f"ORR input={n_in}, output={n_out}"),
        ("ASYLUM_KEYS_UNIQUE",   "CRITICAL", ic(ASY, "DUPLICATE_KEY") == 0,
         f"{ic(ASY, 'DUPLICATE_KEY')} asylum case rows share an a_id (causes fan-out)"),
        ("CONFLICT_RATE",        "CRITICAL" if n_matched >= MIN_MATCHES_FOR_RATE else "HIGH",
         conflict_rate <= MAX_CONFLICT_RATE, f"{conflict_rate:.2%} of {n_matched} matched rows conflict"),
        ("ORR_KEYS_NOT_NULL",    "HIGH", ic(ORR, "NULL_KEY") == 0, f"{ic(ORR, 'NULL_KEY')} ORR records have no a_id"),
        ("ORR_KEYS_UNIQUE",      "HIGH", ic(ORR, "DUPLICATE_KEY") == 0,
         f"{ic(ORR, 'DUPLICATE_KEY')} ORR records share an a_id"),
        ("POSSIBLE_KEY_ERRORS",  "HIGH", status.get("POSSIBLE_KEY_ERROR", 0) == 0,
         f"{status.get('POSSIBLE_KEY_ERROR', 0)} unmatched ORR records match an asylum case under another a_id"),
    ]
    passed = all(ok for _, sev, ok, _ in rows if sev == "CRITICAL")
    gate_df = orr_df.sparkSession.createDataFrame(rows, "check string, severity string, passed boolean, detail string")
    return gate_df, passed


# =============================================================================
# 6. ENTRY POINTS
# =============================================================================

def _cache(df):
    if not CACHE:
        return df
    try:
        return df.persist()
    except Exception:  # e.g. serverless compute
        return df


def run_dq_merge(orr_df: DataFrame, asylum_df: DataFrame) -> dict:
    """Run every step and return the merged result plus all reports. Writes nothing.

    Returns: merged, issues, gate, passed, status_summary, field_summary (and _cached, for release()).
    """
    for name, df, cols in [(ORR, orr_df, ORR_COLS), (ASY, asylum_df, ASY_COLS)]:
        missing = [c for c in cols.values() if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
    clash = [c for c in orr_df.columns
             if c == MATCH_FLAG_COL or c.startswith(("dq_", ASY_PREFIX, "o__", "a__"))]
    if clash:
        raise ValueError(f"{ORR} columns collide with generated output columns: {clash}")

    # additional_notes is optional: without it every notes step is skipped
    orr_cols = {**ORR_COLS, "notes": NOTES_COL} if NOTES_COL in orr_df.columns else ORR_COLS

    # 1. Normalize (asylum columns are prefixed first so nothing collides)
    asy_prefixed = asylum_df.select([_col(c).alias(ASY_PREFIX + c) for c in asylum_df.columns])
    o = _cache(normalize(orr_df, orr_cols, "o__"))
    a = _cache(normalize(asy_prefixed, {f: ASY_PREFIX + c for f, c in ASY_COLS.items()}, "a__"))

    # 2. Single-table checks
    issues = _cache(table_checks(o, orr_cols, "o__", ORR).unionByName(
        table_checks(a, {f: ASY_PREFIX + c for f, c in ASY_COLS.items()}, "a__", ASY)))

    # 3. LEFT join on the canonical a_id. Null keys never match.
    j = o.join(a.where(F.col("a__key").isNotNull()), F.col("o__key") == F.col("a__key"), "left")
    matched = F.col("a__key").isNotNull()
    j = compare(j.withColumn(MATCH_FLAG_COL, matched), matched)

    # 4. Other asylum a_ids for the same person, plus each ORR row's own issues
    row_issues = (issues.where(F.col("table") == ORR)
                  .groupBy(F.col("row_id").alias("o__row_id"))
                  .agg(F.collect_list(F.concat_ws(":", "severity", "check")).alias("dq_row_issues")))
    j = (j.join(find_other_keys(o, a), "o__row_id", "left")
          .join(row_issues, "o__row_id", "left")
          .withColumn("dq_status", F.when((F.col("dq_status") == "NO_MATCH") & F.col(OTHER_KEYS_COL).isNotNull(),
                                          "POSSIBLE_KEY_ERROR").otherwise(F.col("dq_status"))))

    # Output: ORR columns, asylum columns, then DQ columns
    merged = _cache(j.select(*[_col(c) for c in orr_df.columns],
                             *[_col(ASY_PREFIX + c) for c in asylum_df.columns],
                             _col(MATCH_FLAG_COL), "dq_status", "dq_match_score", "dq_nonexact_fields",
                             NOTES_RESOLVED_COL, *STATUS.values(), OTHER_KEYS_COL, OTHER_KEYS_SOURCE_COL,
                             "dq_row_issues"))

    # 5. Gate
    gate_df, passed = gate(orr_df, merged, issues)
    matched_rows = merged.where(_col(MATCH_FLAG_COL))
    # unionByName on the instance (not the class) so it works on Spark Connect / shared clusters
    field_summary = functools.reduce(lambda x, y: x.unionByName(y), [
        matched_rows.groupBy(F.lit(c.removeprefix("dq_").removesuffix("_status")).alias("field"),
                             F.col(c).alias("status")).count()
        for c in STATUS.values()] + [
        # How many matched rows the notes resolved, per field
        matched_rows.select(F.lit("notes").alias("field"), F.explode(NOTES_RESOLVED_COL).alias("status"))
                    .groupBy("field", "status").count()])

    return {"merged": merged, "issues": issues, "gate": gate_df, "passed": passed,
            "status_summary": merged.groupBy("dq_status").count(),
            "field_summary": field_summary, "_cached": [o, a, issues, merged]}


def release(results: dict):
    """Free the cached DataFrames once you are done with a run."""
    for df in results.get("_cached", []):
        try:
            df.unpersist()
        except Exception:
            pass


def publish(results: dict, target_table: str, mode: str = "overwrite"):
    """Write the merged table only if the gate passed."""
    if not results["passed"]:
        failed = results["gate"].where("NOT passed AND severity = 'CRITICAL'").collect()
        raise RuntimeError("Merge blocked by DQ gate: " + "; ".join(f"{r.check} ({r.detail})" for r in failed))
    w = results["merged"].write.mode(mode)
    if mode == "overwrite":
        w = w.option("overwriteSchema", "true")
    if " " in MATCH_FLAG_COL:  # Delta needs column mapping for names with spaces
        w = (w.option("delta.columnMapping.mode", "name")
              .option("delta.minReaderVersion", "2").option("delta.minWriterVersion", "5"))
    w.saveAsTable(target_table)


# =============================================================================
# USAGE
# =============================================================================
# res = run_dq_merge(spark.table("hhs_orr"), spark.table("affirmative_asylum_cases"))
#
# display(res["gate"])                                       # 1. Did the batch pass?
# display(res["status_summary"])                             # 2. How many in each status?
# display(res["field_summary"].orderBy("field", "status"))   # 3. Which fields disagree most?
# display(res["issues"].groupBy("table", "severity", "check").count())
# display(res["merged"].where("dq_status IN ('CONFLICT', 'POSSIBLE_KEY_ERROR', 'NEEDS_REVIEW')"))
#
# publish(res, "catalog.schema.hhs_orr_asylum_merged")
# release(res)
