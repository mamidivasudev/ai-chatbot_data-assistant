from skills import load_skills
import re
from ollama_client import ask_ollama


def render_history(history, max_turns=4):
    """
    Recent turns, so a follow-up like "name of it" can be resolved.

    Without this the endpoint is stateless and every pronoun or elision fails.
    """
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        q = (turn.get("question") or "").strip()
        s = (turn.get("sql") or "").strip()
        a = (turn.get("answer") or "").strip()
        if not q:
            continue
        lines.append(f"Previous question: {q}")
        if s:
            lines.append(f"Previous T-SQL: {s}")
        if a:
            lines.append(f"Previous answer: {a[:300]}")
        lines.append("")
    if not lines:
        return ""

    rule = "=" * 60
    return (
        f"\n\n{rule}\nCONVERSATION SO FAR\n{rule}\n"
        "The current question may be a follow-up that refers to the turns below "
        "using words like 'it', 'that one', 'its name', or by omitting the subject. "
        "Resolve those references from this history, then answer the CURRENT question. "
        "If the current question stands on its own, ignore this history.\n\n"
        + "\n".join(lines)
        + f"{rule}\n"
    )


def generate_tsql(question, schema_text, business_rules="", model=None,
                  db_identifier=None, dynamic_rules="", examples_text="",
                  history=None):
    # Skills are gated on the question, the schema actually supplied, and the
    # connected database — see skills.py.
    rules_text = load_skills(
        question,
        schema_text=schema_text,
        db_identifier=db_identifier,
    )

    # Rules derived from the tables the admin actually selected — allowed table
    # list, soft-delete filters, real FK join paths, sampled column values.
    # See schema_profiler.py.
    if dynamic_rules:
        rules_text += dynamic_rules

    # Verified (question, SQL) pairs resembling this question. Examples teach
    # conventions that rules struggle to state. See query_library.py.
    if examples_text:
        rules_text += examples_text

    # Prior turns, so follow-up questions resolve their references.
    rules_text += render_history(history)

    # Per-database rules configured by an admin, injected alongside the skills
    # rather than discarded.
    custom_rules = ""
    if business_rules and str(business_rules).strip():
        custom_rules = (
            "\n\nCustom Business Rules for this database:\n"
            f"{str(business_rules).strip()}\n"
        )

    prompt = f"""You are a MS SQL AI Assistant Server (T-SQL) expert.

Database Schema:
{schema_text}
{rules_text}{custom_rules}

Rules:
1. Return ONLY a valid T-SQL SELECT statement.
2. Do NOT use markdown code fences or ```sql.
3. Do NOT explain anything.
4. Output must start directly with SELECT.
5. Use SELECT statements only — never UPDATE, DELETE, INSERT, DROP, ALTER, TRUNCATE, EXEC, or EXECUTE.
6. Use square bracket quoting for table and column names: [SchemaName].[TableName].
7. Use TOP instead of LIMIT for row limiting.
8. For string patterns use LIKE with % wildcards.
9. For date functions use GETDATE(), DATEADD(), DATEDIFF(), FORMAT() — not MySQL syntax.
10. Every table and column you reference must appear in the Database Schema above. Never substitute a similarly named table.

Question:
{question}
"""

    kwargs = {"model": model} if model else {}
    raw = ask_ollama(prompt, **kwargs)

    # Strip markdown fences if model still adds them
    raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "")

    # Extract the first SELECT…; block
    match = re.search(r"(SELECT[\s\S]*?)(;|$)", raw, re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        if match.group(2) == ";":
            sql += ";"
    else:
        sql = raw.strip()

    return sql


def detect_question_language(text):
    """
    Language of the question, decided by script rather than by the model.

    Asking the model to "detect the language" made it answer English questions
    in Hindi, because the result rows are full of Indian place names. Script
    detection is deterministic: unless the question itself is written in Telugu
    or Devanagari, the answer must be English.
    """
    for ch in text or "":
        code = ord(ch)
        if 0x0C00 <= code <= 0x0C7F:
            return "Telugu"
        if 0x0900 <= code <= 0x097F:
            return "Hindi"
        if 0x0B80 <= code <= 0x0BFF:
            return "Tamil"
        if 0x0980 <= code <= 0x09FF:
            return "Bengali"
        if 0x0A80 <= code <= 0x0AFF:
            return "Gujarati"
        if 0x0C80 <= code <= 0x0CFF:
            return "Kannada"
    return "English"


def generate_answer_summary(question, sql, columns, rows, model=None, simple_mode=False):
    """Ask Ollama to summarise query results in plain English."""
    if not rows:
        return "No records were returned for your question."

    # Format rows as a compact text table (cap at 50 rows to stay in context)
    preview_rows = rows[:50]
    header = " | ".join(str(c) for c in columns)
    separator = "-" * len(header)
    row_lines = "\n".join(
        " | ".join(str(v) for v in row) for row in preview_rows
    )
    table_text = f"{header}\n{separator}\n{row_lines}"

    if len(rows) > 50:
        table_text += f"\n... ({len(rows) - 50} more rows not shown)"

    prompt = f"""You are a helpful data analyst. The user asked:

"{question}"

The SQL query returned these results:

{table_text}

"""
    if simple_mode:
        prompt += "Write a very concise, direct answer based on the data above. NEVER start with phrases like 'According to the data' or 'Based on the provided data'. Just state the facts immediately. Do not repeat the SQL."
    else:
        prompt += "Write a clear, concise natural-language answer (2–4 sentences) that directly answers the question based on the data above. Do not repeat the SQL. Do not use bullet points."

    # The language is decided here, not by the model. Place names in the result
    # rows used to make it switch to Hindi on English questions.
    language = detect_question_language(question)
    if language == "English":
        prompt += (
            "\nIMPORTANT: Write the answer in ENGLISH. The question is in English, "
            "so the answer must be in English. Place names in the data do not change "
            "this — never reply in Hindi, Telugu or any other language."
        )
    else:
        prompt += (
            f"\nIMPORTANT: The question is written in {language}. "
            f"Write your entire answer in {language}."
        )

    kwargs = {"model": model} if model else {}
    return ask_ollama(prompt, **kwargs)


def generate_rule_from_sql(question, sql, schema_text, model=None):
    """Ask Ollama to deduce a business rule from a question and its correct SQL."""
    prompt = f"""You are a database AI assistant expert. Your task is to extract a business rule from a human's question and their provided correct SQL query.

Database Schema:
{schema_text}

Question:
"{question}"

Correct SQL Query:
{sql}

Instruction:
Deduce the business rule or formula from this example. 
Return ONLY the plain English rule as a single, concise sentence.
Do not use bullet points, do not say "The rule is", just state the rule directly.
For example: "To find 'Busiest' roads, filter where AADT > 10000."
"""

    kwargs = {"model": model} if model else {}
    raw = ask_ollama(prompt, **kwargs)
    
    # Clean up any potential markdown or prefixes
    rule = raw.strip()
    rule = re.sub(r'^\d+\.\s*', '', rule)
    rule = rule.replace('"', '').replace("'", "")
    return rule
