"""Create a colour-coded local review copy and an unmarked Overleaf copy.

Categories record the reason for the October supervisor revision, not merely
whether a span differs from the previous highlighted draft. Mixed passages are
split where an additional check went beyond the requested reviewer response.
"""
from __future__ import annotations

import re

# Prefixes are deliberately explicit: additions cannot silently inherit a colour.
CATEGORIES = {
    'orange': [
        'Positive tone toward AI technology',
        'All variables are aligned with the US equity-market',
        'Each text is scored individually',
        'The 100-observation validation',
        'The same-size comparison adds Cisco',
        'Forward-looking AI expectations',
        'Kalshi (2024)',
        'The Metaculus series uses',
        'Metaculus, question 5121',
        'The first-difference transformations',
        r'S\&P 500 returns enter Equation',
        r'Table \ref{tab:btc_pairwise} makes',
        'Bitcoin-Specific Net Pairwise Connectedness',
        r'\textit{Notes:} Means over 501 dates',
        'Same-size comparison system:',
        'To assess whether the higher average TCI',
        'The three new return series',
        r'Panel C of Table \ref{tab:average_connectedness_table}',
        'BTC--AI-equity system',
        'BTC--comparison-equity system',
        'Total Connectedness in Equally Sized Systems',
        r'\textit{Notes:} Both systems contain BTC',
        'The sentiment regression does not support exclusivity',
        'EAIS and Total Connectedness in Equally Sized Systems',
        r'\textit{Notes:} The final outcome is',
        'Several limitations should be considered.',
        'The scores are aggregated separately',
        'where $N_d^{(l)}$ counts',
        'One-Day Granger Tests of EAIS on Connectedness',
        r'\textit{Notes:} Null: no predictive contribution',
    ],
    'green': [
        'The AI case studied here motivates',
        'The continuous text scores, rather than the discrete labels',
        'although the difference reflects the fit of the entire regression',
        'The firms are not matched in capitalization',
        'Cisco (2025)',
        r'Adding S\&P 500 returns to the same HAC specification',
    ],
    'yellow': [
        'Investment narratives may be associated',
        'The analysis distinguishes between a broad technology-heavy index',
        'This study addresses these gaps',
        'The contribution of this study is threefold.',
        'The findings motivate supplementary firm-level',
        'The residual is the component unexplained',
        'this study views AI-related sentiment',
        'The system comparisons assess',
        'The residual defines macro- and expectation-adjusted',
        r'Table \ref{tab:orthogonalization_regression} shows',
        'A positive coefficient denotes',
        'has a positive and statistically significant conditional association',
        'This result concerns adjusted textual sentiment',
        r'Table \ref{tab:average_connectedness_table} reports',
        'a one-unit increase in lagged EAIS',
        'The opposite signs describe',
        'describe the conditional directional associations.',
        'a one-unit increase in EAIS is associated with 3.45',
        'The BTC--AI-equity system displays more precisely',
        'with opposite-signed lag coefficients',
        'Overall, the regressions document same-day',
        'including sentiment as an endogenous variable',
        'The ARMA-EGARCH models are re-estimated',
        'The higher AI-system TCI also appears',
        'This divergence shows that the estimated timing pattern',
        'Overall, raw AIS does not consistently',
        'As a further robustness check,',
        'The corresponding network and total-connectedness figures',
        'The TVP-VAR results show differences across',
        'The central regression finding is that',
        'Within the AI basket, higher contemporaneous',
        'The raw-sentiment results are a substantive qualification.',
        'For institutional investors, the findings motivate',
        'The scope is one narrative episode,',
        'the observed sentiment association is specific',
        'In conclusion, expectation-adjusted AI sentiment',
    ],
}
MACROS = {'orange': 'reviewadd', 'yellow': 'reviewchange', 'green': 'reviewclarify'}

OLD_PREAMBLE = r'''% Yellow marks all added or substantively revised text for supervisor review.
\usepackage{xcolor}
\usepackage{soul}
\sethlcolor{yellow}
\DeclareRobustCommand{\rev}[1]{\hl{#1}}
\soulregister\ref7
\soulregister\eqref7
\soulregister\citep7
\soulregister\citet7
\soulregister\textit1
'''
LOCAL_PREAMBLE = r'''% Review colours are local only; the Overleaf export removes all markup.
\usepackage{xcolor}
\usepackage{soul}
\definecolor{ReviewOrange}{HTML}{FFC078}
\definecolor{ReviewYellow}{HTML}{FFF176}
\definecolor{ReviewGreen}{HTML}{B9E7AE}
\definecolor{ReviewAqua}{HTML}{A7E8E8}
\DeclareRobustCommand{\reviewadd}[1]{{\sethlcolor{ReviewOrange}\hl{#1}}}
\DeclareRobustCommand{\reviewchange}[1]{{\sethlcolor{ReviewYellow}\hl{#1}}}
\DeclareRobustCommand{\reviewclarify}[1]{{\sethlcolor{ReviewGreen}\hl{#1}}}
\DeclareRobustCommand{\revieworiginal}[1]{{\sethlcolor{ReviewAqua}\hl{#1}}}
\newcommand{\reviewaddbox}[1]{\colorbox{ReviewOrange}{#1}}
\newcommand{\reviewclarifybox}[1]{\colorbox{ReviewGreen}{#1}}
\soulregister\ref7
\soulregister\eqref7
\soulregister\citep7
\soulregister\citet7
\soulregister\textit1
'''
OLD_PDFSTRINGS = r'\pdfstringdefDisableCommands{\def\rev#1{#1}}'
LOCAL_PDFSTRINGS = r'\pdfstringdefDisableCommands{\def\reviewadd#1{#1}\def\reviewchange#1{#1}\def\reviewclarify#1{#1}}'
LEGEND = r'''% BEGIN LOCAL REVIEW KEY
\par\vspace{1em}
{\footnotesize
\colorbox{ReviewOrange}{Orange: additions responding to feedback}\par\smallskip
\colorbox{ReviewYellow}{Yellow: existing wording revised in light of findings}\par\smallskip
\colorbox{ReviewGreen}{Green: additional clarifications and improvements}\par\smallskip
\colorbox{ReviewAqua}{Aqua: original paragraph before the yellow revisions}\par}
% END LOCAL REVIEW KEY
'''


def group_end(text: str, opening: int) -> int:
    """Return the exclusive end of a balanced TeX brace group."""
    assert text[opening] == '{'
    depth = 0
    for i in range(opening, len(text)):
        backslashes = 0
        j = i - 1
        while j >= 0 and text[j] == '\\':
            backslashes += 1
            j -= 1
        if backslashes % 2:
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError('Unclosed review group')


def wrap(text: str, colour: str) -> str:
    return '\\' + MACROS[colour] + '{' + text + '}' if text else ''


def split_span(text: str, default: str) -> list[tuple[str, str]]:
    """Preserve precise provenance within passages containing extra analysis."""
    if text.startswith('Each text is scored individually'):
        cut = text.index('Daily alignment does not establish')
        return [(default, text[:cut]), ('green', text[cut:])]
    if text.startswith(r'S\&P 500 returns enter Equation'):
        cut = text.index('This is a specification choice,')
        return [(default, text[:cut]), ('green', text[cut:])]
    if text.startswith('The first-difference transformations'):
        cut = text.index('; the mixed evidence')
        return [(default, text[:cut]), ('green', text[cut:])]
    if text.startswith('Overall, the regressions document same-day'):
        first = text.index(r'Table \ref{tab:connectedness_granger}')
        extra = text.index(', which does not survive HAC')
        last = text.index(' Thus, these diagnostics')
        return [('yellow', text[:first]), ('orange', text[first:extra]),
                ('green', text[extra:last]), ('yellow', text[last:])]
    if text.startswith('The sentiment regression does not support exclusivity'):
        first = text.index('Regressing the same-date TCI difference')
        last = text.index(' Its lower average TCI')
        return [(default, text[:first]), ('green', text[first:last]), (default, text[last:])]
    if text.startswith('Several limitations should be considered.'):
        cut = text.index(', and full-sample estimation')
        return [(default, text[:cut]), ('green', text[cut:])]
    if text.startswith(r'\textit{Notes:} Null:'):
        cut = text.index('HAC uses five lags')
        return [(default, text[:cut]), ('green', text[cut:])]
    if text.startswith(r'\textit{Notes:} The final outcome is'):
        cut = text.index(' HAC(5) standard errors')
        return [('green', text[:cut]), (default, text[cut:])]
    return [(default, text)]


def mark_extra_columns(source: str, label: str, columns: tuple[int, ...]) -> str:
    """Mark optional checks in green within an otherwise orange new table."""
    start = source.index('\\label{' + label + '}')
    begin = source.index(r'\begin{tabular}', start)
    end = source.index(r'\end{tabular}', begin)
    lines = source[begin:end].splitlines(True)
    result = []
    for line in lines:
        if '&' in line and r'\\' in line:
            cells = line.rstrip().removesuffix(r'\\').rstrip().split('&')
            for col in columns:
                cell = cells[col].strip()
                cells[col] = r' \reviewclarifybox{' + cell + '} '
            line = '&'.join(cells).rstrip() + r' \\' + '\n'
        result.append(line)
    return source[:begin] + ''.join(result) + source[end:]


def remove_review_markup(source: str) -> str:
    """Unwrap content, including nested boxes, rather than disabling colours."""
    names = [*MACROS.values(), 'reviewaddbox', 'reviewclarifybox']
    pattern = re.compile(r'\\(?:' + '|'.join(names) + r')\{')
    result = []
    position = 0
    while match := pattern.search(source, position):
        opening = match.end() - 1
        end = group_end(source, opening)
        result.append(source[position:match.start()])
        result.append(remove_review_markup(source[opening + 1:end - 1]))
        position = end
    result.append(source[position:])
    return ''.join(result)


def original_paragraph(source: str, anchor: str) -> str:
    """Find the unchanged source paragraph, excluding surrounding TeX structure."""
    at = source.index(anchor)
    start = source.rfind('\n\n', 0, at) + 2
    # Display equations delimit the surrounding prose even without blank lines.
    for display_end in re.finditer(r'\\end\{(?:equation\*?|align\*?)\}', source[:at]):
        start = max(start, display_end.end())
    end = source.find('\n\n', at + len(anchor))
    block = source[start:end if end >= 0 else len(source)].strip()
    block = re.sub(r'\\(?:begin|end)\{abstract\}|\\noindent', '', block)
    block = re.sub(r'\\(?:subsection|subsubsection|section|label)\{[^{}]*\}', '', block)
    return block.strip()


def record_original(provenance: list, replacement: str, original: str) -> None:
    position = 0
    while (start := replacement.find(r'\rev{', position)) >= 0:
        end = group_end(replacement, start + 4)
        provenance.append((replacement[start + 5:end - 1], original))
        position = end


def add_original_paragraphs(local: str, provenance: list, expected_count: int | None = 33) -> tuple[str, list[str]]:
    """Insert one verbatim original beneath each paragraph containing yellow."""
    preamble, body = local.split(r'\begin{document}', 1)
    blocks = body.split('\n\n')
    audit = []
    for i, block in enumerate(blocks):
        position = 0
        originals = set()
        while (start := block.find(r'\reviewchange{', position)) >= 0:
            opening = start + len(r'\reviewchange')
            end = group_end(block, opening)
            changed = block[opening + 1:end - 1]
            matches = {original for replacement, original in provenance
                       if changed in replacement}
            assert len(matches) == 1, (changed[:100], matches)
            originals.update(matches)
            position = end
        if originals:
            assert len(originals) == 1, 'A revised paragraph maps to multiple original paragraphs'
            original = originals.pop()
            assert not re.search(r'\\(?:begin|end|section|subsection)\b', original), original[:100]
            blocks[i] += '\n\n' + r'\noindent\revieworiginal{\textit{Original wording:} ' + original + '}'
            audit.append('AQUA ORIGINAL: ' + original)
    if expected_count is not None:
        assert len(audit) == expected_count, f'Expected {expected_count} original comparisons, found {len(audit)}'
    return preamble + r'\begin{document}' + '\n\n'.join(blocks), audit


def render_review_versions(source: str, provenance: list) -> tuple[str, str, str]:
    assert OLD_PREAMBLE in source and OLD_PDFSTRINGS in source
    source = source.replace(OLD_PREAMBLE, LOCAL_PREAMBLE, 1)
    source = source.replace(OLD_PDFSTRINGS, LOCAL_PDFSTRINGS, 1)
    audit = [
        'Local review colours (relative to the original supervisor-review manuscript)',
        'ORANGE: requested additions and explanations responding to supervisor/reviewer feedback.',
        'YELLOW: existing wording or interpretation revised in light of the findings.',
        'GREEN: additional clarity, factual checks or analytical improvements beyond the requested response.',
        'Mixed passages are split; new table bodies are orange, with optional contrast/HAC/Holm columns green.',
        'Formatting-only changes (A8 resizebox and shorter notes) do not recolour unchanged text.',
        'The Overleaf copy contains identical paper content, without review markup or the local colour key.',
        '',
    ]
    result = []
    position = 0
    used = set()
    while (start := source.find(r'\rev{', position)) >= 0:
        opening = start + 4
        end = group_end(source, opening)
        text = source[opening + 1:end - 1]
        matches = [(colour, prefix) for colour, prefixes in CATEGORIES.items()
                   for prefix in prefixes if text.startswith(prefix)]
        assert len(matches) == 1, (text[:100], matches)
        colour, prefix = matches[0]
        used.add(prefix)
        pieces = split_span(text, colour)
        result.append(source[position:start])
        result.append(''.join(wrap(piece, category) for category, piece in pieces))
        audit.extend(category.upper() + ': ' + piece for category, piece in pieces)
        position = end
    result.append(source[position:])
    assert len(used) == sum(map(len, CATEGORIES.values())), 'Stale category mapping'
    local = ''.join(result).replace(r'\colorbox{yellow}', r'\reviewaddbox')
    local = mark_extra_columns(local, 'tab:placebo_regressions', (3,))
    local = mark_extra_columns(local, 'tab:connectedness_granger', (4, 5))
    anchor = r'    {\normalsize \thesisdate}'
    assert anchor in local
    local = local.replace(anchor, anchor + '\n' + LEGEND, 1)
    clean = local.replace(LOCAL_PREAMBLE, '').replace(LOCAL_PDFSTRINGS, '')
    clean = clean.replace(LEGEND, '')
    clean = remove_review_markup(clean)
    assert not re.search(r'\\(?:review\w*|rev|hl|sethlcolor|colorbox)\b', clean)
    assert '\\rev{' not in local
    # Comparison paragraphs exist only in the local copy, after clean export.
    local, originals = add_original_paragraphs(local, provenance)
    audit.extend(originals)
    return local, clean, '\n\n'.join(audit) + '\n'
