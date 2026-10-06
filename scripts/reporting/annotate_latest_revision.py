"""Highlight substantive source changes on the user's supplied PDF, without reflow."""
from pathlib import Path
import difflib
import json
import re
import unicodedata

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Highlight
from pypdf.generic import ArrayObject, FloatObject, NameObject, TextStringObject

ROOT = Path(__file__).resolve().parents[2]
TMP = ROOT / 'tmp/revision_comparison'
OLD = Path('/Users/boazmuller/Downloads/main.tex').read_text()
NEW = Path('/Users/boazmuller/Downloads/main (1).tex').read_text()
PDF = Path('/Users/boazmuller/Downloads/Muller__Boaz_15165051BSc_ETRICS_revised.pdf')
OUT = ROOT / 'output/pdf/paper_changes_highlighted.pdf'


def canonical(s):
    return re.sub(r'\s+', ' ', s).strip()


def prose(source):
    """Separate narrative text from display equations, tables and figure code."""
    source = re.sub(r'(?<!\\)%[^\n]*', '', source)
    source = source.split(r'\begin{document}', 1)[1]
    blocks = []
    for block in re.split(r'\n\s*\n', source):
        if any(x in block for x in [r'\begin{table}', r'\begin{tabular}', r'\begin{figure}', r'\begin{tikzpicture}']):
            continue
        block = re.sub(r'\\(?:subsubsection|subsection|section|label)\{[^{}]*\}', '', block).strip()
        for part in re.split(r'\\begin\{(?:equation\*?|align\*?)\}.*?\\end\{(?:equation\*?|align\*?)\}', block, flags=re.S):
            part = part.strip()
            if not re.match(r'[A-Za-z]', part):
                continue
            if len(part) < 45 or part.startswith(('node distance=', 'box/.style=')):
                continue
            # Environment tails are not prose.
            part = re.split(r'\\end\{(?:abstract|document)\}', part)[0].strip()
            blocks.append(part)
    return blocks


def sentences(text):
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', canonical(text)) if x.strip()]


def readable(tex):
    text = re.sub(r'\\(?:citep|citet|cite|ref|eqref)\{[^{}]*\}', '§', tex)
    text = re.sub(r'\\label\{[^{}]*\}', '', text)
    for command, value in [('ln','ln'), ('beta','β'), ('eta','η'), ('lambda','λ'), ('alpha','α'), ('gamma','γ'), ('varepsilon','ε')]:
        text = re.sub(r'\\' + command + r'\b', lambda _: value, text)
    text = re.sub(r'\\(?:textit|textbf|text|emph|mathrm)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\[A-Za-z]+', '', text)
    return text


def normalize(s):
    return ''.join(c.lower() for c in unicodedata.normalize('NFKD', s) if c.isalnum())


old_blocks = prose(OLD)
new_blocks = prose(NEW)
old_sentences = {canonical(t) for b in old_blocks for t in sentences(b)}
passages = []
for block in new_blocks:
    if canonical(block) in {canonical(x) for x in old_blocks}:
        continue
    current = []
    for sentence in sentences(block):
        if canonical(sentence) not in old_sentences:
            current.append(sentence)
        elif current:
            passages.append(' '.join(current)); current = []
    if current:
        passages.append(' '.join(current))
# Deduplicate text moved unchanged between environments, and invalid table fragments.
passages = list(dict.fromkeys(passages))
passages = [x for x in passages if not re.search(r'\\(?:end|begin)\{(?:tabular|table|figure)\}', x)]

chars = []
stream = []
stream_map = []
pages = []
with pdfplumber.open(PDF) as pdf:
    for pnum, page in enumerate(pdf.pages):
        pages.append({'width': float(page.width), 'height': float(page.height),
                      'images': [dict(x0=float(i['x0']), x1=float(i['x1']), top=float(i['top']), bottom=float(i['bottom'])) for i in page.images]})
        for char in page.chars:
            if char['top'] > float(page.height) - 70:
                continue  # folios are layout, not manuscript wording
            char = dict(char, page=pnum)
            index = len(chars)
            chars.append(char)
            for c in normalize(char['text']):
                stream.append(c); stream_map.append(index)
TEXT = ''.join(stream)


def locate(text, details=False):
    cleaned = readable(text)
    segments = [normalize(x) for x in cleaned.split('§')]
    target = ''.join(segments)
    anchors = []
    off = 0
    for segment in segments:
        if len(segment) >= 24:
            for local in sorted(set([0, max(0, len(segment)//2-35), max(0, len(segment)-70)])):
                anchor = segment[local:local+70]
                if len(anchor) >= 24:
                    anchors.append((off+local, anchor))
        off += len(segment)
    candidates = set()
    for offset, anchor in anchors:
        at = TEXT.find(anchor)
        while at >= 0:
            candidates.add(max(0, at-offset))
            at = TEXT.find(anchor, at+1)
    assert candidates, ('No PDF anchor', text[:150])
    best = None
    for candidate in candidates:
        lo = max(0, candidate-100)
        # Floats can interrupt a sentence at a page break. Include enough of the
        # next page to find its continuation, then exclude the inserted float.
        hi = min(len(TEXT), candidate+len(target)+3000)
        sm = difflib.SequenceMatcher(None, target, TEXT[lo:hi], autojunk=False)
        matches = [m for m in sm.get_matching_blocks() if m.size]
        score = sum(m.size for m in matches)/len(target)
        start = lo+matches[0].b
        end = lo+matches[-1].b+matches[-1].size
        rank = score - abs((end-start)-len(target)) / max(len(target), 1) * .015
        if best is None or rank > best[0]:
            spans = []
            a_end = b_end = None
            for m in matches:
                if b_end is None or (m.b-b_end)-(m.a-a_end)>60:
                    spans.append([lo+m.b,lo+m.b+m.size])
                else:
                    spans[-1][1]=lo+m.b+m.size
                a_end=m.a+m.size; b_end=m.b+m.size
            best = (rank, start, end, score, spans)
    assert best[3] > .88, ('Poor PDF alignment', best, text[:200])
    return (best[1], best[2], best[3], best[4]) if details else (best[1], best[2], best[3])


def glyphs(start, end):
    return chars[stream_map[start]:stream_map[end-1]+1]


def line_rects(glyph_list):
    """One quad per printed line; leave untouched text on other lines unmarked."""
    result = {}
    for page in sorted({c['page'] for c in glyph_list}):
        lines = []
        for c in sorted([c for c in glyph_list if c['page']==page], key=lambda c:(c['top'],c['x0'])):
            if not c['text'].strip(): continue
            center = (c['top']+c['bottom'])/2
            hit = next((row for row in lines if abs(row[0]-center)<4), None)
            if hit is None:
                hit=[center, []];lines.append(hit)
            hit[1].append(c)
        result[page] = [(min(c['x0'] for c in row)-1, min(c['top'] for c in row)-.8,
                         max(c['x1'] for c in row)+2, max(c['bottom'] for c in row)+.8)
                        for _,row in lines]
    return result


reader = PdfReader(PDF)
writer = PdfWriter()
writer.clone_document_from_reader(reader)
audit = []


def annotate(rectangles, note):
    for pnum, rects in rectangles.items():
        height = pages[pnum]['height']
        quads = []
        for x0,top,x1,bottom in rects:
            quads.extend([x0,height-top,x1,height-top,x0,height-bottom,x1,height-bottom])
        bounds=(min(r[0] for r in rects), height-max(r[3] for r in rects),
                max(r[2] for r in rects), height-min(r[1] for r in rects))
        annotation = Highlight(rect=bounds, quad_points=ArrayObject([FloatObject(v) for v in quads]), highlight_color='FFFF00', printing=True)
        annotation[NameObject('/Contents')]=TextStringObject(note)
        annotation[NameObject('/T')]=TextStringObject('Changes from original paper')
        annotation[NameObject('/CA')]=FloatObject(.30)
        writer.add_annotation(page_number=pnum, annotation=annotation)


for i,text in enumerate(passages,1):
    start,end,score,spans=locate(text,details=True)
    rects=line_rects([c for a,b in spans for c in glyphs(a,b)])
    annotate(rects, f'Added or revised passage {i} compared with the original paper.\n'+canonical(readable(text)).replace('§','[reference]'))
    audit.append({'kind':'passage','number':i,'text':text,'pdf_pages':[p+1 for p in rects], 'match':round(score,4)})

# New tables/panel. Highlight the entire new body and its caption/notes.
special = [
    ('Table 4 Panel C', 'Panel C: BTC--comparison-equity system', '19.15', 'panel'),
    ('Table 5', 'Table 5: Bitcoin-Specific Net Pairwise Connectedness', 'negative: net reception.', 'table'),
    ('Table 9', 'Table 9: EAIS and Total Connectedness in Equally Sized Systems', 'HAC standard errors in parentheses. ***, ** and * denote 1%, 5% and 10% significance.', 'table'),
    ('Table A8', 'Table A8: One-Day Granger Tests of EAIS on Connectedness', 'Unadjusted $p$-values.', 'table'),
]
for name,first,last,kind in special:
    a,b,_=locate(first)
    tail=normalize(readable(last))
    finish=TEXT.find(tail,b)
    assert finish>=0 and finish-b<6000,(name,finish-b)
    end=finish+len(tail)
    rects=line_rects(glyphs(a,end))
    assert len(rects)==1,(name,rects)
    annotate(rects,f'New {name} added in response to the reviewer; existing results in other panels are unchanged.')
    audit.append({'kind':kind,'name':name,'pdf_pages':[p+1 for p in rects]})

# New figure: translucent full-area highlighting includes both panels and the notes.
a,b,_=locate('Figure 3: Total Connectedness in Equally Sized Systems')
pnum=chars[stream_map[a]]['page']
images=pages[pnum]['images']
assert len(images)==2,(pnum,len(images))
tail=normalize('Both systems contain BTC and three equities; the comparison basket comprises CSCO, BKNG and MDLZ.')
end=TEXT.find(tail,b)+len(tail)
assert end>=b
fc=glyphs(a,end)
rect=(min(min(i['x0'] for i in images),min(c['x0'] for c in fc))-3,
      min(i['top'] for i in images)-3,
      max(max(i['x1'] for i in images),max(c['x1'] for c in fc))+3,
      max(c['bottom'] for c in fc)+3)
annotate({pnum:[rect]}, 'New comparison figure, including the existing AI-system series and the new comparison-system series.')
audit.append({'kind':'figure','name':'Figure 3','pdf_pages':[pnum+1]})

# The only new subsection heading, including its entry in the contents.
heading='5.4.5 Same-size comparison system: Cisco, Booking and Mondelez'
needle=normalize(heading)
at=TEXT.find(needle)
while at>=0:
    rects=line_rects(glyphs(at,at+len(needle)))
    annotate(rects,'New subsection for the reviewer-requested same-size comparison.')
    audit.append({'kind':'heading','name':'Section 5.4.5','pdf_pages':[p+1 for p in rects]})
    at=TEXT.find(needle,at+1)

writer.write(OUT)
(TMP/'highlight_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
(TMP/'extracted_pages.json').write_text(json.dumps([p.extract_text() for p in reader.pages],ensure_ascii=False))
for item in audit:
    print(item['kind'],item.get('number',item.get('name')),item['pdf_pages'],item.get('text','')[:90])
print('OUTPUT',OUT)
