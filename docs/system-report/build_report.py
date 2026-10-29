"""Build the six-page report; explicit frames reject overflow instead of hiding it.

Run from the repository root:
    uv run --extra paper python docs/system-report/build_report.py
"""
from __future__ import annotations

import argparse
from html import escape
from io import BytesIO
from pathlib import Path
import re

from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    Flowable, Frame, Image, Paragraph, Preformatted, Spacer, Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
PAGE_W, PAGE_H = 612, 792
MARGIN, GAP = 43, 20
COL_W = (PAGE_W - 2 * MARGIN - GAP) / 2
INK = colors.HexColor('#162933')
ACCENT = colors.HexColor('#2b6477')
MUTED = colors.HexColor('#53636b')
BODY = ParagraphStyle('Body', fontName='Times-Roman', fontSize=10,
                      leading=12.2, alignment=TA_JUSTIFY, spaceAfter=6)
HEADING = ParagraphStyle('Heading', parent=BODY, fontName='Times-Bold', fontSize=11,
                         leading=13, alignment=TA_LEFT, spaceBefore=8, spaceAfter=5,
                         keepWithNext=True, textColor=INK)
SUBHEAD = ParagraphStyle('Subhead', parent=HEADING, fontSize=9.9,
                         leading=11.5, spaceBefore=4, spaceAfter=4)
CAPTION = ParagraphStyle('Caption', parent=BODY, fontSize=8.4, leading=10,
                        alignment=TA_LEFT, textColor=MUTED, spaceAfter=7)
REFERENCE = ParagraphStyle('Reference', parent=BODY, fontSize=8.3, leading=9.8,
                          alignment=TA_LEFT, spaceAfter=5)
CODE = ParagraphStyle('Code', fontName='Courier', fontSize=7.7, leading=9.5,
                     spaceBefore=4, spaceAfter=7, leftIndent=3)


def rich(text):
    out = escape(text)
    for old, new in [('q_i^0', 'q<sub>i</sub><super>0</super>'), ('q_i', 'q<sub>i</sub>'), ('S_i', 'S<sub>i</sub>'), ('a_v', 'a<sub>v</sub>'), ('u_1', 'u<sub>1</sub>'), ('u_L', 'u<sub>L</sub>'), ('delta_j', 'delta<sub>j</sub>')]:
        out = out.replace(old, new)
    out = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', out)
    out = re.sub(r'`([^`]+)`', r'<font name="Courier" size="8.2">\1</font>', out)
    out = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="#2b6477">\1</link>', out)
    return out


def equation(text):
    text = text.strip().strip('$').strip()
    buffer = BytesIO()
    mathtext.math_to_image('$' + text + '$', buffer, format='png', dpi=350,
                          prop=FontProperties(size=10.5, math_fontfamily='stix'))
    buffer.seek(0)
    im = PILImage.open(buffer)
    w, h = im.size
    factor = min(72 / 350, COL_W / w)
    buffer.seek(0)
    return Image(buffer, width=w * factor, height=h * factor, hAlign='CENTER')


class Architecture(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = COL_W, 196

    def draw(self):
        c = self.canv
        nodes = [(0, 154, COL_W, 33, ['Frozen corpus + canonical encoder', '384-D embeddings -> saved 2-D PCA']),
                 (0, 107, COL_W, 33, ['Clicked target + nearest examples', 'Fixed map, prompt policy and model IDs']),
                 (0, 58, 117, 35, ['Diffusion / Ollama', 'draft proposals']),
                 (136, 58, COL_W-136, 35, ['Text edit search', 'repair + combinations']),
                 (0, 8, COL_W, 35, ['Canonical verifier + optional quality gate', 'Best measured text -> browser / SQLite'])]
        for x, y, w, h, lines in nodes:
            c.setFillColor(colors.HexColor('#f1f5f6'))
            c.setStrokeColor(colors.HexColor('#9eb3bb'))
            c.roundRect(x, y, w, h, 3, fill=1, stroke=1)
            c.setFillColor(INK)
            c.setFont('Helvetica-Bold', 8.2)
            c.drawCentredString(x+w/2, y+h-12, lines[0])
            c.setFont('Helvetica', 7.6)
            c.drawCentredString(x+w/2, y+8, lines[1])
        c.setStrokeColor(ACCENT)
        for x1,y1,x2,y2 in [(COL_W/2,154,COL_W/2,140),(58,107,58,93),
                              (192,107,192,93),(58,58,58,43),(192,58,192,43)]:
            c.line(x1,y1,x2,y2)
            c.line(x2,y2,x2-2.5,y2+4)
            c.line(x2,y2,x2+2.5,y2+4)
        c.setFont('Helvetica', 7.2)
        c.setFillColor(ACCENT)
        c.drawCentredString(COL_W/2, 1, 'Canonical feedback preserves a verified incumbent')


class Displacements(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = COL_W, 142

    def draw(self):
        c = self.canv
        x0,y0=38,36
        c.setStrokeColor(colors.HexColor('#bdc9cc'))
        c.line(22,20,230,20);c.line(22,20,22,130)
        def arrow(a,b,color,dash=False):
            c.setStrokeColor(color);c.setLineWidth(1)
            c.setDash([3,2] if dash else [])
            c.line(*a,*b);c.setDash([])
            import math
            angle=math.atan2(b[1]-a[1],b[0]-a[0])
            for sign in (-1,1):
                c.line(*b,b[0]-5*math.cos(angle+sign*.45),b[1]-5*math.sin(angle+sign*.45))
        source=(x0,y0);a=(123,112);pred=(193,77);actual=(184,65);target=(201,76)
        arrow(source,a,ACCENT);arrow(a,pred,ACCENT)
        arrow(source,pred,colors.HexColor('#a16d32'),True)
        c.setFillColor(INK);c.circle(*source,2.5,fill=1,stroke=0)
        c.setFillColor(ACCENT);c.circle(*actual,3,fill=1,stroke=0)
        c.setStrokeColor(INK);c.circle(*target,4,fill=0,stroke=1)
        c.setFont('Helvetica',8)
        c.setFillColor(INK)
        c.drawString(29,25,'source');c.drawString(58,96,'measured edit 1')
        c.drawString(133,108,'measured edit 2')
        c.drawString(137,43,'actual combined text')
        c.drawString(204,79,'target')
        c.drawString(87,3,'Schematic only; no measured coordinates')


def parse(text):
    lines=text.strip().splitlines(); story=[];i=0;refs=False
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line.startswith('# '): i+=1;continue
        if line.startswith('## '):
            refs=line[3:]=='References'
            story.append(Paragraph(rich(line[3:]),HEADING));i+=1;continue
        if line.startswith('### '):
            story.append(Paragraph(rich(line[4:]),SUBHEAD));i+=1;continue
        if line.startswith('$$'):
            story.extend([Spacer(1,2),equation(line),Spacer(1,7)]);i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]);i+=1
            story.append(Preformatted('\n'.join(code),CODE));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[t.strip() for t in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[-: ]+',t) for t in cells):rows.append(cells)
                i+=1
            count=len(rows[0])
            widths=[COL_W*.33,COL_W*.67] if count==2 else [COL_W*.43,COL_W*.28,COL_W*.29]
            cell=ParagraphStyle('Cell',parent=BODY,fontSize=8.4,leading=10,spaceAfter=0,alignment=TA_LEFT)
            rendered=[[Paragraph(rich(t),cell) for t in row] for row in rows]
            table=Table(rendered,colWidths=widths,hAlign='LEFT')
            table.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,INK),
                ('LINEBELOW',(0,0),(-1,0),.5,INK),('LINEBELOW',(0,-1),(-1,-1),.7,INK),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef3f5')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),
                ('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),
                ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([Spacer(1,3),table,Spacer(1,7)]);continue
        if line=='[FIGURE:architecture]':story.append(Architecture());i+=1;continue
        if line=='[FIGURE:displacements]':story.append(Displacements());i+=1;continue
        if line.startswith('@caption '):
            story.append(Paragraph(rich(line[9:]),CAPTION));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip():
            para.append(lines[i].strip());i+=1
        story.append(Paragraph(rich(' '.join(para)),REFERENCE if refs else BODY))
    return story


def build(source,output):
    text=source.read_text(encoding='utf-8')
    pages=text.split('<!-- page -->')
    if len(pages)!=6:raise ValueError('Exactly six explicit source pages required')
    output.parent.mkdir(parents=True,exist_ok=True)
    c=Canvas(str(output),pagesize=(PAGE_W,PAGE_H),pageCompression=1)
    c.setTitle('point2text: Frozen-Map Text Optimization - System Technical Report')
    c.setAuthor('point2text contributors - loopmath')
    c.setCreator('point2text docs/system-report/build_report.py')
    for n,page in enumerate(pages,1):
        cols=page.split('<!-- column -->')
        if len(cols)!=2:raise ValueError(f'Page {n} must have two explicit columns')
        top=744
        c.setFillColor(MUTED);c.setFont('Helvetica',8)
        c.drawString(MARGIN,766,'point2text / SYSTEM TECHNICAL REPORT')
        c.drawRightString(PAGE_W-MARGIN,766,'29 October 2026')
        c.setStrokeColor(INK);c.setLineWidth(.7);c.line(MARGIN,756,PAGE_W-MARGIN,756)
        if n==1:
            c.setFillColor(INK);c.setFont('Times-Bold',20)
            c.drawString(MARGIN,728,'Text Optimization on a Frozen Semantic Map')
            c.setFont('Times-Roman',11)
            c.drawString(MARGIN,707,'Architecture, DiffusionGemma guidance, canonical refinement and evidence')
            c.setFont('Helvetica',8.5);c.setFillColor(MUTED)
            c.drawString(MARGIN,688,'point2text contributors - loopmath  |  Experimental MIT-licensed implementation')
            top=674
        for k,col in enumerate(cols):
            story=parse(col)
            frame=Frame(MARGIN+k*(COL_W+GAP),43,COL_W,top-43,
                        leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,
                        id=f'p{n}c{k+1}')
            frame.addFromList(story,c)
            if story:
                preview = getattr(story[0], 'text', type(story[0]).__name__)[:100]
                raise RuntimeError(f'Page {n}, column {k+1} overflow: {len(story)} flowables; first={preview}')
            print(f'Page {n} column {k+1}: used {top-frame._y:.1f}/{top-43:.1f} pt')
        c.setFont('Helvetica',7.6);c.setFillColor(MUTED)
        c.drawString(MARGIN,25,'Frozen weights. Canonical text scores. Quality and distance reported separately.')
        c.drawRightString(PAGE_W-MARGIN,25,f'{n} / 6')
        c.showPage()
    c.save()
    reader=PdfReader(output)
    if len(reader.pages)!=6:raise RuntimeError('Six-page contract failed')
    extracted='\n'.join(p.extract_text() for p in reader.pages)
    for forbidden in ('/Users/','PLACEHOLDER','TODO','TBD'):
        if forbidden in extracted:raise RuntimeError(f'Forbidden artifact text: {forbidden}')
    print(f'Built {output}: 6 pages, {len(extracted.split())} extracted words')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=HERE/'report.md')
    parser.add_argument('--output',type=Path,default=HERE/'point2text-system-report.pdf')
    args=parser.parse_args()
    build(args.source,args.output)
