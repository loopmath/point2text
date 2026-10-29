"""Build the four-page technical report from its readable Markdown source.

Run: uv run --extra paper python docs/paper/build.py
The layout is conference-inspired, not an official NeurIPS submission template.
"""

from __future__ import annotations

import argparse
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re

from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
    Preformatted, Flowable, HRFlowable,
)

HERE = Path(__file__).resolve().parent
WIDTH = 396


def rich(text):
    value = escape(text)
    value = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(https?://[^\s]+)", r'<link href="\1" color="#244f67">\1</link>', value)
    return value


class MapFigure(Flowable):
    def __init__(self):
        Flowable.__init__(self)
        self.width, self.height = WIDTH, 138
        self.data = json.loads((HERE / "observations.json").read_text())

    def draw(self):
        c = self.canv
        x0, y0, w, h = 28, 25, 233, 111
        all_points = self.data["points"] + self.data["candidates"]
        xs = [p["x"] for p in all_points] + [self.data["target"][0]]
        ys = [p["y"] for p in all_points] + [self.data["target"][1]]
        scale = min(w / (max(xs)-min(xs)), h / (max(ys)-min(ys))) * .86
        mx, my = (max(xs)+min(xs))/2, (max(ys)+min(ys))/2
        project = lambda x,y: (x0+w/2+(x-mx)*scale, y0+h/2+(y-my)*scale)
        c.setStrokeColor(colors.HexColor("#d5d5d5")); c.setLineWidth(.5)
        c.rect(x0,y0,w,h)
        c.setFillColor(colors.HexColor("#959da0"))
        for p in self.data["points"]:
            x,y = project(p["x"],p["y"]); c.circle(x,y,1.6,fill=1,stroke=0)
        tx,ty=project(*self.data["target"])
        c.setStrokeColor(colors.HexColor("#bd633d"));c.setLineWidth(.7)
        for p in self.data["candidates"]:
            x,y=project(p["x"],p["y"]);c.line(tx,ty,x,y)
            c.setFillColor(colors.HexColor("#922e2e"));c.circle(x,y,2.4,fill=1,stroke=0)
        c.setStrokeColor(colors.HexColor("#111111"));c.circle(tx,ty,4,fill=0,stroke=1)
        c.line(tx-6,ty,tx+6,ty);c.line(tx,ty-6,tx,ty+6)
        c.setFillColor(colors.black);c.setFont("Times-Roman",8)
        c.drawCentredString(x0+w/2,12,"PC1")
        c.saveState();c.translate(12,y0+h/2);c.rotate(90);c.drawCentredString(0,0,"PC2");c.restoreState()
        c.setFont("Times-Bold",9);c.drawString(276,125,"Actual-text verification")
        c.setFont("Times-Roman",8.5)
        for n,label in enumerate(["Gray: corpus paragraphs","Crosshair: target","Red: generated candidates","Lines: remaining error"]):
            c.drawString(276,108-13*n,label)
        c.drawString(276,42,"Distance is measured in")
        c.drawString(276,30,"fixed PCA coordinates.")


def equation(text):
    text = text.strip().strip("$").strip()
    text = text.replace(r"\operatorname*{argmin}", r"\mathrm{argmin}")
    text = text.replace(r"\mathsf T", r"\mathsf{T}")
    buffer=BytesIO()
    mathtext.math_to_image(f"${text}$", buffer, format="png", dpi=350,
                          prop=FontProperties(size=11, math_fontfamily="stix"))
    buffer.seek(0)
    image=PILImage.open(buffer);w,h=image.size
    factor=min(72/350, WIDTH/w)
    buffer.seek(0)
    return Image(buffer,width=w*factor,height=h*factor,hAlign="CENTER")


def build(output):
    body = ParagraphStyle("Body",fontName="Times-Roman",fontSize=10,leading=12,
                          alignment=TA_JUSTIFY,spaceAfter=6)
    heading=ParagraphStyle("Heading",parent=body,fontName="Times-Bold",fontSize=11,
                           leading=13,spaceBefore=8,spaceAfter=6,keepWithNext=True,
                           alignment=TA_LEFT)
    abstract=ParagraphStyle("Abstract",parent=body,fontSize=9.5,leading=11.5,
                            leftIndent=15,rightIndent=15,spaceAfter=8)
    reference=ParagraphStyle("Reference",parent=body,fontSize=8.1,leading=9.3,
                             spaceAfter=4,alignment=TA_LEFT)
    caption=ParagraphStyle("Caption",parent=body,fontSize=8.5,leading=10,spaceAfter=7)
    code=ParagraphStyle("Code",fontName="Courier",fontSize=8.0,leading=10,
                        leftIndent=9,spaceBefore=6,spaceAfter=7)
    story=[]
    pages=(HERE/"paper.md").read_text().split("<!-- page -->")
    if len(pages)!=4: raise ValueError("Source must define exactly four pages")
    for page_number,page in enumerate(pages):
        if page_number: story.append(PageBreak())
        lines=page.strip().splitlines();i=0;in_abstract=False;in_refs=False
        while i<len(lines):
            line=lines[i].strip()
            if not line: i+=1;continue
            if line.startswith("# "):
                story.append(HRFlowable(width=WIDTH,thickness=1.5,color=colors.black,spaceAfter=10))
                title=ParagraphStyle("Title",fontName="Times-Bold",fontSize=16,leading=19,
                                     alignment=TA_CENTER,spaceAfter=9)
                story.append(Paragraph(rich(line[2:]),title));i+=1;continue
            if line.startswith("@authors ") or line.startswith("@date "):
                value=line.split(" ",1)[1]
                style=ParagraphStyle("Meta",parent=body,alignment=TA_CENTER,fontSize=9,
                                     leading=11,spaceAfter=3)
                story.append(Paragraph(rich(value),style));i+=1;continue
            if line.startswith("## "):
                text=line[3:];in_abstract=text=="Abstract";in_refs=text=="References"
                style=heading
                if in_abstract:
                    story.append(Spacer(1,6))
                    style=ParagraphStyle("AbstractHeading",parent=heading,alignment=TA_CENTER)
                story.append(Paragraph(rich(text),style));i+=1;continue
            if line.startswith("$$"):
                story.extend([Spacer(1,2),equation(line),Spacer(1,7)]);i+=1;continue
            if line.startswith("```"):
                content=[];i+=1
                while i<len(lines) and not lines[i].startswith("```"):
                    content.append(lines[i]);i+=1
                story.append(Preformatted("\n".join(content),code));i+=1;continue
            if line.startswith("|"):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith("|"):
                    cells=[s.strip() for s in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r"[-: ]+",s) for s in cells): rows.append(cells)
                    i+=1
                cell_style=ParagraphStyle("Cell",parent=body,fontSize=8.8,leading=10.5,spaceAfter=0,alignment=TA_LEFT)
                rows=[[Paragraph(rich(t),cell_style) for t in row] for row in rows]
                table=Table(rows,colWidths=[159,237],hAlign="LEFT")
                table.setStyle(TableStyle([("LINEABOVE",(0,0),(-1,0),.7,colors.black),
                                           ("LINEBELOW",(0,0),(-1,0),.5,colors.black),
                                           ("LINEBELOW",(0,-1),(-1,-1),.7,colors.black),
                                           ("VALIGN",(0,0),(-1,-1),"TOP"),
                                           ("LEFTPADDING",(0,0),(-1,-1),4),
                                           ("RIGHTPADDING",(0,0),(-1,-1),4),
                                           ("TOPPADDING",(0,0),(-1,-1),3),
                                           ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
                story.extend([Spacer(1,3),table,Spacer(1,6)]);continue
            if line=="[FIGURE:map]":
                story.append(MapFigure())
                story.append(Paragraph("<b>Figure 1.</b> The fixed corpus map and three actual generated positions from the demonstration. Most embedding variance lies outside this two-dimensional view.",caption))
                i+=1;continue
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip():
                paragraph.append(lines[i].strip());i+=1
            story.append(Paragraph(rich(" ".join(paragraph)),reference if in_refs else abstract if in_abstract else body))

    def footer(canvas,doc):
        canvas.setTitle("point2text: Local Text Generation Toward a Point in Semantic Space")
        canvas.setAuthor("point2text contributors - loopmath")
        canvas.setFont("Times-Roman",8);canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(108,35,"point2text | Technical report | 29 October 2026")
        canvas.drawRightString(504,35,str(doc.page))

    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=letter,leftMargin=102,rightMargin=102,
                          topMargin=48,bottomMargin=49,author="point2text contributors - loopmath")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    count=len(PdfReader(output).pages)
    if count!=4: raise RuntimeError(f"Expected 4 pages, rendered {count}; revise source/layout")
    print(f"Built {output}: {count} pages")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"point2text-technical-report.pdf")
    build(parser.parse_args().output)
