# -*- coding: utf-8 -*-
"""Builds the BT4222 final presentation (.pptx)."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

BASE=Path("/Users/steff/Documents/GitHub/bt4212_finalProj/Final Project")
FIG=BASE/"figures"; SL=BASE/"slides"; SL.mkdir(exist_ok=True)
prs=Presentation(); prs.slide_width=Inches(13.333); prs.slide_height=Inches(7.5)
BLANK=prs.slide_layouts[6]
NAVY=RGBColor(0x1F,0x3A,0x5F); BLUE=RGBColor(0x2B,0x7B,0xBA); DARK=RGBColor(0x22,0x22,0x22)

def slide(title):
    s=prs.slides.add_slide(BLANK)
    b=s.shapes.add_textbox(Inches(0.5),Inches(0.25),Inches(12.3),Inches(0.9)); tf=b.text_frame
    tf.text=title; p=tf.paragraphs[0]; p.runs[0].font.size=Pt(30); p.runs[0].font.bold=True
    p.runs[0].font.color.rgb=NAVY
    return s
def bullets(s,items,left=0.6,top=1.4,width=7.2,height=5.6,size=18):
    tb=s.shapes.add_textbox(Inches(left),Inches(top),Inches(width),Inches(height)); tf=tb.text_frame
    tf.word_wrap=True
    for i,(txt,lvl) in enumerate(items):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        p.text=("• " if lvl==0 else "– ")+txt; p.level=lvl
        p.font.size=Pt(size if lvl==0 else size-3); p.font.color.rgb=DARK
        p.space_after=Pt(6)
    return tb
def pic(s,name,left,top,width):
    if (FIG/name).exists(): s.shapes.add_picture(str(FIG/name),Inches(left),Inches(top),width=Inches(width))

# 1 title
s=prs.slides.add_slide(BLANK)
t=s.shapes.add_textbox(Inches(0.8),Inches(2.3),Inches(11.7),Inches(2)); tf=t.text_frame; tf.word_wrap=True
tf.text="Personalised Product Search on JDSearch"; tf.paragraphs[0].runs[0].font.size=Pt(40)
tf.paragraphs[0].runs[0].font.bold=True; tf.paragraphs[0].runs[0].font.color.rgb=NAVY
p=tf.add_paragraph(); p.text="A Dual-Horizon Re-Ranker Driven by Log-Retention Dynamics"
p.font.size=Pt(24); p.font.color.rgb=BLUE
p=tf.add_paragraph(); p.text="BT4222 Final Project · Group 13 · Prof. Wang Qiuhong"; p.font.size=Pt(16); p.font.color.rgb=DARK

# 2 problem
s=slide("The Problem: Personalised Re-Ranking")
bullets(s,[("Business goal: shorten search→purchase by re-ranking each query's candidate slate",0),
 ("Input: current query + product metadata + user's multi-behaviour history",0),
 ("Label (ordinal): 0 none / 1 click / 2 cart / 3 purchase",0),
 ("Metrics: NDCG@10, MRR@200, Precision@1",0),
 ("173,831 users · 15.5M query–candidate pairs · 26.7M interactions",0),
 ("Hard: 97.9% of candidates are negative; 80.5% of products are cold",1)])

# 3 anomaly
s=slide("A Data Anomaly Worth Chasing")
bullets(s,[("Normal e-commerce funnel: clicks ≫ carts ≫ orders",0),
 ("JDSearch history: CART 43% > CLICK 31% > ORD 26%",0),
 ("Millions of orders & carts have NO clicks at all",0),
 ("Instructor's prompt: trace the timeline back in days — what happens to the mix?",1)],width=7.0)
pic(s,"fig_temporal_clickshare.png",7.6,1.5,5.4)

# 4 key insight
s=slide("Key Finding: a Telescoping Log-Retention Window")
bullets(s,[("Clicks retained ~7 days (day6: 520,760 → day7: 5)",0),
 ("Carts retained ~90 days (day85: 100,902 → day90: 365)",0),
 ("Orders retained ≥1 year",0),
 ("⇒ interaction TYPE is confounded with RECENCY, not intensity",0),
 ("The 'carts/orders > clicks' anomaly = long tail where only durable signals survive",1)],width=6.6,size=17)
pic(s,"fig_retention_cutoffs.png",7.3,1.6,5.7)

# 5 implications
s=slide("From Insight to Feature Engineering")
bullets(s,[("Do NOT weight ORD>CART>CLICK as intent — it just encodes old vs recent",0),
 ("Absence of old clicks is a logging artifact, not a negative",0),
 ("Distinct roles of fields:",0),
 ("Recent clicks/carts (≤7–90d) → short-term query INTENT",1),
 ("Old orders (≤1yr) → long-term brand/category PREFERENCE",1),
 ("Engineered: horizon-split category/brand affinity, recency-decay, re-engagement",0),
 ("Reality check: anonymised query↔name overlap alone ≈ random (0.087)",0)],size=17)

# 6 baseline gap
s=slide("Baseline Gap: Why CF Fails Here")
bullets(s,[("Each user = one record with an almost-always-new query",0),
 ("80.5% of products never appear in any history (14.5% warm on test)",0),
 ("NCF (ID-based) → NDCG@10 0.086 ≈ random (0.083)",0),
 ("SVD/NMF only reaches popularity level (0.274)",0),
 ("Gap = new-query generalisation under massive item cold-start",1)])

# 7 architecture
s=slide("Our Model: Dual-Horizon Attention Ranker (DHAR)")
bullets(s,[("Candidate category/brand embedding = attention query",0),
 ("Tower A: attends over RECENT (≤7d) history categories → short intent",0),
 ("Tower B: attends over ORDER-history categories → long preference",0),
 ("Fuse [cand ⊕ short ⊕ long ⊕ engineered feats] → MLP",0),
 ("Listwise softmax with ordinal gains [0,1,3,7]; trained on Apple MPS",0),
 ("Architecture = direct materialisation of the two retention horizons",1)],size=17)

# 8 results
s=slide("Results (held-out test slates)")
rows=[["Model","NDCG@10","MRR@200","P@1"],
 ["Random","0.083","0.102","0.033"],
 ["NCF (ID-CF)","0.086","0.104","0.032"],
 ["Popularity","0.270","0.283","0.167"],
 ["MatrixFact (SVD)","0.274","0.289","0.170"],
 ["LambdaMART","0.313","0.325","0.204"],
 ["DHAR (ours)","0.335","0.350","0.225"]]
tb=s.shapes.add_table(len(rows),4,Inches(0.6),Inches(1.5),Inches(6.1),Inches(4.6)).table
for r,row in enumerate(rows):
    for c,v in enumerate(row):
        cell=tb.cell(r,c); cell.text=v
        for p in cell.text_frame.paragraphs:
            p.runs[0].font.size=Pt(15); p.runs[0].font.bold=(r==0 or row[0]=="DHAR (ours)")
            if r==0: p.runs[0].font.color.rgb=RGBColor(0xFF,0xFF,0xFF)
pic(s,"fig_model_comparison.png",7.0,1.6,6.1)

# 9 diagnostics
s=slide("Diagnostics Validate the Hypothesis")
bullets(s,[("Rank of eventual positives (lower=better):",0),
 ("Purchase median 7  <  Cart 15  <  Click 26",1),
 ("Model ranks purchases best — they align with long-term order signal",1),
 ("Warm vs cold candidate: NDCG 0.586 vs 0.180",0),
 ("Cold-product ceiling is THE binding limitation",1),
 ("Ablation: temporal-aware features add +0.013 NDCG@10",0)],width=7.0,size=17)
pic(s,"fig_feature_importance.png",8.0,1.5,5.0)

# 10 conclusion
s=slide("Business Insights & Conclusion")
bullets(s,[("A data-collection artifact (retention windows) reshaped our entire model design",0),
 ("Separating short-term intent from long-term preference beats CF, popularity & GBM",0),
 ("Purchase-oriented ranking should use the long-horizon tower; CTR-oriented, the short one",0),
 ("Biggest lever = content representations for cold products (0.586→0.180 gap)",0),
 ("Reproducible pipeline scales from 20k-user laptop run to full data via one knob",0)],size=18)

out=SL/"BT4222_Group13_Final_Presentation.pptx"; prs.save(str(out)); print("saved",out)
