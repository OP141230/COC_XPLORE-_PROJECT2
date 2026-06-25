"""
generate_proposal.py — builds the FantasyXI GSoC-style proposal PDF
using reportlab, matching the structure of the sample EXXA proposal:
mentors/contact, about me, project description, objectives, deliverables,
implementation + architecture diagrams, timeline table, tasks, references.
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    ListFlowable, ListItem, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

PRIMARY = colors.HexColor("#1B3A6B")
ACCENT = colors.HexColor("#2E7D32")
LIGHTBG = colors.HexColor("#F2F6FA")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("TitleBig", parent=styles["Title"], fontSize=24, textColor=PRIMARY, spaceAfter=4))
styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12, textColor=colors.grey,
                           alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle("H1", parent=styles["Heading1"], textColor=PRIMARY, fontSize=15,
                           spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle("H2", parent=styles["Heading2"], textColor=ACCENT, fontSize=12.5,
                           spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.3, leading=14.5, spaceAfter=6))
styles.add(ParagraphStyle("BodyBold", parent=styles["Body"], fontName="Helvetica-Bold"))


story = []

def H1(text): story.append(Paragraph(text, styles["H1"])); story.append(HRFlowable(width="100%", color=LIGHTBG, thickness=1))
def H2(text): story.append(Paragraph(text, styles["H2"]))
def P(text): story.append(Paragraph(text, styles["Body"]))
def BULLETS(items):
    story.append(ListFlowable([ListItem(Paragraph(i, styles["Body"])) for i in items],
                               bulletType="bullet", leftIndent=16))

# ---------- Cover ----------
story.append(Spacer(1, 40))
story.append(Paragraph("FantasyXI", styles["TitleBig"]))
story.append(Paragraph("An Intelligent Decision-Support Framework for Fantasy Premier League", styles["Subtitle"]))
story.append(Paragraph("Google Summer of Code 2026", styles["Subtitle"]))
story.append(Spacer(1, 10))

mentor_data = [["Mentors"], ["Ojas Alai"], ["Kavish Nasta"]]
t = Table(mentor_data, colWidths=[3.2*inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), PRIMARY),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("BACKGROUND", (0,1), (-1,-1), LIGHTBG),
    ("GRID", (0,0), (-1,-1), 0.5, colors.white),
    ("FONTSIZE", (0,0), (-1,-1), 10),
    ("TOPPADDING", (0,0), (-1,-1), 6),
    ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
]))
story.append(t)
story.append(Spacer(1, 16))

H2("Contact Information")
contact_data = [
    ["Full Name", "Anushree Ashish Upasham"],
    ["Degree & Year", "B.Tech Computer Engineering"],
    ["University", "Veermata Jijabai Technological Institute, Mumbai"],
    ["Location", "Mumbai, Maharashtra, India"],
    ["Time Zone", "Indian Standard Time (UTC+5:30)"],
    ["Project Size", "350 Hours (Large Project)"],
]
ct = Table(contact_data, colWidths=[1.7*inch, 4.3*inch])
ct.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#FCE4D6")),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D9D9D9")),
    ("FONTSIZE", (0,0), (-1,-1), 9.5),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(ct)
story.append(PageBreak())

# ---------- About Me ----------
H1("About Me")
P("I am an undergraduate student passionate about combining machine learning with "
  "competitive, data-rich domains. Fantasy Premier League is exactly that: millions of "
  "managers making sequential decisions under uncertainty, budget constraints, and direct "
  "competition with each other. I am drawn to FantasyXI because it lets me apply deep learning, "
  "game theory, and reinforcement learning together to a single, well-defined competitive system, "
  "rather than treating each technique in isolation.")
H2("Relevant Background")
BULLETS([
    "Strong foundation in Data Structures, Algorithms, Probability &amp; Statistics, Artificial Intelligence, "
    "and Database Management Systems.",
    "Hands-on machine learning project experience, including time-series and sequence models.",
    "Active open-source contributor with a track record of shipping complete, documented pipelines.",
    "Comfortable with PyTorch, scikit-learn, and optimisation libraries (PuLP) used throughout this proposal.",
])
H2("Availability")
P("I can commit 25–30 hours per week for the duration of the program, with the ability to extend "
  "during my summer vacation window. I will provide biweekly blog updates, attend all mentor check-ins, "
  "and flag any reduced-availability periods (e.g. examinations) well in advance so milestones stay on track.")

# ---------- Project Description ----------
H1("1. Project Description")
H2("1.1 About")
P("Fantasy Premier League (FPL) is one of the world's most popular fantasy sports platforms, where "
  "managers continuously optimise squad selection, transfers, and captaincy under a fixed budget and "
  "transfer-cost rules, while competing against millions of other managers. Existing tools largely "
  "rely on short-horizon historical statistics and do not explicitly model the long-term, sequential, "
  "and competitive nature of the game. FantasyXI proposes a unified decision-support framework that "
  "combines <b>deep learning</b> for performance forecasting, <b>game theory</b> for ownership-aware, "
  "rank-sensitive recommendations, and <b>reinforcement learning</b> for sequential transfer and "
  "captaincy planning across a full season.")

H2("1.2 Objectives")
BULLETS([
    "<b>O1.</b> Build a reliable historical + live data pipeline from the official FPL API covering player "
    "statistics, fixtures, ownership, and price changes.",
    "<b>O2.</b> Train a BiLSTM/Attention deep learning model that forecasts each player's expected points "
    "over the next 1–5 gameweeks from time-series performance features.",
    "<b>O3.</b> Develop a game-theoretic scoring layer that converts raw point forecasts and ownership data "
    "into rank-aware recommendations — distinguishing 'template' (rank-protecting) from 'differential' "
    "(rank-gaining) picks.",
    "<b>O4.</b> Formulate transfers and captaincy as a Markov Decision Process and train a reinforcement "
    "learning agent that optimises cumulative season performance rather than greedy single-gameweek gains.",
    "<b>O5.</b> Combine all three components into a single, reproducible, open-source FPL management "
    "assistant validated against real historical seasons.",
])

H2("1.3 Deliverables")
deliv = [
    ["ID", "Deliverable"],
    ["D1", "Automated FPL data pipeline producing clean, versioned per-gameweek player time series."],
    ["D2", "BiLSTM + Attention forecasting model predicting next-gameweek and multi-gameweek expected points, "
           "benchmarked by R² / MAE against the FPL API's own predicted points and the OpenFPL baseline."],
    ["D3", "Game-theoretic recommendation engine producing a ranked list of template vs. differential picks "
           "for any risk-tolerance setting."],
    ["D4", "RL transfer/captaincy agent (DQN) trained on simulated seasons, benchmarked against a greedy "
           "points-maximising baseline over a full 38-gameweek season."],
    ["D5", "ILP-based squad optimizer enforcing official FPL rules (budget, club limits, formation), used as "
           "the action filter for both manual recommendations and the RL agent."],
    ["D6", "Backtested validation report comparing FantasyXI's recommendations against actual outcomes from "
           "a completed historical FPL season."],
]
dt = Table(deliv, colWidths=[0.5*inch, 5.5*inch])
dt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), PRIMARY),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHTBG]),
    ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#D9D9D9")),
    ("FONTSIZE", (0,0), (-1,-1), 9.3),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]))
story.append(dt)
story.append(PageBreak())

# ---------- Implementation ----------
H1("2. Implementation")

H2("2.1 Data Collection &amp; Feature Engineering")
P("Data is pulled from the official FPL public API (<i>bootstrap-static</i>, <i>fixtures</i>, and "
  "<i>element-summary</i> endpoints), producing per-player, per-gameweek records of minutes, goals, assists, "
  "Expected Goals (xG), Expected Assists (xA), Expected Goal Involvements (xGI), Expected Goals Conceded "
  "(xGC), ICT index, bonus points, ownership percentage, and price. Rolling 3- and 5-gameweek form windows, "
  "ownership momentum, and price-change features are engineered, and the series are reshaped into sliding "
  "windows for sequence modelling.")

H2("2.2 Deep Learning Forecaster — BiLSTM + Attention")
P("A 2-layer Bidirectional LSTM encodes the last 5 gameweeks of each player's feature vector, with an "
  "additive attention layer over the hidden states so the model can weight the most informative recent "
  "gameweeks (e.g. a sudden injury return or a favourable fixture run) more heavily than older ones. A "
  "small dense head maps the attended context vector to a single expected-points prediction. The model "
  "is trained separately for short (1 GW) and medium (5 GW) horizons, and evaluated using R² and MAE per "
  "position, mirroring the evaluation style used in prior open FPL forecasting work such as OpenFPL.")

H2("2.3 Game Theory — Differential Ownership &amp; Rank-Gain Scoring")
P("Raw expected points alone do not capture what actually moves a manager's rank: a highly-owned premium "
  "player protects rank but rarely gains it, while a low-ownership differential can swing rank "
  "significantly if it outperforms. FantasyXI scores each player using an ownership-weighted expected "
  "rank-gain function:")
P("<b>rank_gain_score(p) = (EV(p) − field_EV) × (1 − ownership(p)) − λ × √variance(p) × (1 − ownership(p))</b>")
P("where <i>λ</i> is a user-controlled risk-tolerance parameter: λ→0 favours rank-protecting template picks, "
  "while λ→1 favours high-upside differentials. This directly operationalises the findings of "
  "'Differential Ownership and Fantasy Football Strategy' and 'Competing with Humans at Fantasy Football' "
  "into an actionable scoring layer.")

H2("2.4 Reinforcement Learning — Sequential Transfer &amp; Captaincy Planning")
P("Transfers and captaincy choices are modelled as a Markov Decision Process: the <b>state</b> captures the "
  "current squad's short-horizon forecasted EV, bank balance, free transfers, and chip availability; the "
  "<b>action</b> space covers hold / single-transfer / point-hit double-transfer decisions (filtered through "
  "the ILP squad optimizer so every action stays rule-compliant); the <b>reward</b> is realised gameweek "
  "points minus any transfer-hit penalty. A Deep Q-Network is trained on simulated seasons built from the "
  "forecaster's predicted EV, learning to trade short-term points for long-term squad value rather than "
  "greedily maximising every single gameweek.")

H2("2.5 Squad Optimization")
P("An Integer Linear Program (solved with PuLP / CBC) enforces the official FPL constraints — 15-man squad "
  "(2 GK / 5 DEF / 5 MID / 3 FWD), ≤£100m budget, maximum 3 players per real-world club, and a valid "
  "starting-XI formation — while maximising total expected points of the starting XI plus a doubled "
  "contribution from the optimal captain. This module is shared between the stand-alone squad-builder tool "
  "and the RL agent's action filter.")

story.append(PageBreak())

# Architecture diagram (drawn as a flow using a simple table-grid replicate of style)
H2("2.6 System Architecture")
arch_rows = [
    ["FPL Official API\n(bootstrap-static, fixtures, element-summary)"],
    ["↓"],
    ["Data Collection & Feature Engineering\n(rolling form, xG/xA/xGI, FDR, ownership Δ, price Δ)"],
    ["↓"],
    ["BiLSTM + Attention Forecaster\n→ per-player expected points (1–5 GW horizon)"],
    ["↓                                              ↓"],
    ["Game Theory Module\n(ownership-aware rank-gain scoring)        ILP Squad Optimizer\n(budget / club / formation rules)"],
    ["↓                                              ↓"],
    ["Reinforcement Learning Agent (DQN)\nSequential transfer + captaincy policy across 38 gameweeks"],
    ["↓"],
    ["Recommendations: Squad · Transfers · Captain · Chip timing"],
    ["↓"],
    ["Validation: Backtest vs. completed historical season + greedy baseline"],
]
at = Table([[r[0]] for r in arch_rows], colWidths=[5.8*inch])
style_cmds = [
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("TOPPADDING", (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
]
for i, r in enumerate(arch_rows):
    if r[0] == "↓" or "↓ " in r[0]:
        style_cmds.append(("FONTSIZE", (0,i), (-1,i), 11))
        style_cmds.append(("TEXTCOLOR", (0,i), (-1,i), PRIMARY))
    else:
        style_cmds.append(("BACKGROUND", (0,i), (-1,i), LIGHTBG))
        style_cmds.append(("BOX", (0,i), (-1,i), 0.7, PRIMARY))
at.setStyle(TableStyle(style_cmds))
story.append(at)
story.append(Spacer(1, 10))
P("<i>Figure 1: End-to-end FantasyXI pipeline — data flows top to bottom from raw API ingestion through "
  "forecasting, game-theoretic and rule-based filtering, into the RL policy that issues the final "
  "weekly recommendation.</i>")

story.append(PageBreak())

# ---------- Timeline ----------
H1("3. Project Timeline (350 Hours)")
timeline = [
    ["Phase", "Duration", "Key Tasks", "Milestone"],
    ["Community Bonding", "~2 weeks",
     "Study FPL API & OpenFPL baseline; review prior FPL ML literature; set up repo and dev environment; "
     "confirm data scope and evaluation plan with mentors.",
     "Dev environment ready; data plan confirmed."],
    ["Phase 1: Data Pipeline", "3 weeks",
     "Build FPL API collector; engineer rolling-form and fixture-difficulty features; construct sliding-"
     "window sequences; run EDA.",
     "D1: Labeled, versioned dataset complete."],
    ["Phase 2: Forecasting Model", "4 weeks",
     "Implement BiLSTM+Attention model; train 1-GW and 5-GW horizon variants; evaluate R²/MAE per position; "
     "benchmark vs. FPL's own predicted points and OpenFPL.",
     "D2: Forecaster trained & benchmarked. MIDTERM EVALUATION."],
    ["Phase 3: Game Theory Module", "3 weeks",
     "Implement ownership-aware rank-gain scoring; calibrate risk-tolerance parameter; backtest template "
     "vs. differential picks on a past season.",
     "D3: Recommendation engine validated."],
    ["Phase 4: Squad Optimizer", "2 weeks",
     "Implement ILP squad/formation optimizer with PuLP; integrate with forecaster output; validate against "
     "real squad-building constraints.",
     "D5: Rule-compliant optimizer integrated."],
    ["Phase 5: RL Agent", "4 weeks",
     "Build season simulation environment; implement DQN agent for transfer/captaincy; train and tune "
     "against simulated seasons; compare to greedy baseline.",
     "D4: RL agent trained & benchmarked."],
    ["Phase 6: Backtesting & Validation", "3 weeks",
     "Run full pipeline on a completed historical FPL season; compare recommended squads/transfers to "
     "actual outcomes and top-rank human managers.",
     "D6: Validation report complete."],
    ["Phase 7: Final Report", "~2 weeks",
     "Clean and document all code; write final report and demo notebooks; prepare blog post; address "
     "mentor feedback; submit final evaluation.",
     "Final submission."],
]
tt = Table(timeline, colWidths=[1.1*inch, 0.7*inch, 3.1*inch, 1.1*inch])
tt.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), PRIMARY),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHTBG]),
    ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#D9D9D9")),
    ("FONTSIZE", (0,0), (-1,-1), 8),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
]))
story.append(tt)
story.append(PageBreak())

# ---------- Evaluation & Benchmarks ----------
H1("4. Evaluation &amp; Benchmarks")
BULLETS([
    "<b>Forecasting:</b> R² and MAE per position and per horizon (1 GW vs 5 GW), benchmarked against the "
    "FPL API's own predicted-points field and the open-source OpenFPL method.",
    "<b>Game theory:</b> backtested rank-gain realised by template vs. differential recommendations across "
    "a completed historical season, at multiple risk-tolerance (λ) settings.",
    "<b>RL agent:</b> total simulated season points and transfer-hit efficiency vs. a greedy "
    "points-maximising baseline with no foresight, over a full 38-gameweek simulated season.",
    "<b>End-to-end:</b> final squad/transfer recommendations compared against actual high-rank human "
    "manager decisions for the same historical season.",
])

H1("5. My Motivation")
P("Machine learning and football analytics are two of my long-standing interests, and FPL is a uniquely "
  "rich environment to combine them: it is a real, large-scale, competitive system with public data, a "
  "well-defined reward signal (points and rank), and genuinely hard sequential decision-making. Building "
  "FantasyXI gives me the opportunity to go beyond a single point-prediction model and build a complete, "
  "reproducible, open-source decision-support system — one that explicitly reasons about uncertainty, "
  "competition, and the long-term consequences of every transfer and captaincy choice.")

# ---------- References ----------
H1("6. References")
refs = [
    "Fantasy Premier League Official Website &amp; API — https://fantasy.premierleague.com/",
    "OpenFPL: An Open-Source Forecasting Method Rivaling State-of-the-Art Fantasy Premier League Services.",
    "Time Series Modeling for Dream Team in Fantasy Premier League.",
    "Differential Ownership and Fantasy Football Strategy.",
    "Competing with Humans at Fantasy Football: Team Formation and Management.",
]
BULLETS(refs)

doc = SimpleDocTemplate(
    "/home/claude/fantasyxi/docs/FantasyXI_GSoC_Proposal.pdf",
    pagesize=letter,
    topMargin=0.6*inch, bottomMargin=0.6*inch, leftMargin=0.7*inch, rightMargin=0.7*inch,
    title="FantasyXI - GSoC 2026 Proposal",
)
doc.build(story)
print("PDF built successfully.")
