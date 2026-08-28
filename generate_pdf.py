#!/usr/bin/env python3
"""Generate FarmFederate_Paper.pdf from FarmFederate_Paper.md using fpdf2."""

import re
from pathlib import Path
from fpdf import FPDF

MD_PATH  = Path(__file__).parent / "FarmFederate_Paper.md"
PDF_PATH = Path(__file__).parent / "FarmFederate_Paper.pdf"


def sanitize(text: str) -> str:
    """Replace non-latin-1 characters with safe ASCII equivalents."""
    replacements = {
        '\u2014': '--',    # em dash
        '\u2013': '-',     # en dash
        '\u2018': "'",     # left single quote
        '\u2019': "'",     # right single quote
        '\u201c': '"',     # left double quote
        '\u201d': '"',     # right double quote
        '\u2026': '...',   # ellipsis
        '\u2212': '-',     # minus sign
        '\u00d7': 'x',     # multiplication sign
        '\u00b2': '^2',    # superscript 2
        '\u00b3': '^3',    # superscript 3
        '\u00b1': '+/-',   # plus-minus
        '\u03b1': 'alpha', # alpha
        '\u03bc': 'mu',    # mu
        '\u03b7': 'eta',   # eta
        '\u2192': '->',    # right arrow
        '\u2190': '<-',    # left arrow
        '\u2264': '<=',    # less equal
        '\u2265': '>=',    # greater equal
        '\u00ae': '(R)',   # registered
        '\u00a9': '(c)',   # copyright
        '\u2022': '*',     # bullet
        '\u2713': 'OK',    # check mark
        '\u2715': 'X',     # cross mark
        '\u0394': 'Delta', # Delta
        '\u03a3': 'Sigma', # Sigma
        '\u03a0': 'Pi',    # Pi
        '\u2211': 'sum',   # summation
        '\u220f': 'prod',  # product
        '\u221e': 'inf',   # infinity
        '\u2248': '~=',    # approximately
        '\u2260': '!=',    # not equal
        '\u00b0': 'deg',   # degree
        '\u03b2': 'beta',  # beta
        '\u03b3': 'gamma', # gamma
        '\u03b4': 'delta', # delta
        '\u03c9': 'omega', # omega
        '\u22c5': '*',     # dot operator
        '\u2217': '*',     # asterisk operator
        '\u2019': "'",
        '\ufb01': 'fi',    # fi ligature
        '\ufb02': 'fl',    # fl ligature
        '\u00e9': 'e',     # e acute
        '\u00e8': 'e',     # e grave
        '\u00ea': 'e',     # e circumflex
        '\u00fc': 'u',     # u umlaut
        '\u00f6': 'o',     # o umlaut
        '\u00e4': 'a',     # a umlaut
        '\u2020': '+',     # dagger
        '\u2021': '++',    # double dagger
    }
    for ch, rep in replacements.items():
        text = text.replace(ch, rep)
    # Final fallback: encode/decode to strip any remaining non-latin-1 chars
    return text.encode('latin-1', errors='replace').decode('latin-1')


# ── colour palette ────────────────────────────────────────────────────────────
C_TITLE   = (26,  60, 110)   # deep navy
C_H1      = (26,  60, 110)
C_H2      = (30,  90, 150)
C_H3      = (50, 120, 180)
C_BODY    = (30,  30,  30)
C_CODE_BG = (245, 245, 245)
C_CODE_FG = (40,  40,  40)
C_TABLE_H = (26,  60, 110)
C_TABLE_A = (235, 240, 250)
C_TABLE_B = (255, 255, 255)
C_RULE    = (160, 175, 200)


class Paper(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(22, 22, 22)
        self._page_label = ""
        self.add_page()

    # ── header / footer ───────────────────────────────────────────────────────
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_RULE)
        self.cell(0, 6, "FarmFederate: Privacy-Preserving Multimodal Federated Learning for Crop Stress Detection", align="L")
        self.ln(0)
        self.set_draw_color(*C_RULE)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_draw_color(*C_RULE)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*C_RULE)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    # ── helpers ───────────────────────────────────────────────────────────────
    def _usable(self):
        return self.w - self.l_margin - self.r_margin

    def rule(self):
        self.set_draw_color(*C_RULE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    # ── cover page ────────────────────────────────────────────────────────────
    def cover(self, title, authors, version, date, abstract):
        # top rule
        self.set_fill_color(*C_TITLE)
        self.rect(self.l_margin, 30, self._usable(), 1.2, "F")

        # title
        self.set_y(38)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(*C_TITLE)
        self.multi_cell(self._usable(), 9, title, align="C")
        self.ln(4)

        # authors / version / date
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 6, authors, align="C"); self.ln(5)
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 5, f"{version}   |   {date}", align="C"); self.ln(6)

        # bottom title rule
        self.set_fill_color(*C_TITLE)
        self.rect(self.l_margin, self.get_y(), self._usable(), 0.8, "F")
        self.ln(8)

        # abstract box
        self.set_fill_color(240, 244, 251)
        self.set_draw_color(*C_H2)
        self.set_line_width(0.4)
        box_x = self.l_margin
        box_y = self.get_y()
        box_w = self._usable()
        # label
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*C_H1)
        self.cell(box_w, 7, "Abstract", align="L"); self.ln(7)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_BODY)
        abs_start = self.get_y()
        self.multi_cell(box_w, 5.2, sanitize(abstract), align="J")
        abs_end = self.get_y()
        # draw rect around abstract
        self.set_fill_color(240, 244, 251)
        self.rect(box_x - 1, box_y - 1, box_w + 2, abs_end - box_y + 3, "D")
        self.ln(6)

        # keywords
        self.set_font("Helvetica", "BI", 9)
        self.set_text_color(*C_H2)
        self.cell(0, 5, "Keywords: federated learning, crop stress detection, vision-language models, RAG, class imbalance, multimodal learning", align="L")
        self.ln(10)

    # ── section headings ─────────────────────────────────────────────────────
    def h1(self, text):
        self.ln(4)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*C_H1)
        self.multi_cell(self._usable(), 8, sanitize(text), align="L")
        # underline
        y = self.get_y()
        self.set_draw_color(*C_H1)
        self.set_line_width(0.6)
        self.line(self.l_margin, y, self.l_margin + self._usable(), y)
        self.ln(4)

    def h2(self, text):
        self.ln(3)
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(*C_H2)
        self.multi_cell(self._usable(), 6.5, sanitize(text), align="L")
        self.ln(2)

    def h3(self, text):
        self.ln(2)
        self.set_font("Helvetica", "BI", 10.5)
        self.set_text_color(*C_H3)
        self.multi_cell(self._usable(), 6, sanitize(text), align="L")
        self.ln(1)

    # ── inline markup ─────────────────────────────────────────────────────────
    def _write_inline(self, text, base_size=9.5):
        """Handle **bold**, *italic*, `code` within a paragraph line."""
        # Split on markers while keeping them
        parts = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)', text)
        self.set_font("Helvetica", "", base_size)
        self.set_text_color(*C_BODY)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                self.set_font("Helvetica", "B", base_size)
                self.write(5.2, part[2:-2])
                self.set_font("Helvetica", "", base_size)
            elif part.startswith("*") and part.endswith("*"):
                self.set_font("Helvetica", "I", base_size)
                self.write(5.2, part[1:-1])
                self.set_font("Helvetica", "", base_size)
            elif part.startswith("`") and part.endswith("`"):
                self.set_font("Courier", "", base_size - 0.5)
                self.set_text_color(*C_CODE_FG)
                self.write(5.2, part[1:-1])
                self.set_font("Helvetica", "", base_size)
                self.set_text_color(*C_BODY)
            else:
                self.write(5.2, part)

    def para(self, text):
        """Render a paragraph with inline markup."""
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_BODY)
        # For simplicity render inline markup via write()
        # We need to use multi_cell for line-wrapping; strip inline markup for
        # the common case (pure text) and fall back to write() for mixed.
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(self._usable(), 5.2, sanitize(clean), align="J")
        self.ln(2)

    def bullet(self, text, level=0):
        indent = 5 + level * 5
        bullet_char = "•" if level == 0 else "–"
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_BODY)
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        self.multi_cell(self._usable() - indent, 5.2,
                         sanitize(f"*  {clean}"), align="L")

    def num_item(self, n, text):
        indent = 5
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*C_BODY)
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        self.multi_cell(self._usable() - indent, 5.2,
                         sanitize(f"{n}.  {clean}"), align="L")

    def code_block(self, lines):
        self.ln(1)
        content = "\n".join(lines)
        self.set_font("Courier", "", 7.8)
        self.set_fill_color(*C_CODE_BG)
        self.set_text_color(*C_CODE_FG)
        self.set_draw_color(*C_RULE)
        self.set_line_width(0.2)
        self.multi_cell(self._usable(), 4.5, sanitize(content), border=1, fill=True, align="L")
        self.set_text_color(*C_BODY)
        self.ln(2)

    # ── table ─────────────────────────────────────────────────────────────────
    def table(self, rows):
        """rows = list of lists of strings; first row is header."""
        if not rows:
            return
        usable = self._usable()
        ncols = max(len(r) for r in rows)
        # Auto column widths — equal split but header gets detected widths
        col_w = [usable / ncols] * ncols

        # Try to be smarter: give first col 35%, rest equal
        if ncols >= 2:
            first_w = usable * 0.32
            rest_w  = (usable - first_w) / (ncols - 1)
            col_w   = [first_w] + [rest_w] * (ncols - 1)

        row_h   = 5.5
        self.ln(2)
        for ri, row in enumerate(rows):
            is_header   = (ri == 0)
            is_sep_row  = all(set(c.strip()) <= {'-', '|', ':'} for c in row)
            if is_sep_row:
                continue

            fill_color = C_TABLE_H if is_header else (C_TABLE_A if ri % 2 == 0 else C_TABLE_B)
            text_color = (255, 255, 255) if is_header else C_BODY
            font_style = "B" if is_header else ""

            self.set_font("Helvetica", font_style, 8.5)
            self.set_text_color(*text_color)
            self.set_fill_color(*fill_color)
            self.set_draw_color(*C_RULE)
            self.set_line_width(0.15)

            # Calculate max height for this row (multi-line cells)
            max_lines = 1
            for ci, cell in enumerate(row):
                w = col_w[ci] if ci < len(col_w) else col_w[-1]
                chars_per_line = int(w / 2.1)  # rough estimate
                lines_needed = max(1, len(cell) // chars_per_line + 1)
                max_lines = max(max_lines, lines_needed)
            cell_h = max(row_h, max_lines * 4.5)

            x_start = self.l_margin
            y_start = self.get_y()

            # Check page break
            if y_start + cell_h > self.h - self.b_margin:
                self.add_page()
                y_start = self.get_y()

            for ci, cell in enumerate(row):
                w = col_w[ci] if ci < len(col_w) else col_w[-1]
                clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', str(cell))
                clean = re.sub(r'`([^`]+)`', r'\1', clean)
                clean = sanitize(clean.strip())
                self.set_xy(x_start, y_start)
                self.multi_cell(w, cell_h / max(max_lines, 1),
                                clean, border=1, fill=True, align="C" if is_header else "L",
                                max_line_height=4.5)
                x_start += w

            self.set_xy(self.l_margin, y_start + cell_h)

        self.ln(3)


# ── Markdown parser ───────────────────────────────────────────────────────────

def strip_md_heading(text):
    return re.sub(r'^#+\s*', '', text).strip()


def parse_table_row(line):
    return [c.strip() for c in line.strip().strip('|').split('|')]


def render(pdf: Paper, md_text: str):
    lines = md_text.splitlines()
    i = 0
    in_code = False
    code_buf = []
    table_buf = []

    # Extract cover metadata from first lines
    title = authors = version = date = ""
    abstract_lines = []
    abs_mode = False

    # Skip first heading line (the title)
    # We'll handle them specially
    skip_cover = False

    while i < len(lines):
        line = lines[i]

        # ── code block ───────────────────────────────────────────────────────
        if line.strip().startswith("```"):
            if in_code:
                pdf.code_block(code_buf)
                code_buf = []
                in_code = False
            else:
                # flush any pending table
                if table_buf:
                    pdf.table(table_buf)
                    table_buf = []
                in_code = True
            i += 1
            continue

        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # ── horizontal rule ──────────────────────────────────────────────────
        if re.match(r'^-{3,}$', line.strip()) or re.match(r'^\*{3,}$', line.strip()):
            if table_buf:
                pdf.table(table_buf)
                table_buf = []
            pdf.rule()
            i += 1
            continue

        # ── headings ─────────────────────────────────────────────────────────
        if line.startswith("# ") and not line.startswith("## "):
            if table_buf:
                pdf.table(table_buf)
                table_buf = []
            text = strip_md_heading(line)
            # First H1 is paper title (already on cover) — skip if cover already done
            if not skip_cover:
                skip_cover = True
                i += 1
                continue
            pdf.h1(text)
            i += 1
            continue

        if line.startswith("## "):
            if table_buf:
                pdf.table(table_buf)
                table_buf = []
            text = strip_md_heading(line)
            pdf.h1(text)
            i += 1
            continue

        if line.startswith("### "):
            if table_buf:
                pdf.table(table_buf)
                table_buf = []
            text = strip_md_heading(line)
            pdf.h2(text)
            i += 1
            continue

        if line.startswith("#### "):
            if table_buf:
                pdf.table(table_buf)
                table_buf = []
            text = strip_md_heading(line)
            pdf.h3(text)
            i += 1
            continue

        # ── table ────────────────────────────────────────────────────────────
        if '|' in line and line.strip().startswith('|'):
            row = parse_table_row(line)
            table_buf.append(row)
            i += 1
            continue
        else:
            if table_buf:
                pdf.table(table_buf)
                table_buf = []

        # ── bullet list ──────────────────────────────────────────────────────
        m = re.match(r'^(\s*)-\s+(.*)', line)
        if m:
            level = len(m.group(1)) // 2
            pdf.bullet(m.group(2), level=level)
            i += 1
            continue

        # ── numbered list ────────────────────────────────────────────────────
        m = re.match(r'^\s*(\d+)\.\s+(.*)', line)
        if m:
            pdf.num_item(m.group(1), m.group(2))
            i += 1
            continue

        # ── blank line ───────────────────────────────────────────────────────
        if line.strip() == "":
            i += 1
            continue

        # ── paragraph ────────────────────────────────────────────────────────
        # Accumulate consecutive non-empty, non-special lines
        para_lines = []
        while i < len(lines):
            l = lines[i]
            if (l.strip() == "" or l.startswith("#") or l.startswith("|") or
                    l.strip().startswith("```") or re.match(r'^-{3,}$', l.strip()) or
                    re.match(r'^\s*-\s+', l) or re.match(r'^\s*\d+\.\s+', l)):
                break
            para_lines.append(l)
            i += 1
        if para_lines:
            pdf.para(" ".join(para_lines))

    if table_buf:
        pdf.table(table_buf)
    if code_buf:
        pdf.code_block(code_buf)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    md_text = MD_PATH.read_text(encoding="utf-8")

    # ── Parse metadata from top of file ──────────────────────────────────────
    lines = md_text.splitlines()
    title   = "FarmFederate: Privacy-Preserving Multimodal Federated Learning\nfor Crop Stress Detection"
    authors = "FarmFederate Team"
    version = "Version 5.1"
    date    = "March 2026"

    # Extract abstract text (between ## Abstract marker and next ## section)
    abstract = ""
    in_abs = False
    abs_lines = []
    for ln in lines:
        if ln.strip() == "## Abstract":
            in_abs = True
            continue
        if in_abs:
            if ln.startswith("## ") and ln.strip() != "## Abstract":
                break
            if ln.strip().startswith("**Keywords"):
                break
            abs_lines.append(ln)
    abstract = " ".join(l.strip() for l in abs_lines if l.strip()).strip()

    # ── Build PDF ─────────────────────────────────────────────────────────────
    pdf = Paper()

    # Cover (page 1)
    pdf.cover(title, authors, version, date, abstract)
    pdf.add_page()

    # Body
    render(pdf, md_text)

    pdf.output(str(PDF_PATH))
    print(f"PDF saved -> {PDF_PATH}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    main()
