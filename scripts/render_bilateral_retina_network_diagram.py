#!/usr/bin/env python3

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "bilateral_retina_spike_network.png"

WIDTH = 2200
HEIGHT = 1400

BG_TOP = (247, 245, 238)
BG_BOTTOM = (232, 236, 241)
INK = (30, 33, 39)
MUTED = (86, 92, 101)
LEFT_FILL = (225, 238, 248)
LEFT_EDGE = (59, 103, 148)
RIGHT_FILL = (244, 232, 216)
RIGHT_EDGE = (163, 96, 47)
FUSION_FILL = (227, 238, 223)
FUSION_EDGE = (77, 121, 71)
SPIKE_FILL = (250, 241, 214)
SPIKE_EDGE = (160, 119, 40)
LEARN_FILL = (239, 225, 241)
LEARN_EDGE = (114, 73, 121)
LEGEND_FILL = (241, 243, 246)
WIRE = (53, 58, 66)
CALL0SAL = (65, 112, 72)
MODULATORY = (133, 76, 132)
REPLAY = (51, 123, 165)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)


FONT_TITLE = load_font(36, bold=True)
FONT_SUBTITLE = load_font(20)
FONT_SECTION = load_font(26, bold=True)
FONT_BOX = load_font(20, bold=True)
FONT_SMALL = load_font(16)
FONT_TINY = load_font(14)


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def draw_gradient(draw: ImageDraw.ImageDraw) -> None:
    for y in range(HEIGHT):
        t = y / float(max(1, HEIGHT - 1))
        color = (
            lerp(BG_TOP[0], BG_BOTTOM[0], t),
            lerp(BG_TOP[1], BG_BOTTOM[1], t),
            lerp(BG_TOP[2], BG_BOTTOM[2], t),
        )
        draw.line((0, y, WIDTH, y), fill=color)


def rounded_box(draw: ImageDraw.ImageDraw, box, fill, outline, text, subtext=None, text_fill=INK):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=22, fill=fill, outline=outline, width=3)
    bbox = draw.multiline_textbbox((0, 0), text, font=FONT_BOX, spacing=4)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + (x2 - x1 - tw) / 2
    ty = y1 + 14
    draw.multiline_text((tx, ty), text, font=FONT_BOX, fill=text_fill, align="center", spacing=4)
    if subtext:
        sb = draw.multiline_textbbox((0, 0), subtext, font=FONT_TINY, spacing=3)
        sw = sb[2] - sb[0]
        sx = x1 + (x2 - x1 - sw) / 2
        sy = y2 - (sb[3] - sb[1]) - 14
        draw.multiline_text((sx, sy), subtext, font=FONT_TINY, fill=MUTED, align="center", spacing=3)


def arrow(draw: ImageDraw.ImageDraw, start, end, color=WIRE, width=5, label=None, label_pos=0.5):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    dx = x2 - x1
    dy = y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux = dx / length
    uy = dy / length
    hx = x2 - ux * 18
    hy = y2 - uy * 18
    px = -uy
    py = ux
    draw.polygon(
        [
            (x2, y2),
            (hx + px * 9, hy + py * 9),
            (hx - px * 9, hy - py * 9),
        ],
        fill=color,
    )
    if label:
        lx = x1 + dx * label_pos
        ly = y1 + dy * label_pos - 20
        bbox = draw.textbbox((0, 0), label, font=FONT_TINY)
        pad = 6
        draw.rounded_rectangle(
            (lx - pad, ly - pad, lx + (bbox[2] - bbox[0]) + pad, ly + (bbox[3] - bbox[1]) + pad),
            radius=8,
            fill=(255, 255, 255),
            outline=None,
        )
        draw.text((lx, ly), label, font=FONT_TINY, fill=color)


def spike_raster(draw: ImageDraw.ImageDraw, x, y, w, h):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=SPIKE_FILL, outline=SPIKE_EDGE, width=3)
    draw.text((x + 18, y + 12), "Temporal spike encoder", font=FONT_BOX, fill=INK)
    draw.text((x + 18, y + 44), "EMNIST image -> time-binned rate-coded spike trains", font=FONT_TINY, fill=MUTED)
    lanes = 4
    left = x + 24
    right = x + w - 24
    top = y + 82
    lane_gap = 28
    spike_positions = [
        [0.06, 0.17, 0.31, 0.62, 0.78, 0.91],
        [0.10, 0.22, 0.41, 0.57, 0.83],
        [0.04, 0.28, 0.36, 0.52, 0.69, 0.74, 0.95],
        [0.13, 0.25, 0.47, 0.58, 0.72, 0.89],
    ]
    for lane in range(lanes):
        ly = top + lane * lane_gap
        draw.line((left, ly, right, ly), fill=(187, 166, 115), width=2)
        for pos in spike_positions[lane]:
            sx = left + pos * (right - left)
            draw.line((sx, ly - 10, sx, ly + 10), fill=SPIKE_EDGE, width=3)
    ticks = ["t0", "t1", "t2", "t3", "t4"]
    for i, tick in enumerate(ticks):
        tx = left + i * (right - left) / (len(ticks) - 1)
        draw.line((tx, top + lanes * lane_gap + 8, tx, top + lanes * lane_gap + 18), fill=MUTED, width=2)
        draw.text((tx - 10, top + lanes * lane_gap + 22), tick, font=FONT_TINY, fill=MUTED)


def hemisphere_panel(draw: ImageDraw.ImageDraw, panel_box, title, fill, edge, branch_names, stage_title):
    x1, y1, x2, y2 = panel_box
    draw.rounded_rectangle(panel_box, radius=28, fill=fill, outline=edge, width=4)
    draw.text((x1 + 22, y1 + 16), title, font=FONT_SECTION, fill=edge)

    view_box = (x1 + 28, y1 + 64, x2 - 28, y1 + 168)
    rounded_box(
        draw,
        view_box,
        fill=(255, 255, 255),
        outline=edge,
        text="Transformed retinal view",
        subtext="small rotation + lateral shift\nsame image, slightly different sampling",
    )

    branch_y = y1 + 232
    branch_w = 146
    gap = 18
    total_w = len(branch_names) * branch_w + (len(branch_names) - 1) * gap
    branch_x = x1 + (x2 - x1 - total_w) / 2
    branch_boxes = []
    for branch in branch_names:
        box = (branch_x, branch_y, branch_x + branch_w, branch_y + 132)
        rounded_box(draw, box, fill=(255, 255, 255), outline=edge, text=branch, subtext="retina branch\nspike-driven features")
        branch_boxes.append(box)
        branch_x += branch_w + gap

    stage_box = (x1 + 110, y1 + 434, x2 - 110, y1 + 562)
    rounded_box(
        draw,
        stage_box,
        fill=(255, 255, 255),
        outline=edge,
        text=stage_title,
        subtext="builds hemisphere evidence from\nbranch spike patterns",
    )

    assoc_box = (x1 + 140, y1 + 626, x2 - 140, y1 + 742)
    rounded_box(
        draw,
        assoc_box,
        fill=(255, 255, 255),
        outline=edge,
        text="Association nucleus",
        subtext="aggregates branch evidence\nfor callosal arbitration",
    )

    for box in branch_boxes:
        arrow(draw, ((view_box[0] + view_box[2]) / 2, view_box[3]), ((box[0] + box[2]) / 2, box[1]), color=edge, width=4)
        arrow(draw, ((box[0] + box[2]) / 2, box[3]), ((stage_box[0] + stage_box[2]) / 2, stage_box[1]), color=edge, width=4)
    arrow(draw, ((stage_box[0] + stage_box[2]) / 2, stage_box[3]), ((assoc_box[0] + assoc_box[2]) / 2, assoc_box[1]), color=edge, width=4)

    return {"view": view_box, "branches": branch_boxes, "stage": stage_box, "assoc": assoc_box}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (WIDTH, HEIGHT), BG_TOP)
    draw = ImageDraw.Draw(image)
    draw_gradient(draw)

    draw.text((60, 32), "Bilateral Retina Continuous-Learning Network", font=FONT_TITLE, fill=INK)
    draw.text(
        (60, 82),
        "Two spike-driven hemispheres observe slightly different retinal views, arbitrate through a corpus-callosum fusion layer, and adapt through delayed replay.",
        font=FONT_SUBTITLE,
        fill=MUTED,
    )

    img_box = (70, 150, 420, 350)
    rounded_box(
        draw,
        img_box,
        fill=(255, 255, 255),
        outline=INK,
        text="Visual stimulus",
        subtext="EMNIST image or any other\nvision input sample",
    )

    spike_box = (500, 140, 1040, 360)
    spike_raster(draw, spike_box[0], spike_box[1], spike_box[2] - spike_box[0], spike_box[3] - spike_box[1])
    arrow(draw, (img_box[2], (img_box[1] + img_box[3]) / 2), (spike_box[0], (spike_box[1] + spike_box[3]) / 2), color=SPIKE_EDGE, width=5, label="encode to spike trains", label_pos=0.35)

    left = hemisphere_panel(
        draw,
        (80, 420, 1000, 1180),
        "Left Hemisphere",
        LEFT_FILL,
        LEFT_EDGE,
        ["Sobel g9", "Sobel g10", "DoG g9"],
        "Stage-1 classifier",
    )
    right = hemisphere_panel(
        draw,
        (1200, 420, 2120, 1180),
        "Right Hemisphere",
        RIGHT_FILL,
        RIGHT_EDGE,
        ["Sobel g9", "Sobel g10", "DoG g9"],
        "Stage-1 classifier",
    )

    arrow(draw, ((spike_box[0] + spike_box[2]) / 2 - 120, spike_box[3]), ((left["view"][0] + left["view"][2]) / 2, left["view"][1]), color=SPIKE_EDGE, width=5, label="left spike stream", label_pos=0.45)
    arrow(draw, ((spike_box[0] + spike_box[2]) / 2 + 120, spike_box[3]), ((right["view"][0] + right["view"][2]) / 2, right["view"][1]), color=SPIKE_EDGE, width=5, label="right spike stream", label_pos=0.45)

    fusion_box = (870, 785, 1330, 970)
    rounded_box(
        draw,
        fusion_box,
        fill=FUSION_FILL,
        outline=FUSION_EDGE,
        text="Corpus-callosum fusion",
        subtext="weighted arbitration over hemisphere evidence\nincluding disagreement-aware bias",
    )

    out_box = (910, 1035, 1290, 1160)
    rounded_box(
        draw,
        out_box,
        fill=(255, 255, 255),
        outline=FUSION_EDGE,
        text="Final classification",
        subtext="top class after bilateral fusion",
    )

    arrow(draw, (left["assoc"][2], (left["assoc"][1] + left["assoc"][3]) / 2), (fusion_box[0], fusion_box[1] + 58), color=CALL0SAL, width=5, label="left evidence", label_pos=0.52)
    arrow(draw, (right["assoc"][0], (right["assoc"][1] + right["assoc"][3]) / 2), (fusion_box[2], fusion_box[1] + 58), color=CALL0SAL, width=5, label="right evidence", label_pos=0.48)
    arrow(draw, ((fusion_box[0] + fusion_box[2]) / 2, fusion_box[3]), ((out_box[0] + out_box[2]) / 2, out_box[1]), color=FUSION_EDGE, width=5)

    learn_box = (760, 1215, 1440, 1360)
    rounded_box(
        draw,
        learn_box,
        fill=LEARN_FILL,
        outline=LEARN_EDGE,
        text="Continuous learning loop",
        subtext="reward/error signal -> delayed replay queue -> decaying eligibility traces\nbounded centroid and exemplar updates after fusion decisions",
    )

    arrow(draw, ((out_box[0] + out_box[2]) / 2, out_box[3]), ((learn_box[0] + learn_box[2]) / 2, learn_box[1]), color=MODULATORY, width=5, label="reward and error", label_pos=0.45)
    arrow(draw, (learn_box[0] + 110, learn_box[1]), (left["stage"][0] + 60, left["stage"][3]), color=REPLAY, width=5, label="replay/update", label_pos=0.42)
    arrow(draw, (learn_box[2] - 110, learn_box[1]), (right["stage"][2] - 60, right["stage"][3]), color=REPLAY, width=5, label="replay/update", label_pos=0.42)
    arrow(draw, ((learn_box[0] + learn_box[2]) / 2, learn_box[1]), ((fusion_box[0] + fusion_box[2]) / 2, fusion_box[3]), color=MODULATORY, width=5, label="update fusion weights", label_pos=0.45)

    legend = (1510, 1190, 2120, 1360)
    draw.rounded_rectangle(legend, radius=24, fill=LEGEND_FILL, outline=(170, 177, 186), width=2)
    draw.text((legend[0] + 20, legend[1] + 16), "Legend", font=FONT_SECTION, fill=INK)
    legend_rows = [
        (SPIKE_EDGE, "temporal spike train drive"),
        (WIRE, "intra-hemisphere feedforward flow"),
        (CALL0SAL, "interhemispheric evidence exchange"),
        (MODULATORY, "reward-modulated plasticity"),
        (REPLAY, "delayed replay / eligibility-trace update"),
    ]
    for idx, (color, label) in enumerate(legend_rows):
        y = legend[1] + 62 + idx * 24
        draw.line((legend[0] + 24, y, legend[0] + 88, y), fill=color, width=5)
        draw.text((legend[0] + 102, y - 10), label, font=FONT_SMALL, fill=INK)

    draw.text(
        (60, HEIGHT - 42),
        "Generated from the current bilateral Retina continuous-learning architecture in SNNFrame.",
        font=FONT_SMALL,
        fill=MUTED,
    )

    image.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
