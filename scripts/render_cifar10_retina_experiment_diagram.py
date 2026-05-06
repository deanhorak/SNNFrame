#!/usr/bin/env python3

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "cifar10_retina_experiment.png"

WIDTH = 2200
HEIGHT = 1400

BG_TOP = (245, 244, 238)
BG_BOTTOM = (231, 235, 242)
INK = (30, 33, 39)
MUTED = (86, 92, 101)
WIRE = (53, 58, 66)
SPIKE = (163, 114, 34)
LEFT_FILL = (224, 237, 248)
LEFT_EDGE = (60, 103, 149)
RIGHT_FILL = (244, 232, 216)
RIGHT_EDGE = (162, 97, 47)
FUSION_FILL = (225, 237, 224)
FUSION_EDGE = (77, 121, 71)
FEATURE_FILL = (239, 226, 241)
FEATURE_EDGE = (114, 73, 121)
EDGE_FILL = (230, 239, 249)
EDGE_EDGE = (46, 103, 165)
LEGEND_FILL = (241, 243, 246)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size=size)


FONT_TITLE = font(36, bold=True)
FONT_SUBTITLE = font(20)
FONT_SECTION = font(26, bold=True)
FONT_BOX = font(20, bold=True)
FONT_SMALL = font(16)
FONT_TINY = font(14)


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
        [(x2, y2), (hx + px * 9, hy + py * 9), (hx - px * 9, hy - py * 9)],
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
        )
        draw.text((lx, ly), label, font=FONT_TINY, fill=color)


def spike_encoder(draw: ImageDraw.ImageDraw, box):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=22, fill=(250, 241, 214), outline=SPIKE, width=3)
    draw.text((x1 + 20, y1 + 14), "Temporal spike encoder", font=FONT_BOX, fill=INK)
    draw.text((x1 + 20, y1 + 45), "RGB CIFAR frame -> time-binned spike packets", font=FONT_TINY, fill=MUTED)
    left = x1 + 24
    right = x2 - 24
    top = y1 + 82
    spike_positions = [
        [0.05, 0.17, 0.33, 0.48, 0.71, 0.91],
        [0.09, 0.25, 0.37, 0.57, 0.79],
        [0.04, 0.20, 0.31, 0.51, 0.76, 0.93],
        [0.11, 0.29, 0.44, 0.63, 0.82],
    ]
    for lane, positions in enumerate(spike_positions):
        ly = top + lane * 28
        draw.line((left, ly, right, ly), fill=(187, 166, 115), width=2)
        for pos in positions:
            sx = left + pos * (right - left)
            draw.line((sx, ly - 10, sx, ly + 10), fill=SPIKE, width=3)


def hemisphere_panel(draw: ImageDraw.ImageDraw, box, title, fill, edge):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=edge, width=4)
    draw.text((x1 + 22, y1 + 16), title, font=FONT_SECTION, fill=edge)

    view_box = (x1 + 26, y1 + 64, x2 - 26, y1 + 170)
    rounded_box(
        draw,
        view_box,
        fill=(255, 255, 255),
        outline=edge,
        text="Transformed RGB retinal view",
        subtext="small rotation + lateral shift\nsame scene, different retinal sampling",
    )

    branch_specs = [
        ("Sobel g9", "appearance bank\ncolor-opponent + luminance"),
        ("Sobel g10", "appearance bank +\ncoarse normalized edge cue"),
        ("DoG g9", "appearance bank\ncoarse contrast cue"),
    ]
    branch_boxes = []
    branch_y = y1 + 232
    branch_w = 160
    gap = 20
    total_w = len(branch_specs) * branch_w + (len(branch_specs) - 1) * gap
    branch_x = x1 + (x2 - x1 - total_w) / 2
    for name, sub in branch_specs:
        outline = EDGE_EDGE if name == "Sobel g10" else edge
        fill_color = EDGE_FILL if name == "Sobel g10" else (255, 255, 255)
        box = (branch_x, branch_y, branch_x + branch_w, branch_y + 138)
        rounded_box(draw, box, fill=fill_color, outline=outline, text=name, subtext=sub)
        branch_boxes.append(box)
        branch_x += branch_w + gap

    stage_box = (x1 + 110, y1 + 438, x2 - 110, y1 + 574)
    rounded_box(
        draw,
        stage_box,
        fill=(255, 255, 255),
        outline=edge,
        text="Stage-1 classifier",
        subtext="weighted-distance evidence from\nbranch spike-derived feature patterns",
    )

    assoc_box = (x1 + 136, y1 + 638, x2 - 136, y1 + 758)
    rounded_box(
        draw,
        assoc_box,
        fill=(255, 255, 255),
        outline=edge,
        text="Association nucleus",
        subtext="hemisphere evidence prepared for\ncorpus-callosum arbitration",
    )

    for branch in branch_boxes:
        arrow(draw, ((view_box[0] + view_box[2]) / 2, view_box[3]), ((branch[0] + branch[2]) / 2, branch[1]), color=edge, width=4)
        bcolor = EDGE_EDGE if branch == branch_boxes[1] else edge
        arrow(draw, ((branch[0] + branch[2]) / 2, branch[3]), ((stage_box[0] + stage_box[2]) / 2, stage_box[1]), color=bcolor, width=4)
    arrow(draw, ((stage_box[0] + stage_box[2]) / 2, stage_box[3]), ((assoc_box[0] + assoc_box[2]) / 2, assoc_box[1]), color=edge, width=4)

    return {"view": view_box, "stage": stage_box, "assoc": assoc_box}


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_TOP)
    draw = ImageDraw.Draw(image)
    draw_gradient(draw)

    draw.text((60, 32), "CIFAR-10 Bilateral Retina Experiment", font=FONT_TITLE, fill=INK)
    draw.text(
        (60, 82),
        "Natural-image path: RGB stimulus -> temporal spikes -> bilateral retinal branches -> feature-mode stage-1 evidence -> corpus-callosum fusion.",
        font=FONT_SUBTITLE,
        fill=MUTED,
    )

    input_box = (70, 150, 430, 350)
    rounded_box(
        draw,
        input_box,
        fill=(255, 255, 255),
        outline=INK,
        text="CIFAR-10 RGB image",
        subtext="32x32 natural scene\n10-way object classification",
    )

    spike_box = (505, 140, 1070, 360)
    spike_encoder(draw, spike_box)
    arrow(
        draw,
        (input_box[2], (input_box[1] + input_box[3]) / 2),
        (spike_box[0], (spike_box[1] + spike_box[3]) / 2),
        color=SPIKE,
        width=5,
        label="encode RGB input",
        label_pos=0.34,
    )

    left = hemisphere_panel(draw, (80, 420, 1000, 1200), "Left Hemisphere", LEFT_FILL, LEFT_EDGE)
    right = hemisphere_panel(draw, (1200, 420, 2120, 1200), "Right Hemisphere", RIGHT_FILL, RIGHT_EDGE)

    arrow(draw, ((spike_box[0] + spike_box[2]) / 2 - 130, spike_box[3]), ((left["view"][0] + left["view"][2]) / 2, left["view"][1]), color=SPIKE, width=5, label="left spike stream", label_pos=0.45)
    arrow(draw, ((spike_box[0] + spike_box[2]) / 2 + 130, spike_box[3]), ((right["view"][0] + right["view"][2]) / 2, right["view"][1]), color=SPIKE, width=5, label="right spike stream", label_pos=0.45)

    fusion_box = (872, 800, 1328, 980)
    rounded_box(
        draw,
        fusion_box,
        fill=FUSION_FILL,
        outline=FUSION_EDGE,
        text="Corpus-callosum fusion",
        subtext="interaction features + disagreement-aware\narbitration across hemisphere evidence",
    )
    out_box = (915, 1045, 1285, 1170)
    rounded_box(
        draw,
        out_box,
        fill=(255, 255, 255),
        outline=FUSION_EDGE,
        text="Final 10-way decision",
        subtext="airplane, automobile, bird,\ncat, deer, dog, frog, horse, ship, truck",
    )

    arrow(draw, (left["assoc"][2], (left["assoc"][1] + left["assoc"][3]) / 2), (fusion_box[0], fusion_box[1] + 58), color=FUSION_EDGE, width=5, label="left evidence", label_pos=0.52)
    arrow(draw, (right["assoc"][0], (right["assoc"][1] + right["assoc"][3]) / 2), (fusion_box[2], fusion_box[1] + 58), color=FUSION_EDGE, width=5, label="right evidence", label_pos=0.48)
    arrow(draw, ((fusion_box[0] + fusion_box[2]) / 2, fusion_box[3]), ((out_box[0] + out_box[2]) / 2, out_box[1]), color=FUSION_EDGE, width=5)

    feature_box = (720, 1215, 1480, 1360)
    rounded_box(
        draw,
        feature_box,
        fill=FEATURE_FILL,
        outline=FEATURE_EDGE,
        text="Natural-image feature path",
        subtext="appearance-bank auxiliary channels:\ncolor opponent, luminance mean, contrast, texture\nplus coarse normalized edge analysis only on Sobel g10",
    )

    legend = (1530, 1210, 2120, 1360)
    draw.rounded_rectangle(legend, radius=24, fill=LEGEND_FILL, outline=(170, 177, 186), width=2)
    draw.text((legend[0] + 20, legend[1] + 16), "Legend", font=FONT_SECTION, fill=INK)
    rows = [
        (SPIKE, "temporal spike-pattern drive"),
        (WIRE, "intra-hemisphere feedforward flow"),
        (FUSION_EDGE, "interhemispheric arbitration / fusion"),
        (EDGE_EDGE, "supplemental coarse normalized edge cue"),
        (FEATURE_EDGE, "feature-mode natural-image representation"),
    ]
    for idx, (color, label) in enumerate(rows):
        y = legend[1] + 62 + idx * 24
        draw.line((legend[0] + 24, y, legend[0] + 88, y), fill=color, width=5)
        draw.text((legend[0] + 102, y - 10), label, font=FONT_SMALL, fill=INK)

    draw.text(
        (60, HEIGHT - 40),
        "Generated from configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json and the current CIFAR-10 bilateral Retina experiment path.",
        font=FONT_SMALL,
        fill=MUTED,
    )

    image.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
