"""
A simple circuit drawer, rendering in matplotlib.

The bottom qubit is index 0, and all qubits from 0
to the maximum targeted/controlled upon in the
circuit are rendered. Circuits are rendered as
compactly as possible without commutation. The main
function draw_circuit() accepts a pyQuEST Circuit
or a list of pyQuEST operators, and spawns a new
matplotlib window.

Quirks:
    - multi-target gates acting upon non-contiguous
      qubits are drawn as a column of one-target
      gates connected by vertical lines (similar to
      how control qubits are rendered).
    - operators without explicit target qubits are
      assumed to apply to the entire state and are
      drawn as N-target gates, where the number of
      state qubits N is inferred from the other
      operators in the circuit. This might differ
      from the actual dimension of operators like
      MixDensityMatrix.
    - decoherence channels are drawn as gates with
      dashed borders.
    - initialisations are drawn as all-target gates
      with dotted borders.

The algorithm is basic; the circuit canvas is
partitioned into a (#qubits x #depth) grid and
each gate is assigned a column index therein. This
is chosen as the leftmost (smallest index) column
which has empty grid squares at every qubit between
the min and max qubits operated upon by the gate.
Note that vertical connectors of a gate (e.g. the
line between target and control qubits) occupy
grid squares, but do not prevent subsequent gates
from being placed left of them. Implementing this
is easy; we track the rightmost targeted column
of each qubit (we can never place new gates left of
this), and also the columns to the right of this
which are occluded (but not targeted) by vertical
connectors.

@author Tyson Jones
@date June 2024
"""

from operator import itemgetter
from itertools import groupby
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines

# all concrete classes herein are drawable
from pyquest.gates import *
from pyquest.unitaries import *
from pyquest.operators import *
from pyquest.decoherence import *
from pyquest.initialisations import *


# visual order of graphic constituents (lower is occluded)
class layer:

    # horizontal qubit lines are drawn at the very bottom
    QUBIT_STAVE = 0

    # vertical lines connecting control and target qubits come next
    VERTICAL_CONNECTOR = 1

    # control qubit circles appear above the connectors
    CONTROL_CIRCLE = 2

    # gate bodies appear on top
    GATE_BODY = 2


# size constants relative to the 1x1 gate grid
class size:

    # minimum padding between circuit graphic and maptlotlib window
    PLOT_PADDING = 0.1

    # padding between gate's grid space and its gate body
    GATE_RECTANGLE_NEG_PADDING = 0.1

    # radius of circle upon control qubits, and phase gate targets
    CONTROL_CIRCLE_RADIUS = 0.1

    # radius of circle upon CX and CCX target qubits
    TARGET_CIRCLE_RADIUS = 0.2


# class holding colors logic
colors = None

class Colors:
    themes = {
        "bw": {
            "fig_color": "white",
            "qubit_stave_color": "lightgray",
            "vertical_connector_color": "gray",
            "label_color": "black",
            "initialisations_color": "white",
            "decoherence_color": "white",
            "operator_color": "white",
            "gate_face_color": "white",
            "gate_edge_color": "black",
            "control_qubit_color": "black",
            "meas_gate_color": "white"
        },
        "dark": {
            "fig_color": "black",
            "qubit_stave_color": "white",
            "vertical_connector_color": "white",
            "label_color": "white",
            "initialisations_color": "black",
            "decoherence_color": "black",
            "operator_color": "black",
            "gate_face_color": "black",
            "gate_edge_color": "white",
            "control_qubit_color": "white",
            "meas_gate_color": "black"
        },
        "qmt": {
            "fig_color": "white",
            "qubit_stave_color": "#2D2E44",
            "vertical_connector_color": "#ff4342",
            "label_color": "white",
            "initialisations_color": "gray",
            "decoherence_color": "#d91328",
            "operator_color": "#ff4342",
            "gate_face_color": "#ff4342",
            "gate_edge_color": "#ff4342",
            "control_qubit_color": "#ff4342",
            "meas_gate_color": "#2D2E44",
        }
    }


    def __init__(self, theme="bw"):

        if theme not in ["bw", "dark", "qmt"]:
            raise ValueError(f"Invalid theme: {theme}. Choose from 'bw', 'dark', or 'qmt'.")

        self.theme = theme
        self.colors = self.themes.get(theme, self.themes["bw"])

    def get_fig_color(self):
        return self.colors["fig_color"]

    def get_qubit_stave_color(self):
        return self.colors["qubit_stave_color"]

    def get_vertical_connector_color(self, gate):

        if self.theme == "qmt":

            if hasattr(pyquest.initialisations, type(gate).__name__):
                return self.colors["initialisations_color"]

            elif hasattr(pyquest.decoherence, type(gate).__name__):
                return self.colors["decoherence_color"]

            elif hasattr(pyquest.operators, type(gate).__name__):
                return self.colors["operator_color"]
        
            elif isinstance(gate, M):
                return self.colors["meas_gate_color"] 

        return self.colors["vertical_connector_color"]

    def get_label_color(self):
        return self.colors["label_color"]

    def get_gate_face_color(self, gate):

        if hasattr(pyquest.initialisations, type(gate).__name__):
            return self.colors["initialisations_color"]

        elif hasattr(pyquest.decoherence, type(gate).__name__):
            return self.colors["decoherence_color"]
        
        elif hasattr(pyquest.operators, type(gate).__name__):
            return self.colors["operator_color"]

        elif isinstance(gate, X) and len(gate.controls) != 0:
            return self.colors["fig_color"]

        elif isinstance(gate, Swap):
            return self.colors["control_qubit_color"]
        
        elif isinstance(gate, Phase):
            return self.colors["control_qubit_color"]

        elif isinstance(gate, M):
            return self.colors["meas_gate_color"]

        return self.colors["gate_face_color"]

    def get_gate_edge_color(self, gate):

        if hasattr(pyquest.decoherence, type(gate).__name__):
            return self.colors["label_color"]

        elif hasattr(pyquest.initialisations, type(gate).__name__):
            return self.colors["label_color"]
        
        elif hasattr(pyquest.operators, type(gate).__name__) and self.theme == "qmt":
            return self.colors["operator_color"]

        elif isinstance(gate, X) and len(gate.controls) != 0:
            return self.colors["control_qubit_color"]

        elif isinstance(gate, M) and self.theme == "qmt":
            return self.colors["meas_gate_color"]

        return self.colors["gate_edge_color"]

    def get_control_qubit_color(self):
        return self.colors["control_qubit_color"]


"""
Logic for deciding gate placement, which...
    - positions all gates within an integer grid by deciding each gate's column
    - assigns gates as far left as is possible without commuting existing gates
    - does not allow gates to coincide with vertical connectors of other gates
"""


def has_explicit_targets(gate):

    # duck-check whether 'targets' was overwritten by operator subclass
    try:
        gate.targets
        return True
    except:
        return False


def has_controls(gate):

    # duck-check whether 'controls' was overwritten by operator subclass
    try:
        gate.controls
        return True
    except:
        return False


def get_operated_qubits(gate, num_qubits):

    # generic gates "operate" upon all their control and target qubits
    if has_explicit_targets(gate):
        return gate.controls + gate.targets

    # un-targeted gates are assumed to operate upon all qubits
    return list(range(0, num_qubits))


def get_num_qubits(gates):

    # find the biggest indexed target/control qubit among explicitly-targeted gates
    return 1 + max(max([*g.targets, *g.controls]) for g in gates if has_explicit_targets(g))


def get_gate_column(gate_qubits, columns_of_last_target, columns_occluded_by_connectors):

    # qubits spanned by connectors between targets & controls
    gate_range = list(range(min(gate_qubits), max(gate_qubits) + 1))

    # initial choice is the leftmost un-targeted column
    column = 1 + max(columns_of_last_target[q] for q in gate_range)

    # but this column may be occluded by control lines
    while any(column in columns_occluded_by_connectors[q] for q in gate_range):
        column += 1

    return column


def get_circuit_columns(gates):

    # choose one column index for each gate
    gate_columns = []

    # the grid height as informed by the highest index targeted qubit
    num_qubits = get_num_qubits(gates)

    # {qubit index: column}
    columns_of_last_target = {i: -1 for i in range(num_qubits)}

    # {qubit index: [columns]}
    columns_occluded_by_connectors = {i: [] for i in range(num_qubits)}

    for gate in gates:

        # all qubits controlled or targeted by gate (global operators return all)
        gate_qubits = get_operated_qubits(gate, num_qubits)

        # there must be room for all qubits between those explicitly targeted, for connectors
        gate_range = (min(gate_qubits), max(gate_qubits) + 1)

        # find the leftmost column which fits the gate
        gate_column = get_gate_column(
            gate_qubits, columns_of_last_target, columns_occluded_by_connectors
        )
        gate_columns.append(gate_column)

        # prevent subsequent gates from commuting left of this gate
        for q in gate_qubits:
            columns_of_last_target[q] = gate_column

        # prevent subsequent gates from occupying the vertical connectors
        for q in range(*gate_range):
            columns_occluded_by_connectors[q].append(gate_column)

        # unnecessary memory cleanup; delete redundant vertical connectors left of targets
        for q in range(*gate_range):
            columns_occluded_by_connectors[q] = [
                c for c in columns_occluded_by_connectors[q] if c > columns_of_last_target[q]
            ]

    # return one column per gate
    return gate_columns


"""
Visual gate styling
    - in matplotlib 3.8+, colours can be in a tuple with an alpha value,
      e.g. ('green', 0.3). We don't make use of this below
    - when the 'reverse_bits' flag = True, non-symmetric symbols (like
      measure) are drawn upside down, so that thet are the correct way up
      after flipping the y-axis
"""


def get_gate_label(gate):

    # all channels get abbreviated
    if isinstance(gate, Damping):
        return "γ"
    if isinstance(gate, Dephasing):
        return "φ"
    if isinstance(gate, Depolarising):
        return "Δ"
    if isinstance(gate, KrausMap):
        return "K"
    if isinstance(gate, PauliNoise):
        return "σ"
    if isinstance(gate, MixDensityMatrix):
        return "ρ"

    # all initialisations get abbreviated
    if isinstance(gate, ZeroState):
        return "0"
    if isinstance(gate, BlankState):
        return "∅"
    if isinstance(gate, ClassicalState):
        return "i"
    if isinstance(gate, PlusState):
        return "+"
    if isinstance(gate, PureState):
        return "ψ"

    # compactly specified unitaries are identical to general unitaries
    if isinstance(gate, CompactU):
        return "U"

    # rotations around vector v look like Rx,Ry,Rz
    if isinstance(gate, RotateAroundAxis):
        return "Rv"

    # some gates have no labels
    if isinstance(gate, Swap):
        raise RuntimeError()
    if isinstance(gate, Phase):
        raise RuntimeError()

    # generic gates use their class name
    return type(gate).__name__


def get_measure_symbol(rect):

    # calculate the center and width of the rectangle
    x, y = (mean(x[i] for x in rect) for i in [0, 1])
    w = rect[2][0] - rect[0][0]

    # arc and line start below middle of the rectangle
    y0 = y + 0.15 * w if reverse_bits_glob else y - 0.15 * w

    # create the arc object
    arc = patches.Arc(
        xy=(x, y0),
        width=w * 0.7,
        height=w * 0.7,
        angle=180 if reverse_bits_glob else 0,
        theta1=0,
        theta2=180,
        fill=False,
        linewidth=1.5,
        color=colors.get_label_color(),
        zorder=layer.GATE_BODY,
    )

    # create the line object
    line = mlines.Line2D(
        [x, x + 0.35 * w],
        [y0, y0 - 0.35 * w if reverse_bits_glob else y0 + 0.35 * w],
        color=colors.get_label_color(),
        zorder=layer.GATE_BODY,
    )


    # Return both objects as a tuple
    return arc, line


def get_gate_rect_style(gate):

    # decoherence channels have dashed rectangles
    if hasattr(pyquest.decoherence, type(gate).__name__):
        return "dashed"

    # initialisations have dotted rectangles
    if hasattr(pyquest.initialisations, type(gate).__name__):
        return "dotted"

    # all other operators have solid lines
    return "solid"


"""
Logic for producing graphics, which...
    - draws phase and control qubits as circles
    - makes decoherence channel borders dashed
    - draws swap gates with X symbols
    - labels gates with concise strings
    - merges gate bodies which target adjacent qubits
    - draws target bullseye for CX and CXX
    - draws bespoke measurement symbol
"""


def get_grouped_consecutive_items(nums):

    # [1,2,4,5,6] -> [(1,2), (4,5,6)]
    indAndNums = enumerate(sorted(nums))
    for _, group in groupby(indAndNums, lambda x: x[0] - x[1]):
        yield list(map(itemgetter(1), group))


def get_gate_graphic_components(gate, column, num_qubits):

    # graphics consist of vertical connector lines, control circles, and gate body rectangles
    lines = []  # item = [(x0,y0), (x1,y1)]
    circles = []  # item = (x0,y0)
    rectangles = []  # item = [(x0,y0), (x0,y1), (x1,y1), (x1,y0)]

    # clarifying (in principle...) constants relative to 1x1 grid
    qubits = get_operated_qubits(gate, num_qubits)
    pad = size.GATE_RECTANGLE_NEG_PADDING
    halfcol = 0.5
    nextcol = column + 1
    midcol = column + halfcol
    midtop = max(qubits) + halfcol
    midbot = min(qubits) + halfcol
    padcol = column + pad
    padnextcol = nextcol - pad

    # note connector lines may be superfluous and occluded by rectangles
    lines.append([(midcol, midbot), (midcol, midtop)])  # only one line needed

    # only attempt drawing controls if any exist (else .controls throws)
    if has_controls(gate):
        circles += [(midcol, q + halfcol) for q in gate.controls]

    # explicitly targeted gates have adjacent targets merged into rectangles
    if has_explicit_targets(gate):

        for group in get_grouped_consecutive_items(gate.targets):
            x0, y0 = padcol, min(group) + pad
            x1, y1 = padnextcol, max(group) + 1 - pad
            rectangles.append([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])

    # whereas untargeted gates are assumed global and act on every qubit
    else:
        x0, y0 = padcol, 0 + pad
        x1, y1 = padnextcol, num_qubits - pad
        rectangles.append([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])

    # returned in order of increasing z-order
    return lines, circles, rectangles


def draw_gate_body(gate, column, rectangles, plt, ax):

    # gate-specific styling for special operators
    special_opts = {"color": colors.get_gate_face_color(gate), "zorder": layer.GATE_BODY}

    # SWAP gates ignore rectangles and draw X at every target
    if isinstance(gate, Swap):
        for q in gate.targets:
            plt.scatter(column + 0.5, q + 0.5, marker="x", **special_opts)
        return

    # Phase gates ignore rectangles and draw circle at every target
    if isinstance(gate, Phase):
        radius = size.CONTROL_CIRCLE_RADIUS
        for q in gate.targets:
            ax.add_patch(plt.Circle((column + 0.5, q + 0.5), radius, **special_opts))
        return

    # ordinary styling for rest
    other_opts = {
        "linestyle": get_gate_rect_style(gate),
        "edgecolor": colors.get_gate_edge_color(gate),
        "facecolor": colors.get_gate_face_color(gate),
        "zorder": layer.GATE_BODY,
    }

    # CX and CCX gates draw bullseyes rather than rectangles at every target
    if isinstance(gate, X) and len(gate.controls) != 0:
        radius = size.TARGET_CIRCLE_RADIUS
        for q in gate.targets:
            # draw a circle
            x, y = column + 0.5, q + 0.5
            ax.add_patch(plt.Circle((x, y), radius, linewidth=1.8, **other_opts))
            # draw the inner cross
            ax.plot([x - radius, x + radius], [y, y], color=colors.get_gate_edge_color(gate))
            ax.plot([x, x], [y - radius, y + radius], color=colors.get_gate_edge_color(gate))
        return

    for rect in rectangles:
        ax.add_patch(plt.Polygon(rect, **other_opts))

    # each rectangle is labelled
    label = get_gate_label(gate)
    label_color = colors.get_label_color()
    for rect in rectangles:

        # measurement gate has a bespoke graphic
        if isinstance(gate, M):
            arc, line = get_measure_symbol(rect)
            ax.add_patch(arc)
            ax.add_line(line)

        # SqrtSWAP gate uses mpl raw text
        elif isinstance(gate, SqrtSwap):
            pos = (mean(x[i] for x in rect) for i in [0, 1])
            plt.text(
                *pos, s=r"$\sqrt{SWAP}$", va="center", ha="center", fontsize=8, color=label_color
            )

        else:
            pos = (mean(x[i] for x in rect) for i in [0, 1])
            plt.text(*pos, s=label, va="center", ha="center", color=label_color)

    return


def draw_gate(gate, column, num_qubits, plt, ax):

    lines, dots, rectangles = get_gate_graphic_components(gate, column, num_qubits)

    # draw vertical connector lines (at back)
    for line in lines:

        # avoid drawing zero-length lines (else matplotlib throws)
        if line[0] == line[1]:
            continue

        (a, b), (c, d) = line
        plt.plot(
            (a, c),
            (b, d),
            color=colors.get_vertical_connector_color(gate),
            zorder=layer.VERTICAL_CONNECTOR,
        )

    # Draw control dots
    for i, dot in enumerate(dots):
        control_color = (
            colors.get_control_qubit_color()
            if not isinstance(gate, U) or not gate.control_pattern
            else (
                colors.get_control_qubit_color()
                if gate.control_pattern[i] == 1
                else colors.get_fig_color()
            )
        )

        ax.add_patch(
            plt.Circle(
                dot,
                size.CONTROL_CIRCLE_RADIUS,
                edgecolor=colors.get_gate_edge_color(gate),
                facecolor=control_color,
                linewidth=1.5,
                zorder=layer.CONTROL_CIRCLE,
            )
        )

    # draw the main body of the gate; possibly labelled rectangles, or bespoke symbols
    draw_gate_body(gate, column, rectangles, plt, ax)


def draw_circuit(gates, filename=None, theme="bw", reverse_bits=False):

    # determine circuit layout
    gate_columns = get_circuit_columns(gates)
    num_columns = 1 + max(gate_columns)
    num_qubits = get_num_qubits(gates)

    # get matplotlib handles and set the canvas size
    mpl_figure = plt.figure()
    mpl_figure.set_size_inches(num_columns, num_qubits)
    ax = plt.gca()

    # set global color theme and bit ordering
    global colors
    colors = Colors(theme)
    global reverse_bits_glob
    reverse_bits_glob = reverse_bits

    # Set the background color
    mpl_figure.patch.set_facecolor(colors.get_fig_color())
    ax.set_facecolor(colors.get_fig_color())

    # draw horizontal qubit stave
    for q in range(num_qubits):
        plt.plot(
            [-0.5, num_columns + 0.5],
            [q + 0.5, q + 0.5],
            color=colors.get_qubit_stave_color(),
            zorder=layer.QUBIT_STAVE,
        )

    # draw each gate above stave
    for gate, column in zip(gates, gate_columns):
        draw_gate(gate, column, num_qubits, plt, ax)

    # set plot range
    pad = size.PLOT_PADDING
    ax.set_xlim(-0.5 - pad, num_columns + pad + 0.5)
    ax.set_ylim(-0.5 - pad, num_qubits + pad + 0.5)

    # all y-coords are negative for reverse bit ordering
    if reverse_bits:
        ax.invert_yaxis()

    # hide frame
    ax.axis("off")

    # force 1:1 aspect ratio (not crucial; fun to relax)
    ax.set_aspect("equal")

    # save the figure
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=300)

    # render circuit immediately
    plt.show()
