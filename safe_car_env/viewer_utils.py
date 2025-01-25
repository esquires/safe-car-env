import dataclasses

# https://stackoverflow.com/a/55769463
import os
import sys
from pathlib import Path
from typing import Optional, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.distributions import Distribution

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""
import pygame  # pylint: disable=wrong-import-position
from pygame import Surface, gfxdraw  # pylint: disable=wrong-import-position
from pygame.font import Font  # pylint: disable=wrong-import-position
from safe_car_env.utils import NpFloat, NpInt  # pylint: disable=wrong-import-position


class Colors:
    BLACK = pygame.colordict.THECOLORS["black"]
    GOLD = pygame.colordict.THECOLORS["gold2"]
    RED = pygame.colordict.THECOLORS["firebrick1"]
    SKY_BLUE = pygame.colordict.THECOLORS["deepskyblue"]
    LIME_GREEN = pygame.colordict.THECOLORS["chartreuse3"]
    MAGENTA = pygame.colordict.THECOLORS["darkmagenta"]
    BROWN = pygame.colordict.THECOLORS["burlywood4"]


COLOR = tuple[int, int, int, int]
T = TypeVar("T")


class Line:
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        point1: Optional[npt.ArrayLike] = None,
        point2: Optional[npt.ArrayLike] = None,
        x: Optional[npt.ArrayLike] = None,
        y: Optional[npt.ArrayLike] = None,
        line: Optional[NpFloat] = None,
    ):
        if point1 is not None and point2 is not None:
            self.line = np.vstack((point1, point2))
        elif x is not None and y is not None:
            self.line = np.array([[x[0], y[0]], [x[1], y[1]]])
        elif line is not None:
            self.line = line
        else:
            raise NotImplementedError()

    def __repr__(self) -> str:
        return self.line.__repr__()

    def x0(self) -> float:
        return self.line[0, 0]

    def x1(self) -> float:
        return self.line[1, 0]

    def y0(self) -> float:
        return self.line[0, 1]

    def y1(self) -> float:
        return self.line[1, 1]

    def x(self) -> NpFloat:
        return self.line[:, 0]

    def y(self) -> NpFloat:
        return self.line[:, 1]

    def p0(self) -> NpFloat:
        return self.line[0, :]

    def p1(self) -> NpFloat:
        return self.line[1, :]


def make_bracket(line: Line, width: float) -> tuple[Line, Line]:

    ang = np.arctan2(line.y1() - line.y0(), line.x1() - line.x0())
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])

    def _make_line(_pt: NpFloat) -> Line:
        _p1 = _pt + R.dot([0, width])
        _p2 = _pt + R.dot([0, -width])
        return Line(_p1, _p2)

    line1 = _make_line(line.line[0, :])
    line2 = _make_line(line.line[1, :])

    return line1, line2


def rot_center(image: Surface, angle: float) -> Surface:
    # https://www.pygame.org/wiki/RotateCenter
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image


@dataclasses.dataclass
class GraphLine:
    x: npt.ArrayLike
    y: npt.ArrayLike
    color: COLOR
    size: int
    legend: bool
    name: str = ""


@dataclasses.dataclass
class Text:
    content: str
    point: tuple[float, float]
    center: bool = True
    color: COLOR = Colors.BLACK
    font: Optional[Font] = None


@dataclasses.dataclass
class Legend:
    pos_in_graph: tuple[float, float]
    y_sep: float
    x_sep: float
    line_len: float


@dataclasses.dataclass
class Graph:
    lines: list[GraphLine]
    x_lim: tuple[float, float]
    y_lim: tuple[float, float]
    x_axis: Optional[GraphLine]
    y_axis: Optional[GraphLine]
    title: Text
    legend: Optional[Legend]


class Viewer:
    def __init__(self, width: int, height: int, caption: str):
        pygame.init()
        pygame.display.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = Font("freesansbold.ttf", 20)
        pygame.display.set_caption(caption)

    def make_surf(self) -> Surface:
        surf = Surface((self.width, self.height))
        surf.fill((255, 255, 255))
        return surf

    def convert(self, data: T) -> T:
        # data should be percentage of screen dimensions
        # output is (0, 0) in the bottom left corner
        if isinstance(data, Line):
            return Line(
                x=(data.x() * self.width).astype(np.int32),
                y=((1 - data.y()) * self.height).astype(np.int32),
            )
        else:
            assert len(data) == 2
            out = [int(data[0] * self.width), int((1 - data[1]) * self.height)]
            if isinstance(data, np.ndarray):
                return np.array(out)
            else:
                return type(data)(out)

    def add_line(self, surf: Surface, line: Line, clr: COLOR, size: int) -> None:
        line = self.convert(line)
        offset_upper = (size // 2 + 1) if size % 2 else size // 2
        offset_lower = size // 2

        coords = (
            (line.x0(), line.y0() + offset_upper),
            (line.x0(), line.y0() - offset_lower),
            (line.x1(), line.y1() - offset_lower),
            (line.x1(), line.y1() + offset_upper),
        )
        gfxdraw.aapolygon(surf, coords, clr)
        gfxdraw.filled_polygon(surf, coords, clr)

    def add_graph(self, surf: Surface, pos: Line, graph: Graph) -> None:

        def _get_x(_x0: float) -> float:
            _pct = (_x0 - graph.x_lim[0]) / (graph.x_lim[1] - graph.x_lim[0])
            return pos.x0() + _pct * (pos.x1() - pos.x0())

        def _get_y(_y0: float) -> float:
            _pct = (_y0 - graph.y_lim[0]) / (graph.y_lim[1] - graph.y_lim[0])
            return pos.y0() + _pct * (pos.y1() - pos.y0())

        def _add_line(_graph_line: GraphLine, _idx: int) -> None:
            _x0 = _get_x(_graph_line.x[_idx])
            _y0 = _get_y(_graph_line.y[_idx])
            _x1 = _get_x(_graph_line.x[_idx + 1])
            _y1 = _get_y(_graph_line.y[_idx + 1])
            _line = Line([_x0, _y0], [_x1, _y1])
            self.add_line(surf, _line, _graph_line.color, _graph_line.size)

        def _add_legend_entry(_i: int, _line: GraphLine) -> None:
            assert graph.legend is not None
            _leg = graph.legend
            _x = _get_x(_leg.pos_in_graph[0])
            _y = _get_y(_leg.pos_in_graph[1]) + _i * _leg.y_sep

            _marker_line = Line(x=[_x, _x + _leg.line_len], y=[_y, _y])
            self.add_line(surf, _marker_line, _line.color, _line.size)
            _text = Text(
                _line.name, [_x + _leg.line_len + _leg.x_sep, _y + 0.01], False
            )
            self.add_text(surf, _text)

        graph_idx = 0
        for graph_line in graph.lines:
            x = np.array(graph_line.x)
            y = np.array(graph_line.y)

            assert len(x) == len(y)

            # plot the lines
            for j in range(len(x) - 1):
                _add_line(graph_line, j)

            if graph.legend is not None and graph_line.legend:
                _add_legend_entry(graph_idx, graph_line)
                graph_idx += 1

        # plot the axes
        if graph.x_axis is not None:
            _add_line(graph.x_axis, 0)

        if graph.y_axis is not None:
            _add_line(graph.y_axis, 0)

        if graph.title is not None:
            self.add_text(surf, graph.title)

    def finish_render(self) -> None:
        pygame.event.pump()
        self.clock.tick(60)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def add_text(self, surf: Surface, text: Text) -> None:

        font = text.font if text.font is not None else self.font
        rendered_text = font.render(text.content, True, text.color)
        point = self.convert(text.point)

        # https://www.reddit.com/r/pygame/comments/qw7fmk/comment/hl13rjc/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
        if text.center:
            rect = rendered_text.get_rect(center=(point[0], point[1]))
        else:
            rect = rendered_text.get_rect()
            rect.x = point[0]
            rect.y = point[1]
        surf.blit(rendered_text, rect)

    def get_img(self) -> NpInt:
        img = pygame.surfarray.array3d(self.screen)
        img = np.transpose(img, (1, 0, 2))
        return img


class FrameSaver:
    def __init__(self, prefix: Path, thresh: int, fps: int):
        self.prefix = prefix / "videos"
        self.prefix.mkdir(parents=True, exist_ok=True)

        self.thresh = thresh
        self.fps = fps

        self.episode_num = 0
        self.frames: list[NpFloat] = []

    def reset(self) -> None:
        if len(self.frames) >= self.thresh:
            # https://blog.finxter.com/5-best-ways-to-convert-a-python-numpy-array-to-video/
            video_file = self.prefix / f"{self.episode_num}.mp4"
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(self.frames, fps=self.fps)
            clip.write_videofile(str(video_file), audio=False)
        self.episode_num += 1

        self.frames = []

    def add_frame(self, frame: NpInt) -> None:
        self.frames.append(frame)


@dataclasses.dataclass
class PolicyGraphParams:
    min_logp: float
    max_logp: float

    impulse_height_pct: float
    impulse_arrow_height_pct: float
    impulse_arrow_width_pct: float


@dataclasses.dataclass
class PolicyGraphData:
    distributions: list[Distribution]
    safe_ac: Optional[Tensor]
    dist_idx: Optional[int]
    colors: list[COLOR]
    safe_ac_color: COLOR
    names: list[str]


# pylint: disable=too-many-locals,too-many-statements
def make_logp_graph(
    ac_idx: int,
    action: float,
    actions: Tensor,
    data: PolicyGraphData,
    params: PolicyGraphParams,
) -> Graph:

    def _get_size(_idx: int) -> int:
        if data.dist_idx is not None and data.dist_idx - 1 == _idx:
            return 5
        else:
            return 1

    def _make_line(
        _dist: Distribution, _color: COLOR, _sz: int, _nm: str
    ) -> list[GraphLine]:
        _logp = _dist.log_prob(actions)[:, ac_idx].squeeze(-1)

        _mask = _logp >= params.min_logp + 1e-3
        _logp = _logp[_mask]
        _actions = actions[_mask, ac_idx]

        try:
            _lim1, _lim2 = _dist.limits()
            _logp_lim1 = _dist.log_prob(_lim1)[:, ac_idx]
            _logp_lim2 = _dist.log_prob(_lim2)[:, ac_idx]

            _lim1 = _lim1[:, ac_idx]
            _lim2 = _lim2[:, ac_idx]

            _mask = torch.logical_and(_lim1 <= _actions, _actions <= _lim2)
            _logp = _logp[_mask]
            _actions = _actions[_mask]

        except AttributeError:
            _lim1, _lim2 = None, None

        if _lim1 is not None and _lim2 is not None:

            def __make_lim_line(__lim: Tensor, __logp: Tensor) -> GraphLine:
                __lim = float(__lim)
                __logp = float(__logp)
                return GraphLine(
                    [__lim, __lim], [params.min_logp, __logp], _color, _sz, False, _nm
                )

            def _make_lim_to_ac_line(
                __lim: Tensor, __logp_lim: Tensor, __ac_idx: int
            ) -> GraphLine:
                __lim = float(__lim)
                __ac = float(_actions[__ac_idx])
                __logp_lim = float(__logp_lim)
                __logp_ac = float(_logp[__ac_idx])

                return GraphLine(
                    [__lim, __ac], [__logp_lim, __logp_ac], _color, _sz, False, _nm
                )

            if len(_logp) == 0:
                _out = [
                    __make_lim_line(_lim1, _logp_lim1),
                    __make_lim_line(_lim2, _logp_lim2),
                ]
            else:
                _out = [
                    __make_lim_line(_lim1, _logp_lim1),
                    _make_lim_to_ac_line(_lim1, _logp_lim1, 0),
                    GraphLine(_actions, _logp, _color, _sz, False, _nm),
                    _make_lim_to_ac_line(_lim2, _logp_lim2, -1),
                    __make_lim_line(_lim2, _logp_lim2),
                ]
            return _out

        else:

            if len(_logp) <= 1:
                _mean = float(torch.clamp(_dist.mean[0, ac_idx], -1, 1))
                return _make_impulse(_mean, _color, _sz, _nm)
            else:
                return [GraphLine(_actions, _logp, _color, _sz, False, _nm)]

    def _make_impulse(_x: float, _color: COLOR, _sz: int, _nm: str) -> list[GraphLine]:
        _p = params
        _dy = _p.max_logp - _p.min_logp
        _y1 = _p.min_logp + _p.impulse_height_pct * _dy
        _y_offset = _p.impulse_arrow_height_pct * _dy
        _x_offset = _p.impulse_arrow_width_pct * (1 - (-1))

        def _helper(__x: list[float], __y: list[float]) -> GraphLine:
            return GraphLine(
                x=__x, y=__y, color=_color, size=_sz, legend=False, name=_nm
            )

        _line1 = _helper([_x, _x], [params.min_logp, _y1])
        _line2 = _helper([_x - _x_offset, _x], [_y1 - _y_offset, _y1])
        _line3 = _helper([_x + _x_offset, _x], [_y1 - _y_offset, _y1])

        return [_line1, _line2, _line3]

    def _make_action_line() -> GraphLine:
        _x = [action, action]
        _dy = 0.05 * (params.max_logp - params.min_logp)
        _y = [params.min_logp - _dy, params.min_logp + _dy]
        _action_line = GraphLine(_x, _y, Colors.BLACK, 1, False, "action")
        return _action_line

    lines: list[GraphLine] = []
    for i, (dist, clr, nm) in enumerate(
        zip(data.distributions, data.colors, data.names)
    ):
        temp_lines = _make_line(dist, clr, _get_size(i), nm)
        temp_lines[0].legend = True
        lines += temp_lines

    if data.safe_ac is not None:
        idx = len(data.distributions)
        lines += _make_impulse(
            float(data.safe_ac[0, ac_idx]),
            data.safe_ac_color,
            _get_size(idx),
            "safe action",
        )

    lines.append(_make_action_line())

    x_axis = GraphLine(
        x=[-1, 1],
        y=[params.min_logp, params.min_logp],
        color=Colors.BLACK,
        legend=False,
        size=0,
    )
    x_lim = (-1.0, 1.0)
    y_lim = (params.min_logp, params.max_logp)

    graph = Graph(
        lines=lines,
        x_lim=x_lim,
        y_lim=y_lim,
        x_axis=x_axis,
        y_axis=None,
        title=None,
        legend=None,
    )
    return graph
