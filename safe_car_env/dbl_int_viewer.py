import dataclasses
import math
import time
from typing import Any, Optional

import numpy as np
import pygame
from pygame import Surface, gfxdraw
from safe_car_env.utils import ActuatorLimits, NpFloat, interp
from safe_car_env.viewer_utils import (
    Colors,
    FrameSaver,
    Line,
    Text,
    Viewer,
)


@dataclasses.dataclass
class DblIntViewerParams:  # pylint: disable=too-many-instance-attributes
    x_axis: tuple[tuple[float, float], tuple[float, float]]
    y_axis: tuple[float, float]

    pos_to_x: tuple[float, float]

    x_tick_sep: float
    x_tick_offsets: tuple[float, float]

    width: int
    height: int

    agent_width: float

    text_pos: tuple[float, float]
    text_y_sep: float

    done_delay_sec: float

    graph_x: tuple[float, float]
    graph_y: tuple[float, float]


class DblIntViewer:
    def __init__(
        self,
        act_lims: ActuatorLimits,
        params: DblIntViewerParams,
        frame_saver: Optional[FrameSaver],
    ):
        self.act_lims = act_lims
        self.viewer = Viewer(params.width, params.height, "DblInt")
        self.params = params
        self.frame_saver = frame_saver
        self.dist_idx: Optional[int] = None
        self.policy: Optional[list[Any]] = None

    def reset(self) -> None:
        if self.frame_saver is not None:
            self.frame_saver.reset()

    def set_dist_idx(self, dist_idx: int) -> None:
        self.dist_idx = dist_idx

    # pylint: disable=too-many-positional-arguments,unused-argument
    def render(
        self, state: NpFloat, action: float, t: float, done: bool, obs: NpFloat
    ) -> None:
        surf = self.viewer.make_surf()

        x_axis = self._add_axes(surf)[0]
        self._add_agent(surf, state[0], x_axis.y0())
        self._add_text(surf, state, action, t)

        self.viewer.screen.blit(surf, (0, 0))

        if self.frame_saver is not None:
            img = self.viewer.get_img()
            self.frame_saver.add_frame(img)

        self.viewer.finish_render()

        if done:
            time.sleep(self.params.done_delay_sec)

    def _get_x(self, state_coord_x: float) -> float:
        p = self.params
        return interp(
            state_coord_x, p.pos_to_x[0], p.pos_to_x[1], p.x_axis[0][0], p.x_axis[0][1]
        )

    def _add_axes(self, surf: Surface) -> tuple[Line, Line]:

        p = self.params
        x_axis = Line(x=self.params.x_axis[0], y=p.x_axis[1])

        origin_x = self._get_x(0)
        y_axis = Line(x=[origin_x, origin_x], y=p.y_axis)

        self.viewer.add_line(surf, x_axis, Colors.BLACK, 0)
        self.viewer.add_line(surf, y_axis, Colors.BLACK, 0)

        def _add_tick(_state_coord_x: float) -> None:
            _x = self._get_x(_state_coord_x)
            if _x > x_axis.x1() + 1e-3 or _x < x_axis.x0() - 1e-3:
                return
            _y = y_axis.y0() + np.array(p.x_tick_offsets)
            _ln = Line(x=[_x, _x], y=_y)
            self.viewer.add_line(surf, _ln, Colors.BLACK, 0)

        if p.pos_to_x[0] < 0:
            for i in range(1, 1 + math.floor(-p.pos_to_x[0] / p.x_tick_sep)):
                _add_tick(i * p.x_tick_sep)

        for i in range(1, 5 + math.floor(p.pos_to_x[1] / p.x_tick_sep)):
            _add_tick(i * p.x_tick_sep)

        return x_axis, y_axis

    def _add_agent(self, surf: Surface, agent_pos: float, render_y: float) -> None:
        p = self.params
        violation = agent_pos < 0
        color = Colors.RED if violation else Colors.BLACK

        pos = self.viewer.convert([self._get_x(agent_pos), render_y])
        width = self.viewer.convert([p.agent_width, p.agent_width])[0]
        agent_rect = pygame.Rect(pos[0], pos[1] - width, width, width)
        gfxdraw.rectangle(surf, agent_rect, color)

    def _add_text(self, surf: Surface, state: NpFloat, action: float, t: float) -> None:

        def _format(_x: float) -> str:
            if _x <= -10:
                _num_zeros = 0
            elif _x < 0:
                _num_zeros = 1
            elif _x < 10:
                _num_zeros = 2
            else:
                _num_zeros = 1
            return (" " * _num_zeros) + f"{float(_x):.1f}"  # NOQA

        def _text(_label: str, _val: float, _y_offset: Any) -> None:
            _p = np.array(p.text_pos).copy()
            _p[1] -= _y_offset
            _t = f"{_label}: {_format(_val)}"
            self.viewer.add_text(surf, Text(_t, _p, False))  # typ3: ignore

        p = self.params
        _text("x", state[0], 0)
        _text("v", state[1], p.text_y_sep)
        _text("a", action, 2 * p.text_y_sep)
        _text("t", t, 3 * p.text_y_sep)
