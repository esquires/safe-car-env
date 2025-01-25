# https://stackoverflow.com/a/55769463
import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import unitpy
from safe_car_env.car import AllVehState, SafetySpec, SafetySummary
from safe_car_env.utils import ActuatorLimits, NpFloat
from safe_car_env.viewer_utils import (
    COLOR,
    Colors,
    FrameSaver,
    Line,
    Text,
    Viewer,
    make_bracket,
    rot_center,
)

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = ""

import pygame  # pylint: disable=wrong-import-position
from pygame import Surface  # pylint: disable=wrong-import-position


@dataclasses.dataclass
class LegendParams:
    y_sep: float
    x_sep: float
    line_len: float


@dataclasses.dataclass
class CarEnvViewerParams:  # pylint: disable=too-many-instance-attributes
    W_lane_render: float
    road_beg_x: float
    road_end_x: float
    road_start_y: float

    dash_width_m: float
    dash_step_m: float

    vel_y: float
    vel_height: float
    vel_text_sep: float

    vel_x: float
    vel_x_sep: float
    vel_width: float

    time_x: float
    time_y: float

    width: int
    height: int
    car_img_scale: float

    done_delay_sec: float
    done_msg_x: float
    done_msg_y: float

    vel_graph_x: tuple[float, float]
    vel_graph_y: tuple[float, float]

    steering_graph_x: tuple[float, float]
    steering_graph_y: tuple[float, float]

    legend_params: LegendParams

    constraint_graph_x: float
    constraint_graph_y: float


# pylint: disable=too-many-instance-attributes
class CarEnvViewer:
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        W_lane: float,
        safety_spec: SafetySpec,
        lead_start_x: float,
        act_lims: ActuatorLimits,
        params: CarEnvViewerParams,
        frame_saver: Optional[FrameSaver],
    ) -> None:

        self.viewer = Viewer(params.width, params.height, "Car")

        self.W_lane = W_lane
        self.safety_spec = safety_spec
        self.act_lims = act_lims
        self.dist_idx: Optional[int] = None
        self.frame_saver = frame_saver
        self.policy: Optional[list[Any]] = None

        # The image is released under public domain
        # https://shmector.com/free-vector/auto/lamborghini_car_top_view/11-0-399]
        car_fname = Path(__file__).parent / "car.png"
        img = pygame.image.load(car_fname)

        scale = params.car_img_scale
        self.car_img = pygame.transform.scale(
            img, (img.get_width() * scale, img.get_height() * scale)
        )
        self.params = params
        self.lead_start_x = lead_start_x

        # the entire screen coordinates vary from 0 to 1 so the screen takes value of 1
        # there are 2 lanes so that takes up 2 * W_lane_render
        # This leaves blank sides of (1 - 2 * W_lane_render) / 2
        lane0_y = params.road_start_y
        self.lane0 = Line(
            point1=[params.road_beg_x, lane0_y], point2=[params.road_end_x, lane0_y]
        )
        self.lane1 = Line(x=self.lane0.x(), y=self.lane0.y() + 2 * params.W_lane_render)

    def reset(self) -> None:
        if self.frame_saver is not None:
            self.frame_saver.reset()

    # pylint: disable=too-many-statements,too-many-locals
    def render(
        self,
        states: AllVehState,
        action: NpFloat,  # pylint: disable=unused-argument
        safety_summary: SafetySummary,
        t: float,
        done: bool,
        info: dict[str, Any],
        obs: NpFloat,  # pylint: disable=unused-argument
    ) -> None:

        surf = self.viewer.make_surf()

        def _draw_car_sep(
            _state_j: NpFloat, _dist_j: float, _too_close_j: bool
        ) -> None:
            _text_pos, _bracket = self._get_car_sep_dims(_state_j, states.c)

            _cmp = ">" if _dist_j >= 0 else "<"
            _clr = Colors.RED if _too_close_j else Colors.BLACK

            for _ln in _bracket:
                self.viewer.add_line(surf, _ln, _clr, 0)

            # associated text
            _txt = f"{_dist_j:.1f}m {_cmp} {safety_summary.dist_thresh:.1f}m"
            self.viewer.add_text(surf, Text(_txt, _text_pos, False, _clr))

        def _draw_thresh_line() -> None:
            _x = self._get_road_x(safety_summary.dist_thresh)
            _line = Line([_x, self.lane0.y0()], [_x, self.lane1.y1()])
            self.viewer.add_line(surf, _line, Colors.BLACK, 0)

        def _draw_lane_sep(_lane_des: str) -> None:
            _vals = self._get_lane_sep_dims(states.c, _lane_des, safety_summary.offset)
            if _vals is None:
                return
            _text_pos, _bracket, _clr, _dist = _vals
            _txt = f"{_dist:.1f}m"
            self.viewer.add_text(surf, Text(_txt, _text_pos, True, _clr))

            for _ln in _bracket:
                self.viewer.add_line(surf, _ln, _clr, 0)

        def _draw_vel() -> None:
            _v_a = states.a[2]
            _v_b = states.b[2]
            _v_c = states.c[2]

            _vel_lines, _vel_text_pos, _vel_text_pos2 = self._get_vel_dims(
                _v_a, _v_b, _v_c
            )

            for _ln_vert, _ln_horz in _vel_lines:
                self.viewer.add_line(surf, _ln_vert, Colors.BLACK, 0)
                self.viewer.add_line(surf, _ln_horz, Colors.BLACK, 2)

            def _text(_p: tuple[float, float], _v: float, _t: str) -> None:
                _clr = Colors.RED if _v > self.safety_spec.v_lim else Colors.BLACK
                self.viewer.add_text(surf, Text(_t, _p, True, _clr))

            for _pos, _vel in zip(_vel_text_pos, [_v_a, _v_b, _v_c]):
                _vel_mph = (float(_vel) * unitpy.Unit("m/s")).to("mile per hour").value
                _text(_pos, _vel, f"{_vel_mph:.1f} mph")

            for _pos2, _des in zip(
                _vel_text_pos2, ["Lower Lane", "Upper Lane", "agent"]
            ):
                _text(_pos2, 0, _des)

        def _draw_time() -> None:
            _pos = (self.params.time_x, self.params.time_y)
            self.viewer.add_text(surf, Text(f"t: {t:.1f}", _pos, False))

        def _draw_done() -> None:
            if info["exit_crash"]:
                _text = "crashed"
                _color = Colors.RED
            elif info["exit_off_road"]:
                _text = "off road"
                _color = Colors.RED
            elif info["exit_t_thresh"]:
                _text = "time elapsed"
                _color = Colors.BLACK
            else:
                raise NotImplementedError("done condition not recognized")

            _pos = (self.params.done_msg_x, self.params.done_msg_y)
            self.viewer.add_text(surf, Text(_text, _pos, False, _color))

        def _add_car_img(_img: Surface, _state_j: NpFloat) -> None:
            _x = self._get_road_x(_state_j[0] - states.c[0])
            _img_height = self.car_img.get_height() / self.viewer.height
            _y = self._rescale_lane(_state_j[1]) + self.lane0.y0() + 0.6 * _img_height
            _x, _y = self.viewer.convert((_x, _y))
            self.viewer.screen.blit(_img, (_x, _y))

        self.viewer.add_line(surf, self.lane0, Colors.BLACK, 4)
        self.viewer.add_line(surf, self.lane1, Colors.BLACK, 4)

        dashes = self._get_lane_dashes(states.c[0])
        for dash in dashes:
            self.viewer.add_line(surf, dash, Colors.GOLD, 4)  # type: disable

        _draw_car_sep(states.a[:2], safety_summary.dist_a, safety_summary.too_close_a)
        _draw_car_sep(states.b[:2], safety_summary.dist_b, safety_summary.too_close_b)
        _draw_thresh_line()

        for lane_des in ["L1", "H1", "L2", "H2"]:
            _draw_lane_sep(lane_des)

        _draw_vel()
        _draw_time()

        if done:
            _draw_done()

        self.viewer.screen.blit(surf, (0, 0))

        _add_car_img(self.car_img, states.a)
        _add_car_img(self.car_img, states.b)

        car_rotate = rot_center(self.car_img, np.rad2deg(states.c[3]))
        _add_car_img(car_rotate, states.c)

        if self.frame_saver is not None:
            img = pygame.surfarray.array3d(self.viewer.screen)
            img = np.transpose(img, (1, 0, 2))
            self.frame_saver.add_frame(img)

        self.viewer.finish_render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if done and self.params.done_delay_sec > 0:
            time.sleep(self.params.done_delay_sec)

    def set_dist_idx(self, dist_idx: int) -> None:
        self.dist_idx = dist_idx

    def _get_lane_dashes(self, x_c: float) -> tuple[Line, Line, list[Line]]:

        p = self.params

        x = (x_c // self.lead_start_x - 1) * self.lead_start_x
        y = float(np.mean([self.lane0.y0(), self.lane1.y0()]))

        dashes = []

        while x < x_c + self.lead_start_x:
            visible = x + p.dash_width_m > x_c
            if visible:
                # meters
                beg_x = max(x_c, x)
                end_x = min(x + p.dash_width_m, x_c + self.lead_start_x)

                # pct of frame
                def _interp(_x: float) -> float:
                    _pct = (_x - x_c) / self.lead_start_x
                    return self.lane0.x0() + _pct * (self.lane0.x1() - self.lane0.x0())

                beg_x = _interp(beg_x)
                end_x = _interp(end_x)

                dashes.append(Line([beg_x, y], [end_x, y]))
            x += p.dash_width_m + p.dash_step_m

        return dashes

    def _get_car_sep_dims(
        self, state_j: NpFloat, state_c: NpFloat
    ) -> tuple[tuple[float, float], list[Line]]:
        text_offset = 0.25 * self.params.W_lane_render
        j_is_upper_lane = state_j[1] > self.W_lane

        if j_is_upper_lane:
            y_bracket = self.lane1.y1() + text_offset
            y_text = y_bracket + 3 * text_offset
        else:
            y_bracket = self.lane0.y0() - text_offset
            y_text = y_bracket - text_offset

        x_bracket_beg = self._get_road_x(0)
        x_bracket_end = self._get_road_x(state_j[0] - state_c[0])

        line = Line([x_bracket_beg, y_bracket], [x_bracket_end, y_bracket])
        bracket_lines = [line] + list(make_bracket(line, 0.02))
        x_text = x_bracket_beg

        text_pos = (x_text, y_text)
        return text_pos, bracket_lines

    # pylint: disable=too-many-locals
    def _get_lane_sep_dims(
        self, state_c: NpFloat, lane_des: str, car_offset: float
    ) -> Optional[tuple[tuple[float, float], list[Line], COLOR, float]]:

        txt_offset = 0.2 * self.W_lane
        y_c = state_c[1]
        if lane_des == "L1":
            y0, y1 = 0.0, y_c - car_offset
            yt = y0
            if y1 > self.W_lane:
                return None
            clr = Colors.RED if y1 < 0 else Colors.BLACK
        elif lane_des == "H1":
            y0, y1 = y_c + car_offset, self.W_lane
            yt = self.W_lane - txt_offset
            if y0 < 0 or y0 > self.W_lane:
                return None
            clr = Colors.BLACK
        elif lane_des == "L2":
            y0, y1 = self.W_lane, y_c - car_offset
            yt = self.W_lane + txt_offset
            if y1 < self.W_lane or y1 > 2 * self.W_lane:
                return None
            clr = Colors.BLACK
        elif lane_des == "H2":
            y0, y1 = y_c + car_offset, 2 * self.W_lane
            yt = 2 * self.W_lane
            if y0 < self.W_lane:
                return None
            clr = Colors.RED if y0 > 2 * self.W_lane else Colors.BLACK
        else:
            raise NotImplementedError()

        dist = y1 - y0
        y0 = self.lane0.y0() + self._rescale_lane(y0)
        y1 = self.lane0.y0() + self._rescale_lane(y1)
        yt = self.lane0.y0() + self._rescale_lane(yt)
        x = self._get_road_x(0)

        # associated text
        xt = x - 0.05

        x = x - 0.02
        line = Line([x, y0], [x, y1])

        w = 0.02 * self.viewer.height / self.viewer.width
        bracket_lines = [line] + list(make_bracket(line, w))

        return (xt, yt), bracket_lines, clr, dist

    def _get_vel_dims(
        self, v_a: float, v_b: float, v_c: float
    ) -> tuple[
        list[tuple[Line, Line]], list[tuple[float, float]], list[tuple[float, float]]
    ]:

        p = self.params
        y_beg = p.vel_y
        y_end = y_beg + p.vel_height

        def _make_vel(
            _v: float, _x: float
        ) -> tuple[tuple[Line, Line], tuple[float, float], tuple[float, float]]:

            _line_vert = Line([_x, p.vel_y], [_x, p.vel_y + p.vel_height])
            _y_horz = y_beg + (_v / self.safety_spec.v_lim) * (y_end - y_beg)
            _line_horz = Line(
                [_x - p.vel_width / 2, _y_horz], [_x + p.vel_width / 2, _y_horz]
            )

            _text_pos = (_x, p.vel_y + p.vel_height + p.vel_text_sep)
            _text_pos2 = (_x, p.vel_y + p.vel_height + 2 * p.vel_text_sep)

            return (_line_vert, _line_horz), _text_pos, _text_pos2

        line_a, text_a_pos, text_a_pos2 = _make_vel(v_a, p.vel_x + 2 * p.vel_x_sep)
        line_b, text_b_pos, text_b_pos2 = _make_vel(v_b, p.vel_x + p.vel_x_sep)
        line_c, text_c_pos, text_c_pos2 = _make_vel(v_c, p.vel_x)

        lines = [line_a, line_b, line_c]
        text_pos = [text_a_pos, text_b_pos, text_c_pos]
        text_pos2 = [text_a_pos2, text_b_pos2, text_c_pos2]

        return lines, text_pos, text_pos2

    def _get_road_x(self, rel_x: float) -> float:
        road_x = (
            self.lane0.x0()
            + (self.lane0.x1() - self.lane0.x0()) * rel_x / self.lead_start_x
        )
        return road_x

    def _rescale_lane(self, y: float) -> float:
        return self.params.W_lane_render * y / self.W_lane
