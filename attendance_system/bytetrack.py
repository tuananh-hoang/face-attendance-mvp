"""
bytetrack.py
Simplified ByteTrack implementation cho camera cố định 1 cửa.

ByteTrack (2022) — key idea:
  SORT/DeepSORT bỏ low-confidence detection → mất track khi bị che
  ByteTrack dùng CẢ high + low confidence → giảm ID switch đáng kể

Pipeline mỗi frame:
  1. Tách detections thành high (>=0.6) và low (0.1–0.6)
  2. Match high detections với active tracks (IoU)
  3. Match low detections với unmatched tracks (vớt lại track bị che)
  4. Unmatched high detections → new tracks
  5. Tracks không match lâu → remove

Kalman Filter state: [x, y, w, h, vx, vy, vw, vh]
  x, y = tâm bbox
  w, h = kích thước
  vx, vy, vw, vh = vận tốc (constant velocity model)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ════════════════════════════════════════════
#  Kalman Filter
# ════════════════════════════════════════════

class KalmanFilter:
    """
    Constant velocity model cho bbox tracking.
    State: [x, y, w, h, vx, vy, vw, vh]
    Observation: [x, y, w, h]
    """

    def __init__(self):
        dt = 1.0  # 1 frame time step

        # State transition matrix (8×8)
        self.F = np.eye(8)
        self.F[0, 4] = dt
        self.F[1, 5] = dt
        self.F[2, 6] = dt
        self.F[3, 7] = dt

        # Observation matrix (4×8) — chỉ observe position/size
        self.H = np.eye(4, 8)

        # Process noise
        self.Q = np.eye(8) * 0.01
        self.Q[4:, 4:] *= 10.0  # vận tốc uncertainty cao hơn

        # Observation noise
        self.R = np.eye(4) * 1.0

        # Initial uncertainty
        self.P0 = np.eye(8)
        self.P0[4:, 4:] *= 100.0

    def initiate(self, bbox_xywh: np.ndarray):
        """Khởi tạo track từ bbox [x, y, w, h]."""
        x = np.zeros(8)
        x[:4] = bbox_xywh
        P = self.P0.copy()
        return x, P

    def predict(self, x, P):
        """Dự đoán state ở frame tiếp theo."""
        x = self.F @ x
        P = self.F @ P @ self.F.T + self.Q
        return x, P

    def update(self, x, P, bbox_xywh: np.ndarray):
        """Cập nhật state dựa trên observation mới."""
        z = bbox_xywh
        y = z - self.H @ x                          # innovation
        S = self.H @ P @ self.H.T + self.R          # innovation covariance
        K = P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        x = x + K @ y
        P = (np.eye(8) - K @ self.H) @ P
        return x, P


# ════════════════════════════════════════════
#  Track
# ════════════════════════════════════════════

class TrackState:
    Tracked  = 0
    Lost     = 1
    Removed  = 2


class Track:
    _id_counter = 0

    def __init__(self, bbox_xywh: np.ndarray, score: float, kf: KalmanFilter):
        Track._id_counter += 1
        self.track_id  = Track._id_counter
        self.state     = TrackState.Tracked
        self.score     = score
        self.hits      = 1          # số frame liên tiếp được detect
        self.age       = 1          # tổng số frame tồn tại
        self.time_since_update = 0  # số frame không được update

        # Kalman state
        self.kf      = kf
        self.x, self.P = kf.initiate(bbox_xywh)

        # Buffer frames tốt nhất để recognize sau khi track exit
        self.best_score     = score
        self.best_frame     = None   # numpy frame
        self.best_bbox      = None   # [x1, y1, x2, y2]
        self.frame_count    = 0

    @property
    def bbox_xywh(self) -> np.ndarray:
        return self.x[:4].copy()

    @property
    def bbox_xyxy(self) -> np.ndarray:
        x, y, w, h = self.x[:4]
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])

    def predict(self):
        self.x, self.P = self.kf.predict(self.x, self.P)
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox_xywh: np.ndarray, score: float,
               frame: np.ndarray = None, bbox_xyxy: np.ndarray = None):
        self.x, self.P = self.kf.update(self.x, self.P, bbox_xywh)
        self.score     = score
        self.hits     += 1
        self.time_since_update = 0
        self.state     = TrackState.Tracked
        self.frame_count += 1

        # Giữ frame có score cao nhất để recognize sau
        if frame is not None and score > self.best_score:
            self.best_score = score
            self.best_frame = frame.copy()
            self.best_bbox  = bbox_xyxy.copy() if bbox_xyxy is not None else None


# ════════════════════════════════════════════
#  IoU + Hungarian matching
# ════════════════════════════════════════════

def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Tính IoU matrix giữa 2 tập bbox [x1,y1,x2,y2].
    bboxes_a: (M, 4), bboxes_b: (N, 4) → returns (M, N)
    """
    if len(bboxes_a) == 0 or len(bboxes_b) == 0:
        return np.zeros((len(bboxes_a), len(bboxes_b)))

    ax1, ay1, ax2, ay2 = bboxes_a[:, 0], bboxes_a[:, 1], bboxes_a[:, 2], bboxes_a[:, 3]
    bx1, by1, bx2, by2 = bboxes_b[:, 0], bboxes_b[:, 1], bboxes_b[:, 2], bboxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def hungarian_match(cost_matrix: np.ndarray, threshold: float):
    """
    Hungarian matching với threshold.
    Trả về (matched_rows, matched_cols, unmatched_rows, unmatched_cols)
    """
    if cost_matrix.size == 0:
        return [], [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_rows, matched_cols = [], []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= threshold:
            matched_rows.append(r)
            matched_cols.append(c)

    unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in matched_rows]
    unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in matched_cols]

    return matched_rows, matched_cols, unmatched_rows, unmatched_cols


# ════════════════════════════════════════════
#  ByteTracker
# ════════════════════════════════════════════

class ByteTracker:
    """
    ByteTrack tracker.

    Params:
        high_thresh   : ngưỡng high-confidence detection (default 0.6)
        low_thresh    : ngưỡng low-confidence detection (default 0.1)
        iou_thresh    : ngưỡng IoU để match (default 0.3)
        max_lost      : số frame tối đa track ở trạng thái Lost trước khi xóa
        min_hits      : số frame tối thiểu để track được confirm
        exit_callback : hàm gọi khi track bị xóa (để trigger recognize)
    """

    def __init__(
        self,
        high_thresh   : float = 0.6,
        low_thresh    : float = 0.1,
        iou_thresh    : float = 0.3,
        max_lost      : int   = 30,   # ~1.2s ở 25FPS
        min_hits      : int   = 3,    # confirm sau 3 frame liên tiếp
        exit_callback = None,
    ):
        self.high_thresh   = high_thresh
        self.low_thresh    = low_thresh
        self.iou_thresh    = iou_thresh
        self.max_lost      = max_lost
        self.min_hits      = min_hits
        self.exit_callback = exit_callback

        self.kf             = KalmanFilter()
        self.tracked_tracks : list = []   # đang active
        self.lost_tracks    : list = []   # tạm mất
        self.frame_id       = 0

    def update(self, detections: list, frame: np.ndarray = None) -> list:
        """
        Update tracker với detections của frame hiện tại.

        detections: list of {
            'bbox_xyxy': [x1,y1,x2,y2],
            'score'    : float,
        }

        Trả về list Track đang được confirmed (hits >= min_hits).
        """
        self.frame_id += 1

        # ── Tách high/low confidence ──
        high_dets = [d for d in detections if d['score'] >= self.high_thresh]
        low_dets  = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]

        # ── Predict tất cả tracks ──
        for t in self.tracked_tracks + self.lost_tracks:
            t.predict()

        # ── Step 1: Match high dets với tracked tracks ──
        active_tracks = [t for t in self.tracked_tracks]

        matched_t, matched_d, unmatched_t, unmatched_d = self._iou_match(
            active_tracks, high_dets
        )

        # Update matched
        for ti, di in zip(matched_t, matched_d):
            det = high_dets[di]
            bxyxy = np.array(det['bbox_xyxy'])
            bxywh = self._xyxy2xywh(bxyxy)
            active_tracks[ti].update(bxywh, det['score'], frame, bxyxy)

        # ── Step 2: Match low dets với unmatched tracks (ByteTrack key idea) ──
        unmatched_tracks_1 = [active_tracks[i] for i in unmatched_t]
        matched_t2, matched_d2, unmatched_t2, _ = self._iou_match(
            unmatched_tracks_1, low_dets
        )

        for ti, di in zip(matched_t2, matched_d2):
            det = low_dets[di]
            bxyxy = np.array(det['bbox_xyxy'])
            bxywh = self._xyxy2xywh(bxyxy)
            unmatched_tracks_1[ti].update(bxywh, det['score'], frame, bxyxy)

        # Tracks không match bước 2 → Lost
        for i in unmatched_t2:
            t = unmatched_tracks_1[i]
            t.state = TrackState.Lost
            self.lost_tracks.append(t)

        # ── Step 3: Match lost tracks với unmatched high dets ──
        unmatched_high_dets = [high_dets[i] for i in unmatched_d]
        matched_lost, matched_hd, _, unmatched_hd2 = self._iou_match(
            self.lost_tracks, unmatched_high_dets
        )

        for ti, di in zip(matched_lost, matched_hd):
            det = unmatched_high_dets[di]
            bxyxy = np.array(det['bbox_xyxy'])
            bxywh = self._xyxy2xywh(bxyxy)
            self.lost_tracks[ti].update(bxywh, det['score'], frame, bxyxy)
            self.lost_tracks[ti].state = TrackState.Tracked

        # ── Step 4: New tracks từ unmatched high dets ──
        for i in unmatched_hd2:
            det = unmatched_high_dets[i]
            bxyxy = np.array(det['bbox_xyxy'])
            bxywh = self._xyxy2xywh(bxyxy)
            new_track = Track(bxywh, det['score'], self.kf)
            new_track.best_frame = frame.copy() if frame is not None else None
            new_track.best_bbox  = bxyxy.copy()
            self.tracked_tracks.append(new_track)

        # ── Step 5: Remove expired lost tracks ──
        removed = []
        still_lost = []
        for t in self.lost_tracks:
            if t.time_since_update > self.max_lost:
                t.state = TrackState.Removed
                removed.append(t)
            else:
                still_lost.append(t)
        self.lost_tracks = still_lost

        # Callback khi track bị remove (trigger recognize)
        if self.exit_callback:
            for t in removed:
                if t.hits >= self.min_hits and t.best_frame is not None:
                    self.exit_callback(t)

        # ── Rebuild tracked list ──
        self.tracked_tracks = [
            t for t in self.tracked_tracks
            if t.state == TrackState.Tracked
        ]
        # Thêm các lost track vừa được re-activate
        for t in self.lost_tracks:
            if t.state == TrackState.Tracked and t not in self.tracked_tracks:
                self.tracked_tracks.append(t)
        self.lost_tracks = [t for t in self.lost_tracks if t.state == TrackState.Lost]

        # Trả về tracks đã confirmed
        return [t for t in self.tracked_tracks if t.hits >= self.min_hits]

    def _iou_match(self, tracks, dets):
        if not tracks or not dets:
            return [], [], list(range(len(tracks))), list(range(len(dets)))

        track_bboxes = np.array([t.bbox_xyxy for t in tracks])
        det_bboxes   = np.array([d['bbox_xyxy'] for d in dets])

        iou_mat  = iou_batch(track_bboxes, det_bboxes)
        cost_mat = 1 - iou_mat

        return hungarian_match(cost_mat, 1 - self.iou_thresh)

    @staticmethod
    def _xyxy2xywh(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1])