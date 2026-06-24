import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QCursor, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except Exception:  # pragma: no cover
    torch = None
    build_sam2 = None
    SAM2ImagePredictor = None


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class Annotation:
    label: str
    color: tuple[int, int, int]
    box: list[int]
    score: float
    mask: np.ndarray


class SAM2Segmenter:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        if SAM2ImagePredictor is None or build_sam2 is None:
            raise RuntimeError("SAM2 未安装或导入失败，请先安装依赖并确认 sam2 可用。")
        self.device = device
        self.model = build_sam2(config_path, checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)

    def predict_box(self, box_xyxy: list[int]):
        box = np.array(box_xyxy, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        return masks[0].astype(bool), float(scores[0])


def infer_sam2_config_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """
    Infer SAM2 Hydra config name from checkpoint file name.
    Supports common names like:
    sam2_hiera_tiny.pt / small.pt / base_plus.pt / large.pt
    """
    ckpt_name = Path(checkpoint_path).name.lower()
    variant_to_hydra_config = [
        ("2.1_hiera_tiny", "configs/sam2.1/sam2.1_hiera_t.yaml"),
        ("2.1_hiera_small", "configs/sam2.1/sam2.1_hiera_s.yaml"),
        ("2.1_hiera_base_plus", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
        ("2.1_hiera_large", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        ("tiny", "configs/sam2/sam2_hiera_t.yaml"),
        ("small", "configs/sam2/sam2_hiera_s.yaml"),
        ("base_plus", "configs/sam2/sam2_hiera_b+.yaml"),
        ("base-plus", "configs/sam2/sam2_hiera_b+.yaml"),
        ("baseplus", "configs/sam2/sam2_hiera_b+.yaml"),
        ("large", "configs/sam2/sam2_hiera_l.yaml"),
    ]
    for key, config_name in variant_to_hydra_config:
        if key in ckpt_name:
            return config_name
    return None


def choose_device(mode: str = "auto") -> str:
    if torch is None:
        return "cpu"
    mode = (mode or "auto").strip().lower()
    if mode == "cpu":
        return "cpu"
    try:
        # Some environments print a CUDA driver mismatch warning here.
        # We suppress it and safely fall back to CPU.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cuda_ready = torch.cuda.is_available()
        if mode == "cuda":
            return "cuda" if cuda_ready else "cpu"
        return "cuda" if cuda_ready else "cpu"
    except Exception:
        return "cpu"


class BoxDrawLabel(QLabel):
    box_finished = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111319; border: 1px solid #2d3345;")
        self.setMinimumSize(420, 420)

        self._base_pixmap: Optional[QPixmap] = None
        self._display_size = (1, 1)
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0

        self._drawing = False
        self._start_point = QPoint()
        self._current_point = QPoint()

    def set_display_pixmap(self, pixmap: QPixmap):
        self._base_pixmap = pixmap
        self._update_scaled_pixmap()

    def clear_box(self):
        self._drawing = False
        self.update()

    def _update_scaled_pixmap(self):
        if self._base_pixmap is None:
            self.setPixmap(QPixmap())
            return

        container_w = max(1, self.width())
        container_h = max(1, self.height())
        pixmap_w = self._base_pixmap.width()
        pixmap_h = self._base_pixmap.height()
        scale = min(container_w / pixmap_w, container_h / pixmap_h)
        display_w = max(1, int(pixmap_w * scale))
        display_h = max(1, int(pixmap_h * scale))

        self._scale = scale
        self._display_size = (display_w, display_h)
        self._offset_x = (container_w - display_w) // 2
        self._offset_y = (container_h - display_h) // 2

        scaled = self._base_pixmap.scaled(
            display_w,
            display_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _is_inside_image(self, point: QPoint):
        x, y = point.x(), point.y()
        return (
            self._offset_x <= x <= self._offset_x + self._display_size[0]
            and self._offset_y <= y <= self._offset_y + self._display_size[1]
        )

    def _to_image_coords(self, point: QPoint):
        px = min(max(point.x() - self._offset_x, 0), self._display_size[0] - 1)
        py = min(max(point.y() - self._offset_y, 0), self._display_size[1] - 1)
        x = int(round(px / self._scale))
        y = int(round(py / self._scale))
        return x, y

    def mousePressEvent(self, event):
        if self._base_pixmap is None or event.button() != Qt.LeftButton:
            return
        if not self._is_inside_image(event.position().toPoint()):
            return
        self._drawing = True
        self._start_point = event.position().toPoint()
        self._current_point = event.position().toPoint()
        self.update()

    def mouseMoveEvent(self, event):
        if not self._drawing:
            return
        self._current_point = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event):
        if not self._drawing or event.button() != Qt.LeftButton:
            return
        self._drawing = False
        self._current_point = event.position().toPoint()
        box = self._build_box_xyxy()
        self.update()
        if box is not None:
            self.box_finished.emit(box)

    def _build_box_xyxy(self):
        x1, y1 = self._to_image_coords(self._start_point)
        x2, y2 = self._to_image_coords(self._current_point)
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        if x_max - x_min < 3 or y_max - y_min < 3:
            return None
        return [x_min, y_min, x_max, y_max]

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._drawing:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor("#00e6ff"), 2, Qt.SolidLine))
        rect = QRect(self._start_point, self._current_point).normalized()
        painter.drawRect(rect)


class FitImageLabel(QLabel):
    erase_point = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111319; border: 1px solid #2d3345;")
        self.setMinimumSize(420, 420)
        self._base_pixmap: Optional[QPixmap] = None
        self._display_size = (1, 1)
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._painting = False

    def set_display_pixmap(self, pixmap: QPixmap):
        self._base_pixmap = pixmap
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._base_pixmap is None:
            self.setPixmap(QPixmap())
            return
        container_w = max(1, self.width())
        container_h = max(1, self.height())
        pixmap_w = self._base_pixmap.width()
        pixmap_h = self._base_pixmap.height()
        scale = min(container_w / pixmap_w, container_h / pixmap_h)
        display_w = max(1, int(pixmap_w * scale))
        display_h = max(1, int(pixmap_h * scale))

        self._scale = scale
        self._display_size = (display_w, display_h)
        self._offset_x = (container_w - display_w) // 2
        self._offset_y = (container_h - display_h) // 2

        scaled = self._base_pixmap.scaled(
            display_w,
            display_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _is_inside_image(self, point: QPoint):
        x, y = point.x(), point.y()
        return (
            self._offset_x <= x <= self._offset_x + self._display_size[0]
            and self._offset_y <= y <= self._offset_y + self._display_size[1]
        )

    def _to_image_coords(self, point: QPoint):
        px = min(max(point.x() - self._offset_x, 0), self._display_size[0] - 1)
        py = min(max(point.y() - self._offset_y, 0), self._display_size[1] - 1)
        x = int(round(px / self._scale))
        y = int(round(py / self._scale))
        return x, y

    def _emit_erase_point(self, point: QPoint):
        if self._base_pixmap is None or not self._is_inside_image(point):
            return
        x, y = self._to_image_coords(point)
        self.erase_point.emit(x, y)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._painting = True
        self._emit_erase_point(event.position().toPoint())

    def mouseMoveEvent(self, event):
        if self._painting:
            self._emit_erase_point(event.position().toPoint())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._painting = False
        super().mouseReleaseEvent(event)


class SpineEndoSegTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spine Endoscope Segmentation Tool")
        self.resize(1600, 940)
        self.project_root = Path(__file__).resolve().parent
        self.default_data_dir = self.project_root 
        self.default_save_dir = self.project_root 

        self.segmenter: Optional[SAM2Segmenter] = None
        self.image_path: Optional[str] = None
        self.image_np: Optional[np.ndarray] = None
        self.color_mask: Optional[np.ndarray] = None
        self.label_mask: Optional[np.ndarray] = None
        self.annotations: list[Annotation] = []
        self.current_box: Optional[list[int]] = None
        self.current_color = (255, 0, 0)
        self.erase_mode = False
        self.erase_radius = 18

        self.color_options = {
            "Red": (255, 0, 0),
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0),
            "Cyan": (0, 255, 255),
            "Magenta": (255, 0, 255),
            "Orange": (255, 165, 0),
            "Purple": (160, 32, 240),
            "White": (255, 255, 255),
        }
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        load_model_btn = QPushButton("加载 SAM2 模型")
        load_model_btn.clicked.connect(self.load_sam2)
        left_layout.addWidget(QLabel("1) 模型"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Auto", "CPU", "CUDA"])
        self.device_combo.setCurrentText("Auto")
        left_layout.addWidget(self.device_combo)
        left_layout.addWidget(load_model_btn)

        load_img_btn = QPushButton("打开图像")
        load_img_btn.clicked.connect(self.load_image)
        left_layout.addWidget(QLabel("2) 图像"))
        left_layout.addWidget(load_img_btn)
        left_layout.addWidget(QLabel("支持: .png .jpg .jpeg .bmp .tif .webp"))

        left_layout.addWidget(QLabel("3) 标签与颜色"))
        self.color_combo = QComboBox()
        self.color_combo.addItems(self.color_options.keys())
        self.color_combo.currentTextChanged.connect(self.on_color_select)
        left_layout.addWidget(self.color_combo)

        custom_color_btn = QPushButton("自定义颜色")
        custom_color_btn.clicked.connect(self.choose_custom_color)
        left_layout.addWidget(custom_color_btn)

        self.label_edit = QLineEdit("lesion")
        self.label_edit.setPlaceholderText("输入标签名")
        left_layout.addWidget(self.label_edit)

        left_layout.addWidget(QLabel("4) 分割与修正"))
        seg_btn = QPushButton("框提示分割")
        seg_btn.clicked.connect(self.segment_current_box)
        self.erase_btn = QPushButton("开启擦除模式")
        self.erase_btn.setCheckable(True)
        self.erase_btn.toggled.connect(self.on_toggle_erase_mode)
        self.erase_radius_spin = QSpinBox()
        self.erase_radius_spin.setRange(3, 80)
        self.erase_radius_spin.setValue(self.erase_radius)
        self.erase_radius_spin.valueChanged.connect(self.on_erase_radius_changed)
        self.erase_radius_spin.setPrefix("半径 ")

        clear_btn = QPushButton("清空全部标注")
        clear_btn.clicked.connect(self.clear_annotations)
        left_layout.addWidget(seg_btn)
        left_layout.addWidget(self.erase_btn)
        left_layout.addWidget(self.erase_radius_spin)
        left_layout.addWidget(clear_btn)

        left_layout.addWidget(QLabel("5) 保存"))
        save_btn = QPushButton("保存标注结果")
        save_btn.clicked.connect(self.save_results)
        left_layout.addWidget(save_btn)

        self.status_label = QLabel("准备就绪")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("statusLabel")
        left_layout.addWidget(self.status_label)

        left_layout.addWidget(QLabel("标注记录"))
        self.log_list = QListWidget()
        left_layout.addWidget(self.log_list, 1)

        root.addWidget(left_panel)

        view_splitter = QSplitter(Qt.Horizontal)
        view_splitter.setChildrenCollapsible(False)
        root.addWidget(view_splitter, 1)

        self.original_view = BoxDrawLabel()
        self.original_view.box_finished.connect(self.on_box_finished)
        self.seg_view = FitImageLabel()
        self.seg_view.erase_point.connect(self.on_erase_point)

        original_wrap = self._build_view_card("原始图像（在此拖拽框选）", self.original_view)
        seg_wrap = self._build_view_card("分割结果", self.seg_view)

        view_splitter.addWidget(original_wrap)
        view_splitter.addWidget(seg_wrap)
        view_splitter.setSizes([700, 700])

    def _build_view_card(self, title: str, inner_widget: QWidget):
        card = QFrame()
        card.setObjectName("viewCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("viewTitle")
        layout.addWidget(title_label)
        layout.addWidget(inner_widget, 1)
        return card

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #0f1218; color: #e6e9f2; }
            QFrame#leftPanel, QFrame#viewCard {
                background: #171b24;
                border: 1px solid #2a3140;
                border-radius: 10px;
            }
            QLabel { color: #e6e9f2; }
            QLabel#viewTitle { font-size: 14px; font-weight: 600; color: #96a6d7; }
            QLabel#statusLabel {
                color: #a9b8de;
                background: #121722;
                border: 1px solid #2a3348;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton {
                background: #2a6df4;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 8px;
                font-weight: 600;
            }
            QPushButton:hover { background: #3b7cff; }
            QPushButton:pressed { background: #1f59cf; }
            QPushButton:checked { background: #c75a1a; }
            QPushButton:checked:hover { background: #dd6822; }
            QLineEdit, QListWidget {
                background: #0f141f;
                border: 1px solid #2a3348;
                border-radius: 7px;
                color: #e6e9f2;
                padding: 6px;
            }
            QComboBox {
                background: #8a8f99;
                border: 1px solid #a4a9b3;
                border-radius: 7px;
                color: #111318;
                padding: 6px;
                font-weight: 600;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QComboBox QAbstractItemView {
                background: #a0a5ae;
                color: #111318;
                border: 1px solid #bcc1c9;
                selection-background-color: #7b808a;
                selection-color: #ffffff;
            }
            """
        )

    def set_status(self, text: str):
        self.status_label.setText(text)

    def show_warn(self, title: str, text: str):
        QMessageBox.warning(self, title, text)

    def show_info(self, title: str, text: str):
        QMessageBox.information(self, title, text)

    def show_error(self, title: str, text: str):
        QMessageBox.critical(self, title, text)

    def on_color_select(self, name: str):
        self.current_color = self.color_options.get(name, (255, 0, 0))
        self.set_status(f"当前颜色: {name} {self.current_color}")

    def choose_custom_color(self):
        color = QColorDialog.getColor(parent=self, title="选择自定义标注颜色")
        if not color.isValid():
            return
        self.current_color = (color.red(), color.green(), color.blue())
        self.set_status(f"当前自定义颜色: {self.current_color}")

    def on_toggle_erase_mode(self, enabled: bool):
        self.erase_mode = enabled
        self._update_erase_cursor()
        self.erase_btn.setText("关闭擦除模式" if enabled else "开启擦除模式")
        if enabled:
            self.set_status("擦除模式已开启：请在右侧分割图上按住左键拖动擦除")
        else:
            self.set_status("擦除模式已关闭")

    def on_erase_radius_changed(self, value: int):
        self.erase_radius = int(value)
        self._update_erase_cursor()
        if self.erase_mode:
            self.set_status(f"擦除半径: {self.erase_radius} 像素")

    def _update_erase_cursor(self):
        if not self.erase_mode:
            self.seg_view.setCursor(Qt.ArrowCursor)
            return

        # Build a circle brush cursor whose size follows erase radius.
        r = max(3, int(self.erase_radius))
        pad = 4
        size = 2 * r + 2 * pad
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)

        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Outer bright ring for visibility on dark/bright backgrounds.
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        painter.drawEllipse(pad, pad, 2 * r, 2 * r)
        # Inner cyan ring to match the app accent.
        painter.setPen(QPen(QColor(0, 230, 255, 220), 1))
        painter.drawEllipse(pad + 1, pad + 1, 2 * r - 2, 2 * r - 2)
        painter.end()

        hotspot = pad + r
        self.seg_view.setCursor(QCursor(pix, hotspot, hotspot))

    def load_sam2(self):
        checkpoint_start_dir = self.project_root
        if not checkpoint_start_dir.exists():
            checkpoint_start_dir = self.project_root
        checkpoint_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 SAM2 权重文件(.pt)",
            str(checkpoint_start_dir),
            "PyTorch files (*.pt *.pth);;All Files (*)",
        )
        if not checkpoint_path:
            return
        try:
            config_path = infer_sam2_config_from_checkpoint(checkpoint_path)
            if config_path is None:
                self.show_warn(
                    "无法自动匹配配置",
                    "当前 .pt 文件名无法自动匹配 SAM2 配置。\n"
                    "请使用官方命名（tiny/small/base_plus/large）。",
                )
                self.set_status("SAM2 加载失败：无法根据 .pt 文件名匹配配置")
                return

            device = choose_device(self.device_combo.currentText())
            self.segmenter = SAM2Segmenter(config_path, checkpoint_path, device=device)
            selected_mode = self.device_combo.currentText()
            if selected_mode.lower() == "cuda" and device == "cpu":
                self.show_warn(
                    "CUDA 不可用",
                    "你选择了 CUDA，但当前环境不可用，已自动回退到 CPU。",
                )
            self.set_status(f"SAM2 模型加载成功，设备: {device}（模式: {selected_mode}）")
            self.show_info(
                "成功",
                f"SAM2 模型已加载。\n"
                f"Checkpoint: {Path(checkpoint_path).name}\n"
                f"Config: {config_path}\n"
                f"Device Mode: {selected_mode}\n"
                f"Device: {device}",
            )
        except Exception as exc:
            self.show_error("模型加载失败", str(exc))
            self.set_status(f"SAM2 加载失败: {exc}")

    def load_image(self):
        image_start_dir = self.default_data_dir if self.default_data_dir.exists() else self.project_root
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像",
            str(image_start_dir),
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(SUPPORTED_EXTS):
            self.show_warn("格式不支持", "请选择常见图像格式（png/jpg/jpeg...）")
            return
        try:
            image = Image.open(file_path).convert("RGB")
            self.image_np = np.array(image)
            self.image_path = file_path
            if self.segmenter is not None:
                self.segmenter.set_image(self.image_np)

            h, w = self.image_np.shape[:2]
            self.color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            self.label_mask = np.zeros((h, w), dtype=np.uint8)
            self.annotations.clear()
            self.log_list.clear()
            self.current_box = None
            self.original_view.clear_box()
            self.refresh_views()
            self.set_status(f"已加载: {os.path.basename(file_path)} ({w}x{h})")
        except Exception as exc:
            self.show_error("图像加载失败", str(exc))
            self.set_status(f"图像加载失败: {exc}")

    def on_box_finished(self, box_xyxy: list[int]):
        if self.erase_mode:
            return
        self.current_box = box_xyxy
        self.set_status(f"已框选: {box_xyxy}，点击“框提示分割”执行")

    def on_erase_point(self, x: int, y: int):
        if not self.erase_mode or self.image_np is None:
            return
        if self.color_mask is None or self.label_mask is None:
            return

        h, w = self.label_mask.shape
        r = int(self.erase_radius)
        x1, x2 = max(0, x - r), min(w - 1, x + r)
        y1, y2 = max(0, y - r), min(h - 1, y + r)
        if x1 > x2 or y1 > y2:
            return

        ys = np.arange(y1, y2 + 1) - y
        xs = np.arange(x1, x2 + 1) - x
        circle = (ys[:, None] ** 2 + xs[None, :] ** 2) <= (r * r)

        color_roi = self.color_mask[y1 : y2 + 1, x1 : x2 + 1]
        label_roi = self.label_mask[y1 : y2 + 1, x1 : x2 + 1]
        had_pixels = np.any(color_roi > 0, axis=2) | (label_roi > 0)
        erase_pixels = circle & had_pixels
        if not np.any(erase_pixels):
            return

        color_roi[erase_pixels] = 0
        label_roi[erase_pixels] = 0
        self.color_mask[y1 : y2 + 1, x1 : x2 + 1] = color_roi
        self.label_mask[y1 : y2 + 1, x1 : x2 + 1] = label_roi
        self.refresh_views()

    def _to_qimage(self, rgb_np: np.ndarray):
        h, w, _ = rgb_np.shape
        arr = np.ascontiguousarray(rgb_np)
        return QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()

    def _overlay_mask(self):
        base = self.image_np.astype(np.float32)
        overlay = self.color_mask.astype(np.float32)
        mask_pixels = np.any(self.color_mask > 0, axis=2)
        mixed = base.copy()
        mixed[mask_pixels] = 0.65 * base[mask_pixels] + 0.35 * overlay[mask_pixels]
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def _draw_box_on_image(self, image_rgb: np.ndarray):
        if self.current_box is None:
            return image_rgb
        x1, y1, x2, y2 = self.current_box
        img = image_rgb.copy()
        img[y1 : y1 + 2, x1 : x2 + 1] = np.array([0, 230, 255], dtype=np.uint8)
        img[y2 - 1 : y2 + 1, x1 : x2 + 1] = np.array([0, 230, 255], dtype=np.uint8)
        img[y1 : y2 + 1, x1 : x1 + 2] = np.array([0, 230, 255], dtype=np.uint8)
        img[y1 : y2 + 1, x2 - 1 : x2 + 1] = np.array([0, 230, 255], dtype=np.uint8)
        return img

    def refresh_views(self):
        if self.image_np is None:
            return
        original_with_box = self._draw_box_on_image(self.image_np)
        seg_overlay = self._overlay_mask()

        original_qimg = self._to_qimage(original_with_box)
        seg_qimg = self._to_qimage(seg_overlay)
        original_pix = QPixmap.fromImage(original_qimg)
        seg_pix = QPixmap.fromImage(seg_qimg)

        self.original_view.set_display_pixmap(original_pix)
        self.seg_view.set_display_pixmap(seg_pix)

    def segment_current_box(self):
        if self.segmenter is None:
            self.show_warn("未加载模型", "请先加载 SAM2 模型")
            return
        if self.image_np is None:
            self.show_warn("未加载图像", "请先打开图像")
            return
        if self.current_box is None:
            self.show_warn("未框选", "请在左图拖拽框选后再分割")
            return
        try:
            self.segmenter.set_image(self.image_np)
            mask, score = self.segmenter.predict_box(self.current_box)
            label = self.label_edit.text().strip() or "lesion"
            color = self.current_color
            ann = Annotation(label=label, color=color, box=self.current_box.copy(), score=score, mask=mask)
            self.annotations.append(ann)

            idx = len(self.annotations)
            self.color_mask[mask] = np.array(color, dtype=np.uint8)
            self.label_mask[mask] = min(idx, 255)
            self.log_list.addItem(
                f"#{idx} label={label}, box={self.current_box}, score={score:.4f}"
            )
            self.current_box = None
            self.original_view.clear_box()
            self.refresh_views()
            self.set_status(f"分割完成，共 {idx} 条标注")
        except Exception as exc:
            self.show_error("分割失败", str(exc))
            self.set_status(f"分割失败: {exc}")

    def clear_annotations(self):
        if self.image_np is None:
            return
        h, w = self.image_np.shape[:2]
        self.color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        self.label_mask = np.zeros((h, w), dtype=np.uint8)
        self.annotations.clear()
        self.log_list.clear()
        self.current_box = None
        self.original_view.clear_box()
        self.refresh_views()
        self.set_status("已清空全部标注")

    def save_results(self):
        if self.image_np is None:
            self.show_warn("未加载图像", "请先打开图像")
            return
        if not self.annotations:
            self.show_warn("无标注", "当前没有分割结果可保存")
            return
        if not self.default_save_dir.exists():
            self.default_save_dir.mkdir(parents=True, exist_ok=True)
        save_dir = QFileDialog.getExistingDirectory(
            self, "选择保存目录", str(self.default_save_dir)
        )
        if not save_dir:
            return

        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        color_mask_path = os.path.join(save_dir, f"{base_name}_mask.png")
        label_mask_path = os.path.join(save_dir, f"{base_name}_label.png")
        meta_path = os.path.join(save_dir, f"{base_name}_boxes.json")

        Image.fromarray(self.color_mask).save(color_mask_path)
        Image.fromarray(self.label_mask).save(label_mask_path)

        metadata = {
            "image_path": self.image_path,
            "mask_path": color_mask_path,
            "label_mask_path": label_mask_path,
            "annotations": [
                {
                    "id": i + 1,
                    "label": ann.label,
                    "color_rgb": list(ann.color),
                    "box_xyxy": ann.box,
                    "sam_score": ann.score,
                }
                for i, ann in enumerate(self.annotations)
            ],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.set_status(f"保存完成: {os.path.basename(color_mask_path)} + {os.path.basename(meta_path)}")
        self.show_info("保存成功", f"已保存:\n{color_mask_path}\n{label_mask_path}\n{meta_path}")


def main():
    app = QApplication(sys.argv)
    window = SpineEndoSegTool()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
