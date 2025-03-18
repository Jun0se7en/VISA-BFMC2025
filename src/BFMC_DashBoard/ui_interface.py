# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'interfaceDsBoir.ui'
##
## Created by: Qt User Interface Compiler version 5.14.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
                         QRect, QSize, QUrl, Qt, QRectF)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
                        QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter, QPixmap,
                        QRadialGradient, QImage)
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import numpy as np

from AnalogGaugeWidget import AnalogGaugeWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView


class Ui_MainWindow(object):
    def __init__(self, width, height):
        """Initialize the UI with specified width and height."""
        self.width = width
        self.height = height

    def setupUi(self, MainWindow):
        """Set up the main window and its UI components."""
        # Main Window Configuration
        if MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(self.width, self.height)
        icon = QIcon()
        icon.addFile(u"images/Logo_UIT.png", QSize(), QIcon.Normal, QIcon.Off)
        MainWindow.setWindowIcon(icon)

        # Central Widget
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QSize(self.width, self.height))

        # Header Frame
        self.Header_Frame = QFrame(self.centralwidget)
        self.Header_Frame.setObjectName(u"Header_Frame")
        self.Header_Frame.setGeometry(QRect(0, 0, self.width, 80))
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.Header_Frame.sizePolicy().hasHeightForWidth())
        self.Header_Frame.setSizePolicy(sizePolicy1)
        self.Header_Frame.setFrameShape(QFrame.StyledPanel)
        self.Header_Frame.setFrameShadow(QFrame.Raised)
        self.Header_Frame.setLineWidth(0)
        self.Header_Frame_Layout = QHBoxLayout(self.Header_Frame)
        self.Header_Frame_Layout.setSpacing(0)
        self.Header_Frame_Layout.setObjectName(u"Header_Frame_Layout")
        self.Header_Frame_Layout.setContentsMargins(0, 0, 0, 0)

        # Header Left
        self.Header_Left = QFrame(self.Header_Frame)
        self.Header_Left.setObjectName(u"Header_Left")
        self.Header_Left.setEnabled(True)
        sizePolicy1.setHeightForWidth(self.Header_Left.sizePolicy().hasHeightForWidth())
        self.Header_Left.setSizePolicy(sizePolicy1)
        self.Header_Left.setFrameShape(QFrame.StyledPanel)
        self.Header_Left.setFrameShadow(QFrame.Raised)
        self.Header_Left.setLineWidth(0)
        self.Header_Left_Layout = QVBoxLayout(self.Header_Left)
        self.Header_Left_Layout.setSpacing(0)
        self.Header_Left_Layout.setObjectName(u"Header_Left_Layout")
        self.Header_Left_Layout.setContentsMargins(0, 0, 0, 0)
        self.lb_Left_Signal = QLabel(self.Header_Left)
        self.lb_Left_Signal.setObjectName(u"lb_Left_Signal")
        self.lb_Left_Signal.setLineWidth(0)
        self.lb_Left_Signal.setPixmap(QPixmap(u"images/left_arrow_small.png"))
        self.lb_Left_Signal.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
        self.Header_Left_Layout.addWidget(self.lb_Left_Signal)
        self.Header_Frame_Layout.addWidget(self.Header_Left)

        # Header Title
        self.Header_Title = QFrame(self.Header_Frame)
        self.Header_Title.setObjectName(u"Header_Title")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.Header_Title.sizePolicy().hasHeightForWidth())
        self.Header_Title.setSizePolicy(sizePolicy2)
        self.Header_Title.setFrameShape(QFrame.StyledPanel)
        self.Header_Title.setFrameShadow(QFrame.Raised)
        self.Header_Title.setLineWidth(0)
        self.Header_Title_Layout = QVBoxLayout(self.Header_Title)
        self.Header_Title_Layout.setSpacing(0)
        self.Header_Title_Layout.setObjectName(u"Header_Title_Layout")
        self.Header_Title_Layout.setContentsMargins(0, 0, 0, 0)
        self.lb_Title = QLabel(self.Header_Title)
        self.lb_Title.setObjectName(u"lb_Title")
        self.lb_Title.setLineWidth(0)
        self.lb_Title.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.Header_Title_Layout.addWidget(self.lb_Title)
        self.Header_Frame_Layout.addWidget(self.Header_Title)

        # Header Right
        self.Header_Right = QFrame(self.Header_Frame)
        self.Header_Right.setObjectName(u"Header_Right")
        sizePolicy1.setHeightForWidth(self.Header_Right.sizePolicy().hasHeightForWidth())
        self.Header_Right.setSizePolicy(sizePolicy1)
        self.Header_Right.setFrameShape(QFrame.StyledPanel)
        self.Header_Right.setFrameShadow(QFrame.Raised)
        self.Header_Right.setLineWidth(0)
        self.Header_Right_Layout = QVBoxLayout(self.Header_Right)
        self.Header_Right_Layout.setSpacing(0)
        self.Header_Right_Layout.setObjectName(u"Header_Right_Layout")
        self.Header_Right_Layout.setContentsMargins(0, 0, 0, 0)
        self.lb_Right_Signal = QLabel(self.Header_Right)
        self.lb_Right_Signal.setObjectName(u"lb_Right_Signal")
        self.lb_Right_Signal.setLineWidth(0)
        self.lb_Right_Signal.setPixmap(QPixmap(u"images/right_arrow_small.png"))
        self.lb_Right_Signal.setAlignment(Qt.AlignRight | Qt.AlignTop | Qt.AlignTrailing)
        self.Header_Right_Layout.addWidget(self.lb_Right_Signal)
        self.Header_Frame_Layout.addWidget(self.Header_Right)

        # Body Frame (Gauges)
        self.BodyFrame = QFrame(self.centralwidget)
        self.BodyFrame.setObjectName(u"BodyFrame")
        self.BodyFrame.setGeometry(QRect(0, int(self.width / 3), self.width, int(self.width / 3)))
        sizePolicy1.setHeightForWidth(self.BodyFrame.sizePolicy().hasHeightForWidth())
        self.BodyFrame.setSizePolicy(sizePolicy1)
        self.BodyFrame.setFrameShape(QFrame.StyledPanel)
        self.BodyFrame.setFrameShadow(QFrame.Raised)
        self.BodyFrame.setLineWidth(0)

        # Body Center Frame
        self.Body_Center_Frame = QFrame(self.BodyFrame)
        self.Body_Center_Frame.setObjectName(u"Body_Center_Frame")
        self.Body_Center_Frame.setGeometry(QRect(0, 0, self.width, int(self.width / 3)))
        self.Body_Center_Frame.setFrameShape(QFrame.StyledPanel)
        self.Body_Center_Frame.setFrameShadow(QFrame.Raised)
        self.Body_Center_Frame.setLineWidth(0)

        # Speed Gauge
        self.Analog_Gauge_Speed = AnalogGaugeWidget(self.Body_Center_Frame)
        self.Analog_Gauge_Speed.setObjectName(u"Analog_Gauge_Speed")
        gauge_width = int(self.width / 4)
        gauge_height = gauge_width  # Keeping it square
        gauge_x = (self.width - gauge_width) // 2  # Center horizontally
        gauge_y = 60  # Maintain the same Y position
        self.Analog_Gauge_Speed.setGeometry(QRect(gauge_x, gauge_y, gauge_width, gauge_height))

        # Images Frame
        self.ImagesFrame = QFrame(self.centralwidget)
        self.ImagesFrame.setObjectName(u"ImagesFrame")
        self.ImagesFrame.setGeometry(QRect(0, 80 + int(self.width / 3), self.width, int(self.width / 3)))
        self.ImagesFrame.setFrameShape(QFrame.StyledPanel)
        self.ImagesFrame.setFrameShadow(QFrame.Raised)
        self.ImagesFrame.setLineWidth(0)

        # Raw Image Frame with White Bounding Box
        self.Raw_Img_Frame = QFrame(self.ImagesFrame)
        self.Raw_Img_Frame.setObjectName(u"Raw_Img_Frame")
        self.Raw_Img_Frame.setGeometry(QRect(int(self.width * 0.6) + 40, 30, 220, 160))
        self.Raw_Img_Frame.setFrameShape(QFrame.StyledPanel)
        self.Raw_Img_Frame.setFrameShadow(QFrame.Raised)
        self.Raw_Img_Frame.setLineWidth(0)
        self.Raw_Img_Frame.setStyleSheet("border: 1px solid white;")
        self.Raw_Img_Layout = QVBoxLayout(self.Raw_Img_Frame)
        self.Raw_Img_Layout.setSpacing(0)
        self.Raw_Img_Layout.setObjectName(u"Raw_Img_Layout")
        self.Raw_Img_Layout.setContentsMargins(0, 0, 0, 0)
        self.lb_Raw_Img = QLabel(self.Raw_Img_Frame)
        self.lb_Raw_Img.setObjectName(u"lb_Raw_Img")
        self.lb_Raw_Img.setLineWidth(0)
        self.lb_Raw_Img.setAlignment(Qt.AlignCenter)
        self.Raw_Img_Layout.addWidget(self.lb_Raw_Img)

        # Output Image Frame with White Bounding Box
        self.Output_Img_Frame = QFrame(self.ImagesFrame)
        self.Output_Img_Frame.setObjectName(u"Output_Img_Frame")
        self.Output_Img_Frame.setGeometry(QRect(int(self.width * 0.6) + 260, 30, 220, 160))
        self.Output_Img_Frame.setFrameShape(QFrame.StyledPanel)
        self.Output_Img_Frame.setFrameShadow(QFrame.Raised)
        self.Output_Img_Frame.setLineWidth(0)
        self.Output_Img_Frame.setStyleSheet("border: 1px solid white;")
        self.Output_Img_Layout = QVBoxLayout(self.Output_Img_Frame)
        self.Output_Img_Layout.setSpacing(0)
        self.Output_Img_Layout.setObjectName(u"Output_Img_Layout")
        self.Output_Img_Layout.setContentsMargins(0, 0, 0, 0)
        self.lb_Output_Img = QLabel(self.Output_Img_Frame)
        self.lb_Output_Img.setObjectName(u"lb_Output_Img")
        self.lb_Output_Img.setLineWidth(0)
        self.lb_Output_Img.setAlignment(Qt.AlignCenter)
        self.Output_Img_Layout.addWidget(self.lb_Output_Img)

        # Slider
        self.slider = QSlider(Qt.Horizontal, self.centralwidget)
        self.slider.setRange(-25, 25)
        self.slider.setValue(0)
        window_width = 1280  # Example window width (adjust if dynamic)
        slider_width = 200
        slider_x = (window_width - slider_width) // 2
        slider_y = 160
        self.slider.setGeometry(slider_x, slider_y, slider_width, 100)
        self.slider.setTracking(True)
        self.slider.setPageStep(1)

        # Angle Label
        self.angle_label = QLabel("Angle: 50°", self.centralwidget)
        self.angle_label.setStyleSheet("""
            color: #FFFFFF;
            font-weight: bold;
            font-size: 24px;
            font-family: Arial, Helvetica, sans-serif;
        """)
        self.angle_label.setAlignment(Qt.AlignCenter)
        self.angle_label.setGeometry(slider_x, slider_y - 20, slider_width, 30)

        # Slider Icon Update Function
        def update_slider_icon(value):
            self.angle_label.setText(f"Angle: {value}°")
            if value < 0:
                self.slider.setStyleSheet("QSlider::handle { image: url('images/left_arrow2.png'); }")
            elif value > 0:
                self.slider.setStyleSheet("QSlider::handle { image: url('images/right_arrow2.png'); }")
            else:
                self.slider.setStyleSheet("QSlider::handle { image: url('images/middle.png'); }")

        # Connect Slider Value Change
        self.slider.valueChanged.connect(update_slider_icon)
        update_slider_icon(self.slider.value())

        # Progress Bars and Labels
        self.progress_bars = []
        self.percentage_labels = []
        temp_labels = []
        labels = ["Core1", "Core2", "Core3", "Core4", "Core5", "Core6", "GPU", "MEM", "Temp"]
        x_offset = 50
        y_offset = 250
        spacing = 53

        for i, label in enumerate(labels):
            # Progress Bar
            progress_bar = QProgressBar(MainWindow)
            progress_bar.setOrientation(Qt.Vertical)
            progress_bar.setGeometry(x_offset + i * spacing, y_offset, 20, 150)
            progress_bar.setMaximum(100)
            progress_bar.setMinimum(0)
            progress_bar.setValue(50)
            progress_bar.setTextVisible(False)
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #3A3A3A;
                    border-radius: 1px;
                    background-color: #000000;
                }
                QProgressBar::chunk {
                    background-color: #219934;
                }
            """)
            self.progress_bars.append(progress_bar)

            # Percentage/Temperature Label
            percentage_label = QLabel("50%" if label != "Temp" else "45°C", MainWindow)
            percentage_label.setStyleSheet("color: #FFFFFF; font-weight: bold;")
            percentage_label.setAlignment(Qt.AlignCenter)
            percentage_label.setGeometry(x_offset + i * spacing - 10, y_offset - 30, 40, 20)
            self.percentage_labels.append(percentage_label)

            # Core/Temp Label
            temp_label = QLabel(label, MainWindow)
            temp_label.setStyleSheet("color: #FFFFFF; font-weight: bold;")
            temp_label.setAlignment(Qt.AlignCenter)
            temp_label.setGeometry(x_offset + i * spacing - 10, y_offset + 160, 40, 20)
            temp_labels.append(temp_label)

        # Example Usage
        self.update_progress([0, 0, 0, 0, 0, 0, 0, 0], 0)

        def create_value_label(prefix, value, x, y, width, height, font_size=14):
            label = QLabel(f"{prefix} <span style='color: blue;'>{value}</span>", MainWindow)
            label.setGeometry(QRect(x, self.height - y, width, height))
            font = label.font()
            font.setFamily("Arial")
            font.setBold(True)
            font.setPointSize(font_size)
            label.setFont(font)
            label.setTextFormat(Qt.RichText)
            label.show()
            return label

        # Location Section
        self.X_Value = create_value_label("X:", "0", 330, 222, 100, 30, font_size=10)
        self.Y_Value = create_value_label("Y:", "0", 380, 222, 100, 30, font_size=10)

        # Vehicle State Section
        self.Current_State = create_value_label("Curr:", "Idle", 99, 110, 200, 20, font_size=10)
        self.Prev_State = create_value_label("Prev:", "None", 100, 90, 120, 20, font_size=10)

        # Jetson Power Section
        self.Current_Power = create_value_label("Current:", "0W", 100, 219, 200, 20, font_size=11)

        # FPS
        self.FPS = create_value_label("", "0", 368, 105, 200, 20, font_size=11)


        # Example usage
        self.update_label(self.X_Value, "0")
        self.update_label(self.Y_Value, "0")
        self.update_label(self.Current_State, "")
        self.update_label(self.Prev_State, "")
        self.update_label(self.Current_Power, "20W")
        self.update_label(self.FPS, "0")

        self.update_angle(0)

        # Temperature vs time plot
        self.plot_graph = pg.PlotWidget(self.centralwidget)  # Parent it to centralwidget
        self.plot_graph.setGeometry(830, 115, 400, 400)  # Adjust size and position
        # Ẩn các trục và khung xung quanh plot_graph
        self.plot_graph.showAxis('bottom', False)
        self.plot_graph.showAxis('left', False)
        self.plot_graph.showAxis('top', False)
        self.plot_graph.showAxis('right', False)

        # Xóa đường viền (border) của plot
        self.plot_graph.setFrameStyle(0)

        # Set background trong suốt (nếu cần)
        self.plot_graph.setBackground(None)

        image = QImage("./BFMC2025_Map.png")
        if image.isNull():
            print("Lỗi tải ảnh: Kiểm tra đường dẫn ảnh của bạn!")
        else:
            # Scale ảnh phù hợp với plot_graph (vd: 400x400)
            plot_width, plot_height = 400, 400
            image = image.scaled(plot_width, plot_height).convertToFormat(QImage.Format_RGB888)
            width, height = image.width(), image.height()

            # Cách chuẩn để convert sang numpy array:
            ptr = image.constBits()
            ptr.setsize(height * width * 3)
            image_array = np.array(ptr, dtype=np.uint8).reshape((height, width, 3))

            # Cần đảo chiều dọc và transpose cho PyQtGraph hiển thị đúng:
            image_array = np.flipud(image_array)
            image_array = image_array.transpose((1, 0, 2))

            # Thêm ảnh vào plot graph
            img_item = pg.ImageItem(image_array)
            self.plot_graph.addItem(img_item)
            img_item.setZValue(-100)

            # Đặt lại vị trí rect
            img_item.setRect(QRectF(0, 0, plot_width, plot_height))

    def update_plot(self, x, y):
        # Kích thước điểm lớn hơn (ví dụ: 15), màu RGB (vd: đỏ: (255,0,0))
        pen = pg.mkPen(color=(255, 0, 0), width=0)  # Viền điểm (width=0 để không có viền)
        brush = pg.mkBrush(255, 0, 0)  # màu điểm RGB đỏ, bạn có thể chỉnh tùy ý
        size = 15  # Kích thước điểm to hơn

        # Plot lên plot_graph của bạn:
        scatter = pg.ScatterPlotItem(x=x, y=y, pen=pen, brush=brush, size=size)
        self.plot_graph.addItem(scatter)
    # Update Speed
    def update_speed(self, value):
        self.Analog_Gauge_Speed.updateValue(value)

    # Update Steer
    def update_angle(self, value):
        self.slider.setValue(value)

    # Update Progress Function
    def update_progress(self, values, temp_value):
        for i, value in enumerate(values):
            self.progress_bars[i].setValue(value)
            self.percentage_labels[i].setText(f"{value}%" if i != 8 else f"{temp_value}°C")

    # Update Label
    def update_label(self, label, value):
        """
        Updates the QLabel text with a new value while retaining the prefix.

        Parameters:
        - label (QLabel): The target label.
        - value (str): The value to display in blue text.
        """
        if label == self.FPS:
            label.setText(f"<span style='color: blue;'>{value}</span>")
        else:
            label.setText(f"{label.text().split(':')[0]}: <span style='color: blue;'>{value}</span>")

# End of class definition