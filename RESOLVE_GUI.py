from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFormLayout, QLineEdit, QCheckBox, QSpinBox, 
    QFileDialog, QToolButton, QComboBox, QGroupBox
)
import sys
import os

try:
    from scripts.resolve import main as resolve_main
except ImportError:
    resolve_main = None

class ConfigPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Configuration Panel")
        layout = QVBoxLayout()

        # Dropdown menu for selecting the configuration
        self.config_selector = QComboBox()
        self.config_selector.addItems(["Refined-Maps", "Micrographs", "Tilt-Series", "Tomograms"])
        layout.addWidget(self.config_selector)

        form_layout = QFormLayout()

        # Helper function to create input rows 
        def create_input_row(label_text, tooltip_text, add_file_button=False, add_dir_button=False):
            row_layout = QHBoxLayout()
            input_field = QLineEdit()
            row_layout.addWidget(input_field)

            if add_file_button:
                file_button = QPushButton("...")
                file_button.setFixedWidth(30)
                file_button.clicked.connect(lambda: self.open_file_dialog(input_field))
                row_layout.addWidget(file_button)

            if add_dir_button:
                file_button = QPushButton("...")
                file_button.setFixedWidth(30)
                file_button.clicked.connect(lambda: self.open_directory_dialog(input_field))
                row_layout.addWidget(file_button)

            help_button = QToolButton()
            help_button.setText("?")
            help_button.setFixedWidth(20)
            help_button.setToolTip(tooltip_text)
            row_layout.addWidget(help_button)

            return row_layout, input_field

        # Input rows
        row1, self.input1 = create_input_row("Input 1:", "Select the first input file.", add_file_button=True)
        row2, self.input2 = create_input_row("Input 2:", "Select the second input file.", add_file_button=True)

        pixel_size_layout = QHBoxLayout()
        self.pixel_size = QLineEdit()
        self.pixel_size.setPlaceholderText("Optional")
        pixel_size_layout.addWidget(self.pixel_size)

        help_button_pixel_size = QToolButton()
        help_button_pixel_size.setText("?")
        help_button_pixel_size.setFixedWidth(20)
        help_button_pixel_size.setToolTip("Pixel Size. If not given, reading header.")
        pixel_size_layout.addWidget(help_button_pixel_size)

        row3, self.outputDir = create_input_row("Output Directory:", "Select output directory.", add_dir_button=True)

        form_layout.addRow("Input 1:", row1)
        form_layout.addRow("Input 2:", row2)
        form_layout.addRow("Pixel Size:", pixel_size_layout)
        form_layout.addRow("Output Directory:", row3)

        layout.addLayout(form_layout)

        # Advanced options toggle
        self.advanced_button = QPushButton("Show Advanced Options")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setChecked(False)
        self.advanced_button.clicked.connect(self.toggle_advanced_options)
        self.advanced_button.setStyleSheet("""
            QPushButton {
                background: none;
                color: #555;
                border: none;
                text-align: left;
                padding: 4px 0;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
        """)
        layout.addWidget(self.advanced_button)

        # Advanced options group
        self.advanced_group = QGroupBox()
        self.advanced_group.setVisible(False)
        advanced_layout = QFormLayout()

        # CPU threads
        cpu_layout = QHBoxLayout()
        self.cpu_threads = QSpinBox()
        self.cpu_threads.setMinimum(1)
        self.cpu_threads.setMaximum(128)
        self.cpu_threads.setValue(4)
        cpu_layout.addWidget(self.cpu_threads)

        help_button_cpu = QToolButton()
        help_button_cpu.setText("?")
        help_button_cpu.setFixedWidth(20)
        help_button_cpu.setToolTip("Specify the number of CPU threads to use.")
        cpu_layout.addWidget(help_button_cpu)

        advanced_layout.addRow("CPU Threads:", cpu_layout)

        # GPU checkbox and input
        gpu_layout = QHBoxLayout()
        self.gpu_checkbox = QCheckBox("Enable GPU")
        self.gpu_checkbox.setChecked(True)  # Set to True by default
        self.gpu_input = QLineEdit()
        self.gpu_input.setPlaceholderText("0,1")
        self.gpu_input.setEnabled(True)  # Enable by default
        self.gpu_input.setInputMask("")  # Clear any input mask
        self.gpu_input.setMaxLength(32767)  # Set to maximum length
        gpu_layout.addWidget(self.gpu_checkbox)
        gpu_layout.addWidget(self.gpu_input)

        help_button_gpu = QToolButton()
        help_button_gpu.setText("?")
        help_button_gpu.setFixedWidth(20)
        help_button_gpu.setToolTip("Check to enable GPU. List GPUs to use, comma separated. By default, the first two available GPUs are used.")
        gpu_layout.addWidget(help_button_gpu)

        self.gpu_checkbox.stateChanged.connect(self.toggle_gpu_input)
        advanced_layout.addRow("GPU:", gpu_layout)

        # Run fast checkbox
        run_fast_layout = QHBoxLayout()
        self.run_fast_checkbox = QCheckBox("fast analysis")
        self.run_fast_checkbox.setChecked(False)  # True by default
        run_fast_layout.addWidget(self.run_fast_checkbox)
        
        help_button_fast = QToolButton()
        help_button_fast.setText("?")
        help_button_fast.setFixedWidth(20)
        help_button_fast.setToolTip("Lower sampling in Fourier space and real space. Faster, needs less memory.")
        run_fast_layout.addWidget(help_button_fast)
        run_fast_layout.addStretch()

        advanced_layout.addRow("Run fast:", run_fast_layout)

        # Input mask option with dropdown for median/average
        row_mask_layout = QHBoxLayout()
        self.input_mask = QLineEdit()
        row_mask_layout.addWidget(self.input_mask)

        mask_file_button = QPushButton("...")
        mask_file_button.setFixedWidth(30)
        mask_file_button.clicked.connect(lambda: self.open_file_dialog(self.input_mask))
        row_mask_layout.addWidget(mask_file_button)

        self.mask_measure_combo = QComboBox()
        self.mask_measure_combo.addItems(["median", "average"])
        self.mask_measure_combo.setToolTip("Measure to calculate global resolution from local measurements.")
        row_mask_layout.addWidget(self.mask_measure_combo)

        help_button_mask = QToolButton()
        help_button_mask.setText("?")
        help_button_mask.setFixedWidth(20)
        help_button_mask.setToolTip("Focus global resolution estimates with an input mask.")
        row_mask_layout.addWidget(help_button_mask)

        self.mask_row_container = QWidget()
        self.mask_row_container.setLayout(row_mask_layout)
        advanced_layout.addRow("Input Mask:", self.mask_row_container)

        self.advanced_group.setLayout(advanced_layout)
        layout.addWidget(self.advanced_group)

        # Run button — visually highlighted
        self.run_button = QPushButton("Run")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #005f99;
            }
        """)
        self.run_button.clicked.connect(self.run_function)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def toggle_advanced_options(self):
        show = self.advanced_button.isChecked()
        self.advanced_group.setVisible(show)
        self.advanced_button.setText("Hide Advanced Options" if show else "Show Advanced Options")

    def toggle_gpu_input(self, state):
        self.gpu_input.setEnabled(state == 2)

    def open_file_dialog(self, input_field):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            input_field.setText(file_path)

    def open_directory_dialog(self, input_field):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            input_field.setText(directory)

    def run_function(self):
        selected_config = self.config_selector.currentText()
        input_1 = self.input1.text()
        input_2 = self.input2.text()
        try:
            apix = float(self.pixel_size.text().strip()) if self.pixel_size.text().strip() else None
        except ValueError:
            apix = None

        outputDir = self.outputDir.text().strip()
        if not outputDir:
            outputDir = os.getcwd()
        else:
            outputDir = os.path.abspath(outputDir)

        cpu_threads = self.cpu_threads.value()
        gpu_enabled = self.gpu_checkbox.isChecked()
        gpu_settings = self.gpu_input.text() if gpu_enabled else "Disabled"
        mask_file = self.input_mask.text().strip()
        mask_measure = self.mask_measure_combo.currentText()
        run_fast = self.run_fast_checkbox.isChecked()

        if resolve_main:
            resolve_main(
                mode="single",
                config=selected_config,
                apix=apix,
                odd_input=input_1,
                even_input=input_2,
                cpu_threads=cpu_threads,
                gpu_enabled=gpu_enabled,
                gpu_settings=gpu_settings,
                run_fast=run_fast,
                signal_mask_input=mask_file,
                mask_measure=mask_measure,
                outputDir=outputDir,
                inputDir=""
            )
        else:
            print("Error: Could not import resolve.py's main function!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigPanel()
    window.show()
    sys.exit(app.exec_())