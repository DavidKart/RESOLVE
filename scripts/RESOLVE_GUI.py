from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFormLayout, QLineEdit, QCheckBox, QSpinBox, 
    QFileDialog, QToolButton, QComboBox, QGroupBox
)
import sys
import os

try:
    from src.resolve import main as resolve_main
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
        self.config_selector.currentTextChanged.connect(self.on_config_changed)
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
        self.gpu_checkbox.setChecked(True)
        self.gpu_input = QLineEdit()
        self.gpu_input.setPlaceholderText("0,1")
        self.gpu_input.setEnabled(True)
        self.gpu_input.setInputMask("")
        self.gpu_input.setMaxLength(32767)
        gpu_layout.addWidget(self.gpu_checkbox)
        gpu_layout.addWidget(self.gpu_input)

        help_button_gpu = QToolButton()
        help_button_gpu.setText("?")
        help_button_gpu.setFixedWidth(20)
        help_button_gpu.setToolTip("Check to enable GPU. List GPUs to use. For single file processing, only one GPU is used, by default the first.")
        gpu_layout.addWidget(help_button_gpu)

        self.gpu_checkbox.stateChanged.connect(self.toggle_gpu_input)
        advanced_layout.addRow("GPU:", gpu_layout)

        # Run fast checkbox
        run_fast_layout = QHBoxLayout()
        self.run_fast_checkbox = QCheckBox("fast analysis")
        self.run_fast_checkbox.setChecked(False)
        run_fast_layout.addWidget(self.run_fast_checkbox)
        
        help_button_fast = QToolButton()
        help_button_fast.setText("?")
        help_button_fast.setFixedWidth(20)
        help_button_fast.setToolTip("Lower sampling in Fourier space and real space. Faster, needs less memory.")
        run_fast_layout.addWidget(help_button_fast)
        run_fast_layout.addStretch()

        advanced_layout.addRow("Run fast:", run_fast_layout)

        # --- Median resolution section ---
        median_res_header = QHBoxLayout()
        median_res_label = QLabel("Median resolution")
        median_res_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        median_res_header.addWidget(median_res_label)

        help_button_median_res = QToolButton()
        help_button_median_res.setText("?")
        help_button_median_res.setFixedWidth(20)
        help_button_median_res.setToolTip("Measuring global resolution with the following settings. Only applicable to Micrographs, Tilt-series and Tomograms.")
        median_res_header.addWidget(help_button_median_res)
        median_res_header.addStretch()

        advanced_layout.addRow(median_res_header)

        # Masking strategy dropdown
        strategy_layout = QHBoxLayout()
        self.mask_strategy_combo = QComboBox()
        self.mask_strategy_combo.addItems(["remove background", "signal mask", "full map"])
        strategy_layout.addWidget(self.mask_strategy_combo)

        help_button_strategy = QToolButton()
        help_button_strategy.setText("?")
        help_button_strategy.setFixedWidth(20)
        help_button_strategy.setToolTip(
            "Choose how to mask the map for global resolution estimation.\n"
            "• remove background: automatically remove regions not passing lowest measured resolution.\n"
            "• signal mask: provide a custom binary mask file.\n"
            "• full map: use the entire map without masking."
        )
        strategy_layout.addWidget(help_button_strategy)

        advanced_layout.addRow("Masking strategy:", strategy_layout)

        # Input mask file row (only visible when "signal mask" is selected)
        row_mask_layout = QHBoxLayout()
        self.input_mask = QLineEdit()
        row_mask_layout.addWidget(self.input_mask)

        mask_file_button = QPushButton("...")
        mask_file_button.setFixedWidth(30)
        mask_file_button.clicked.connect(lambda: self.open_file_dialog(self.input_mask))
        row_mask_layout.addWidget(mask_file_button)

        help_button_mask = QToolButton()
        help_button_mask.setText("?")
        help_button_mask.setFixedWidth(20)
        help_button_mask.setToolTip("Provide a binary mask file to focus the resolution estimate on the region of interest.")
        row_mask_layout.addWidget(help_button_mask)

        self.mask_row_container = QWidget()
        self.mask_row_container.setLayout(row_mask_layout)
        self.mask_row_container.setVisible(False)  # hidden by default
        advanced_layout.addRow("Input Mask:", self.mask_row_container)

        # Also hide the label for the mask row; we track it to show/hide together
        self.mask_row_label = advanced_layout.labelForField(self.mask_row_container)
        if self.mask_row_label:
            self.mask_row_label.setVisible(False)

        self.mask_strategy_combo.currentTextChanged.connect(self.toggle_mask_input)

        # Measure dropdown (independent, always visible under this section)
        measure_layout = QHBoxLayout()
        self.mask_measure_combo = QComboBox()
        self.mask_measure_combo.addItems([["median", "mean"]])
        measure_layout.addWidget(self.mask_measure_combo)

        help_button_measure = QToolButton()
        help_button_measure.setText("?")
        help_button_measure.setFixedWidth(20)
        help_button_measure.setToolTip("Measure to calculate global resolution from local measurements.")
        measure_layout.addWidget(help_button_measure)

        advanced_layout.addRow("Measure:", measure_layout)

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

    def on_config_changed(self, text):
        if text == "Micrographs":
            self.gpu_checkbox.setChecked(True)
        else:
            self.gpu_checkbox.setChecked(True)

    def toggle_mask_input(self, text):
        visible = (text == "signal mask")
        self.mask_row_container.setVisible(visible)
        if self.mask_row_label:
            self.mask_row_label.setVisible(visible)

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
        mask_strategy = self.mask_strategy_combo.currentText()
        mask_file = self.input_mask.text().strip() if mask_strategy == "signal mask" else ""
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
                mask_strategy=mask_strategy.replace(" ", "_"),
                signal_mask_input=mask_file,
                mask_measure=mask_measure,
                outputDir=outputDir,
                inputDir=""
            )
        else:
            print("Error: Could not import resolve.py's main function!")


def main():
    app = QApplication(sys.argv)
    window = ConfigPanel()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
