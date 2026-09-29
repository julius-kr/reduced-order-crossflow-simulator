#!/usr/bin/env python
# -*-coding:utf-8 -*-

import os
import numpy as np
from PIL import Image, ImageTk
from computation import flowsculpting
from computation_3D import particleengineering
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkFont
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# use discrete value to represent a slider
class DiscreteSlider(tk.Frame):
    def __init__(self, parent, values=None, command=None, width=220, font=("Arial", 12), **kwargs):
        super().__init__(parent)

        self._values = []
        self._command = command
        self._suspend_callback = False
        self._committed_values = []
        self._committed_value = ""

        self.value_var = tk.StringVar(value="")
        self.index_var = tk.IntVar(value=0)

        self.scale = tk.Scale(
            self,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            variable=self.index_var,
            command=self._on_scale_move,  
            length=width,
            resolution=1
        )
        self.scale.pack(side="left", fill="x", expand=True)

        self.value_label = tk.Label(self, textvariable=self.value_var, width=15, anchor="w", font=font)
        self.value_label.pack(side="left", padx=(8, 0))

        self.scale.bind("<ButtonRelease-1>", self._on_scale_release)

        self.set_values(values or ["-- Please choose --"])

    def _normalize_values(self, values):
        result = []
        for v in values:
            result.append(str(v))
        if not result:
            result = ["-- Please choose --"]
        return result

    def _emit_selected(self):
        if self._suspend_callback:
            return
        if self._command:
            self._command()

    def _sync_label_from_index(self):
        if not self._values:
            self.value_var.set("")
            return

        idx = self.index_var.get()
        idx = max(0, min(idx, len(self._values) - 1))
        self.index_var.set(idx)

        current_value = self._values[idx]

        valid_values = [v for v in self._values if v != "-- Please choose --"]
        if not valid_values or current_value == "-- Please choose --":
            self.value_var.set("")
            return

        try:
            max_value = max(float(v) for v in valid_values)

            self.value_var.set(f"{current_value} / {max_value}")

        except ValueError:
            self.value_var.set(current_value)

    def _on_scale_move(self, _=None):
        self._sync_label_from_index()

    def _on_scale_release(self, event=None):
        self._sync_label_from_index()
        self._emit_selected()

    def set_values(self, values):
        self._values = self._normalize_values(values)

        self._suspend_callback = True
        try:
            self.scale.config(to=max(len(self._values) - 1, 0))
            self.index_var.set(0)
            self._sync_label_from_index()
        finally:
            self._suspend_callback = False

    def get(self):
        if not self._values:
            return ""
        idx = self.index_var.get()
        idx = max(0, min(idx, len(self._values) - 1))
        return self._values[idx]

    def set(self, value):
        value = str(value)
        if value in self._values:
            idx = self._values.index(value)
            self._suspend_callback = True
            try:
                self.index_var.set(idx)
                self._sync_label_from_index()
            finally:
                self._suspend_callback = False

    def current(self, index):
        if not self._values:
            return
        index = max(0, min(index, len(self._values) - 1))
        self._suspend_callback = True
        try:
            self.index_var.set(index)
            self._sync_label_from_index()
        finally:
            self._suspend_callback = False

    def __setitem__(self, key, value):
        if key == "values":
            self.set_values(value)
        else:
            raise KeyError(key)

    def __getitem__(self, key):
        if key == "values":
            return self._values
        raise KeyError(key)
    
    def commit(self):
        self._committed_values = self._values[:]
        self._committed_value = self.get()

    def rollback(self):
        if not self._committed_values:
            return

        self.set_values(self._committed_values)

        target = str(self._committed_value)
        if target in self._values:
            self.set(target)
            return

        try:
            target_float = float(target)
            for v in self._values:
                try:
                    if float(v) == target_float:
                        self.set(v)
                        return
                except ValueError:
                    pass
        except ValueError:
            pass

class FlowSimulationApp:
    def __init__(self, 
                 root,
                 inlet_lib='generated_library_inlet',
                 pillar_lib='generated_library_pillar',
                 outlet_lib='generated_library_outlet',
                 pillar_params_keys=['cw', 'ch', 'ph', 'ya', 'yb', 'xc', 'yc'],
                 inlet_params_keys=['ch','ow', 'inlet', 'ip', 'pin'],
                 ):

        self.inlet_lib = inlet_lib
        self.pillar_lib = pillar_lib 
        self.outlet_lib = outlet_lib
        self.pillar_params_keys = pillar_params_keys
        self.inlet_params_keys = inlet_params_keys
        self.inlet_ui_keys = ['ch', 'ow', 'inlet', 'ip']
        
        self.root = root
        self.root.title("Flow Profile Simulation")
        
        self.setup_styles()
        self.init_variables()
        self.build_ui()

        ## Global params
        self.inlet_lib_data = None
        self.inlet_dict_data = None
        self.pillar_lib_data = None
        self.pillar_dict_data = None
        self.outlet_lib_data = None
        self.outlet_dict_data = None

        self.Hsculpt = flowsculpting()
        self.Pengine = particleengineering()

        self.auto_load_default_libraries()

    def setup_styles(self):
        """Unify global font and layout parameters"""
        
        # --- Font ---
        self.default_font = tkFont.nametofont("TkDefaultFont")
        self.default_font.configure(family="Arial", size=16)
        self.bold_font = tkFont.Font(family="Arial", size=25, weight="bold")
        # For the font of ttk.Combobox
        self.root.option_add('*TCombobox*Listbox.font', ('Arial', 16))

        # --- Core typography style dictionary ---
        # Basic styles of the container Frame
        self.style_frame_base = {"bd": 1, "relief": "ridge", "padx": 5, "pady": 5}
        # Layout of a container frame placed in a grid
        self.style_grid_frame = {"sticky": "nsew", "padx": 5}
        # Normal horizontal fill row
        self.style_pack_row = {"fill": "x", "pady": 5}
        # Title Label Layout
        self.style_pack_title = {"anchor": "w", "pady": (0, 10)}
        # Standardized layout for input boxes (Entry / Combobox)
        self.style_widget_entry = {"font": ("Arial", 16)}
        self.style_pack_entry_sec1 = {"side": "left", "fill": "x", "expand": True, "ipady": 8, "padx": (0, 10)}
        self.style_pack_entry_sec2 = {"fill": "x", "pady": (0, 10), "ipady": 8}

    def build_ui(self):
        """Build three-column main layout"""
        self.root.columnconfigure(0, weight=2)  
        self.root.columnconfigure(1, weight=3)  
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left panel
        self.left_panel = tk.Frame(self.root)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        self.left_panel.columnconfigure(0, weight=1)
        self.left_panel.rowconfigure(0, weight=0)
        self.left_panel.rowconfigure(1, weight=0)

        # right panel
        self.right_panel = tk.Frame(self.root)
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        self.right_panel.columnconfigure(0, weight=1)
        self.right_panel.columnconfigure(1, weight=1)
        self.right_panel.rowconfigure(0, weight=0)
        self.right_panel.rowconfigure(1, weight=0)
        self.right_panel.rowconfigure(2, weight=0)

        # particle panel
        self.particle_panel = tk.Frame(self.root)
        self.particle_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        self.particle_panel.columnconfigure(0, weight=1)

        self.build_left_panel()
        self.build_right_panel()
        self.build_particle_panel()

    def build_left_panel(self):
        self.build_left_top(self.left_panel, row=0, col=0)
        self.build_left_down(self.left_panel, row=1, col=0)

    def build_right_panel(self):
        self.build_right_section1(self.right_panel, row=0, col=0)
        self.build_right_section2(self.right_panel, row=1, col=0)
        self.build_right_section3(self.right_panel, row=1, col=1)
        self.build_right_section4(self.right_panel, row=2, col=0)

    def build_left_top(self, parent, row, col):
        frame = tk.Frame(parent, **self.style_frame_base)
        frame.grid(row=row, column=col, sticky="nsew", padx=5)

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=0)
        frame.rowconfigure(1, weight=0)
        frame.rowconfigure(2, weight=0)
        frame.rowconfigure(3, weight=0)

        tk.Label(frame, text="1. Library loading & initial flow profile:", font=self.bold_font).pack(anchor="w", pady=20)

        # --- One row for both libraries ---
        lib_row = tk.Frame(frame)
        lib_row.pack(fill="x", pady=5)

        # ===== Inlet =====
        inlet_frame = tk.Frame(lib_row)
        inlet_frame.pack(side="left", expand=True, fill="x", padx=(0, 10))

        tk.Label(inlet_frame, text="Inlet:", width=5, anchor="w").pack(side="left")

        self.inlet_lib_entry = tk.Entry(inlet_frame, width=5, **self.style_widget_entry)
        self.inlet_lib_entry.pack(side="left", fill="x", expand=True)
        self.inlet_lib_entry.insert(0, self.inlet_lib)

        tk.Button(
            inlet_frame,
            text="...",
            command=self.browse_inlet_file,
            width=5
        ).pack(side="right")

        # ===== Pillar =====
        pillar_frame = tk.Frame(lib_row)
        pillar_frame.pack(side="left", expand=True, fill="x")

        tk.Label(pillar_frame, text="Pillar:", width=5, anchor="w").pack(side="left")

        self.pillar_lib_entry = tk.Entry(pillar_frame, width=5, **self.style_widget_entry)
        self.pillar_lib_entry.pack(side="left", fill="x", expand=True)
        self.pillar_lib_entry.insert(0, self.pillar_lib)

        tk.Button(
            pillar_frame,
            text="...",
            command=self.browse_pillar_file,
            width=5
        ).pack(side="right")

        # ===== Outlet =====
        outlet_frame = tk.Frame(lib_row)
        outlet_frame.pack(side="left", expand=True, fill="x", padx=(10, 0))

        tk.Label(outlet_frame, text="Outlet:", width=6, anchor="w").pack(side="left")

        self.outlet_lib_entry = tk.Entry(outlet_frame, width=5, **self.style_widget_entry)
        self.outlet_lib_entry.pack(side="left", fill="x", expand=True)
        self.outlet_lib_entry.insert(0, self.outlet_lib)

        tk.Button(
            outlet_frame,
            text="...",
            command=self.browse_outlet_file,
            width=5
        ).pack(side="right")

        stream_frame = tk.Frame(frame)
        stream_frame.pack(fill="x", pady=10)

        tk.Label(
            stream_frame,
            text="Initial inlet number:",
            width=16,
            anchor="w",
            pady=10,
        ).pack(side="left")

        self.stream_num_combo = ttk.Combobox(
            stream_frame,
            values=["-- Please choose --", "1", "2", "3", "4", "5"],
            state="readonly",
            width=30,
            **self.style_widget_entry
        )
        self.stream_num_combo.current(0)

        self.stream_num_combo.pack(side="right")
        self.stream_num_combo.bind("<<ComboboxSelected>>", self.on_stream_number_change)

        spacer = tk.Frame(frame)
        spacer.pack(fill="both", expand=True)

        self.stream_canvas = tk.Canvas(frame, width=525, height=525, bg="lightgrey", highlightthickness=0, bd=0, relief="flat")
        self.stream_canvas.pack(pady=(0,15))

    def build_left_down(self, parent, row, col):
        frame = tk.Frame(parent, **self.style_frame_base)
        frame.grid(row=row, column=col, sticky="nsew", padx=5)

        # Configure internal grid
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=0)  # Title
        frame.rowconfigure(1, weight=0)  # Counter
        frame.rowconfigure(2, weight=1)  # Spacer
        frame.rowconfigure(3, weight=0)  # Canvas

        title_row = tk.Frame(frame)
        title_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(20, 35))

        tk.Label(
            title_row,
            text="3. Output flow profile",
            font=self.bold_font
        ).pack(side="left")

        tk.Button(
            title_row,
            text="Save image",
            command=self.export_output_flow_profile,
            width=12
        ).pack(side="right")
                    
        tk.Label(
            frame,
            textvariable=self.operator_count_var
        ).grid(row=1, column=0, pady=(0, 5))

        self.lower_result_canvas = tk.Canvas(
            frame,
            width=525,
            height=525,
            bg="lightgrey",
            highlightthickness=0,
            bd=0,
            relief="flat"
        )
        self.lower_result_canvas.grid(row=3, column=0, pady=15)
        self.stream_canvas.create_text(250, 250, text="Initial flow profile")

    def build_right_section1(self, parent, row,  col):
        frame = tk.Frame(parent, **self.style_frame_base)
        frame.grid(row=row, column=col, columnspan=2, sticky="nsew", padx=5)

        # Configure internal grid for the operator area
        for i in range(self.num_operators):
            frame.columnconfigure(i, weight=1)

        frame.rowconfigure(0, weight=0)  # Title row
        frame.rowconfigure(1, weight=0)  # Operator label row
        frame.rowconfigure(2, weight=0)  # Radio button row
        frame.rowconfigure(3, weight=0)  # Preview canvas row

        tk.Label(
            frame,
            text="2. Three types of operators:",
            font=self.bold_font
        ).grid(
            row=0,
            column=0,
            columnspan=self.num_operators,
            sticky="w",
            padx=10,
            pady=20
        )

        for i in range(self.num_operators):
            initial_state = "normal" if i == 0 else "disabled"

            # Operator title
            tk.Label(
                frame,
                text=f"Operator {i+1}"
            ).grid(row=1, column=i, pady=(0, 5))

            # Operator control area
            radio_frame = tk.Frame(frame, **self.style_frame_base)
            radio_frame.grid(row=2, column=i, sticky="nsew", padx=5)

            # inner frame for centering
            inner = tk.Frame(radio_frame)
            inner.pack(expand=True)

            # Inlet
            rb_a = tk.Radiobutton(
                inner,
                text="Inlet",
                variable=self.operator_vars[i],
                value=1,
                state=initial_state,
                command=lambda idx=i: self.on_operator_change(idx)
            )
            rb_a.pack(side="left")

            # Pillar
            rb_b = tk.Radiobutton(
                inner,
                text="Pillar",
                variable=self.operator_vars[i],
                value=2,
                state=initial_state,
                command=lambda idx=i: self.on_operator_change(idx)
            )
            rb_b.pack(side="left")

            # Clear button
            btn_clear = tk.Button(
                inner,
                text="X",
                fg="red",
                relief="flat",
                overrelief="raised",
                state=initial_state,
                command=lambda idx=i: self.clear_step(idx)
            )
            btn_clear.pack(side="left", padx=2)

            self.radio_widgets.append((rb_a, rb_b, btn_clear))

            # Operator preview area
            canvas = tk.Canvas(
                frame,
                width=180,
                height=180,
                bg="lightgrey",
                highlightthickness=0
            )
            canvas.grid(row=3, column=i, padx=5, pady=15, sticky="n")
            canvas.create_text(90, 90, text=f"Operator {i+1}")
            canvas.bind("<Button-1>", lambda event, idx=i: self.select_step(idx))

            self.upper_canvases.append(canvas)  

    def build_right_section2(self, parent, row, col):
        frame = tk.Frame(parent, **self.style_frame_base)
        frame.grid(row=row, column=col, **self.style_grid_frame)

        tk.Label(frame, text="⭐ Inlet operator:", font=self.bold_font).pack(anchor="w", pady=20)

        data_inlet_frame = tk.Frame(frame)
        data_inlet_frame.pack(**self.style_pack_row)
        tk.Label(data_inlet_frame, text="Inlet configuration available:").pack(side="left")
        tk.Label(data_inlet_frame, textvariable=self.data_inlet_var, fg="blue", font=self.bold_font).pack(side="left", padx=5)

        label_map = {
            'ch': 'Channel height (mm)',
            'ow': 'Operator width (mm)',
            'inlet': 'Inlet number at this row',
            'ip': 'Position of the inlets'
        }

        for i, name in enumerate(self.inlet_ui_keys):
            row_frame = tk.Frame(frame)
            row_frame.pack(**self.style_pack_row)

            tk.Label(row_frame, text=f"{label_map.get(name, name)}:", width=22, anchor="w").pack(side="left")

            slider = DiscreteSlider(
                row_frame,
                values=[""],
                command=lambda idx=i: self.on_inlet_combo_cascade(idx),
                width=150,
                font=self.style_widget_entry.get("font", ("Arial", 16))
            )
            slider.pack(side="left", fill="x", expand=True)

            self.inlet_comboboxes[name] = slider

        total_inlet_frame = tk.Frame(frame)
        total_inlet_frame.pack(**self.style_pack_row)
        tk.Label(total_inlet_frame, text="Total inlet number:").pack(side="left")
        tk.Label(total_inlet_frame, textvariable=self.total_inlet_var, fg="blue", font=self.bold_font).pack(side="left", padx=5)

        spacer = tk.Frame(frame)
        spacer.pack(fill="both", expand=True)

        inlet_preview_frame = tk.Frame(frame)
        inlet_preview_frame.pack(side="bottom")
        
        tk.Label(inlet_preview_frame, textvariable=self.inlet_preview_title_var).pack(pady=(20, 5))

        self.inlet_canvas = tk.Canvas(
            inlet_preview_frame,
            width=210,
            height=210,
            bg="lightgrey",
            highlightthickness=0,
            bd=0,
            relief="flat"
        )
        self.inlet_canvas.pack(side="bottom", pady=(0, 10))

    def build_right_section3(self, parent, row, col):
        frame = tk.Frame(parent, **self.style_frame_base)
        frame.grid(row=row, column=col, **self.style_grid_frame)

        tk.Label(frame, text="⭐ Indented pillar operator:", font=self.bold_font).pack(anchor="w", pady=20)

        data_pillar_frame = tk.Frame(frame)
        data_pillar_frame.pack(fill="x", pady=5)
        tk.Label(data_pillar_frame, text="Pillar configuration available:").pack(side="left")
        tk.Label(data_pillar_frame, textvariable=self.data_pillar_var, fg="blue", font=self.bold_font).pack(side="left", padx=5)
 
        label_map = {
        "cw": "Channel width (mm)",
        "ch": "Channel height (mm)",
        "ph": "Indented pillar depth (mm)",
        "ya": "y-coordinate of point A (mm)",
        "yb": "y-coordinate of point B (mm)",
        "xc": "x-coordinate of point C (mm)",
        "yc": "y-coordinate of point C (mm)"
        }

        for i, name in enumerate(self.pillar_params_keys):
            row_frame = tk.Frame(frame)
            row_frame.pack(**self.style_pack_row)
            tk.Label(row_frame, text=f"{label_map.get(name, name)}:", width=25, anchor="w").pack(side="left")
            
            slider = DiscreteSlider(
                row_frame,
                values=[""],
                command=lambda idx=i: self.on_pillar_combo_cascade(idx),
                width=150,
                font=self.style_widget_entry.get("font", ("Arial", 16))
            )
            slider.pack(side="left", fill="x", expand=True)

            self.pillar_comboboxes[name] = slider

        spacer = tk.Frame(frame)
        spacer.pack(fill="both", expand=True)
        
        pillar_preview_frame = tk.Frame(frame)
        pillar_preview_frame.pack(side="bottom")

        tk.Label(pillar_preview_frame, textvariable=self.pillar_preview_title_var).pack(pady=(20, 5))

        self.pillar_canvas = tk.Canvas(
            pillar_preview_frame,
            width=210,
            height=210,
            bg="lightgrey",
            highlightthickness=0,
            bd=0,
            relief="flat"
        )
        self.pillar_canvas.pack(side="bottom", pady=(0, 10))

    def build_right_section4(self, parent, row, col):
        frame = tk.Frame(parent, **self.style_frame_base)
        frame.grid(row=row, column=col, columnspan=2, sticky="nsew", padx=5)

        tk.Label(frame, text="⭐ Outlet operator (the last operator):", font=self.bold_font).pack(anchor="w", pady=(20, 5))

        self.outlet_canvases = []

        # Container to hold all outlet canvases
        self.outlet_canvas_container = tk.Frame(frame)
        self.outlet_canvas_container.pack(fill="x", pady=(12,6))

        # Store references to all outlet canvases
        self.outlet_canvases = []

    def build_right_section4_outlet_canvases(self):
        """
        Dynamically create outlet canvases in right panel section 4.
        The canvases are arranged in multiple columns and can scroll vertically.
        """

        if not hasattr(self, "outlet_canvas_container"):
            return

        # Clear existing outlet canvases
        for widget in self.outlet_canvas_container.winfo_children():
            widget.destroy()

        self.outlet_canvases = []

        if not self.outlet_lib_data:
            tk.Label(
                self.outlet_canvas_container,
                text="No outlet data"
            ).grid(row=0, column=0, padx=5, pady=10)
            return

        max_cols = 5

        # Let outlet columns expand evenly, similar to right section 1
        for col in range(max_cols):
            self.outlet_canvas_container.columnconfigure(col, weight=1)

        # Add default outlet at the beginning
        outlet_keys = ["square"] + list(self.outlet_lib_data.keys())

        for i, key in enumerate(outlet_keys):
            r = i // max_cols
            c = i % max_cols

            # Each outlet item contains a selector row and a preview canvas
            item_frame = tk.Frame(self.outlet_canvas_container)
            item_frame.grid(row=r, column=c, padx=5, pady=(10,5), sticky="nsew")

            # Selection control area, similar to right section 1
            select_frame = tk.Frame(item_frame, **self.style_frame_base)
            select_frame.pack(fill="x", pady=(0, 15))

            inner = tk.Frame(select_frame)
            inner.pack(expand=True)

            tk.Label(
                inner,
                text=key
            ).pack(side="left", padx=(5, 10))

            radio = tk.Radiobutton(
                inner,
                text="default" if key == "square" else "select",
                variable=self.outlet_var,
                value=key,
                command=lambda k=key: self.select_outlet(k)
            )
            radio.pack(side="left")

            # Outlet preview canvas
            canvas = tk.Canvas(
                item_frame,
                width=180,
                height=180,
                bg="lightgrey",
                highlightthickness=1
            )
            canvas.pack(pady=0)

            canvas.bind("<Button-1>", lambda event, k=key: self.select_outlet(k))

            self.outlet_canvases.append((canvas, key, radio))

            if key == "square":
                img = self.Hsculpt.show_the_outlet_shape_in_outlet_operator(None)
                self.display_image_on_canvas(img, canvas)

            elif key in self.outlet_lib_data:
                entry = self.outlet_lib_data[key]

                if isinstance(entry, dict) and "valid_mask" in entry:
                    valid_mask = entry["valid_mask"]

                    img = self.Hsculpt.show_the_outlet_shape_in_outlet_operator(valid_mask)

                    self.display_image_on_canvas(img, canvas)

    def build_particle_panel(self):
        outer = tk.Frame(self.particle_panel, **self.style_frame_base)
        outer.grid(row=0, column=0, sticky="nsew", padx=5)

        self.particle_panel.columnconfigure(0, weight=1)

        outer.columnconfigure(0, weight=1)

        tk.Label(
            outer,
            text="4. 3D particle engineering workspace",
            font=self.bold_font
        ).grid(row=0, column=0, sticky="w", padx=12, pady=20)

        self.build_flow_profile_panel(outer, row=1)
        self.build_photomask_panel(outer, row=2)
        self.build_intersection_panel(outer, row=3)
        self.build_3d_particle_panel(outer, row=4)


    def build_flow_profile_panel(self, parent, row):
        section = tk.Frame(parent, bd=0, relief="flat")
        section.grid(row=row, column=0, sticky="nsew", padx=10, pady=(0, 8))

        tk.Label(section, text="Flow profile (structured flow):").grid(
            row=0, column=0, sticky="w", pady=(0,8)
        )

        content_row = tk.Frame(section)
        content_row.grid(row=1, column=0, sticky="w")

        self.flow_profile_canvas = tk.Canvas(
            content_row,
            width=250,
            height=250,
            bg="lightgrey",
            highlightthickness=0
        )
        self.flow_profile_canvas.pack(side="left")

        button_col = tk.Frame(content_row)
        button_col.pack(side="left", padx=(20, 0), anchor="n")

        tk.Button(
            button_col,
            text="Use/renew output flow profile",
            width=25,
            command=self.use_output_flow_profile
        ).pack(padx=10, pady=10)

        tk.Button(
            button_col,
            text="Stream segmentation",
            width=36,
            command=self.enable_flow_segmentation
        ).pack(padx=10, pady=10)

        tk.Button(
            button_col,
            text="Finalize curable and non-curable regions\n(black is curable region)",
            width=36,
            command=self.finalize_flow_mask
        ).pack(padx=10, pady=10)


    def build_photomask_panel(self, parent, row):
        section = tk.Frame(parent, bd=0, relief="flat")
        section.grid(row=row, column=0, sticky="nsew", padx=10, pady=(0, 8))

        top_row = tk.Frame(section)
        top_row.grid(row=0, column=0, sticky="w", pady=(0, 3))

        tk.Label(top_row, text="Photomask (patterned UV exposure):").pack(anchor="w")

        content_row = tk.Frame(section)
        content_row.grid(row=1, column=0, sticky="w")

        self.photomask_canvas = tk.Canvas(
            content_row,
            width=250,
            height=250,
            bg="lightgrey",
            highlightthickness=0
        )
        self.photomask_canvas.pack(side="left")

        
        button_col = tk.Frame(content_row)
        button_col.pack(side="left", padx=(20,0), anchor="n")

        upload_row = tk.Frame(button_col)
        upload_row.pack(padx=10, pady=10, fill="x")

        tk.Label(
            upload_row,
            text="Upload photomask image:",
            width=20,
            anchor="w"
        ).pack(side="left")

        self.photomask_path_entry = tk.Entry(
            upload_row,
            width=8,
            font=("Arial", 12)
        )
        self.photomask_path_entry.pack(side="left", expand=True)

        tk.Button(
            upload_row,
            text="...",
            width=8,
            command=self.browse_photomask_file
        ).pack(side="right")

        tk.Button(
            button_col,
            text="Photomask segmentation",
            width=36,
            command=self.enable_photomask_segmentation
        ).pack(padx=10, pady=10)

        tk.Button(
            button_col,
            text="Finalize exposured and non-exposured regions\n(black is exposured region)",
            width=36,
            command=self.finalize_photomask_mask
        ).pack(padx=10, pady=10)

    def build_intersection_panel(self, parent, row):
        section = tk.Frame(parent, bd=0, relief="flat")
        section.grid(row=row, column=0, sticky="nsew", padx=10, pady=(0, 8))

        top_row = tk.Frame(section)
        top_row.grid(row=0, column=0, sticky="ew", pady=(0, 3))

        tk.Label(top_row, text="Top view intersection of flow channel (gray lines) and photomask:").pack(side="left", pady=(10, 8))

        canvas_row = tk.Frame(section)
        canvas_row.grid(row=1, column=0, sticky="w")

        self.intersection_canvas = tk.Canvas(
            canvas_row,
            width=250,
            height=250,
            bg="lightgrey",
            highlightthickness=0
        )
        self.intersection_canvas.pack(side="left")
        control_col = tk.Frame(canvas_row)
        control_col.pack(side="left", padx=(15, 0), anchor="n")

        self.intersection_scale_var = tk.DoubleVar(value=1.0)
        self.intersection_x_var = tk.DoubleVar(value=0.0)
        self.intersection_y_var = tk.DoubleVar(value=0.0)
        self.intersection_rotation_var = tk.IntVar(value=0)

        row = tk.Frame(control_col)
        row.pack(anchor="w", pady=(0, 6))

        # Scale slider
        scale_row = tk.Frame(control_col)
        scale_row.pack(anchor="w", pady=(0, 8))

        tk.Label(
            scale_row,
            text="Scale",
            width=15,
            anchor="w"
        ).pack(side="left")

        tk.Scale(
            scale_row,
            from_=0.5,
            to=2.0,
            resolution=0.05,
            orient="horizontal",
            variable=self.intersection_scale_var,
            command=lambda v: self.redraw_intersection(),
            length=160
        ).pack(side="left")

        # Left / right movement slider
        x_row = tk.Frame(control_col)
        x_row.pack(anchor="w", pady=(0, 8))

        tk.Label(
            x_row,
            text="Move left / right",
            width=15,
            anchor="w"
        ).pack(side="left")

        tk.Scale(
            x_row,
            from_=-100,
            to=100,
            resolution=1,
            orient="horizontal",
            variable=self.intersection_x_var,
            command=lambda v: self.redraw_intersection(),
            length=160
        ).pack(side="left")

        rotate_row = tk.Frame(control_col)
        rotate_row.pack(anchor="w", pady=(0, 8))

        tk.Label(
            rotate_row,
            text="Rotate",
            width=15,
            anchor="w"
        ).pack(side="left")

        tk.Scale(
            rotate_row,
            from_=-180,
            to=180,
            resolution=90,
            orient="horizontal",
            variable=self.intersection_rotation_var,
            command=lambda v: self.redraw_intersection(),
            length=160
        ).pack(side="left")


    def build_3d_particle_panel(self, parent, row):

        section = tk.Frame(parent, bd=0, relief="flat")
        section.grid(row=row, column=0, sticky="nsew", padx=10, pady=(0, 0))

        section.columnconfigure(0, weight=0)
        section.columnconfigure(1, weight=1)

        control_col = tk.Frame(section)
        control_col.grid(
            row=0,
            column=0,
            sticky="nw",
            padx=(0, 20)
        )

        tk.Label(
            control_col,
            text="3D particle preview:"
        ).pack(anchor="w", pady=(10, 10))

        tk.Button(
            control_col,
            text="Generate",
            width=10,
            command=self.render_3d_particle_intersection
        ).pack(anchor="w")

        tk.Label(
            control_col,
            text="View angle:"
        ).pack(anchor="w", pady=(20, 5))

        tk.Button(
            control_col,
            text="Photomask direction",
            width=20,
            command=lambda: self.set_3d_view(elev=180, azim=270)
        ).pack(anchor="w", pady=2)

        tk.Button(
            control_col,
            text="Channel direction",
            width=20,
            command=lambda: self.set_3d_view(elev=90, azim=270)
        ).pack(anchor="w", pady=2)

        tk.Button(
            control_col,
            text="Isometric",
            width=20,
            command=lambda: self.set_3d_view(elev=25, azim=-60)
        ).pack(anchor="w", pady=2)

        canvas_holder = tk.Frame(section)
        canvas_holder.grid(
            row=0,
            column=1,
            sticky="e"
        )

        self.particle_3d_canvas = tk.Canvas(
            canvas_holder,
            width=470,
            height=470,
            bg="lightgrey",
            highlightthickness=0
        )

        self.particle_3d_canvas.pack()

    def auto_load_default_libraries(self):
        current_dir = os.getcwd()

        inlet_folder = os.path.join(current_dir, self.inlet_lib)
        pillar_folder = os.path.join(current_dir, self.pillar_lib)
        outlet_folder = os.path.join(current_dir, self.outlet_lib)

        inlet_ok = False
        pillar_ok = False
        outlet_ok = False

        try:
            self.load_inlet_library(inlet_folder)
            inlet_ok = True
        except Exception as e:
            print(f"Auto-load inlet library failed: {e}")

        try:
            self.load_pillar_library(pillar_folder)
            pillar_ok = True
        except Exception as e:
            print(f"Auto-load pillar library failed: {e}")

        try:
            self.load_outlet_library(outlet_folder)
            outlet_ok = True
        except Exception as e:
            print(f"Auto-load outlet library failed: {e}")

        if inlet_ok or pillar_ok or outlet_ok:
            self.refresh_all()
            self.commit_inlet_sliders()

    def browse_inlet_file(self):
        current_dir = os.getcwd()
        target_dir = os.path.join(current_dir, getattr(self, 'inlet_lib', ''))

        filepath = filedialog.askopenfilename(
            title="Select Inlet Library File",
            initialdir=target_dir,
            filetypes=[("Numpy Archives", "*.npy")]
        )

        if not filepath:
            return

        folder_path = os.path.dirname(filepath)

        try:
            self.load_inlet_library(folder_path)
            self.refresh_all()
        except Exception as e:
            messagebox.showerror("Loading Error", str(e))

    def browse_photomask_file(self):

        current_dir = os.getcwd()

        filepath = filedialog.askopenfilename(
            title="Select photomask image",
            initialdir=current_dir,
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.gif"),
                ("All Files", "*.*")
            ]
        )

        if filepath:
            self.photomask_path_entry.delete(0, tk.END)
            self.photomask_path_entry.insert(0, filepath)
            self.load_photomask_image()

    def load_photomask_image(self):
        filepath = self.photomask_path_entry.get().strip()

        if not filepath:
            messagebox.showwarning("No image selected", "Please select a photomask image first.")
            return

        try:
            img = Image.open(filepath).convert("RGB")
        except Exception as e:
            messagebox.showerror("Loading Error", f"Cannot load image:\n{e}")
            return

        self.photomask_img = img
        self.photomask_mask_state = {}
        self.photomask_segmentation_mode = False

        self.display_image_on_canvas(self.photomask_img, self.photomask_canvas)

    def load_inlet_library(self, folder_path):
        file_dict_path = os.path.join(folder_path, 'library_inlet_dict.npy')
        file_lib_path = os.path.join(folder_path, 'library_inlet.npy')

        if not (os.path.exists(file_dict_path) and os.path.exists(file_lib_path)):
            raise FileNotFoundError(
                f"Inlet library files not found in:\n{folder_path}\n\n"
                "Required files:\n"
                "1. streamline_files_inlet_dict.npy\n"
                "2. streamline_files_inlet.npy"
            )

        self.inlet_dict_data = np.load(file_dict_path, allow_pickle=True).item()
        self.inlet_lib_data = np.load(file_lib_path, allow_pickle=True).item()

        self.inlet_lib_entry.delete(0, tk.END)
        self.inlet_lib_entry.insert(0, folder_path)

        self.update_inlet_comboboxes()
        self.compute_inlet_stream_numbers()

        print("✅ Successfully loaded inlet library files!")

    def browse_pillar_file(self):
        current_dir = os.getcwd()
        target_dir = os.path.join(current_dir, getattr(self, 'pillar_lib', ''))

        filepath = filedialog.askopenfilename(
            title="Select Pillar Library File",
            initialdir=target_dir,
            filetypes=[("Numpy Archives", "*.npy")]
        )

        if not filepath:
            return

        folder_path = os.path.dirname(filepath)

        try:
            self.load_pillar_library(folder_path)
            self.refresh_all()
        except Exception as e:
            messagebox.showerror("Loading Error", str(e))

    def browse_outlet_file(self):
        current_dir = os.getcwd()
        target_dir = os.path.join(current_dir, getattr(self, 'outlet_lib', ''))

        filepath = filedialog.askopenfilename(
            title="Select Outlet Library File",
            initialdir=target_dir,
            filetypes=[("Numpy Archives", "*.npy")]
        )

        if not filepath:
            return

        folder_path = os.path.dirname(filepath)

        try:
            self.load_outlet_library(folder_path)
            self.refresh_all()
        except Exception as e:
            messagebox.showerror("Loading Error", str(e))

    def load_pillar_library(self, folder_path):
        file_dict_path = os.path.join(folder_path, 'library_pillar_dict.npy')
        file_lib_path = os.path.join(folder_path, 'library_pillar.npy')

        if not (os.path.exists(file_dict_path) and os.path.exists(file_lib_path)):
            raise FileNotFoundError(
                f"Pillar library files not found in:\n{folder_path}\n\n"
                "Required files:\n"
                "1. streamline_files_pillar_dict.npy\n"
                "2. streamline_files_pillar.npy"
            )

        self.Hsculpt.load_libraries(file_lib_path)
        self.pillar_dict_data = np.load(file_dict_path, allow_pickle=True).item()
        self.pillar_lib_data = np.load(file_lib_path, allow_pickle=True).item()

        self.pillar_lib_entry.delete(0, tk.END)
        self.pillar_lib_entry.insert(0, folder_path)

        self.update_pillar_comboboxes()
        self.compute_pillar_stream_numbers()

        print("✅ Successfully loaded pillar library files!")

    def load_outlet_library(self, folder_path):
        file_dict_path = os.path.join(folder_path, 'library_outlet_dict.npy')
        file_lib_path = os.path.join(folder_path, 'library_outlet.npy')

        if not (os.path.exists(file_dict_path) and os.path.exists(file_lib_path)):
            raise FileNotFoundError(
                f"Outlet library files not found in:\n{folder_path}\n\n"
                "Required files:\n"
                "1. streamline_files_outlet_dict.npy\n"
                "2. streamline_files_outlet.npy"
            )

        self.outlet_dict_data = np.load(file_dict_path, allow_pickle=True).item()
        self.outlet_lib_data = np.load(file_lib_path, allow_pickle=True).item()

        self.outlet_lib_entry.delete(0, tk.END)
        self.outlet_lib_entry.insert(0, folder_path)

        print("✅ Successfully loaded outlet library files!")
        self.build_right_section4_outlet_canvases()

    def on_stream_number_change(self, event=None):
        """
        Triggered whenever the user selects a new value from the stream number Combobox 
        in lower section 1.
        """
        
        snapshot = self.make_state_snapshot()
        
        # Get the current string value from the combobox
        selected_value = self.stream_num_combo.get()
        
        # Make sure the user actually selected a number, not the placeholder
        if selected_value == "-- Please choose --":
            return
        stream_count = int(selected_value)

        inlet_sum = 0
        for step in self.steps:
            if step["type"] == "none":
                break

            if step["type"] == "inlet":
                params = step["params"] or {}
                inlet_sum += int(float(params.get("inlet", 0)))

        if stream_count + inlet_sum > 5:
            self.is_reverting_stream = True
            try:
                self.stream_num_combo.set(self.last_valid_stream_num)
                self.update_total_inlet_number()
            finally:
                self.is_reverting_stream = False

            self.root.after_idle(
                lambda: messagebox.showwarning(
                    "Warning",
                    "no more than 5 inlets in total"
                )
            )
            return

        # Calculate the splitting points (excluding 1.0)
        self.dividers = [i / stream_count for i in range(1, stream_count)]
        self.dividers.append(1.0)
        
        # Reset the engine and create the new inlet
        self.Hsculpt.reset()
        self.Hsculpt.create_initial_flow_profile(self.dividers)
        
        # Display the newly generated inlet color image on the stream canvas
        self.display_image_on_canvas(self.Hsculpt.inlet_color, self.stream_canvas)

        self.last_valid_stream_num = selected_value

        self.recompute_all_inlet_steps()

        self.refresh_radio_states()
        self.refresh_upper_canvases()
        self.refresh_preview_panel()
        self.render_cumulative_result()
        self.update_operator_count()
        self.update_total_inlet_number()

    def on_combobox_change(self, event=None):
        if self.is_loading_params or self.is_reverting_stream:
            return
        self.update_active_step_params()

    def on_inlet_combo_cascade(self, changed_index):
        if (
            self.active_step_index != -1
            and self.steps[self.active_step_index]["type"] == "inlet"
            and self.inlet_ui_keys[changed_index] == "inlet"
        ):
            old_params = self.steps[self.active_step_index]["params"] or {}
            old_inlet = int(float(old_params.get("inlet", 0)))

            try:
                new_inlet = int(float(self.inlet_comboboxes["inlet"].get()))
            except ValueError:
                return

            current_total = self.get_total_inlet_number()
            candidate_total = current_total - old_inlet + new_inlet

            if candidate_total > 5:
                self.rollback_inlet_sliders()

                self.update_total_inlet_number()

                self.root.after_idle(
                    lambda: messagebox.showwarning(
                        "Warning",
                        "no more than 5 inlets in total"
                    )
                )
                return
        
        tree = self.inlet_dict_data.get('tree', {})
        if not tree:
            return

        curr_dict = tree
        valid_path = True

        for i in range(changed_index + 1):
            name = self.inlet_ui_keys[i]
            val_str = self.inlet_comboboxes[name].get()
            try:
                val_float = float(val_str)
                if val_float in curr_dict:
                    curr_dict = curr_dict[val_float]
                else:
                    valid_path = False
                    break
            except ValueError:
                valid_path = False
                break

        if valid_path:
            for i in range(changed_index + 1, len(self.inlet_ui_keys)):
                name = self.inlet_ui_keys[i]
                combo = self.inlet_comboboxes[name]

                options = sorted(list(curr_dict.keys()))
                combo['values'] = [str(v) for v in options]

                if options:
                    combo.current(0)
                    curr_dict = curr_dict[options[0]]
                else:
                    combo.set("")
                    curr_dict = {}

        self.on_combobox_change()
        self.update_total_inlet_number()
        
    
    def on_pillar_combo_cascade(self, changed_index):
        """Handling tree-like cascading refreshes of dropdown lists"""
        tree = self.pillar_dict_data.get('tree', {})
        if not tree:
            return
            
        curr_dict = tree
        valid_path = True
        
        for i in range(changed_index + 1):
            name = self.pillar_params_keys[i]
            val_str = self.pillar_comboboxes[name].get()
            try:
                val_float = float(val_str)
                if val_float in curr_dict:
                    curr_dict = curr_dict[val_float]
                else:
                    valid_path = False
                    break
            except ValueError:
                valid_path = False
                break
                
        if valid_path:
            for i in range(changed_index + 1, len(self.pillar_params_keys)):
                name = self.pillar_params_keys[i]
                combo = self.pillar_comboboxes[name]
                
                options = sorted(list(curr_dict.keys()))
                combo['values'] = [str(v) for v in options]

                if options:
                    combo.current(0)
                    curr_dict = curr_dict[options[0]]
                else:
                    combo.set("")
                    curr_dict = {}

        self.on_combobox_change()

    def on_operator_change(self, clicked_index=None):
        if clicked_index is None:
            return

        choice = self.operator_vars[clicked_index].get()
        step_type = self.get_step_type_from_choice(choice)
        self.change_step_type(clicked_index, step_type)

    def render_step_image(self, index, target_canvas):
        step = self.steps[index]
        if step["type"] == "none":
            self.show_no_data_on_canvas(target_canvas)
            return

        active_keys = []
        for i in range(index + 1):
            s = self.steps[i]
            if s["type"] == "none":
                break
            if s["key"] is not None:
                active_keys.append((s["type"], s["key"]))

        if not active_keys:
            target_canvas.delete("all")
            target_canvas.create_text(100, 100, text=f"Step {index+1}\n(No Data)", justify="center")
            return

        self.Hsculpt.reset()
        self.Hsculpt.create_initial_flow_profile(self.dividers)

        if self.Hsculpt.map_lib is None:
            self.Hsculpt.map_lib = {}

        op_sequence = []

        for op_type, key in active_keys:
            if op_type == "pillar":
                if key not in self.pillar_lib_data:
                    continue
                self.Hsculpt.map_lib[key] = self.pillar_lib_data[key]

            elif op_type == "inlet":
                if key not in self.inlet_lib_data:
                    continue
                self.Hsculpt.map_lib[key] = self.inlet_lib_data[key]

            op_sequence.append((op_type, key))

        self.Hsculpt.compute_and_generate_new_flow_profile(
            op_sequence=op_sequence,
            treat_empty_as_new_source=True
        )

        final_img = self.Hsculpt.outlet_color
        self.display_image_on_canvas(final_img, target_canvas)
    
    def render_cumulative_result(self):
        # Return immediately if the canvas has not been created yet
        if not hasattr(self, 'lower_result_canvas'):
            return

        selected_stream = self.stream_num_combo.get()

        if selected_stream == "-- Please choose --":
            self.lower_result_canvas.delete("all")
            self.lower_result_canvas.create_text(
                250, 250,
                text="Cumulative result\n(No Data)",
                justify="center"
            )
            return

        active_keys = []

        for i in range(self.num_operators):
            step = self.steps[i]
            if step["type"] == "none":
                break
            if step["key"] is not None:
                active_keys.append((step["type"], step["key"]))

        # --- Start calling the ActiSculpt engine for cumulative chained calculation ---
        self.Hsculpt.reset()
        
        # Set the initial fluid state (you can change this to dynamically read from an input box later)
        self.Hsculpt.create_initial_flow_profile(self.dividers) 

        if self.Hsculpt.map_lib is None:
            self.Hsculpt.map_lib = {}

        op_sequence = []

        for op_type, key in active_keys:
            if op_type == "pillar":
                if key not in self.Hsculpt.map_lib:
                    self.Hsculpt.map_lib[key] = self.pillar_lib_data[key]
            elif op_type == "inlet":
                if key not in self.Hsculpt.map_lib:
                    self.Hsculpt.map_lib[key] = self.inlet_lib_data[key]

            op_sequence.append((op_type, key))

        # Apply selected outlet as the last operator
        selected_outlet = self.outlet_var.get()

        if selected_outlet and selected_outlet != "square":
            if self.outlet_lib_data is not None and selected_outlet in self.outlet_lib_data:
                self.Hsculpt.map_lib[selected_outlet] = self.outlet_lib_data[selected_outlet]
                op_sequence.append(("outlet", selected_outlet))

        self.Hsculpt.compute_and_generate_new_flow_profile(
            op_sequence=op_sequence,
            treat_empty_as_new_source=True
        )
        
        final_img = self.Hsculpt.outlet_color

        self.display_image_on_canvas(final_img, self.lower_result_canvas)

    def export_output_flow_profile(self):
        if not hasattr(self.Hsculpt, "outlet_color") or self.Hsculpt.outlet_color is None:
            messagebox.showwarning(
                "No output flow profile",
                "Please generate an output flow profile first."
            )
            return

        filepath = os.path.join(os.getcwd(), "output_flow_profile.png")

        if not filepath:
            return

        try:
            img = self.Hsculpt.outlet_color.resize(
                (1024, 1024),
                Image.NEAREST
            )
            img.save(filepath, dpi=(300,300))
            messagebox.showinfo("Saved", f"Saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Cannot save:\n{e}")

    def update_inlet_comboboxes(self):
        if not hasattr(self, 'inlet_dict_data') or self.inlet_dict_data is None:
            return
        
        tree = self.inlet_dict_data.get('tree', {})
        first_key = self.inlet_ui_keys[0]

        if tree:
            options = sorted(list(tree.keys()))
            options = [str(v) for v in options]
            self.inlet_comboboxes[first_key]['values'] = options

            if options:
                self.inlet_comboboxes[first_key].current(0)

            self.on_inlet_combo_cascade(changed_index=0)
    
    def update_pillar_comboboxes(self):
        """After reading the file, only the first dropdown list is initialized, triggering a cascading effect to update subsequent dropdown lists."""
        if not hasattr(self, 'pillar_dict_data'):
            return
            
        tree = self.pillar_dict_data.get('tree', {})
        first_key = self.pillar_params_keys[0] # 'cw'
        
        if tree:
            options = [str(v) for v in sorted(list(tree.keys()))]
            self.pillar_comboboxes[first_key]['values'] = options
            
            if options:
                self.pillar_comboboxes[first_key].current(0)
                
            self.on_pillar_combo_cascade(changed_index=0)
    
    def compute_inlet_stream_numbers(self):

        if not hasattr(self, 'inlet_lib_data') or self.inlet_lib_data is None:
            self.data_inlet_var.set("0")
            return
        total_num = len(self.inlet_lib_data)
        self.data_inlet_var.set(str(total_num))
    
    def compute_pillar_stream_numbers(self):
        total_num = len(self.pillar_lib_data)
        self.data_pillar_var.set(str(total_num))
    
    def make_inlet_key_from_params(self, params):
        parts = []
        for name in self.inlet_params_keys:
            try:
                val_float = float(params[name])
            except (KeyError, ValueError, TypeError):
                return None
            parts.append(f"{name}{val_float:.4f}")
        return "_".join(parts)

    def get_current_inlet_key(self,step_index):
        parts = []

        for name in self.inlet_params_keys:
            if name == 'pin':
                val_float = self.get_pin_before_step(step_index)
            else:
                val_str = self.inlet_comboboxes[name].get()
                try:
                    val_float = float(val_str)
                except ValueError:
                    return None
            parts.append(f"{name}{val_float:.4f}")    

        return "_".join(parts)

    def get_current_pillar_key(self):
        """Generate the formatted dictionary key from the 7 comboboxes"""
        prefixes = self.pillar_params_keys
        parts = []
        
        for i, name in enumerate(self.pillar_params_keys):
            val_str = self.pillar_comboboxes[name].get()
            try:
                val_float = float(val_str)
                parts.append(f"{prefixes[i]}{val_float:.4f}")
            except ValueError:
                return None 
                
        final_key = "_".join(parts)
        return final_key
    
    def get_current_inlet_params(self, step_index):
        params = {}

        for name in self.inlet_params_keys:
            if name == 'pin':
                params[name] = self.get_pin_before_step(step_index)
            else:
                val_str = self.inlet_comboboxes[name].get()
                try:
                    params[name] = float(val_str)
                except ValueError:
                    return None

        return params
    
    def restore_inlet_ui_from_params(self, params):
        if not params:
            return

        self.is_loading_params = True
        try:
            for name in self.inlet_ui_keys:
                if name in params and name in self.inlet_comboboxes:
                    self.inlet_comboboxes[name].set(str(params[name]))
        finally:
            self.is_loading_params = False


    def restore_inlet_ui_from_params_with_values(self, params):
        if not params or not self.inlet_dict_data:
            return

        tree = self.inlet_dict_data.get("tree", {})
        curr_dict = tree

        self.is_loading_params = True
        try:
            for name in self.inlet_ui_keys:
                combo = self.inlet_comboboxes[name]

                options = sorted(list(curr_dict.keys()))
                combo["values"] = [str(v) for v in options]

                target = float(params[name])
                combo.set(str(target))

                if target in curr_dict:
                    curr_dict = curr_dict[target]
                else:
                    break
        finally:
            self.is_loading_params = False

    def restore_pillar_ui_from_params(self, params):
        if not params or not self.pillar_dict_data:
            return

        tree = self.pillar_dict_data.get("tree", {})
        curr_dict = tree

        self.is_loading_params = True
        try:
            for name in self.pillar_params_keys:
                combo = self.pillar_comboboxes[name]

                options = sorted(list(curr_dict.keys()))
                combo["values"] = [str(v) for v in options]

                target = float(params[name])
                combo.set(str(target))

                if target in curr_dict:
                    curr_dict = curr_dict[target]
                else:
                    break
        finally:
            self.is_loading_params = False

    def get_current_pillar_params(self):
        params = {}
        for name in self.pillar_params_keys:
            val_str = self.pillar_comboboxes[name].get()
            try:
                params[name] = float(val_str)
            except ValueError:
                return None
        return params
    
    def get_step_type_from_choice(self, choice):
        mapping = {
            0: "none",
            1: "inlet",
            2: "pillar",
        }
        return mapping.get(choice, "none")

    def get_choice_from_step_type(self, step_type):
        mapping = {
            "none": 0,
            "inlet": 1,
            "pillar": 2,
        }
        return mapping.get(step_type, 0)

    def reset_step_state(self, index):
        self.steps[index] = {
            "type": "none",
            "key": None,
            "params": None,
            "saved": {
                "inlet": None,
                "pillar": None,
            }
        }

    def set_step_state(self, index, step_type, key, params):
        step = self.steps[index]

        saved = step.get("saved", {"inlet": None, "pillar": None})
        if step_type in saved:
            saved[step_type] = {
                "key": key,
                "params": None if params is None else params.copy()
            }

        self.steps[index] = {
            "type": step_type,
            "key": key,
            "params": params,
            "saved": saved,
        }

    def reset_steps_from(self, start_index):
        for i in range(start_index, self.num_operators):
            self.reset_step_state(i)

    def is_step_enabled(self, index):
        if index == 0:
            return True
        return self.steps[index - 1]["type"] != "none"

    def get_active_step(self):
        if 0 <= self.active_step_index < self.num_operators:
            return self.steps[self.active_step_index]
        return None

    def copy_saved(self, saved):
        return {
            k: None if v is None else {
                "key": v["key"],
                "params": None if v["params"] is None else v["params"].copy()
            }
            for k, v in saved.items()
        }

    def make_state_snapshot(self):
        return {
            "steps": [
                {
                    "type": s["type"],
                    "key": s["key"],
                    "params": None if s["params"] is None else s["params"].copy(),
                    "saved": self.copy_saved(
                        s.get("saved", {"inlet": None, "pillar": None})
                    )
                }
                for s in self.steps
            ],
            "operator_vars": [v.get() for v in self.operator_vars],
            "active_step_index": self.active_step_index,
            "dividers": self.dividers[:],
            "stream_num": self.stream_num_combo.get(),
        }

    def restore_state_snapshot(self, snapshot):
        self.steps = [
            {
                "type": s["type"],
                "key": s["key"],
                "params": None if s["params"] is None else s["params"].copy(),
                "saved": self.copy_saved(
                    s.get("saved", {"inlet": None, "pillar": None})
                )
            }
            for s in snapshot["steps"]
        ]

        for var, value in zip(self.operator_vars, snapshot["operator_vars"]):
            var.set(value)

        self.active_step_index = snapshot["active_step_index"]
        self.dividers = snapshot["dividers"][:]

        self.stream_num_combo.set(snapshot["stream_num"])

        self.upper_canvas_cache = [None for _ in range(self.num_operators)]

        self.refresh_editor_panel()

        self.refresh_radio_states()
        self.refresh_upper_canvases()
        self.refresh_preview_panel()
        self.render_cumulative_result()
        self.update_operator_count()
        self.update_total_inlet_number()

    def refresh_all(self):
        if not self.check_total_inlet_or_warn():
            return

        self.refresh_radio_states()
        self.refresh_upper_canvases()
        self.refresh_preview_panel()
        self.render_cumulative_result()
        self.update_operator_count()
        self.update_total_inlet_number()

    def refresh_radio_states(self):
        for i in range(self.num_operators):
            state = "normal" if self.is_step_enabled(i) else "disabled"
            rb_inlet, rb_pillar, btn_clear = self.radio_widgets[i]
            rb_inlet.config(state=state)
            rb_pillar.config(state=state)
            btn_clear.config(state=state)

            if state == "disabled":
                self.operator_vars[i].set(0)

    def refresh_upper_canvases(self):
        for i in range(self.num_operators):
            step = self.steps[i]
            canvas = self.upper_canvases[i]

            params_for_cache = None

            if step["params"] is not None:
                params_for_cache = step["params"].copy()

                # pin only affects flow computation, not inlet geometry preview
                if step["type"] == "inlet":
                    params_for_cache.pop("pin", None)

            cache_key = (
                step["type"],
                str(params_for_cache)
            )

            if self.upper_canvas_cache[i] == cache_key:
                continue

            self.upper_canvas_cache[i] = cache_key

            canvas.delete("all")

            if step["type"] == "none":
                canvas.create_text(90, 90, text=f"Operator {i+1}")
            elif step["type"] == "pillar":
                self.draw_pillar_geometry(canvas, step["params"])
            elif step["type"] == "inlet":
                self.draw_inlet_geometry(canvas, step["params"])

    def refresh_editor_panel(self):
        step = self.get_active_step()
        if step is None:
            return

        if step["type"] == "pillar" and step["params"]:
            self.is_loading_params = True
            try:
                self.restore_pillar_ui_from_params(step["params"])
            finally:
                self.is_loading_params = False

        elif step["type"] == "inlet" and step["params"]:
            self.restore_inlet_ui_from_params_with_values(step["params"])
            self.commit_inlet_sliders()

    def refresh_preview_panel(self):
        step = self.get_active_step()

        if step is None or step["type"] == "none":
            self.inlet_preview_title_var.set("Intermediate profile - No Data")
            self.pillar_preview_title_var.set("Intermediate profile - No Data")
            self.show_no_data_on_canvas(self.inlet_canvas)
            self.show_no_data_on_canvas(self.pillar_canvas)
            return

        if step["type"] == "inlet":
            self.inlet_preview_title_var.set(f"Intermediate profile - Step {self.active_step_index + 1}")
            self.pillar_preview_title_var.set("Intermediate profile - No Data")
            self.show_no_data_on_canvas(self.pillar_canvas)
            self.render_step_image(self.active_step_index, self.inlet_canvas)

        elif step["type"] == "pillar":
            self.pillar_preview_title_var.set(f"Intermediate profile - Step {self.active_step_index + 1}")
            self.inlet_preview_title_var.set("Intermediate profile - No Data")
            self.show_no_data_on_canvas(self.inlet_canvas)
            self.render_step_image(self.active_step_index, self.pillar_canvas)

    def refresh_for_step_selection(self):
        self.refresh_editor_panel()
        self.refresh_preview_panel()

    def select_step(self, index):
        if not self.is_step_enabled(index):
            return

        self.active_step_index = index
        self.refresh_for_step_selection()
        step = self.steps[index]
        if step["type"] == "inlet":
            self.commit_inlet_sliders()

    def change_step_type(self, index, new_type):
        snapshot = self.make_state_snapshot()

        old_step_type = self.steps[index]["type"]
        snapshot["operator_vars"][index] = self.get_choice_from_step_type(old_step_type)

        self.active_step_index = index

        # Keep existing parameters if the operator type does not change
        current_step = self.steps[index]
        if current_step["type"] == new_type and current_step["params"] is not None:
            self.refresh_for_step_selection()
            return
        
        saved = current_step.get("saved", {}).get(new_type)

        if saved and saved.get("params") is not None:
            params = saved["params"].copy()
            key = saved["key"]

            if new_type == "inlet":
                candidate_steps = [
                    {
                        "type": s["type"],
                        "key": s["key"],
                        "params": None if s["params"] is None else s["params"].copy()
                    }
                    for s in self.steps
                ]

                candidate_steps[index] = {
                    "type": "inlet",
                    "key": key,
                    "params": params
                }

                if self.get_total_inlet_number(candidate_steps) > 5:
                    self.operator_vars[index].set(snapshot["operator_vars"][index])
                    self.active_step_index = snapshot["active_step_index"]
                    self.refresh_editor_panel()
                    self.update_total_inlet_number()

                    messagebox.showwarning(
                        "Warning",
                        "no more than 5 inlets in total"
                    )
                    return

            self.set_step_state(index, new_type, key, params)

            self.operator_vars[index].set(self.get_choice_from_step_type(new_type))
            self.refresh_all()
            self.refresh_editor_panel()

            if new_type == "inlet":
                self.commit_inlet_sliders()

            return

        if new_type == "none":
            self.clear_step(index)
            return

        if new_type == "inlet":
            self.is_loading_params = True
            try:
                self.update_inlet_comboboxes()
            finally:
                self.is_loading_params = False

            key = self.get_current_inlet_key(index)
            params = self.get_current_inlet_params(index)

            if params is None:
                return

            current_total = self.get_total_inlet_number()
            add_num = int(float(params.get("inlet", 0)))

            if current_total + add_num > 5:
                self.operator_vars[index].set(snapshot["operator_vars"][index])
                self.active_step_index = snapshot["active_step_index"]

                self.is_loading_params = True
                try:
                    self.refresh_editor_panel()
                    self.update_total_inlet_number()
                finally:
                    self.is_loading_params = False

                messagebox.showwarning(
                    "Warning",
                    "no more than 5 inlets in total"
                )
                return

        elif new_type == "pillar":
            self.is_loading_params = True
            try:
                self.update_pillar_comboboxes()
            finally:
                self.is_loading_params = False

            key = self.get_current_pillar_key()
            params = self.get_current_pillar_params()
        else:
            return

        self.set_step_state(index, new_type, key, params)

        if not self.check_total_inlet_or_warn():
            self.restore_state_snapshot(snapshot)
            return

        self.operator_vars[index].set(self.get_choice_from_step_type(new_type))
        self.refresh_all()

    def clear_step(self, index):
        self.reset_steps_from(index)

        for i in range(index,self.num_operators):
            self.operator_vars[i].set(0)

        prev_valid = index - 1
        while prev_valid >= 0 and self.steps[prev_valid]["type"] == "none":
            prev_valid -= 1

        self.active_step_index = prev_valid
        self.refresh_all()

    def update_active_step_params(self):
        if self.active_step_index == -1:
            return
        
        snapshot = self.make_state_snapshot()

        step = self.steps[self.active_step_index]

        if step["type"] == "inlet":
            key = self.get_current_inlet_key(self.active_step_index)
            params = self.get_current_inlet_params(self.active_step_index)

            if params is None:
                return

            candidate_steps = [
                {
                    "type": s["type"],
                    "key": s["key"],
                    "params": None if s["params"] is None else s["params"].copy()
                }
                for s in self.steps
            ]

            candidate_steps[self.active_step_index] = {
                "type": "inlet",
                "key": key,
                "params": params
            }

            if self.get_total_inlet_number(candidate_steps) > 5:
                self.rollback_inlet_sliders()
                self.update_total_inlet_number()

                self.root.after_idle(
                    lambda: messagebox.showwarning(
                        "Warning",
                        "no more than 5 inlets in total"
                    )
                )
                return

            self.set_step_state(self.active_step_index, "inlet", key, params)

            if not self.check_total_inlet_or_warn():
                self.restore_state_snapshot(snapshot)
                return

        elif step["type"] == "pillar":
            key = self.get_current_pillar_key()
            params = self.get_current_pillar_params()

            if key is None or self.pillar_lib_data is None or key not in self.pillar_lib_data:
                messagebox.showwarning(
                    "Invalid pillar",
                    f"No pillar library entry:\n{key}"
                )
                return

            self.set_step_state(self.active_step_index, "pillar", key, params)

        self.refresh_all()
        if step["type"] == "inlet":
            self.commit_inlet_sliders() 
    
    def draw_inlet_geometry(self, canvas, params):
        canvas.delete("all")

        if not params:
            canvas.create_text(90, 90, text="No Geometry")
            return

        try:
            ow = float(params["ow"])
            inlet_num = int(params["inlet"])
            ip = float(params["ip"])
        except (KeyError, ValueError, TypeError):
            canvas.create_text(90, 90, text="Invalid Inlet")
            return

        if inlet_num <= 0 or ow <= 0:
            canvas.create_text(90, 90, text="No Inlet")
            return

        d = 0.75
        r = d / 2.0

        # canvas dimensions
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        scale = canvas_w/d * 0.18
        offset_x = canvas_w / 2.0
        offset_y = canvas_h / 2.0

        # channel dimensions and line
        line_y_bottom = 0
        line_y_top = ow
        line_x_left = -1
        line_x_right = 1

        circle_min_y = ip - r
        circle_max_y = ip + (inlet_num - 1) * d + r

        shape_bottom = min(line_y_bottom, circle_min_y)
        shape_top = max(line_y_top, circle_max_y)

        shape_left = min(line_x_left, -r)
        shape_right = max(line_x_right, r)

        center_x = (shape_left + shape_right) / 2.0
        center_y = (shape_bottom + shape_top) / 2.0
        
        def map_x(x):
            return offset_x + (x - center_x) * scale

        def map_y(y):
            return offset_y - (y - center_y) * scale

        canvas.create_line(
            map_x(line_x_left), map_y(line_y_bottom),
            map_x(line_x_right), map_y(line_y_bottom),
            fill="black", width=2
        )
        canvas.create_line(
            map_x(line_x_left), map_y(line_y_top),
            map_x(line_x_right), map_y(line_y_top),
            fill="black", width=2
        )

        cx = 0.0
        for j in range(inlet_num):
            cy = ip + j * d
            canvas.create_oval(
                map_x(cx - r), map_y(cy + r),
                map_x(cx + r), map_y(cy - r),
                fill="white", outline="black", width=2
            )

    def draw_pillar_geometry(self, canvas, params):
        # canvas dimensions
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        # read parameter
        cw = float(params["cw"])
        ya = float(params["ya"])
        yb = float(params["yb"])
        xc = float(params["xc"])
        yc = float(params["yc"])

        # pillar triangle points
        xa = 4.0
        xb = 4.0

        # channel line x range should cover pillar
        padding_real = 0.8
        line_x_left = min(xa, xb, xc) - padding_real
        line_x_right = max(xa, xb, xc) + padding_real

        line_y_bottom = 0
        line_y_top = cw

        # bounds include BOTH channel lines and pillar points
        shape_left = min(line_x_left, xa, xb, xc)
        shape_right = max(line_x_right, xa, xb, xc)
        shape_bottom = min(line_y_bottom, line_y_top, ya, yb, yc)
        shape_top = max(line_y_bottom, line_y_top, ya, yb, yc)

        real_w = max(shape_right - shape_left, 1e-6)
        real_h = max(shape_top - shape_bottom, 1e-6)

        margin = 15
        scale = min(
            (canvas_w - 2 * margin) / real_w,
            (canvas_h - 2 * margin) / real_h
        )

        center_x = (shape_left + shape_right) / 2.0
        center_y = (shape_bottom + shape_top) / 2.0

        offset_x = canvas_w / 2.0
        offset_y = canvas_h / 2.0

        def map_x(x):
            return offset_x + (x - center_x) * scale

        def map_y(y):
            return offset_y - (y - center_y) * scale

        # draw line 0 
        canvas.create_line(
            map_x(line_x_left), map_y(line_y_bottom),
            map_x(line_x_right), map_y(line_y_bottom),
            fill="black",
            width=2
            )

        # draw line cw 
        canvas.create_line(
            map_x(line_x_left), map_y(line_y_top),
            map_x(line_x_right),  map_y(line_y_top),
            fill="black",
            width=2
            )
        
        A = (map_x(xa), map_y(ya))
        B = (map_x(xb), map_y(yb))
        C = (map_x(xc), map_y(yc))

        # triangle
        canvas.create_polygon(
        A[0], A[1],
        B[0], B[1],
        C[0], C[1],
        fill="gray",
        outline="black",
        width=2
        )

        r = 3
        for label, (px, py) in [("A", A), ("B", B), ("C", C)]:
            canvas.create_oval(px-r, py-r, px+r, py+r, fill="red", outline="red")
            canvas.create_text(px + 8, py - 8, text=label, anchor="w")

    def get_intersection_channel_bounds(self):
        canvas_w = int(self.intersection_canvas.cget("width"))
        canvas_h = int(self.intersection_canvas.cget("height"))

        margin_y = 20

        return {
            "x_left": canvas_w * 0.25,
            "x_right": canvas_w * 0.75,
            "y_top": margin_y,
            "y_bottom": canvas_h - margin_y,
        }

    def get_channel_fit_size_from_mask(self, mask, scale=1.0):
        bounds = self.get_intersection_channel_bounds()
        channel_w = bounds["x_right"] - bounds["x_left"]

        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            return 1, 1, mask

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

        mask_h, mask_w = cropped_mask.shape

        fit_scale = channel_w / mask_w

        display_w = max(1, int(mask_w * fit_scale * scale))
        display_h = max(1, int(mask_h * fit_scale * scale))

        return display_w, display_h, cropped_mask

    def draw_photomask_on_intersection(self):

        if not hasattr(self, "photomask_mask_original"):
            return

        if not hasattr(self, "photomask_img"):
            return

        canvas_w = int(self.intersection_canvas.cget("width"))
        canvas_h = int(self.intersection_canvas.cget("height"))

        scale = self.intersection_scale_var.get()
        offset_x = self.intersection_x_var.get()
        offset_y = self.intersection_y_var.get()
        angle = self.intersection_rotation_var.get()

        display_w, display_h, cropped_mask = self.get_channel_fit_size_from_mask(
            self.photomask_mask_original,
            scale=scale
        )

        mask_arr = cropped_mask.astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_arr, mode="L")

        mask_img = mask_img.resize(
            (display_w, display_h),
            Image.NEAREST
        )

        resized_mask = np.array(mask_img) > 0

        rgba_arr = np.zeros(
            (display_h, display_w, 4),
            dtype=np.uint8
        )

        rgba_arr[resized_mask] = [0, 0, 0, 255]

        img = Image.fromarray(rgba_arr, mode="RGBA")

        if angle != 0:
            img = img.rotate(
                angle,
                resample=Image.BICUBIC,
                expand=True
            )

        img_tk = ImageTk.PhotoImage(img)

        x0 = canvas_w / 2 + offset_x
        y0 = canvas_h / 2 + offset_y

        self.intersection_canvas.create_image(
            x0,
            y0,
            anchor=tk.CENTER,
            image=img_tk
        )

        self.intersection_canvas.photomask_img_tk = img_tk

    def draw_channel_on_intersection(self):
        bounds = self.get_intersection_channel_bounds()

        self.intersection_canvas.create_line(
            bounds["x_left"],
            bounds["y_top"],
            bounds["x_left"],
            bounds["y_bottom"],
            fill="gray",
            width=3
        )

        self.intersection_canvas.create_line(
            bounds["x_right"],
            bounds["y_top"],
            bounds["x_right"],
            bounds["y_bottom"],
            fill="gray",
            width=3
        )

    def redraw_intersection(self):
        self.intersection_canvas.delete("all")

        if getattr(self, "channel_intersection_ready", False):
            self.draw_channel_on_intersection()

        if getattr(self, "photomask_intersection_ready", False):
            self.draw_photomask_on_intersection()

    def build_particle_facecolors(self, particle_volume, nx, ny, nz):
        flow_color_img = self.flow_profile_img.resize((nx, nz), Image.NEAREST)
        flow_color_2d = np.array(flow_color_img).astype(float) / 255.0

        flow_color_xz = np.fliplr(
            np.transpose(flow_color_2d, (1, 0, 2))
        )

        flow_color_volume = np.repeat(
            flow_color_xz[:, :, None, :],
            ny,
            axis=2
        )

        facecolors = np.zeros(
            particle_volume.shape + (4,),
            dtype=float
        )

        facecolors[particle_volume, :3] = flow_color_volume[particle_volume]
        facecolors[particle_volume, 3] = 1.0

        return facecolors
    
    def render_3d_particle_intersection(self):
        if not hasattr(self, "flow_mask"):
            return

        if not hasattr(self, "photomask_mask_original"):
            return

        canvas_w = int(self.intersection_canvas.cget("width"))
        canvas_h = int(self.intersection_canvas.cget("height"))

        particle_volume = self.Pengine.compute_particle_volume(
            flow_mask=self.flow_mask,
            photomask_mask=self.photomask_mask_original,
            canvas_size=(canvas_w, canvas_h),
            scale=self.intersection_scale_var.get(),
            offset_x=self.intersection_x_var.get(),
            offset_y=self.intersection_y_var.get(),
            rotation=self.intersection_rotation_var.get()
        )

        self.particle_volume = particle_volume

        nx = self.Pengine.nx
        ny = self.Pengine.ny
        nz = self.Pengine.nz

        if particle_volume is None or not np.any(particle_volume):
            self.particle_3d_canvas.delete("all")
            self.particle_3d_canvas.create_text(
                140,
                140,
                text="No 3D intersection",
                justify="center"
            )
            return

        # Render 3D voxel preview
        fig = plt.Figure(figsize=(1, 1), dpi=100)
        self.ax3d = fig.add_subplot(111, projection="3d")
        ax = self.ax3d
        ax.set_anchor("C")
        ax.set_position([0, 0, 1, 1])

        facecolors = self.build_particle_facecolors(
            particle_volume,
            nx,
            ny,
            nz
        )

        ax.voxels(
            particle_volume,
            facecolors=facecolors,
            edgecolor=None,
            linewidth=0
        )

        ax.view_init(
            elev=getattr(self, "particle_elev", 25),
            azim=getattr(self, "particle_azim", -60)
        )

        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])

        ax.set_box_aspect((nx, nx, nz))

        coords = np.argwhere(particle_volume)

        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        cz = (z_min + z_max) / 2

        size = max(
            x_max - x_min,
            y_max - y_min,
            z_max - z_min
        ) * 0.44

        ax.set_xlim(cx - size, cx + size)
        ax.set_ylim(cy - size, cy + size)
        ax.set_zlim(cz - size, cz + size)

        ax.dist = 3

        if hasattr(self, "particle_3d_figure_canvas"):
            self.particle_3d_figure_canvas.get_tk_widget().destroy()

        self.particle_3d_figure_canvas = FigureCanvasTkAgg(
            fig,
            master=self.particle_3d_canvas
        )
        self.particle_figure = fig

        self.particle_3d_figure_canvas.draw()

        canvas_w = self.particle_3d_canvas.winfo_width()
        canvas_h = self.particle_3d_canvas.winfo_height()

        self.particle_3d_figure_canvas.get_tk_widget().place(
            x=0,
            y=0,
            width=canvas_w,
            height=canvas_h
        )

    def set_3d_view(self, elev, azim):

        if not hasattr(self, "ax3d"):
            return

        self.ax3d.view_init(
            elev=elev,
            azim=azim
        )

        self.particle_3d_figure_canvas.draw_idle()


    def update_operator_count(self):
        count = 0
        for step in self.steps:
            if step["type"] != "none":
                count += 1
        self.operator_count_var.set(f"Total applied operators: {count}")
    
    def get_committed_stream_num(self):
        val = self.last_valid_stream_num
        if val == "-- Please choose --" or val == "":
            return 0
        return int(float(val))

    def get_pin_before_step(self, step_index):
        initial_num = self.get_committed_stream_num()

        pin = initial_num

        for i in range(step_index):
            step = self.steps[i]
            if step["type"] == "inlet":
                params = step["params"]
                if params and "inlet" in params:
                    try:
                        pin += int(params["inlet"])
                    except (ValueError, TypeError):
                        pass

        return float(pin)
    
    def get_total_inlet_number(self, steps=None):
        if steps is None:
            steps = self.steps

        try:
            initial_num = self.get_committed_stream_num()

        except (ValueError, AttributeError):
            initial_num = 0

        inlet_sum = 0
        for step in steps:
            if step["type"] == "none":
                break

            if step["type"] == "inlet":
                params = step["params"] or {}
                try:
                    inlet_sum += int(float(params.get("inlet", 0)))
                except (ValueError, TypeError):
                    pass

        return initial_num + inlet_sum


    def check_total_inlet_or_warn(self, steps=None):
        MAX_INLET = 5
        total = self.get_total_inlet_number(steps)

        if total > MAX_INLET:
            messagebox.showwarning(
                "Warning",
                "no more than 5 inlets in total"
            )
            return False

        return True
    
    def update_total_inlet_number(self):
        initial_num = self.get_committed_stream_num()

        inlet_sum = 0

        for step in self.steps:
            if step["type"] == "none":
                break

            if step["type"] == "inlet":
                params = step["params"] or {}
                try:
                    inlet_sum += int(float(params.get("inlet", 0)))
                except (ValueError, TypeError):
                    pass

        MAX_INLET = 5
        total = min(initial_num + inlet_sum, MAX_INLET)
        self.total_inlet_var.set(f"{total}/{MAX_INLET}")
    
    def show_no_data_on_canvas(self, canvas):
        canvas.delete("all")
        canvas.create_text(100, 100, text="No Data")

    def use_output_flow_profile(self):
        if not hasattr(self.Hsculpt, "outlet_color") or self.Hsculpt.outlet_color is None:
            messagebox.showwarning(
                "No output flow profile",
                "Please generate an output flow profile first."
            )
            return

        self.flow_profile_img = self.Hsculpt.outlet_color.copy().convert("RGB")
        self.flow_mask_state = {}

        arr = np.array(self.flow_profile_img)

        is_gray_background = (
            np.abs(arr[:, :, 0] - arr[:, :, 1]) < 5
        ) & (
            np.abs(arr[:, :, 1] - arr[:, :, 2]) < 5
        ) & (
            arr[:, :, 0] > 180
        )

        self.flow_valid_region = ~is_gray_background

        self.display_image_on_canvas(self.flow_profile_img, self.flow_profile_canvas)


    def enable_flow_segmentation(self):
        if not hasattr(self, "flow_profile_img"):
            messagebox.showwarning(
                "No flow profile",
                "Please click 'Use/renew output flow profile' first."
            )
            return

        self.flow_segmentation_mode = True
        self.flow_profile_canvas.bind("<Button-1>", self.on_flow_profile_click)

        messagebox.showinfo(
            "Segmentation enabled",
            "Click a region to switch curable/non-curable.\nBlack = curable."
        )


    def on_flow_profile_click(self, event):
        if not getattr(self, "flow_segmentation_mode", False):
            return

        canvas_w = int(self.flow_profile_canvas.cget("width"))
        canvas_h = int(self.flow_profile_canvas.cget("height"))

        img_w, img_h = self.flow_profile_img.size

        display_scale = min(
            canvas_w / img_w,
            canvas_h / img_h
        )

        display_w = img_w * display_scale
        display_h = img_h * display_scale

        offset_x = (canvas_w - display_w) / 2
        offset_y = (canvas_h - display_h) / 2

        if not (
            offset_x <= event.x <= offset_x + display_w and
            offset_y <= event.y <= offset_y + display_h
        ):
            return

        x = int((event.x - offset_x) / display_scale)
        y = int((event.y - offset_y) / display_scale)

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))

        if not self.flow_valid_region[y, x]:
            return

        arr = np.array(self.flow_profile_img)
        color = tuple(arr[y, x])

        self.flow_mask_state[color] = not self.flow_mask_state.get(color, False)

        self.render_flow_mask_preview()


    def render_flow_mask_preview(self):
        arr = np.array(self.flow_profile_img)

        preview = np.ones_like(arr) * 255

        preview[~self.flow_valid_region] = arr[~self.flow_valid_region]

        for color, selected in self.flow_mask_state.items():
            if selected:
                region = np.all(arr == np.array(color), axis=2)
                region = region & self.flow_valid_region
                preview[region] = [0, 0, 0]

        self.flow_mask_preview_img = Image.fromarray(preview.astype(np.uint8))
        self.display_image_on_canvas(self.flow_mask_preview_img, self.flow_profile_canvas)


    def finalize_flow_mask(self):
        if not hasattr(self, "flow_profile_img"):
            messagebox.showwarning(
                "No flow mask",
                "Please segment the flow profile first."
            )
            return

        arr = np.array(self.flow_profile_img)

        final_mask = np.zeros(arr.shape[:2], dtype=bool)

        for color, selected in self.flow_mask_state.items():
            if selected:
                region = np.all(arr == np.array(color), axis=2)
                final_mask[region] = True

        if not np.any(final_mask):
            messagebox.showwarning(
                "No flow mask",
                "Please select at least one curable region."
            )
            return

        self.flow_mask_original = final_mask
        self.flow_mask = self.flow_mask_original

        self.flow_segmentation_mode = False
        self.flow_profile_canvas.unbind("<Button-1>")
        self.channel_intersection_ready = True
        self.redraw_intersection()

        messagebox.showinfo(
            "Finalized",
            "Flow mask has been finalized."
        )

    def enable_photomask_segmentation(self):
        if not hasattr(self, "photomask_img"):
            messagebox.showwarning(
                "No photomask image",
                "Please upload a photomask image first."
            )
            return

        self.photomask_segmentation_mode = True
        self.photomask_canvas.bind("<Button-1>", self.on_photomask_click)

        messagebox.showinfo(
            "Segmentation enabled",
            "Click a region to switch exposed/non-exposed.\nBlack = exposed."
        )


    def on_photomask_click(self, event):
        if not getattr(self, "photomask_segmentation_mode", False):
            return

        canvas_w = int(self.photomask_canvas.cget("width"))
        canvas_h = int(self.photomask_canvas.cget("height"))

        img_w, img_h = self.photomask_img.size

        display_scale = min(
            canvas_w / img_w,
            canvas_h / img_h
        )

        display_w = img_w * display_scale
        display_h = img_h * display_scale

        offset_x = (canvas_w - display_w) / 2
        offset_y = (canvas_h - display_h) / 2

        if not (
            offset_x <= event.x <= offset_x + display_w and
            offset_y <= event.y <= offset_y + display_h
        ):
            return

        x = int((event.x - offset_x) / display_scale)
        y = int((event.y - offset_y) / display_scale)

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))

        arr = np.array(self.photomask_img)
        color = tuple(arr[y, x])

        self.photomask_mask_state[color] = not self.photomask_mask_state.get(color, False)

        self.render_photomask_mask_preview()


    def render_photomask_mask_preview(self):
        arr = np.array(self.photomask_img)

        preview = np.ones_like(arr) * 255

        for color, selected in self.photomask_mask_state.items():
            if selected:
                region = np.all(arr == np.array(color), axis=2)
                preview[region] = [0, 0, 0]

        self.photomask_mask_preview_img = Image.fromarray(preview.astype(np.uint8))
        self.display_image_on_canvas(self.photomask_mask_preview_img, self.photomask_canvas)


    def finalize_photomask_mask(self):
        if not hasattr(self, "photomask_mask_preview_img"):
            messagebox.showwarning(
                "No photomask mask",
                "Please segment the photomask image first."
            )
            return

        gray = np.array(self.photomask_mask_preview_img.convert("L"))

        self.photomask_mask_original = gray < 128

        ys, xs = np.where(self.photomask_mask_original)

        if len(xs) == 0 or len(ys) == 0:
            messagebox.showwarning(
                "No exposed region",
                "Please select at least one exposed photomask region."
            )
            return

        mask_width = xs.max() - xs.min() + 1
        mask_height = ys.max() - ys.min() + 1

        length_to_width_ratio = mask_height / mask_width

        if length_to_width_ratio > 2:
            messagebox.showwarning(
                "Photomask aspect ratio warning",
                f"The photomask length/width ratio is {length_to_width_ratio:.2f}, which is larger than 2.\n"
                "Please check whether the exposed region is too long."
            )
            return

        self.photomask_mask = self.photomask_mask_original

        self.photomask_segmentation_mode = False
        self.photomask_intersection_ready = True
        self.redraw_intersection()

        messagebox.showinfo(
            "Finalized",
            "Photomask mask has been finalized."
        )

    def display_image_on_canvas(self, img_display, canvas):
        canvas_width = int(canvas.cget("width"))
        canvas_height = int(canvas.cget("height"))

        img_w, img_h = img_display.size

        scale = min(
            canvas_width / img_w,
            canvas_height / img_h
        )

        display_w = max(1, int(img_w * scale))
        display_h = max(1, int(img_h * scale))

        img_resized = img_display.resize(
            (display_w, display_h),
            Image.NEAREST
        )

        img_tk = ImageTk.PhotoImage(img_resized)

        canvas.delete("all")

        x0 = (canvas_width - display_w) / 2
        y0 = (canvas_height - display_h) / 2

        canvas.create_image(
            x0,
            y0,
            anchor=tk.NW,
            image=img_tk
        )

        canvas.image = img_tk

    def select_outlet(self, key):
        """Handle outlet selection."""
        self.outlet_var.set(key)
        self.refresh_all()
    
    def init_variables(self):
        """Initialize all Tkinter tracking variables"""
        self.num_operators = 5
        self.operator_vars = [tk.IntVar(value=0) for _ in range(self.num_operators)]
        
        self.total_inlet_var = tk.StringVar(value="0/5")
        self.data_inlet_var = tk.StringVar(value="0")
        self.data_pillar_var = tk.StringVar(value="0")
        self.inlet_preview_title_var = tk.StringVar(value="Intermediate profile")
        self.pillar_preview_title_var = tk.StringVar(value="Intermediate profile")
        self.operator_count_var = tk.StringVar(value="Total applied operators: 0")
        self.outlet_var = tk.StringVar(value="square")
        
        self.radio_widgets = []

        self.inlet_comboboxes = {}
        self.pillar_comboboxes = {}
        self.upper_canvases = []
        self.upper_canvas_cache = [None for _ in range(self.num_operators)]
        self.steps = [
            {"type": "none", "key": None, "params": None}
            for _ in range(self.num_operators)
        ]

        self.is_loading_params = False  
        self.is_reverting_stream = False
        self.active_step_index = -1  
        self.dividers = [1.0]
        self.last_valid_stream_num = "-- Please choose --"

    def recompute_all_inlet_steps(self):
        for i, step in enumerate(self.steps):
            if step["type"] == "none":
                break

            if step["type"] == "inlet":
                params = step["params"]
                if not params:
                    continue

                params = params.copy()
                params["pin"] = self.get_pin_before_step(i)

                self.steps[i]["params"] = params
                self.steps[i]["key"] = self.make_inlet_key_from_params(params)

        self.update_total_inlet_number()

    def commit_inlet_sliders(self):
        for combo in self.inlet_comboboxes.values():
            combo.commit()

    def rollback_inlet_sliders(self):
        self.is_loading_params = True
        try:
            for combo in self.inlet_comboboxes.values():
                combo.rollback()
        finally:
            self.is_loading_params = False

if __name__ == "__main__":
    root = tk.Tk()
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    
    root.geometry("2800x1500")  
    app = FlowSimulationApp(root)
    root.mainloop()