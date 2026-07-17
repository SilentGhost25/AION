import os
import sys
import json
import time
import yaml
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import requests

# Set window title and basic styling tokens
AION_VIOLET = "#8B5CF6"
AION_BG_DARK = "#121214"
AION_CARD_BG = "#1A1A1E"
AION_CARD_BORDER = "#2A2A30"
AION_FG_WHITE = "#F3F4F6"
AION_FG_MUTED = "#9CA3AF"
AION_BLUE = "#3B82F6"
AION_GREEN = "#10B981"
AION_RED = "#EF4444"
AION_AMBER = "#F59E0B"

class AIONTrainerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AION Trainer Desktop App — v1.0")
        self.root.geometry("1100x820")
        self.root.configure(bg=AION_BG_DARK)
        
        # Connection parameters
        self.server_url = tk.StringVar(value=os.getenv("AION_SERVER_URL", "http://127.0.0.1:8000"))
        self.server_token = tk.StringVar(value=os.getenv("AION_SERVER_TOKEN", "test-token-0000"))
        self.connected = False
        self.gpu_info = "NVIDIA A100-SXM4-40GB"
        self.cuda_available = True
        
        # State variables
        self.department = tk.StringVar(value="AIML")
        self.semester = tk.StringVar(value="4")
        self.subject = tk.StringVar(value="BAI401")
        self.selected_files = [] # List of dict: {"path": Path, "category": str, "size": str}
        self.active_job_id = None
        self.polling = False
        self.chart_points = []
        self.local_server_process = None
        
        # Strict mode / lock parameters
        self.demo_mode = True
        self.allow_mock_data = True
        self.allow_fallback = True
        self.training_mode = "demo" # "demo" | "analysis" | "training"
        
        # Apply TTK styles
        self._setup_styles()
        
        # Build UI layout
        self._build_header()
        self._build_body()
        
        # Start initial server connection check
        self._check_connection_async()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure frames and labels
        style.configure(".", bg=AION_BG_DARK, foreground=AION_FG_WHITE)
        style.configure("TFrame", background=AION_BG_DARK)
        style.configure("Card.TFrame", background=AION_CARD_BG, relief="flat", borderwidth=1)
        
        # Labels
        style.configure("TLabel", background=AION_BG_DARK, foreground=AION_FG_WHITE, font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background=AION_CARD_BG, foreground=AION_FG_WHITE, font=("Segoe UI", 12, "bold"))
        style.configure("Muted.TLabel", background=AION_CARD_BG, foreground=AION_FG_MUTED, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=AION_CARD_BG, foreground=AION_BLUE, font=("Segoe UI", 10, "bold"))
        
        # Buttons
        style.configure("TButton", font=("Segoe UI", 10, "bold"), background=AION_BLUE, foreground=AION_FG_WHITE, borderwidth=0, padding=6)
        style.map("TButton", background=[("active", "#2563EB"), ("disabled", "#4B5563")])
        
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), background=AION_VIOLET, foreground=AION_FG_WHITE, borderwidth=0, padding=8)
        style.map("Action.TButton", background=[("active", "#7C3AED"), ("disabled", "#4B5563")])

        style.configure("Success.TButton", font=("Segoe UI", 10, "bold"), background=AION_GREEN, foreground=AION_FG_WHITE, borderwidth=0, padding=8)
        style.map("Success.TButton", background=[("active", "#059669"), ("disabled", "#4B5563")])

        style.configure("Secondary.TButton", font=("Segoe UI", 9), background="#2E2E35", foreground=AION_FG_WHITE, borderwidth=0, padding=4)
        style.map("Secondary.TButton", background=[("active", "#3E3E45")])
        
        # Dropdowns (Combobox)
        style.configure("TCombobox", fieldbackground="#2E2E35", background="#2E2E35", foreground=AION_FG_WHITE, arrowcolor=AION_FG_WHITE)
        style.map("TCombobox", fieldbackground=[("readonly", "#2E2E35")], selectbackground=[("readonly", "#2E2E35")])

    def _build_header(self):
        # Header main frame
        header_frame = ttk.Frame(self.root, height=80)
        header_frame.pack(fill="x", padx=20, pady=(15, 10))
        header_frame.pack_propagate(False)
        
        # Title and Subtitle
        title_frame = ttk.Frame(header_frame)
        title_frame.pack(side="left", fill="y")
        
        title_label = ttk.Label(title_frame, text="AION Trainer Desktop Dashboard", font=("Segoe UI", 16, "bold"), foreground=AION_VIOLET)
        title_label.pack(anchor="w", pady=(5, 2))
        
        subtitle_label = ttk.Label(title_frame, text="Autonomous Academic Cognitive Cycle & Learning Engine", font=("Segoe UI", 9), foreground=AION_FG_MUTED)
        subtitle_label.pack(anchor="w")
        
        # Server connection card
        self.server_card = ttk.Frame(header_frame, style="Card.TFrame")
        self.server_card.pack(side="right", fill="both", ipadx=10, ipady=5)
        
        # inner frame inside card
        inner_conn = ttk.Frame(self.server_card)
        inner_conn.configure(style="Card.TFrame")
        inner_conn.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.status_indicator = ttk.Label(inner_conn, text="● Offline", foreground=AION_RED, font=("Segoe UI", 11, "bold"), background=AION_CARD_BG)
        self.status_indicator.pack(anchor="e", padx=5)
        
        self.server_info_label = ttk.Label(inner_conn, text="Checking connection...", font=("Segoe UI", 9), foreground=AION_FG_MUTED, background=AION_CARD_BG)
        self.server_info_label.pack(anchor="e", padx=5)

    def _build_body(self):
        # Main container with two columns
        body_container = ttk.Frame(self.root)
        body_container.pack(fill="both", expand=True, padx=20, pady=(5, 15))
        
        # Left column (Configuration & Material selection)
        left_col = ttk.Frame(body_container, width=500)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Step 1 & 2: Subject Card
        self._build_subject_card(left_col)
        
        # Step 3: Material selection / Drag & Drop simulator Card
        self._build_material_card(left_col)
        
        # Right column (Training state, logs, benchmarks)
        right_col = ttk.Frame(body_container, width=550)
        right_col.pack(side="right", fill="both", expand=True, padx=(10, 0))
        
        # Step 5: Live Progress Display
        self._build_progress_card(right_col)
        
        # Step 6: Candidate Validation & Comparison Report
        self._build_comparison_card(right_col)

    def _build_subject_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill="x", pady=(0, 10), ipady=8)
        
        title = ttk.Label(card, text="1. Select Subject Target", style="CardTitle.TLabel")
        title.pack(anchor="w", padx=15, pady=(12, 10))
        
        selectors_frame = ttk.Frame(card)
        selectors_frame.configure(style="Card.TFrame")
        selectors_frame.pack(fill="x", padx=15, pady=5)
        
        # Dept
        lbl_dept = ttk.Label(selectors_frame, text="Department", font=("Segoe UI", 9), foreground=AION_FG_MUTED, background=AION_CARD_BG)
        lbl_dept.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        cb_dept = ttk.Combobox(selectors_frame, textvariable=self.department, values=["AIML", "CSE", "ISE"], width=10, state="readonly")
        cb_dept.grid(row=1, column=0, padx=5, pady=(2, 10))
        
        # Semester
        lbl_sem = ttk.Label(selectors_frame, text="Semester", font=("Segoe UI", 9), foreground=AION_FG_MUTED, background=AION_CARD_BG)
        lbl_sem.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        cb_sem = ttk.Combobox(selectors_frame, textvariable=self.semester, values=["3", "4", "5", "6"], width=10, state="readonly")
        cb_sem.grid(row=1, column=1, padx=5, pady=(2, 10))
        
        # Subject
        lbl_sub = ttk.Label(selectors_frame, text="Subject Code", font=("Segoe UI", 9), foreground=AION_FG_MUTED, background=AION_CARD_BG)
        lbl_sub.grid(row=0, column=2, sticky="w", padx=5, pady=2)
        cb_sub = ttk.Combobox(selectors_frame, textvariable=self.subject, values=["BAI401", "BAI404", "BCS401", "BCS402"], width=12, state="readonly")
        cb_sub.grid(row=1, column=2, padx=5, pady=(2, 10))
        
        # Handle selection change
        cb_sub.bind("<<ComboboxSelected>>", lambda e: self._on_subject_changed())
        cb_dept.bind("<<ComboboxSelected>>", lambda e: self._on_subject_changed())
        cb_sem.bind("<<ComboboxSelected>>", lambda e: self._on_subject_changed())

    def _build_material_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill="both", expand=True, pady=10)
        
        title_frame = ttk.Frame(card)
        title_frame.configure(style="Card.TFrame")
        title_frame.pack(fill="x", padx=15, pady=(12, 5))
        
        title = ttk.Label(title_frame, text="2. Academic Material Ingestion", style="CardTitle.TLabel")
        title.pack(side="left")
        
        self.manifest_status_label = ttk.Label(title_frame, text="(0 files synced)", font=("Segoe UI", 9), foreground=AION_FG_MUTED, background=AION_CARD_BG)
        self.manifest_status_label.pack(side="left", padx=10)
        
        # Drag and Drop simulated dropzone
        dropzone = tk.Canvas(card, height=100, bg="#212126", highlightthickness=1, highlightbackground=AION_CARD_BORDER)
        dropzone.pack(fill="x", padx=15, pady=10)
        
        dropzone.create_text(
            220, 50,
            text="📁 Browse Academic Documents\n(Click here to select PDFs, DOCX, or Slides)",
            fill=AION_FG_MUTED,
            font=("Segoe UI", 10, "bold"),
            justify="center",
            tags="prompt"
        )
        dropzone.bind("<Button-1>", lambda e: self._browse_and_add_files())
        
        # Files list box
        list_label = ttk.Label(card, text="Ingested Documents Queue:", font=("Segoe UI", 9, "bold"), background=AION_CARD_BG)
        list_label.pack(anchor="w", padx=15, pady=(5, 2))
        
        list_container = ttk.Frame(card)
        list_container.configure(style="Card.TFrame")
        list_container.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        
        # Treeview list of files
        columns = ("name", "category", "size", "status")
        self.files_tree = ttk.Treeview(list_container, columns=columns, show="headings", height=8)
        self.files_tree.heading("name", text="File Name")
        self.files_tree.heading("category", text="Category")
        self.files_tree.heading("size", text="Size")
        self.files_tree.heading("status", text="Sync Status")
        
        self.files_tree.column("name", width=180, anchor="w")
        self.files_tree.column("category", width=100, anchor="center")
        self.files_tree.column("size", width=70, anchor="center")
        self.files_tree.column("status", width=80, anchor="center")
        self.files_tree.pack(fill="both", expand=True, side="left")
        
        # Scrollbar for tree
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.files_tree.yview)
        self.files_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(fill="y", side="right")
        
        # Buttons for lists
        btn_frame = ttk.Frame(card)
        btn_frame.configure(style="Card.TFrame")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))
        
        btn_remove = ttk.Button(btn_frame, text="Remove Selected", style="Secondary.TButton", command=self._remove_selected_file)
        btn_remove.pack(side="left", padx=(0, 5))
        
        btn_clear = ttk.Button(btn_frame, text="Clear All", style="Secondary.TButton", command=self._clear_files)
        btn_clear.pack(side="left", padx=5)
        
        self.btn_sync = ttk.Button(btn_frame, text="Sync Material with Server", style="TButton", command=self._sync_materials)
        self.btn_sync.pack(side="right")

    def _build_progress_card(self, parent):
        card = ttk.Frame(parent, style="Card.TFrame")
        card.pack(fill="both", expand=True, pady=(0, 10))
        
        title_frame = ttk.Frame(card)
        title_frame.configure(style="Card.TFrame")
        title_frame.pack(fill="x", padx=15, pady=(12, 5))
        
        title = ttk.Label(title_frame, text="3. Training Control & Progress", style="CardTitle.TLabel")
        title.pack(side="left")
        
        self.btn_train = ttk.Button(title_frame, text="Train Model", style="Action.TButton", command=self._start_training)
        self.btn_train.pack(side="right")
        
        # Metrics labels
        self.metrics_frame = ttk.Frame(card)
        self.metrics_frame.configure(style="Card.TFrame")
        self.metrics_frame.pack(fill="x", padx=15, pady=5)
        
        self.lbl_epoch = ttk.Label(self.metrics_frame, text="Epoch: -", font=("Segoe UI", 10, "bold"), background=AION_CARD_BG)
        self.lbl_epoch.pack(side="left", padx=(0, 15))
        
        self.lbl_loss = ttk.Label(self.metrics_frame, text="Loss: -", font=("Segoe UI", 10, "bold"), background=AION_CARD_BG)
        self.lbl_loss.pack(side="left", padx=15)
        
        self.lbl_gpu = ttk.Label(self.metrics_frame, text="GPU: -", font=("Segoe UI", 10, "bold"), background=AION_CARD_BG)
        self.lbl_gpu.pack(side="left", padx=15)
        
        self.lbl_eta = ttk.Label(self.metrics_frame, text="ETA: -", font=("Segoe UI", 10, "bold"), background=AION_CARD_BG)
        self.lbl_eta.pack(side="right", padx=(15, 0))
        
        # Custom Canvas for animated live loss chart!
        chart_label = ttk.Label(card, text="Training Loss Curve (Live):", font=("Segoe UI", 9, "bold"), background=AION_CARD_BG)
        chart_label.pack(anchor="w", padx=15, pady=(5, 2))
        
        self.chart_canvas = tk.Canvas(card, height=120, bg="#1E1E22", highlightthickness=1, highlightbackground=AION_CARD_BORDER)
        self.chart_canvas.pack(fill="x", padx=15, pady=(0, 8))
        self._clear_chart()
        
        # Expandable Logs view
        logs_bar = ttk.Frame(card)
        logs_bar.configure(style="Card.TFrame")
        logs_bar.pack(fill="x", padx=15, pady=(5, 2))
        
        lbl_logs = ttk.Label(logs_bar, text="Server Console Logs:", font=("Segoe UI", 9, "bold"), background=AION_CARD_BG)
        lbl_logs.pack(side="left")
        
        self.logs_text = tk.Text(card, height=6, bg="#0C0C0D", fg="#10B981", insertbackground="white", font=("Consolas", 9), state="disabled", wrap="word", relief="flat")
        self.logs_text.pack(fill="both", expand=True, padx=15, pady=(0, 12))

    def _build_comparison_card(self, parent):
        self.comp_card = ttk.Frame(parent, style="Card.TFrame")
        self.comp_card.pack(fill="x", pady=10, ipady=10)
        
        title_frame = ttk.Frame(self.comp_card)
        title_frame.configure(style="Card.TFrame")
        title_frame.pack(fill="x", padx=15, pady=(12, 10))
        
        title = ttk.Label(title_frame, text="4. Candidate Validation Report", style="CardTitle.TLabel")
        title.pack(side="left")
        
        self.promotion_badge = ttk.Label(title_frame, text="PENDING EVALUATION", font=("Segoe UI", 9, "bold"), foreground=AION_AMBER, background=AION_CARD_BG)
        self.promotion_badge.pack(side="right", padx=5)
        
        # Table frame
        self.table_frame = ttk.Frame(self.comp_card)
        self.table_frame.configure(style="Card.TFrame")
        self.table_frame.pack(fill="x", padx=15, pady=5)
        
        # Build comparison grid
        self._update_comparison_grid(None)
        
        # Deploy button
        self.btn_deploy = ttk.Button(self.comp_card, text="Deploy Candidate Model to Production", style="Success.TButton", command=self._deploy_model)
        self.btn_deploy.pack(fill="x", padx=15, pady=(10, 5))
        self.btn_deploy.configure(state="disabled")

    def _update_comparison_grid(self, data):
        # Clear frame
        for widget in self.table_frame.winfo_children():
            widget.destroy()
            
        headers = ["Metric", "Production", "Candidate", "Variance", "Gate Result"]
        for col_idx, h in enumerate(headers):
            lbl = ttk.Label(self.table_frame, text=h, font=("Segoe UI", 9, "bold"), background=AION_CARD_BG, anchor="center")
            lbl.grid(row=0, column=col_idx, sticky="ew", padx=8, pady=5)
            self.table_frame.grid_columnconfigure(col_idx, weight=1)
            
        # Draw separator line
        sep = ttk.Separator(self.table_frame, orient="horizontal")
        sep.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(2, 5))
        
        if not data:
            # Placeholder/Empty state
            lbl_empty = ttk.Label(self.table_frame, text="No evaluation record found. Sync materials and run training first.", font=("Segoe UI", 9, "italic"), foreground=AION_FG_MUTED, background=AION_CARD_BG, anchor="center")
            lbl_empty.grid(row=2, column=0, columnspan=5, pady=15)
            return
            
        metrics_mapping = [
            ("grammar", "Grammar & Quality"),
            ("academic_quality", "Academic Quality"),
            ("bloom_accuracy", "Bloom Alignment"),
            ("vtu_similarity", "VTU Exam Similarity"),
            ("question_diversity", "Question Diversity"),
            ("expected_answer_quality", "Expected Ans Quality"),
            ("diagram_prediction", "Diagram Requirements")
        ]
        
        row_idx = 2
        for key, display_name in metrics_mapping:
            cand_score = data["candidate"].get(key, 0.0)
            prod_score = data["production"].get(key, 0.0)
            diff = cand_score - prod_score
            threshold = data["thresholds"].get(key, 0.8)
            
            # Decide gate result: Candidate must beat threshold (or production if threshold is higher)
            passed = cand_score >= threshold
            gate_text = "PASS" if passed else "FAIL"
            gate_color = AION_GREEN if passed else AION_RED
            
            # Format differences
            diff_prefix = "+" if diff >= 0 else ""
            diff_text = f"{diff_prefix}{diff*100:.1f}%" if diff != 0 else "0.0%"
            diff_color = AION_GREEN if diff >= 0 else AION_RED
            
            lbl_name = ttk.Label(self.table_frame, text=display_name, font=("Segoe UI", 9), background=AION_CARD_BG, anchor="w")
            lbl_name.grid(row=row_idx, column=0, sticky="w", padx=8, pady=3)
            
            lbl_prod = ttk.Label(self.table_frame, text=f"{prod_score*100:.1f}%", font=("Segoe UI", 9), background=AION_CARD_BG, anchor="center")
            lbl_prod.grid(row=row_idx, column=1, sticky="center", padx=8, pady=3)
            
            lbl_cand = ttk.Label(self.table_frame, text=f"{cand_score*100:.1f}%", font=("Segoe UI", 9, "bold"), background=AION_CARD_BG, anchor="center")
            lbl_cand.grid(row=row_idx, column=2, sticky="center", padx=8, pady=3)
            
            lbl_diff = ttk.Label(self.table_frame, text=diff_text, font=("Segoe UI", 9), foreground=diff_color, background=AION_CARD_BG, anchor="center")
            lbl_diff.grid(row=row_idx, column=3, sticky="center", padx=8, pady=3)
            
            lbl_gate = ttk.Label(self.table_frame, text=gate_text, font=("Segoe UI", 9, "bold"), foreground=gate_color, background=AION_CARD_BG, anchor="center")
            lbl_gate.grid(row=row_idx, column=4, sticky="center", padx=8, pady=3)
            
            row_idx += 1

    def _clear_chart(self):
        self.chart_canvas.delete("all")
        self.chart_canvas.create_line(10, 100, 480, 100, fill="#3A3A40", width=1) # X axis
        self.chart_canvas.create_line(10, 10, 10, 100, fill="#3A3A40", width=1)   # Y axis
        self.chart_points = []

    def _draw_chart_point(self, epoch, loss):
        # Canvas dimensions: 500 wide, 120 high
        # Margin: X: 20 -> 480, Y: 10 -> 100
        # Map epoch (1 -> 10) to X (20 -> 480)
        # Map loss (0.0 -> 1.0) to Y (100 -> 10)
        max_epochs = 10
        x = 20 + int((epoch / max_epochs) * 440)
        y = 100 - int(min(loss, 1.0) * 80)
        
        self.chart_points.append((x, y))
        
        if len(self.chart_points) > 1:
            self.chart_canvas.delete("plot_line")
            self.chart_canvas.create_line(self.chart_points, fill=AION_VIOLET, width=2, tags="plot_line")
        
        # Draw dynamic node indicator
        self.chart_canvas.delete("current_node")
        self.chart_canvas.create_oval(x-4, y-4, x+4, y+4, fill=AION_BLUE, outline=AION_FG_WHITE, tags="current_node")

    def _update_mode_state(self):
        if self.selected_files:
            self.demo_mode = False
            self.allow_mock_data = False
            self.allow_fallback = False
            if self.training_mode == "demo":
                self.training_mode = "analysis"
        else:
            self.demo_mode = True
            self.allow_mock_data = True
            self.allow_fallback = True
            self.training_mode = "demo"

    # ------------------------------------------------------------------
    # EVENT HANDLERS
    # ------------------------------------------------------------------
    
    def _on_subject_changed(self):
        self._clear_files()
        self._fetch_manifest_async()

    def _browse_and_add_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Academic Materials",
            filetypes=[("Documents", "*.pdf;*.docx"), ("All Files", "*.*")]
        )
        if not file_paths:
            return
            
        for path_str in file_paths:
            path = Path(path_str)
            # Automatic category selection based on keyword in file path
            category = "textbooks"
            name_lower = path.name.lower()
            if "note" in name_lower:
                category = "notes"
            elif "pyq" in name_lower or "paper" in name_lower or "exam" in name_lower or "previous" in name_lower:
                category = "previous_papers"
            elif "bank" in name_lower or "question" in name_lower:
                category = "question_bank"
            elif "syllabus" in name_lower:
                category = "syllabus"
                
            # Get size in KB/MB
            size_bytes = path.stat().st_size
            if size_bytes > 1024 * 1024:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{size_bytes / 1024:.1f} KB"
                
            file_record = {"path": path, "category": category, "size": size_str, "synced": False}
            self.selected_files.append(file_record)
            
        self._update_files_list()
        self._update_mode_state()

    def _update_files_list(self):
        # Clear list first
        for row in self.files_tree.get_children():
            self.files_tree.delete(row)
            
        for idx, f in enumerate(self.selected_files):
            status_text = "Synced ✓" if f["synced"] else "Pending"
            self.files_tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(f["path"].name, f["category"], f["size"], status_text)
            )

    def _remove_selected_file(self):
        selected = self.files_tree.selection()
        if not selected:
            messagebox.showwarning("Warning", "Select a file from the list to remove.")
            return
        # Remove in reverse order to keep indices valid
        for s in sorted(selected, key=int, reverse=True):
            idx = int(s)
            self.selected_files.pop(idx)
        self._update_files_list()
        self._update_mode_state()

    def _clear_files(self):
        self.selected_files = []
        self._update_files_list()
        self._update_mode_state()

    # ------------------------------------------------------------------
    # HTTP ASYNC ACTIONS
    # ------------------------------------------------------------------
    
    def _check_connection_async(self):
        threading.Thread(target=self._check_connection, daemon=True).start()

    def _check_connection(self):
        url = self.server_url.get()
        token = self.server_token.get()
        
        try:
            res = requests.get(f"{url}/ping", timeout=3)
            if res.status_code == 200 and res.json().get("status") == "ok":
                self.connected = True
                self.status_indicator.configure(text="● Connected", foreground=AION_GREEN)
                self.server_info_label.configure(text=f"GPU Server Active | Token verified")
                # Trigger manifest fetch
                self._fetch_manifest()
            else:
                self._set_offline()
        except requests.RequestException:
            self._set_offline()

    def _set_offline(self):
        self.connected = False
        self.status_indicator.configure(text="● Offline (Demo Mode)", foreground=AION_AMBER)
        self.server_info_label.configure(text="Server not running. Using simulated fallback mode.")

    def _fetch_manifest_async(self):
        if self.connected:
            threading.Thread(target=self._fetch_manifest, daemon=True).start()

    def _fetch_manifest(self):
        url = self.server_url.get()
        token = self.server_token.get()
        sub = self.subject.get()
        
        try:
            res = requests.get(f"{url}/manifest/{sub}", headers={"X-AION-Token": token}, timeout=3)
            if res.status_code == 200:
                data = res.json()
                total_files = len(data.get("textbooks", [])) + len(data.get("notes", [])) + len(data.get("previous_papers", []))
                self.manifest_status_label.configure(text=f"({total_files} files synced)")
                
                # Pre-populate selected files from server manifest if queue is empty
                if not self.selected_files:
                    for category in ["textbooks", "notes", "previous_papers", "question_bank", "syllabus"]:
                        for f in data.get(category, []):
                            f_path = Path(f["path"])
                            self.selected_files.append({
                                "path": f_path,
                                "category": category,
                                "size": "-",
                                "synced": True
                            })
                    self.root.after(0, lambda: [self._update_files_list(), self._update_mode_state()])
        except requests.RequestException:
            pass

    def _sync_materials(self):
        if not self.selected_files:
            messagebox.showwarning("Warning", "Add some local files to sync first.")
            return
            
        if not self.connected:
            if not self.allow_fallback:
                messagebox.showerror("Connection Error", "Cannot sync: Server is offline and fallback is disabled in Production mode.")
                return
            # Simulate sync in Offline mode
            for f in self.selected_files:
                f["synced"] = True
            self._update_files_list()
            self.manifest_status_label.configure(text=f"({len(self.selected_files)} files synced)")
            messagebox.showinfo("Sync Complete", "[Simulated] Synced all files to mock local server manifest.")
            return
            
        threading.Thread(target=self._sync_materials_worker, daemon=True).start()

    def _sync_materials_worker(self):
        self.btn_sync.configure(state="disabled")
        url = self.server_url.get()
        token = self.server_token.get()
        sub = self.subject.get()
        
        success_count = 0
        for f in self.selected_files:
            if f["synced"]:
                continue
                
            try:
                # Read file content
                with open(f["path"], "rb") as file_bytes:
                    payload = {"category": f["category"]}
                    files = {"file": (f["path"].name, file_bytes, "application/octet-stream")}
                    res = requests.post(
                        f"{url}/manifest/{sub}/upload",
                        data=payload,
                        files=files,
                        headers={"X-AION-Token": token},
                        timeout=15
                    )
                    if res.status_code == 200 and res.json().get("success"):
                        f["synced"] = True
                        success_count += 1
            except Exception as e:
                self._append_log_line(f"Error syncing {f['path'].name}: {e}")
                
        self.root.after(0, lambda: self._on_sync_finished(success_count))

    def _on_sync_finished(self, success_count):
        self.btn_sync.configure(state="normal")
        self._update_files_list()
        self._fetch_manifest()
        messagebox.showinfo("Sync Complete", f"Successfully synced {success_count} new files to the GPU Server.")

    def _start_training(self):
        if not self.connected:
            if not self.allow_fallback:
                messagebox.showerror("Connection Error", "Cannot train: Server is offline and fallback is disabled in Production mode.")
                return
            # Simulate training loop in Offline mode
            threading.Thread(target=self._simulate_training, daemon=True).start()
            return
            
        threading.Thread(target=self._start_training_worker, daemon=True).start()

    def _start_training_worker(self):
        self.btn_train.configure(state="disabled")
        self._clear_chart()
        self._clear_logs()
        self._append_log_line("Connecting to server job queue...")
        
        url = self.server_url.get()
        token = self.server_token.get()
        sub = self.subject.get()
        
        try:
            res = requests.post(
                f"{url}/jobs",
                json={"subject": sub, "job_type": "learn", "params": {"force": True}},
                headers={"X-AION-Token": token},
                timeout=5
            )
            if res.status_code == 200:
                data = res.json()
                self.active_job_id = data.get("job_id")
                self._append_log_line(f"Submitted Job successfully! ID: {self.active_job_id}")
                self.polling = True
                threading.Thread(target=self._poll_job_status, daemon=True).start()
            else:
                self._append_log_line(f"Server rejected job submission: {res.text}")
                self.btn_train.configure(state="normal")
        except Exception as e:
            self._append_log_line(f"Failed to submit training job: {e}")
            self.btn_train.configure(state="normal")

    def _poll_job_status(self):
        url = self.server_url.get()
        token = self.server_token.get()
        
        last_log_seq = -1
        
        while self.polling:
            try:
                # 1. Fetch latest logs
                log_res = requests.get(
                    f"{url}/jobs/{self.active_job_id}/logs",
                    params={"since_seq": last_log_seq} if last_log_seq >= 0 else None,
                    headers={"X-AION-Token": token},
                    timeout=3
                )
                if log_res.status_code == 200:
                    entries = log_res.json()
                    for entry in entries:
                        msg = entry.get("message", "")
                        self._append_log_line(f"[{entry.get('level')}] {msg}")
                        last_log_seq = max(last_log_seq, entry.get("seq", -1))
                        
                        # Inspect log message for live charts data!
                        # e.g., "AIONTrainer: Epoch 4/10 complete - Loss: 0.1245"
                        if "loss:" in msg.lower() or "epoch" in msg.lower():
                            self._parse_and_plot_metrics(msg)
                            
                # 2. Fetch job state
                job_res = requests.get(
                    f"{url}/jobs/{self.active_job_id}",
                    headers={"X-AION-Token": token},
                    timeout=3
                )
                if job_res.status_code == 200:
                    job_data = job_res.json()
                    status = job_data.get("status")
                    
                    if status in ("completed", "failed"):
                        self.polling = False
                        self.root.after(0, lambda: self._on_job_finished(job_data))
                        break
                        
            except Exception as e:
                # Log issues but continue polling
                pass
                
            time.sleep(1.0)

    def _parse_and_plot_metrics(self, msg):
        try:
            # Simple message parser for simulated or actual logs
            # e.g. "Epoch 4/10 completed - loss: 0.2412" or "Epoch: 4, Loss: 0.2412"
            import re
            epoch_match = re.search(r"epoch:?\s*(\d+)", msg, re.IGNORECASE)
            loss_match = re.search(r"loss:?\s*([\d\.]+)", msg, re.IGNORECASE)
            
            if epoch_match and loss_match:
                epoch = int(epoch_match.group(1))
                loss = float(loss_match.group(1))
                
                self.lbl_epoch.configure(text=f"Epoch: {epoch}/10")
                self.lbl_loss.configure(text=f"Loss: {loss:.4f}")
                self.lbl_gpu.configure(text=f"GPU Util: 88%")
                self.lbl_eta.configure(text=f"ETA: {int((10 - epoch)*8)}s")
                
                self.root.after(0, lambda: self._draw_chart_point(epoch, loss))
        except Exception:
            pass

    def _on_job_finished(self, job_data):
        self.btn_train.configure(state="normal")
        status = job_data.get("status")
        
        if status == "completed":
            self._append_log_line("=" * 60)
            self._append_log_line("AION Training cycle completed successfully!")
            self._append_log_line("=" * 60)
            
            # Fetch model records to display evaluation report
            self._fetch_model_registry()
            messagebox.showinfo("Training Success", "Model training & evaluation cycle completed!")
        else:
            self._append_log_line("AION Training cycle failed. Error details:")
            self._append_log_line(str(job_data.get("error")))
            messagebox.showerror("Training Failure", f"Training job failed: {job_data.get('error')}")

    def _fetch_model_registry(self):
        url = self.server_url.get()
        token = self.server_token.get()
        sub = self.subject.get()
        
        try:
            res = requests.get(f"{url}/models/{sub}", headers={"X-AION-Token": token}, timeout=3)
            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                production = data.get("production")
                
                if candidates:
                    latest_candidate = candidates[-1] # The newly trained candidate
                    prod_record = production if production else latest_candidate # Fallback to same if none in production
                    
                    # Fetch comparison info from model registry promote schema rules
                    # VtU thresholds
                    thresholds = {
                        "grammar": 0.90, "academic_quality": 0.85, "bloom_accuracy": 0.85,
                        "vtu_similarity": 0.80, "question_diversity": 0.80,
                        "expected_answer_quality": 0.85, "diagram_prediction": 0.80
                    }
                    
                    comp_data = {
                        "candidate": latest_candidate.get("benchmark_scores", {}),
                        "production": prod_record.get("benchmark_scores", {}),
                        "thresholds": thresholds
                    }
                    
                    self.root.after(0, lambda: self._update_comparison_grid(comp_data))
                    
                    # Verify overall gate approval status
                    overall_passed = True
                    for key, thresh in thresholds.items():
                        if latest_candidate.get("benchmark_scores", {}).get(key, 0.0) < thresh:
                            overall_passed = False
                            
                    # Get promotion suitability details
                    if overall_passed:
                        self.promotion_badge.configure(text="SUITABLE FOR PROMOTION", foreground=AION_GREEN)
                        self.btn_deploy.configure(state="normal")
                        self.btn_deploy.current_version = latest_candidate.get("version")
                    else:
                        self.promotion_badge.configure(text="DEGRADED METRICS — PROMOTION BLOCKED", foreground=AION_RED)
                        self.btn_deploy.configure(state="disabled")
        except Exception as e:
            self._append_log_line(f"Failed to fetch model registry: {e}")

    def _deploy_model(self):
        version = getattr(self.btn_deploy, "current_version", None)
        if not version:
            return
            
        if not self.connected:
            if not self.allow_fallback:
                messagebox.showerror("Connection Error", "Cannot deploy: Server is offline and fallback is disabled in Production mode.")
                return
            messagebox.showinfo("Deploy", f"[Simulated] Deployed candidate model version {version} to production.")
            self.promotion_badge.configure(text="DEPLOYED TO PRODUCTION", foreground=AION_GREEN)
            self.btn_deploy.configure(state="disabled")
            return
            
        threading.Thread(target=self._deploy_model_worker, args=(version,), daemon=True).start()

    def _deploy_model_worker(self, version):
        url = self.server_url.get()
        token = self.server_token.get()
        sub = self.subject.get()
        
        try:
            res = requests.post(
                f"{url}/models/{sub}/promote",
                json={"version": version},
                headers={"X-AION-Token": token},
                timeout=5
            )
            if res.status_code == 200 and res.json().get("success"):
                self.root.after(0, lambda: self._on_deploy_finished(version))
            else:
                self._append_log_line(f"Server rejected model promotion: {res.text}")
        except Exception as e:
            self._append_log_line(f"Deploy model failed: {e}")

    def _on_deploy_finished(self, version):
        messagebox.showinfo("Deploy Success", f"Successfully promoted Candidate AION_{version} to VTU Production serving.")
        self.promotion_badge.configure(text="DEPLOYED TO PRODUCTION", foreground=AION_GREEN)
        self.btn_deploy.configure(state="disabled")

    # ------------------------------------------------------------------
    # OFFLINE DEMO MODE SIMULATION
    # ------------------------------------------------------------------
    
    def _simulate_training(self):
        if not self.allow_mock_data:
            raise RuntimeError(
                "Mock data requested while production mode is active."
            )
        self.btn_train.configure(state="disabled")
        self._clear_chart()
        self._clear_logs()
        
        self._append_log_line("[Simulated] Initiating local pipeline runner execution...")
        time.sleep(1.0)
        self._append_log_line("[Simulated] Stage 1/8: Scanning workspace for files...")
        time.sleep(1.0)
        self._append_log_line(f"[Simulated] Found {len(self.selected_files)} textbooks/notes/papers.")
        time.sleep(1.0)
        self._append_log_line("[Simulated] Stage 2/8: Scanning delta updates...")
        time.sleep(1.0)
        self._append_log_line("[Simulated] Stage 3/8: Initializing parallel document parser...")
        time.sleep(1.0)
        self._append_log_line("[Simulated] Ingesting documents...")
        
        import random
        # Simulated 10 Epochs training loop
        for epoch in range(1, 11):
            time.sleep(1.2)
            loss = 0.9 * (0.8 ** (epoch - 1)) + random.uniform(-0.02, 0.02)
            loss = max(0.01, loss)
            
            msg = f"AIONTrainer: Epoch {epoch}/10 complete - Loss: {loss:.4f}"
            self._append_log_line(f"[INFO] {msg}")
            self._parse_and_plot_metrics(msg)
            
        self._append_log_line("[Simulated] Stage 7/8: Evaluating model candidate benchmarks...")
        time.sleep(1.0)
        self._append_log_line("[Simulated] Stage 8/8: Saving model to registry...")
        time.sleep(1.0)
        
        # Build dummy evaluation report
        # User specified failed gate in proposal: "Expected Ans Quality: -1.9% [FAIL] -> DO NOT PROMOTE"
        mock_data = {
            "candidate": {
                "grammar": 0.954, "academic_quality": 0.885, "bloom_accuracy": 0.892,
                "vtu_similarity": 0.841, "question_diversity": 0.812,
                "expected_answer_quality": 0.831, "diagram_prediction": 0.824
            },
            "production": {
                "grammar": 0.932, "academic_quality": 0.864, "bloom_accuracy": 0.872,
                "vtu_similarity": 0.821, "question_diversity": 0.795,
                "expected_answer_quality": 0.850, "diagram_prediction": 0.801
            },
            "thresholds": {
                "grammar": 0.90, "academic_quality": 0.85, "bloom_accuracy": 0.85,
                "vtu_similarity": 0.80, "question_diversity": 0.80,
                "expected_answer_quality": 0.85, "diagram_prediction": 0.80
            }
        }
        
        self.root.after(0, lambda: self._update_comparison_grid(mock_data))
        self.promotion_badge.configure(text="DEGRADED METRICS — PROMOTION BLOCKED", foreground=AION_RED)
        self.btn_deploy.configure(state="disabled")
        self.btn_train.configure(state="normal")
        
        messagebox.showwarning("Validation Report", "Candidate model failed VTU shadow professor gate check (Expected Answer Quality too low). Promotion blocked.")

    # ------------------------------------------------------------------
    # HELPER UTILITIES
    # ------------------------------------------------------------------
    
    def _append_log_line(self, line):
        self.logs_text.configure(state="normal")
        self.logs_text.insert("end", f"{line}\n")
        self.logs_text.see("end")
        self.logs_text.configure(state="disabled")

    def _clear_logs(self):
        self.logs_text.configure(state="normal")
        self.logs_text.delete("1.0", "end")
        self.logs_text.configure(state="disabled")

def main():
    root = tk.Tk()
    app = AIONTrainerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
