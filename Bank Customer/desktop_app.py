from __future__ import annotations
 
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
 
import customtkinter as ctk
import pandas as pd
import joblib
from tkinter import filedialog, messagebox
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("churniq")
 
class Theme:
 
    NAVY = "#14213D"
    NAVY_HOVER = "#23385F"
    BLUE = "#053698"
    BLUE_HOVER = "#315FCB"
 
    BG = "#F3F5FA"
    WHITE = "#FFFFFF"
    TEXT = "#202B43"
    MUTED = "#7C879B"
    BORDER = "#E6EAF1"
 
    GREEN = "#16845B"
    GREEN_BG = "#EAF7F0"
    RED = "#D64545"
    RED_BG = "#FFF0F0"
 
    SIDEBAR_MUTED = "#AAB8D2"
    SIDEBAR_LABEL = "#8292B1"
    SIDEBAR_CARD = "#263653"
 
FEATURE_ORDER = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary",
]
 
ENCODER_FILES = {
    "geo_encoder": "geo_encoder.pkl",
    "gender_encoder": "gender_encoder.pkl",
    "scaler": "scaler.pkl",
}

# Models are loaded automatically from here on startup (no file picking needed).
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR_CANDIDATES = [APP_DIR / "model_files", APP_DIR]

# Old single-model file name (the notebook used to save only the SVM).
LEGACY_MODEL_FILE = "churn_model.pkl"
RECOMMENDED_KEY = "svm"


@dataclass(frozen=True)
class ModelInfo:
    key: str
    name: str
    filename: str
    summary: str
    best_for: str
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    # Scores from the notebook (20% stratified hold-out test set).
    accuracy: float
    precision: float
    recall: float
    f1: float
    has_probability: bool = True


MODEL_CATALOG: dict[str, ModelInfo] = {
    m.key: m
    for m in [
        ModelInfo(
            "logistic_regression", "Logistic Regression",
            "churn_model_logistic_regression.pkl",
            "A linear model that combines the customer's features into a single "
            "churn score. Simple, fast and a common baseline.",
            "A quick baseline and easy-to-explain results.",
            ("Very fast to train and to run", "Simple and easy to interpret",
             "Good recall (72.9%)"),
            ("Cannot capture non-linear patterns", "Lowest precision (43.6%): many false alarms",
             "Lowest accuracy of all models"),
            0.743085, 0.435948, 0.728952, 0.545601,
        ),
        ModelInfo(
            "svm", "SVM",
            "churn_model_svm.pkl",
            "Finds the boundary that best separates customers who stay from those "
            "who leave. This is the final model chosen in the project.",
            "Catching as many at-risk customers as possible (retention campaigns).",
            ("Highest recall (79.2%): misses the fewest churners",
             "Highest F1-score (0.634)", "Handles complex boundaries"),
            ("Precision is 52.8%: about 47% of flagged customers would have stayed",
             "Slower on large datasets",
             "Does not output a churn probability in this setup"),
            0.806465, 0.528473, 0.791953, 0.633926, has_probability=False,
        ),
        ModelInfo(
            "knn", "KNN",
            "churn_model_knn.pkl",
            "Looks at the 5 most similar customers in the training data and "
            "predicts what most of them did.",
            "Predictions based on similar past customers.",
            ("No assumptions about the data shape", "Good precision (67.8%)",
             "Easy to understand"),
            ("Misses almost half of churners (recall 53.1%)",
             "Slower predictions: compares against all training data",
             "Model file includes the training data"),
            0.847245, 0.677514, 0.530641, 0.595150,
        ),
        ModelInfo(
            "decision_tree", "Decision Tree",
            "churn_model_decision_tree.pkl",
            "Asks a chain of yes/no questions about the customer (for example age "
            "or number of products) until it reaches a decision.",
            "Explaining decisions as simple rules.",
            ("Easy to visualize and explain", "Fast predictions",
             "Captures non-linear patterns"),
            ("Tends to overfit the training data", "Lowest F1 among tree models (0.526)",
             "Recall is only 52.1%"),
            0.801072, 0.530449, 0.521334, 0.525852,
        ),
        ModelInfo(
            "naive_bayes", "Naive Bayes",
            "churn_model_naive_bayes.pkl",
            "A probabilistic model that assumes every feature acts independently "
            "of the others.",
            "Very fast screening when speed matters more than recall.",
            ("Extremely fast and lightweight", "Decent precision (65.2%)",
             "Works with little data"),
            ("Lowest recall (39.6%): misses about 6 of 10 churners",
             "Independence assumption is unrealistic here",
             "Lowest F1-score (0.493)"),
            0.827431, 0.651815, 0.395905, 0.492606,
        ),
        ModelInfo(
            "random_forest", "Random Forest",
            "churn_model_random_forest.pkl",
            "Builds 100 decision trees on different samples and lets them vote "
            "on the final answer.",
            "A stable all-round choice with few false alarms.",
            ("High accuracy (85.7%)", "High precision (73.0%)",
             "Robust and handles non-linear patterns"),
            ("Misses about half of churners (recall 51.5%)",
             "Larger file and slower to load", "Harder to explain than one tree"),
            0.857000, 0.729615, 0.515034, 0.603827,
        ),
        ModelInfo(
            "gradient_boosting", "Gradient Boosting",
            "churn_model_gradient_boosting.pkl",
            "Builds trees one after another, each one correcting the mistakes of "
            "the previous ones.",
            "When false alarms are costly and you want the most precise flags.",
            ("Highest accuracy (86.4%)", "Highest precision (75.5%)",
             "When it flags churn, it is usually right"),
            ("Misses almost half of churners (recall 52.8%)",
             "Slower to train", "Less transparent than simple models"),
            0.863877, 0.754964, 0.528064, 0.621451,
        ),
    ]
}

METRIC_LABELS = (
    ("accuracy", "ACCURACY"), ("precision", "PRECISION"),
    ("recall", "RECALL"), ("f1", "F1-SCORE"),
)
BEST_METRIC = {
    metric: max(getattr(info, metric) for info in MODEL_CATALOG.values())
    for metric, _ in METRIC_LABELS
}

@dataclass
class PredictionResult:
    will_churn: bool
    probability: float | None
 
class ChurnModelError(Exception):
    pass
 
 
@dataclass
class ChurnModelBundle:

    geo_encoder: Any = None
    gender_encoder: Any = None
    scaler: Any = None
    model: Any = None
    active_key: str | None = None
    model_paths: dict[str, Path] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_loaded(self) -> bool:
        return all(
            item is not None
            for item in (self.model, self.geo_encoder, self.gender_encoder, self.scaler)
        )

    @property
    def available_keys(self) -> list[str]:
        return [key for key in MODEL_CATALOG if key in self.model_paths]

    @staticmethod
    def _discover_models(folder: Path) -> dict[str, Path]:
        paths = {
            key: folder / info.filename
            for key, info in MODEL_CATALOG.items()
            if (folder / info.filename).is_file()
        }
        legacy = folder / LEGACY_MODEL_FILE
        if RECOMMENDED_KEY not in paths and legacy.is_file():
            paths[RECOMMENDED_KEY] = legacy
        return paths

    @staticmethod
    def _read_model(path: Path) -> Any:
        try:
            return joblib.load(path)
        except Exception as exc:
            raise ChurnModelError(f"Could not load {path.name}: {exc}") from exc

    def load(self, folder: Path) -> None:
        missing = [n for n in ENCODER_FILES.values() if not (folder / n).is_file()]
        model_paths = self._discover_models(folder)
        if not model_paths:
            missing.append("churn_model_<name>.pkl (at least one model)")
        if missing:
            raise ChurnModelError(
                "These files are missing from the selected folder:\n\n"
                + "\n".join(missing)
            )

        try:
            geo_encoder = joblib.load(folder / ENCODER_FILES["geo_encoder"])
            gender_encoder = joblib.load(folder / ENCODER_FILES["gender_encoder"])
            scaler = joblib.load(folder / ENCODER_FILES["scaler"])
        except Exception as exc:
            raise ChurnModelError(f"Could not load model files: {exc}") from exc

        default_key = (
            RECOMMENDED_KEY if RECOMMENDED_KEY in model_paths
            else next(k for k in MODEL_CATALOG if k in model_paths)
        )
        model = self._read_model(model_paths[default_key])

        # Commit only after everything loaded successfully.
        self.geo_encoder, self.gender_encoder, self.scaler = geo_encoder, gender_encoder, scaler
        self.model_paths = model_paths
        self._cache = {default_key: model}
        self.model, self.active_key = model, default_key
        logger.info("Loaded %d model(s) from %s (active: %s)",
                    len(model_paths), folder, default_key)

    def select_model(self, key: str) -> None:
        if key not in self.model_paths:
            raise ChurnModelError("This model file was not found in the loaded folder.")
        if key not in self._cache:
            self._cache[key] = self._read_model(self.model_paths[key])
        self.model, self.active_key = self._cache[key], key
        logger.info("Active model switched to %s", key)

    def predict(self, raw: dict[str, Any]) -> PredictionResult:
        if not self.is_loaded:
            raise ChurnModelError("Model files were not found. Put the model_files folder next to the app.")
 
        try:
            geo_value = self.geo_encoder.transform([raw["Geography"]])[0]
        except ValueError as exc:
            raise ChurnModelError(
                f"Unrecognized Geography value: {raw['Geography']!r}"
            ) from exc
 
        try:
            gender_value = self.gender_encoder.transform([raw["Gender"]])[0]
        except ValueError as exc:
            raise ChurnModelError(
                f"Unrecognized Gender value: {raw['Gender']!r}"
            ) from exc
 
        row = pd.DataFrame(
            [[
                raw["CreditScore"], geo_value, gender_value, raw["Age"],
                raw["Tenure"], raw["Balance"], raw["NumOfProducts"],
                raw["HasCrCard"], raw["IsActiveMember"], raw["EstimatedSalary"],
            ]],
            columns=FEATURE_ORDER,
        )
 
        try:
            scaled = self.scaler.transform(row)
            prediction = int(self.model.predict(scaled)[0])
            probability = None
            if hasattr(self.model, "predict_proba"):
                probability = float(self.model.predict_proba(scaled)[0][1])
        except Exception as exc:
            raise ChurnModelError(f"Prediction failed: {exc}") from exc
 
        return PredictionResult(will_churn=bool(prediction), probability=probability)
 
@dataclass
class NumericField:
    label: str
    minimum: float
    maximum: float
    is_int: bool = False
 
    def parse(self, raw: str) -> float:
        raw = raw.strip()
        if not raw:
            raise ChurnModelError(f"Please enter {self.label}.")
        try:
            value = float(raw)
        except ValueError:
            raise ChurnModelError(f"{self.label} must be a number.") from None
        if self.is_int and not value.is_integer():
            raise ChurnModelError(f"{self.label} must be a whole number.")
        if not (self.minimum <= value <= self.maximum):
            raise ChurnModelError(
                f"{self.label} must be between {self.minimum:g} and {self.maximum:g}."
            )
        return value
 
NUMERIC_FIELDS: dict[str, NumericField] = {
    "CreditScore": NumericField("Credit Score", 300, 850, is_int=True),
    "Age": NumericField("Age", 18, 100, is_int=True),
    "Tenure": NumericField("Tenure", 0, 15, is_int=True),
    "Balance": NumericField("Account Balance", 0, 10_000_000),
    "NumOfProducts": NumericField("Number of Products", 1, 10, is_int=True),
    "EstimatedSalary": NumericField("Estimated Salary", 0, 10_000_000),
}
 
class ModelInfoDialog(ctk.CTkToplevel):
    """Lists every model with its description and scores, and lets the user pick one."""

    def __init__(
        self, master: ctk.CTk, available: set[str], active_key: str | None,
        loaded: bool, on_use: Callable[[str], bool],
    ) -> None:
        super().__init__(master)
        self.title("Model Information")
        self.geometry("930x640")
        self.minsize(880, 620)
        self.configure(fg_color=Theme.BG)
        self.transient(master)

        self._available = available
        self._active = active_key
        self._loaded = loaded
        self._on_use = on_use
        self._selected = active_key or RECOMMENDED_KEY
        self._list_buttons: dict[str, ctk.CTkButton] = {}

        self._build()
        self._show(self._selected)
        self.after(150, self.lift)
        self.after(150, self.focus_force)

    def _build(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20, pady=20)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(
            body, width=235, fg_color=Theme.WHITE, corner_radius=14,
            border_width=1, border_color=Theme.BORDER,
        )
        left.grid(row=0, column=0, sticky="ns", padx=(0, 15))
        left.pack_propagate(False)

        ctk.CTkLabel(
            left, text="MODELS", text_color=Theme.MUTED,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(20, 10))

        for key, info in MODEL_CATALOG.items():
            button = ctk.CTkButton(
                left, text=self._list_text(key), anchor="w", height=40,
                corner_radius=8, fg_color="transparent", hover_color="#EEF2FB",
                text_color=Theme.TEXT if key in self._available or not self._loaded
                else Theme.MUTED,
                font=ctk.CTkFont(size=13, weight="bold"),
                command=lambda k=key: self._show(k),
            )
            button.pack(fill="x", padx=10, pady=2)
            self._list_buttons[key] = button

        ctk.CTkLabel(
            left, text="★  Final model of the project\n✓  Currently in use",
            text_color=Theme.MUTED, justify="left", font=ctk.CTkFont(size=11),
        ).pack(side="bottom", anchor="w", padx=18, pady=18)

        right = ctk.CTkFrame(
            body, fg_color=Theme.WHITE, corner_radius=14,
            border_width=1, border_color=Theme.BORDER,
        )
        right.grid(row=0, column=1, sticky="nsew")

        header = ctk.CTkFrame(right, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(22, 0))
        self.name_label = ctk.CTkLabel(
            header, text="", text_color=Theme.TEXT,
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        self.name_label.pack(side="left")
        self.final_badge = ctk.CTkLabel(
            header, text="FINAL MODEL", text_color=Theme.BLUE, fg_color="#E8EEFC",
            corner_radius=6, height=22, width=92, font=ctk.CTkFont(size=10, weight="bold"),
        )
        self.status_badge = ctk.CTkLabel(
            header, text="", corner_radius=6, height=22, width=92,
            font=ctk.CTkFont(size=10, weight="bold"),
        )

        self.summary_label = ctk.CTkLabel(
            right, text="", text_color=Theme.TEXT, justify="left", anchor="w",
            wraplength=590, font=ctk.CTkFont(size=13),
        )
        self.summary_label.pack(fill="x", padx=24, pady=(10, 4))

        self.best_for_label = ctk.CTkLabel(
            right, text="", text_color=Theme.BLUE, justify="left", anchor="w",
            wraplength=590, font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.best_for_label.pack(fill="x", padx=24, pady=(0, 14))

        metrics = ctk.CTkFrame(right, fg_color="transparent")
        metrics.pack(fill="x", padx=18)
        self.metric_values: dict[str, ctk.CTkLabel] = {}
        self.metric_tags: dict[str, ctk.CTkLabel] = {}
        for col, (metric, caption) in enumerate(METRIC_LABELS):
            metrics.grid_columnconfigure(col, weight=1, uniform="metric")
            tile = ctk.CTkFrame(metrics, fg_color="#F6F8FC", corner_radius=10)
            tile.grid(row=0, column=col, sticky="ew", padx=6)
            ctk.CTkLabel(
                tile, text=caption, text_color=Theme.MUTED,
                font=ctk.CTkFont(size=10, weight="bold"),
            ).pack(anchor="w", padx=12, pady=(10, 0))
            value = ctk.CTkLabel(
                tile, text="", text_color=Theme.TEXT,
                font=ctk.CTkFont(size=20, weight="bold"),
            )
            value.pack(anchor="w", padx=12)
            tag = ctk.CTkLabel(
                tile, text="", text_color=Theme.GREEN, height=16,
                font=ctk.CTkFont(size=10, weight="bold"),
            )
            tag.pack(anchor="w", padx=12, pady=(0, 8))
            self.metric_values[metric] = value
            self.metric_tags[metric] = tag

        lists = ctk.CTkFrame(right, fg_color="transparent")
        lists.pack(fill="x", padx=24, pady=(18, 0))
        lists.grid_columnconfigure((0, 1), weight=1, uniform="lists")
        self.strengths_label = self._add_list_block(lists, 0, "STRENGTHS", Theme.GREEN)
        self.limits_label = self._add_list_block(lists, 1, "LIMITATIONS", Theme.RED)

        self.proba_label = ctk.CTkLabel(
            right, text="", text_color=Theme.MUTED, anchor="w",
            font=ctk.CTkFont(size=11),
        )
        self.proba_label.pack(fill="x", padx=24, pady=(14, 0))

        footer = ctk.CTkFrame(right, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=24, pady=(0, 22))
        ctk.CTkLabel(
            footer, text="Scores come from the notebook's 20% hold-out test set "
                         "(33,007 customers).",
            text_color=Theme.MUTED, anchor="w", font=ctk.CTkFont(size=10),
        ).pack(fill="x", pady=(0, 10))
        self.use_button = ctk.CTkButton(
            footer, text="Use this model", command=self._on_use_clicked,
            height=43, corner_radius=9, fg_color=Theme.BLUE,
            hover_color=Theme.BLUE_HOVER, font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.use_button.pack(fill="x")

    @staticmethod
    def _add_list_block(parent: ctk.CTkFrame, col: int, title: str, color: str) -> ctk.CTkLabel:
        block = ctk.CTkFrame(parent, fg_color="transparent")
        block.grid(row=0, column=col, sticky="nsew", padx=(0, 10) if col == 0 else (10, 0))
        ctk.CTkLabel(
            block, text=title, text_color=color,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w", pady=(0, 6))
        label = ctk.CTkLabel(
            block, text="", text_color=Theme.TEXT, justify="left", anchor="w",
            wraplength=270, font=ctk.CTkFont(size=12),
        )
        label.pack(anchor="w", fill="x")
        return label

    def _list_text(self, key: str) -> str:
        marks = ""
        if key == RECOMMENDED_KEY:
            marks += "  ★"
        if key == self._active:
            marks += "  ✓"
        return MODEL_CATALOG[key].name + marks

    def _show(self, key: str) -> None:
        info = MODEL_CATALOG[key]
        self._selected = key

        for k, button in self._list_buttons.items():
            button.configure(fg_color="#E8EEFC" if k == key else "transparent")

        self.name_label.configure(text=info.name)
        self.final_badge.pack_forget()
        self.status_badge.pack_forget()
        if key == RECOMMENDED_KEY:
            self.final_badge.pack(side="left", padx=(12, 0))
        if self._loaded and key == self._active:
            self.status_badge.configure(text="IN USE", text_color=Theme.GREEN, fg_color=Theme.GREEN_BG)
            self.status_badge.pack(side="left", padx=(8, 0))
        elif self._loaded and key not in self._available:
            self.status_badge.configure(text="FILE NOT FOUND", text_color=Theme.RED, fg_color=Theme.RED_BG)
            self.status_badge.pack(side="left", padx=(8, 0))

        self.summary_label.configure(text=info.summary)
        self.best_for_label.configure(text=f"Best for: {info.best_for}")

        for metric, _caption in METRIC_LABELS:
            value = getattr(info, metric)
            text = f"{value:.3f}" if metric == "f1" else f"{value * 100:.1f}%"
            self.metric_values[metric].configure(text=text)
            is_best = abs(value - BEST_METRIC[metric]) < 1e-9
            self.metric_tags[metric].configure(text="★ best" if is_best else "")

        self.strengths_label.configure(text="\n".join(f"•  {s}" for s in info.strengths))
        self.limits_label.configure(text="\n".join(f"•  {s}" for s in info.limitations))
        self.proba_label.configure(
            text="Churn probability: available" if info.has_probability
            else "Churn probability: not available (the result panel will show N/A)"
        )

        if self._loaded and key not in self._available:
            self.use_button.configure(text="Model file not found", state="disabled")
        elif self._loaded and key == self._active:
            self.use_button.configure(text="Currently in use", state="disabled")
        else:
            self.use_button.configure(text=f"Use {info.name}", state="normal")

    def _on_use_clicked(self) -> None:
        if self._on_use(self._selected):
            self.destroy()


class ChurnIQApp(ctk.CTk):
 
    def __init__(self) -> None:
        super().__init__()
 
        self.bundle = ChurnModelBundle()
        self.model_dialog: ModelInfoDialog | None = None
        self._name_to_key: dict[str, str] = {}
        self.inputs: dict[str, ctk.CTkEntry | ctk.CTkComboBox] = {}
 
        self.title("ChurnIQ | Customer Analytics")
        self.geometry("1150x760")
        self.minsize(1000, 680)
        self.configure(fg_color=Theme.BG)
 
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
 
        self._build_sidebar()
        self._build_workspace()
 
        self.bind("<Return>", lambda _event: self.on_predict())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._auto_load_models)
 
    def _on_close(self) -> None:
        logger.info("Closing ChurnIQ.")
        self.destroy()
 
    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=215, fg_color=Theme.NAVY, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
 
        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.pack(fill="x", padx=22, pady=(28, 35))
 
        ctk.CTkLabel(
            brand, text="◈", text_color="#8EAEFF",
            font=ctk.CTkFont(size=30, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand, text="ChurnIQ", text_color=Theme.WHITE,
            font=ctk.CTkFont(size=25, weight="bold"),
        ).pack(anchor="w", pady=(3, 0))
        ctk.CTkLabel(
            brand, text="CUSTOMER ANALYTICS", text_color=Theme.SIDEBAR_MUTED,
            font=ctk.CTkFont(size=10, weight="bold"),
        ).pack(anchor="w", pady=(3, 0))
 
        ctk.CTkLabel(
            sidebar, text="WORKSPACE", text_color=Theme.SIDEBAR_LABEL,
            font=ctk.CTkFont(size=10, weight="bold"),
        ).pack(anchor="w", padx=23, pady=(0, 10))
 
        ctk.CTkLabel(
            sidebar, text="▦   Churn Prediction", text_color=Theme.WHITE,
            fg_color=Theme.NAVY_HOVER, corner_radius=9, anchor="w", height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(fill="x", padx=13)
 
        ctk.CTkLabel(
            sidebar, text="MODEL", text_color=Theme.SIDEBAR_LABEL,
            font=ctk.CTkFont(size=10, weight="bold"),
        ).pack(anchor="w", padx=23, pady=(32, 12))
 
        self.model_badge = ctk.CTkLabel(
            sidebar, text="● MODEL NOT LOADED", text_color="#CBD5E1",
            fg_color=Theme.SIDEBAR_CARD, corner_radius=8, height=32,
            font=ctk.CTkFont(size=10, weight="bold"),
        )
        self.model_badge.pack(fill="x", padx=14)
 
        # Fallback only: shown when the models folder cannot be found automatically.
        self.locate_button = ctk.CTkButton(
            sidebar, text="Locate Models Folder", command=self.on_load_model,
            height=39, corner_radius=8, fg_color=Theme.BLUE,
            hover_color=Theme.BLUE_HOVER, font=ctk.CTkFont(size=12, weight="bold"),
        )

        ctk.CTkLabel(
            sidebar, text="Machine Learning Project\nBank Customer Churn",
            text_color=Theme.SIDEBAR_MUTED, justify="left",
            font=ctk.CTkFont(size=11),
        ).pack(side="bottom", anchor="w", padx=20, pady=22)
 
    def _build_workspace(self) -> None:
        workspace = ctk.CTkFrame(self, fg_color=Theme.BG, corner_radius=0)
        workspace.pack(side="left", fill="both", expand=True, padx=25, pady=22)
 
        topbar = ctk.CTkFrame(workspace, fg_color="transparent")
        topbar.pack(fill="x", pady=(0, 20))

        titles = ctk.CTkFrame(topbar, fg_color="transparent")
        titles.pack(side="left")
        ctk.CTkLabel(
            titles, text="Customer Churn Prediction", text_color=Theme.TEXT,
            font=ctk.CTkFont(size=25, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            titles, text="Analyze customer information and estimate churn risk.",
            text_color=Theme.MUTED, font=ctk.CTkFont(size=13),
        ).pack(anchor="w", pady=(4, 0))

        chooser = ctk.CTkFrame(topbar, fg_color="transparent")
        chooser.pack(side="right", anchor="n")
        ctk.CTkLabel(
            chooser, text="CHOOSE MODEL", text_color=Theme.MUTED,
            font=ctk.CTkFont(size=10, weight="bold"),
        ).pack(anchor="w", pady=(0, 5))
        chooser_row = ctk.CTkFrame(chooser, fg_color="transparent")
        chooser_row.pack()
        self.model_selector = ctk.CTkOptionMenu(
            chooser_row, values=[m.name for m in MODEL_CATALOG.values()],
            command=self.on_model_selected, width=190, height=38, corner_radius=8,
            fg_color=Theme.WHITE, text_color=Theme.TEXT,
            button_color=Theme.BLUE, button_hover_color=Theme.BLUE_HOVER,
            dropdown_fg_color=Theme.WHITE, dropdown_text_color=Theme.TEXT,
            dropdown_hover_color="#EEF2FB", font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.model_selector.set(MODEL_CATALOG[RECOMMENDED_KEY].name)
        self.model_selector.pack(side="left")
        ctk.CTkButton(
            chooser_row, text="ⓘ", width=38, height=38, corner_radius=8,
            fg_color=Theme.WHITE, hover_color="#EEF2FB", text_color=Theme.BLUE,
            border_width=1, border_color=Theme.BORDER,
            font=ctk.CTkFont(size=16, weight="bold"), command=self.on_show_model_info,
        ).pack(side="left", padx=(8, 0))

        content = ctk.CTkFrame(workspace, fg_color="transparent")
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)
        content.grid_rowconfigure(0, weight=1)
 
        self._build_form(content)
        self._build_result_panel(content)
 
        ctk.CTkLabel(
            workspace, text="ChurnIQ  •  Bank Customer Churn Classification",
            text_color=Theme.MUTED, font=ctk.CTkFont(size=10),
        ).pack(pady=(12, 0))
 
    def _build_form(self, parent: ctk.CTkFrame) -> None:
        form_card = ctk.CTkFrame(
            parent, fg_color=Theme.WHITE, corner_radius=14,
            border_width=1, border_color=Theme.BORDER,
        )
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
 
        ctk.CTkLabel(
            form_card, text="Customer Details", text_color=Theme.TEXT,
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(20, 4))
        ctk.CTkLabel(
            form_card, text="Fill in the fields below", text_color=Theme.MUTED,
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w", padx=20, pady=(0, 12))
 
        fields = ctk.CTkFrame(form_card, fg_color="transparent")
        fields.pack(fill="both", expand=True, padx=13)
        fields.grid_columnconfigure(0, weight=1)
        fields.grid_columnconfigure(1, weight=1)
 
        self._add_entry(fields, "CreditScore", "Credit Score", 0, 0, "e.g. 650")
        self._add_combo(fields, "Geography", "Geography", ["France", "Germany", "Spain"], 0, 1)
        self._add_combo(fields, "Gender", "Gender", ["Female", "Male"], 1, 0)
        self._add_entry(fields, "Age", "Age", 1, 1, "e.g. 35")
        self._add_entry(fields, "Tenure", "Tenure (Years)", 2, 0, "e.g. 5")
        self._add_entry(fields, "Balance", "Account Balance", 2, 1, "e.g. 50000")
        self._add_entry(fields, "NumOfProducts", "Number of Products", 3, 0, "e.g. 2")
        self._add_combo(fields, "HasCrCard", "Has Credit Card?", ["Yes", "No"], 3, 1)
        self._add_combo(fields, "IsActiveMember", "Active Member?", ["Yes", "No"], 4, 0)
        self._add_entry(fields, "EstimatedSalary", "Estimated Salary", 4, 1, "e.g. 75000")
 
        button_row = ctk.CTkFrame(form_card, fg_color="transparent")
        button_row.pack(fill="x", padx=20, pady=(10, 20))
 
        self.predict_button = ctk.CTkButton(
            button_row, text="Predict Churn  →", command=self.on_predict,
            height=43, corner_radius=9, fg_color=Theme.BLUE,
            hover_color=Theme.BLUE_HOVER, font=ctk.CTkFont(size=13, weight="bold"),
            state="disabled",
        )
        self.predict_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
 
        ctk.CTkButton(
            button_row, text="Clear", command=self.on_clear, height=43,
            corner_radius=9, fg_color="#E9EDF5", hover_color="#DDE4F0",
            text_color=Theme.TEXT, font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=(6, 0))
 
    def _build_result_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(
            parent, width=245, fg_color=Theme.WHITE, corner_radius=14,
            border_width=1, border_color=Theme.BORDER,
        )
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)
 
        ctk.CTkLabel(
            panel, text="PREDICTION", text_color=Theme.MUTED,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w", padx=19, pady=(22, 15))
 
        self.result_icon = ctk.CTkLabel(
            panel, text="?", width=55, height=55, corner_radius=16,
            fg_color="#EDF0F5", text_color=Theme.MUTED,
            font=ctk.CTkFont(size=26, weight="bold"),
        )
        self.result_icon.pack(anchor="w", padx=19)
 
        self.result_title = ctk.CTkLabel(
            panel, text="No prediction yet", text_color=Theme.TEXT,
            font=ctk.CTkFont(size=17, weight="bold"), wraplength=205, justify="left",
        )
        self.result_title.pack(anchor="w", padx=19, pady=(17, 8))
 
        self.result_desc = ctk.CTkLabel(
            panel, text="Complete the customer details and run the model.",
            text_color=Theme.MUTED, font=ctk.CTkFont(size=12),
            wraplength=205, justify="left",
        )
        self.result_desc.pack(anchor="w", padx=19)
 
        ctk.CTkFrame(panel, height=1, fg_color=Theme.BORDER).pack(
            fill="x", padx=19, pady=20
        )
 
        ctk.CTkLabel(
            panel, text="CHURN PROBABILITY", text_color=Theme.MUTED,
            font=ctk.CTkFont(size=10, weight="bold"),
        ).pack(anchor="w", padx=19)
 
        self.probability_label = ctk.CTkLabel(
            panel, text="—", text_color=Theme.BLUE,
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.probability_label.pack(anchor="w", padx=19, pady=(8, 0))

        ctk.CTkFrame(panel, height=1, fg_color=Theme.BORDER).pack(
            fill="x", padx=19, pady=20
        )

        ctk.CTkLabel(
            panel, text="MODEL USED", text_color=Theme.MUTED,
            font=ctk.CTkFont(size=10, weight="bold"),
        ).pack(anchor="w", padx=19)

        self.result_model_label = ctk.CTkLabel(
            panel, text="—", text_color=Theme.TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.result_model_label.pack(anchor="w", padx=19, pady=(8, 0))
 
    def _add_entry(
        self, parent: ctk.CTkFrame, name: str, label: str,
        row: int, col: int, placeholder: str = "Enter value",
    ) -> None:
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row, column=col, sticky="ew", padx=7, pady=8)
        ctk.CTkLabel(
            box, text=label, text_color=Theme.TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 5))
        entry = ctk.CTkEntry(
            box, height=38, corner_radius=8, border_color=Theme.BORDER,
            fg_color="#FAFBFE", text_color=Theme.TEXT,
            placeholder_text=placeholder, font=ctk.CTkFont(size=12),
        )
        entry.pack(fill="x")
        self.inputs[name] = entry
 
    def _add_combo(
        self, parent: ctk.CTkFrame, name: str, label: str,
        values: list[str], row: int, col: int,
    ) -> None:
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row, column=col, sticky="ew", padx=7, pady=8)
        ctk.CTkLabel(
            box, text=label, text_color=Theme.TEXT,
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(anchor="w", pady=(0, 5))
        combo = ctk.CTkComboBox(
            box, values=values, height=38, corner_radius=8,
            border_color=Theme.BORDER, fg_color="#FAFBFE", text_color=Theme.TEXT,
            button_color=Theme.BLUE, button_hover_color=Theme.BLUE_HOVER,
            state="readonly", font=ctk.CTkFont(size=12),
        )
        combo.pack(fill="x")
        self.inputs[name] = combo
 
    def _auto_load_models(self) -> None:
        """Load models from the default folder so the user only has to pick a model."""
        for folder in MODEL_DIR_CANDIDATES:
            if not folder.is_dir() or not self.bundle._discover_models(folder):
                continue
            try:
                self._load_folder(folder)
                return
            except ChurnModelError as exc:
                logger.warning("Auto-load from %s failed: %s", folder, exc)
        self.model_badge.configure(text="● NO MODELS FOUND", text_color="#FCA5A5")
        self.locate_button.pack(fill="x", padx=14, pady=(12, 0))

    def _load_folder(self, folder: Path) -> None:
        self.configure(cursor="watch")
        self.update_idletasks()
        try:
            self.bundle.load(folder)
        finally:
            self.configure(cursor="")

        self.model_badge.configure(text="● MODEL READY", text_color="#6EE7B7")
        self.locate_button.pack_forget()
        self.predict_button.configure(state="normal")
        self._refresh_model_widgets()
        self._reset_result_panel()

    def on_load_model(self) -> None:
        folder = filedialog.askdirectory(title="Choose the folder that contains the models")
        if not folder:
            return
        try:
            self._load_folder(Path(folder))
        except ChurnModelError as exc:
            messagebox.showerror("Loading error", str(exc))
            return
        messagebox.showinfo(
            "Success",
            f"Loaded {len(self.bundle.available_keys)} model(s).\n"
            f"Active model: {MODEL_CATALOG[self.bundle.active_key].name}",
        )

    def _refresh_model_widgets(self) -> None:
        self._name_to_key = {m.name: k for k, m in MODEL_CATALOG.items()}
        if not self.bundle.is_loaded:
            self.result_model_label.configure(text="—")
            return
        active_name = MODEL_CATALOG[self.bundle.active_key].name
        self.model_selector.set(active_name)
        self.result_model_label.configure(text=active_name)

    def on_model_selected(self, name: str) -> None:
        key = self._name_to_key.get(name)
        if key:
            self._activate_model(key)

    def _activate_model(self, key: str) -> bool:
        """Switch to `key`, loading its file on demand. Returns True on success."""
        try:
            if not self.bundle.is_loaded:
                self._auto_load_models()
                if not self.bundle.is_loaded:
                    raise ChurnModelError(
                        "Model files were not found.\n\n"
                        "Put the 'model_files' folder next to the app, or use "
                        "'Locate Models Folder' in the sidebar."
                    )
            self.configure(cursor="watch")
            self.update_idletasks()
            try:
                self.bundle.select_model(key)
            finally:
                self.configure(cursor="")
        except ChurnModelError as exc:
            messagebox.showerror("Model error", str(exc))
            self._restore_selector()
            return False

        self._refresh_model_widgets()
        self._reset_result_panel()
        return True

    def _restore_selector(self) -> None:
        key = self.bundle.active_key if self.bundle.is_loaded else RECOMMENDED_KEY
        self.model_selector.set(MODEL_CATALOG[key].name)

    def on_show_model_info(self) -> None:
        if self.model_dialog is not None and self.model_dialog.winfo_exists():
            self.model_dialog.destroy()
        self.model_dialog = ModelInfoDialog(
            self,
            available=set(self.bundle.available_keys),
            active_key=self.bundle.active_key,
            loaded=self.bundle.is_loaded,
            on_use=self._activate_model,
        )
 
    def on_clear(self) -> None:
        for widget in self.inputs.values():
            if isinstance(widget, ctk.CTkEntry):
                widget.delete(0, "end")
            else:
                widget.set("")
        self._reset_result_panel()
 
    def _reset_result_panel(self) -> None:
        self.result_icon.configure(text="?", text_color=Theme.MUTED, fg_color="#EDF0F5")
        self.result_title.configure(text="No prediction yet", text_color=Theme.TEXT)
        self.result_desc.configure(
            text="Enter customer details and run the prediction.",
            text_color=Theme.MUTED,
        )
        self.probability_label.configure(text="—")
 
    def _read_form(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
 
        for name, spec in NUMERIC_FIELDS.items():
            values[name] = spec.parse(self.inputs[name].get())
 
        geography = self.inputs["Geography"].get()
        gender = self.inputs["Gender"].get()
        has_card = self.inputs["HasCrCard"].get()
        is_active = self.inputs["IsActiveMember"].get()
 
        if not all([geography, gender, has_card, is_active]):
            raise ChurnModelError("Please complete all dropdown fields.")
 
        values["Geography"] = geography
        values["Gender"] = gender
        values["HasCrCard"] = 1 if has_card == "Yes" else 0
        values["IsActiveMember"] = 1 if is_active == "Yes" else 0
        return values
 
    def on_predict(self) -> None:
        if str(self.predict_button.cget("state")) == "disabled":
            messagebox.showwarning("Model not loaded", "Model files were not found. Put the model_files folder next to the app.")
            return
 
        try:
            form_values = self._read_form()
            result = self.bundle.predict(form_values)
        except ChurnModelError as exc:
            messagebox.showerror("Prediction error", str(exc))
            return
        except Exception as exc:
            logger.exception("Unexpected error during prediction")
            messagebox.showerror("Unexpected error", str(exc))
            return
 
        self._render_result(result)
 
    def _render_result(self, result: PredictionResult) -> None:
        if result.will_churn:
            self.result_icon.configure(text="!", text_color=Theme.RED, fg_color=Theme.RED_BG)
            self.result_title.configure(text="Churn Risk Detected", text_color=Theme.RED)
            self.result_desc.configure(
                text="The model predicts this customer may leave the bank.",
                text_color=Theme.TEXT,
            )
        else:
            self.result_icon.configure(text="✓", text_color=Theme.GREEN, fg_color=Theme.GREEN_BG)
            self.result_title.configure(text="Customer Likely to Stay", text_color=Theme.GREEN)
            self.result_desc.configure(
                text="The model predicts this customer may remain with the bank.",
                text_color=Theme.TEXT,
            )
 
        if result.probability is not None:
            self.probability_label.configure(
                text=f"{result.probability * 100:.1f}%"
            )
        else:
            self.probability_label.configure(text="N/A")
 
def main() -> None:
    app = ChurnIQApp()
    app.mainloop()
 
if __name__ == "__main__":
    main()