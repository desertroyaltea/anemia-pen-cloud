#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert PMML (.xml/.pmml) models to .joblib models that mimic them.

Two strategies:
  1) Proxy (default): wrap the original PMML via PyPMML in a scikit-learn-like estimator.
     - Perfect fidelity, requires pypmml at prediction time.
  2) Native linear attempt (--try-native): for RegressionModel/GLM (identity link, no transforms),
     parse coefficients and build a pure estimator. Validate against PMML on sample CSV.
     If mismatch > tolerance, fall back to proxy.

Logging:
  - Writes detailed logs to console and to <out>.log (same basename as output path).
  - Includes parsed feature names, intercept, coefficients, validation comparisons.

Usage:
  python pmml_to_joblib.py \
      --pmml /path/to/Neural1.xml \
      --out  outputs/models/Neural1.joblib \
      --features-csv features.csv \
      --target hb \
      --try-native

Minimal:
  python pmml_to_joblib.py --pmml GeneralizedLinear.xml --out GeneralizedLinear.joblib

Requires:
  pip install joblib pandas numpy pypmml lxml
"""

import argparse
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

# XML parsing
import xml.etree.ElementTree as ET

# PyPMML for proxy scoring
try:
    from pypmml import Model as PyPMMLModel
except Exception:
    PyPMMLModel = None


# ----------------------------- Logging ----------------------------- #

def setup_logger(out_path: Path) -> logging.Logger:
    logger = logging.getLogger("pmml2joblib")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(levelname)s] %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File log next to output model
    log_path = out_path.with_suffix(".log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_path}")
    return logger


# --------------------------- Utilities ---------------------------- #

def read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def detect_namespace(root: ET.Element) -> Dict[str, str]:
    # Extract namespace from root tag like {http://www.dmg.org/PMML-4_4}PMML
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0].strip("{")
        return {"p": uri}
    return {"p": ""}


def findall(root: ET.Element, xpath: str, ns: Dict[str, str]) -> List[ET.Element]:
    return root.findall(xpath, ns)


def findone(root: ET.Element, xpath: str, ns: Dict[str, str]) -> Optional[ET.Element]:
    el = root.find(xpath, ns)
    return el


# ----------------------- Feature extraction ----------------------- #

def mining_active_fields(root: ET.Element, ns: Dict[str, str]) -> Tuple[List[str], Optional[str]]:
    """
    Return (active_inputs, target_name) from MiningSchema.
    """
    inputs: List[str] = []
    target: Optional[str] = None
    # Most models have exactly one MiningModel/RegressionModel/GeneralRegressionModel etc.
    # Search all MiningSchema nodes.
    schemas = findall(root, ".//p:MiningSchema", ns)
    for schema in schemas:
        for mf in list(schema):
            if mf.tag.endswith("MiningField"):
                name = mf.attrib.get("name")
                utype = mf.attrib.get("usageType", "active")
                if utype in ("active", "supplementary"):
                    if name and name not in inputs:
                        inputs.append(name)
                elif utype in ("predicted", "target"):
                    target = name or target
    return inputs, target


def output_field_candidates(root: ET.Element, ns: Dict[str, str]) -> List[str]:
    """
    Collect possible output field names for prediction (e.g., 'hb', 'predicted_hb').
    """
    cands = set(["hb", "predicted_hb", "Prediction", "Predicted_hb"])
    outs = findall(root, ".//p:Output/p:OutputField", ns)
    for of in outs:
        name = of.attrib.get("name")
        if name:
            cands.add(name)
    return list(cands)


# ------------------- Native linear parsing (best-effort) ------------------- #

def parse_regression_linear(root: ET.Element, ns: Dict[str, str]) -> Optional[Tuple[float, Dict[str, float]]]:
    """
    Parse PMML <RegressionModel> with linear function.
    Return (intercept, {feature: coef}) or None if not supported.
    """
    rm = findone(root, ".//p:RegressionModel", ns)
    if rm is None:
        return None

    # Verify functionName
    if rm.attrib.get("functionName", "").lower() not in ("regression", "regressionmodel", "timeSeries".lower()):
        return None

    # If there are LocalTransformations, we can't safely reproduce without re-implementing them
    lt = findone(rm, "./p:LocalTransformations", ns)
    if lt is not None:
        # Abort native; caller will use proxy
        return None

    # Find first RegressionTable
    table = findone(rm, ".//p:RegressionTable", ns)
    if table is None:
        return None

    # Intercept
    intercept = float(table.attrib.get("intercept", "0.0"))

    coefs: Dict[str, float] = {}

    # NumericPredictors (linear terms)
    for npred in findall(table, "./p:NumericPredictor", ns):
        name = npred.attrib.get("name")
        coef = float(npred.attrib.get("coefficient", "0.0"))
        exponent = int(npred.attrib.get("exponent", "1"))
        if exponent != 1:
            # polynomial terms not supported
            return None
        if name:
            coefs[name] = coefs.get(name, 0.0) + coef

    # If there are CategoricalPredictors or Interaction terms, abort native
    if findone(table, "./p:CategoricalPredictor", ns) is not None:
        return None
    if findone(table, "./p:PredictorTerm", ns) is not None:
        return None

    return intercept, coefs


def is_general_regression_identity(root: ET.Element, ns: Dict[str, str]) -> bool:
    """
    Detect GeneralRegressionModel with identity link and no transforms.
    If detected, we still need to parse coefficients (non-trivial across vendors), so
    we keep proxy unless clearly simple. For now, return False to prefer proxy.
    """
    gr = findone(root, ".//p:GeneralRegressionModel", ns)
    if gr is None:
        return False
    link = gr.attrib.get("linkFunction", "").lower()
    if link not in ("identity", ""):
        return False
    # Skip if there are LocalTransformations
    if findone(gr, "./p:LocalTransformations", ns) is not None:
        return False
    return False  # keep proxy until we implement full parser


# --------------------- Estimators to be saved --------------------- #

class PMMLProxyEstimator:
    """
    A scikit-learn-like estimator that delegates prediction to the stored PMML XML
    via PyPMML. This ensures perfect fidelity to the original model.

    Parameters saved in joblib:
      - pmml_xml: the raw XML text
      - feature_names: preferred feature order for DataFrame inputs
      - output_candidates: list of acceptable output column names
    """

    def __init__(self, pmml_xml: str, feature_names: List[str], output_candidates: List[str]):
        self.pmml_xml = pmml_xml
        self.feature_names = list(feature_names) if feature_names else []
        self.output_candidates = list(output_candidates) if output_candidates else []
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is not None:
            return
        if PyPMMLModel is None:
            raise RuntimeError("pypmml is not installed; cannot load PMML proxy model.")
        # Load from bytes using a NamedTemporaryFile (PyPMML expects path/string)
        with tempfile.NamedTemporaryFile("w", suffix=".pmml", delete=False, encoding="utf-8") as tf:
            tf.write(self.pmml_xml)
            tmp_path = tf.name
        self._model = PyPMMLModel.load(tmp_path)

    def predict(self, X: Any) -> np.ndarray:
        import pandas as pd
        self._ensure_model()
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
        # Ensure float dtype
        X = X.copy()
        for c in X.columns:
            X[c] = X[c].astype(float)

        df = self._model.predict(X)

        # Try output candidates
        for key in self.output_candidates:
            if key in df.columns:
                return df[key].to_numpy(dtype=float).ravel()

        # Fallback to first numeric column
        for col in df.columns:
            try:
                return df[col].astype(float).to_numpy().ravel()
            except Exception:
                continue

        raise RuntimeError("PMML proxy: couldn't find numeric prediction column in model output.")


class NativeLinearEstimator:
    """
    A minimal linear predictor: y = intercept + sum_i coef[i] * x[i]
    Only used when we can parse a plain RegressionModel with numeric predictors and no transforms.
    """

    def __init__(self, intercept: float, coefs: Dict[str, float], feature_names: List[str]):
        self.intercept = float(intercept)
        self.coefs = dict(coefs)
        self.feature_names = list(feature_names)

    def predict(self, X: Any) -> np.ndarray:
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
        X = X.copy()
        for c in X.columns:
            X[c] = X[c].astype(float)
        y = np.full((len(X),), self.intercept, dtype=float)
        for name, coef in self.coefs.items():
            if name not in X.columns:
                raise ValueError(f"Missing required feature for native linear model: {name}")
            y += coef * X[name].to_numpy(dtype=float)
        return y


# ----------------------- Conversion pipeline ---------------------- #

def build_joblib_from_pmml(pmml_path: Path,
                           out_path: Path,
                           features_csv: Optional[Path],
                           target_name: Optional[str],
                           try_native: bool,
                           logger: logging.Logger) -> None:

    pmml_xml = read_text(pmml_path)
    root = ET.fromstring(pmml_xml)
    ns = detect_namespace(root)

    # Features & target detection
    active_inputs, detected_target = mining_active_fields(root, ns)
    out_candidates = output_field_candidates(root, ns)
    logger.info(f"Active inputs (from MiningSchema): {active_inputs}")
    logger.info(f"Target detected: {detected_target}")
    logger.info(f"Output candidates: {out_candidates}")

    # If no inputs found, let caller provide or fall back to 14 known names (your pipeline)
    default_features = [
        "R_norm_p50", "a_mean", "R_p50", "R_p10", "RG", "S_p50",
        "gray_p90", "gray_kurt", "gray_std", "gray_mean",
        "B_mean", "B_p10", "B_p75", "G_kurt",
    ]
    feature_names = active_inputs or default_features
    if target_name:
        detected_target = target_name

    # Decide estimator
    estimator = None
    chosen_mode = "proxy"

    # Attempt native parse only if asked
    if try_native:
        parsed = parse_regression_linear(root, ns)
        if parsed is not None:
            intercept, coefs = parsed
            logger.info("Parsed RegressionModel (native) with intercept and coefficients.")
            logger.debug(f"Intercept: {intercept}")
            logger.debug(f"Coefficients: {json.dumps(coefs, indent=2)}")
            native_est = NativeLinearEstimator(intercept, coefs, feature_names)

            # Validate on sample CSV against PMML proxy; if mismatch, fall back
            if features_csv is not None and features_csv.exists():
                if PyPMMLModel is None:
                    logger.warning("pypmml not installed: cannot validate native vs PMML; choosing proxy to be safe.")
                else:
                    proxy_est = PMMLProxyEstimator(pmml_xml, feature_names, out_candidates)
                    df = pd.read_csv(features_csv)
                    # Keep only intersecting columns
                    cols = [c for c in feature_names if c in df.columns]
                    if not cols:
                        logger.warning("Sample CSV has no matching feature columns; skipping validation.")
                    else:
                        X = df[cols]
                        y_proxy = proxy_est.predict(X)
                        y_native = native_est.predict(X)
                        diff = np.abs(y_proxy - y_native)
                        mae = float(np.mean(diff))
                        mxe = float(np.max(diff))
                        logger.info(f"Validation (native vs PMML): MAE={mae:.6g}, MaxErr={mxe:.6g}, N={len(diff)}")
                        if np.isnan(mae) or mxe > 1e-9:
                            logger.warning("Native mismatch > 1e-9 detected; falling back to proxy for perfect fidelity.")
                        else:
                            estimator = native_est
                            chosen_mode = "native"
            else:
                logger.info("No features CSV provided; defaulting to proxy to guarantee fidelity.")
        else:
            logger.info("Model not a simple RegressionModel without transforms; using proxy.")

    if estimator is None:
        estimator = PMMLProxyEstimator(pmml_xml, feature_names, out_candidates)
        chosen_mode = "proxy"

    # Final info
    logger.info(f"Chosen mode: {chosen_mode}")
    logger.info(f"Saving joblib to: {out_path}")
    joblib.dump({
        "estimator": estimator,
        "meta": {
            "mode": chosen_mode,
            "pmml_path": str(pmml_path),
            "feature_names": feature_names,
            "target": detected_target,
            "output_candidates": out_candidates,
        }
    }, out_path)
    logger.info("Done.")


# ------------------------------ CLI ------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Convert PMML (.xml/.pmml) to .joblib models with perfect mimicry.")
    parser.add_argument("--pmml", required=True, type=str, help="Path to PMML (.xml/.pmml)")
    parser.add_argument("--out", required=True, type=str, help="Output .joblib path")
    parser.add_argument("--features-csv", type=str, default=None,
                        help="Optional CSV with feature columns for validation (only used for --try-native)")
    parser.add_argument("--target", type=str, default=None, help="Optional target field name (e.g., hb)")
    parser.add_argument("--try-native", action="store_true",
                        help="Attempt native linear extraction for simple RegressionModel (falls back to proxy if mismatch)")
    args = parser.parse_args()

    pmml_path = Path(args.pmml)
    out_path = Path(args.out)
    features_csv = Path(args.features_csv) if args.features_csv else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_path)

    # Sanity checks
    if not pmml_path.exists():
        logger.error(f"PMML file not found: {pmml_path}")
        sys.exit(2)

    if args.try_native and PyPMMLModel is None and features_csv is not None:
        # We can still build native without validating; but we prefer validation.
        logger.warning("pypmml not installed — native attempt cannot be validated against PMML.")

    try:
        build_joblib_from_pmml(
            pmml_path=pmml_path,
            out_path=out_path,
            features_csv=features_csv,
            target_name=args.target,
            try_native=args.try_native,
            logger=logger,
        )
    except Exception as e:
        logger.exception(f"Failed to convert PMML to joblib: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
