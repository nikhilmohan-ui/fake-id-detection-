#MANASA'S CODE
"""
XAI-DocGuard
Risk Management Engine

Combines the ACTUAL outputs from:
1. OCR Module
2. Face Matching Module
3. Tamper Detection Module

Produces:
- Overall Risk Score
- Risk Level
- Risk Contributions
- Reasons
- Recommendation
- XAI-ready evidence
"""


class RiskEngine:

    # ---------------------------------------------------------
    # WEIGHTS
    # ---------------------------------------------------------
    # Actual modules available in the current repository:
    #
    # Tamper Detection -> 45%
    # Face Matching    -> 35%
    # OCR              -> 20%
    #
    # Total = 100%
    # ---------------------------------------------------------

    WEIGHTS = {
        "tamper": 45,
        "face": 35,
        "ocr": 20
    }

    def __init__(self):
        self.weights = self.WEIGHTS.copy()

    # =========================================================
    # UTILITY FUNCTIONS
    # =========================================================

    @staticmethod
    def clamp(value, minimum=0.0, maximum=1.0):
        """Keep a value between minimum and maximum."""

        return max(
            minimum,
            min(maximum, value)
        )

    @staticmethod
    def to_float(value, default=0.0):
        """Safely convert a value to float."""

        try:
            return float(value)

        except (TypeError, ValueError):

            return default

    # =========================================================
    # OCR RISK
    # =========================================================

    def calculate_ocr_risk(self, ocr_result):

        if not isinstance(ocr_result, dict):
            ocr_result = {}

        reasons = []

        # -----------------------------------------------------
        # ACTUAL OCR OUTPUT FROM REPO
        #
        # mean_confidence
        # is_valid_format
        # flags
        # structured_fields
        # -----------------------------------------------------

        confidence = self.clamp(
            self.to_float(
                ocr_result.get(
                    "mean_confidence",
                    0.0
                )
            )
        )

        is_valid_format = ocr_result.get(
            "is_valid_format",
            False
        )

        flags = ocr_result.get(
            "flags",
            []
        )

        if not isinstance(flags, list):
            flags = []

        # -----------------------------------------------------
        # Confidence based risk
        # -----------------------------------------------------

        confidence_risk = 1.0 - confidence

        # -----------------------------------------------------
        # Format validation risk
        # -----------------------------------------------------

        format_risk = 0.0

        if not is_valid_format:

            format_risk = 1.0

            reasons.append(
                "OCR document format validation failed."
            )

        # -----------------------------------------------------
        # OCR flags
        # -----------------------------------------------------

        if flags:

            reasons.append(
                "OCR detected: " +
                ", ".join(str(flag) for flag in flags)
            )

        # -----------------------------------------------------
        # Final OCR risk
        # -----------------------------------------------------

        risk = max(
            confidence_risk,
            format_risk
        )

        if confidence < 0.65:

            reasons.append(
                "OCR confidence is below the accepted threshold."
            )

        elif confidence < 0.80:

            reasons.append(
                "OCR confidence is moderate."
            )

        return {
            "risk": self.clamp(risk),
            "confidence": confidence,
            "reasons": reasons,
            "flags": flags
        }

    # =========================================================
    # FACE MATCHING RISK
    # =========================================================

    def calculate_face_risk(self, face_result):

        if not isinstance(face_result, dict):
            face_result = {}

        reasons = []

        # -----------------------------------------------------
        # ACTUAL FACE MODULE OUTPUT
        #
        # matched
        # similarity
        # distance
        # threshold
        # risk_score
        # risk_level
        # risk_flags
        # -----------------------------------------------------

        matched = face_result.get(
            "matched",
            False
        )

        similarity = self.clamp(
            self.to_float(
                face_result.get(
                    "similarity",
                    0.0
                )
            ) / 100.0
        )

        # -----------------------------------------------------
        # BEST SOURCE:
        # The repository's face_match.py already calculates
        # its own biometric risk_score from 0-100.
        # -----------------------------------------------------

        if "risk_score" in face_result:

            risk_score = self.clamp(
                self.to_float(
                    face_result.get(
                        "risk_score",
                        0.0
                    )
                ) / 100.0
            )

        else:

            # Fallback using similarity
            risk_score = 1.0 - similarity

        risk_flags = face_result.get(
            "risk_flags",
            []
        )

        if not isinstance(risk_flags, list):
            risk_flags = []

        # -----------------------------------------------------
        # Reasons
        # -----------------------------------------------------

        if not matched:

            reasons.append(
                "Face does not match the document identity."
            )

        if similarity < 0.50:

            reasons.append(
                "Very low facial similarity detected."
            )

        elif similarity < 0.70:

            reasons.append(
                "Low facial similarity detected."
            )

        if risk_flags:

            reasons.extend(
                str(flag)
                for flag in risk_flags
            )

        return {
            "risk": self.clamp(risk_score),
            "confidence": similarity,
            "similarity": round(
                similarity * 100,
                2
            ),
            "matched": bool(matched),
            "reasons": reasons,
            "risk_flags": risk_flags
        }

    # =========================================================
    # TAMPER DETECTION RISK
    # =========================================================

    def calculate_tamper_risk(self, tamper_result):

        if not isinstance(tamper_result, dict):
            tamper_result = {}

        reasons = []

        # -----------------------------------------------------
        # ACTUAL TAMPER MODULE OUTPUT
        #
        # tamper_score
        # reasons
        # heatmap
        # ela_image
        # -----------------------------------------------------

        tamper_score = self.clamp(
            self.to_float(
                tamper_result.get(
                    "tamper_score",
                    0.0
                )
            ) / 100.0
        )

        module_reasons = tamper_result.get(
            "reasons",
            []
        )

        if not isinstance(module_reasons, list):
            module_reasons = []

        # -----------------------------------------------------
        # Copy actual reasons
        # -----------------------------------------------------

        reasons.extend(
            str(reason)
            for reason in module_reasons
        )

        # -----------------------------------------------------
        # Additional explanation
        # -----------------------------------------------------

        if tamper_score >= 0.70:

            reasons.append(
                "High probability of document tampering."
            )

        elif tamper_score >= 0.40:

            reasons.append(
                "Moderate tampering indicators detected."
            )

        elif tamper_score > 0:

            reasons.append(
                "Minor tampering indicators detected."
            )

        return {
            "risk": tamper_score,
            "confidence": tamper_score,
            "tamper_score": round(
                tamper_score * 100,
                2
            ),
            "reasons": reasons
        }

    # =========================================================
    # RISK CLASSIFICATION
    # =========================================================

    @staticmethod
    def classify_risk(score):

        if score <= 30:

            return "LOW"

        elif score <= 60:

            return "MEDIUM"

        elif score <= 80:

            return "HIGH"

        else:

            return "CRITICAL"

    # =========================================================
    # RECOMMENDATION
    # =========================================================

    @staticmethod
    def get_recommendation(level):

        recommendations = {

            "LOW":
                "Document appears low risk. "
                "Proceed with normal verification.",

            "MEDIUM":
                "Document requires additional "
                "manual verification.",

            "HIGH":
                "Document should be flagged "
                "for detailed verification.",

            "CRITICAL":
                "Document is highly suspicious "
                "and should be escalated for "
                "manual investigation."
        }

        return recommendations.get(
            level,
            "Manual verification recommended."
        )

    # =========================================================
    # MAIN RISK ASSESSMENT
    # =========================================================

    def assess(
        self,
        ocr_result=None,
        face_result=None,
        tamper_result=None
    ):

        # -----------------------------------------------------
        # Make sure missing results don't crash the system
        # -----------------------------------------------------

        ocr_result = (
            ocr_result
            if isinstance(ocr_result, dict)
            else {}
        )

        face_result = (
            face_result
            if isinstance(face_result, dict)
            else {}
        )

        tamper_result = (
            tamper_result
            if isinstance(tamper_result, dict)
            else {}
        )

        # -----------------------------------------------------
        # Calculate individual risks
        # -----------------------------------------------------

        ocr = self.calculate_ocr_risk(
            ocr_result
        )

        face = self.calculate_face_risk(
            face_result
        )

        tamper = self.calculate_tamper_risk(
            tamper_result
        )

        # =====================================================
        # WEIGHTED RISK CONTRIBUTIONS
        # =====================================================

        contributions = {

            "tamper": round(
                tamper["risk"] *
                self.weights["tamper"],
                2
            ),

            "face": round(
                face["risk"] *
                self.weights["face"],
                2
            ),

            "ocr": round(
                ocr["risk"] *
                self.weights["ocr"],
                2
            )
        }

        # -----------------------------------------------------
        # Final risk score
        # -----------------------------------------------------

        risk_score = round(
            sum(
                contributions.values()
            ),
            2
        )

        risk_score = max(
            0.0,
            min(
                100.0,
                risk_score
            )
        )

        # -----------------------------------------------------
        # Risk Level
        # -----------------------------------------------------

        risk_level = self.classify_risk(
            risk_score
        )

        # =====================================================
        # REASONS
        # =====================================================

        reasons = []

        reasons.extend(
            tamper["reasons"]
        )

        reasons.extend(
            face["reasons"]
        )

        reasons.extend(
            ocr["reasons"]
        )

        # Remove duplicates
        reasons = list(
            dict.fromkeys(
                reasons
            )
        )

        if not reasons:

            reasons.append(
                "No significant anomalies detected."
            )

        # =====================================================
        # OVERALL CONFIDENCE
        # =====================================================

        confidence_values = [

            tamper["confidence"],
            face["confidence"],
            ocr["confidence"]

        ]

        overall_confidence = (

            sum(confidence_values)
            /
            len(confidence_values)

        )

        overall_confidence = round(
            overall_confidence * 100,
            2
        )

        # =====================================================
        # RECOMMENDATION
        # =====================================================

        recommendation = (
            self.get_recommendation(
                risk_level
            )
        )

        # =====================================================
        # FINAL XAI RESULT
        # =====================================================

        return {

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "confidence":
                overall_confidence,

            "risk_contributions":
                contributions,

            "reasons":
                reasons,

            "recommendation":
                recommendation,

            "evidence": {

                "ocr":
                    ocr,

                "face":
                    face,

                "tamper":
                    tamper
            },

            "xai_ready":
                True
        }


# =============================================================
# EASY FUNCTION FOR OTHER MODULES
# =============================================================

def assess_document_risk(
    ocr_result=None,
    face_result=None,
    tamper_result=None
):

    engine = RiskEngine()

    return engine.assess(

        ocr_result=ocr_result,

        face_result=face_result,

        tamper_result=tamper_result
    )


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # ACTUAL OCR MODULE STYLE
    # ---------------------------------------------------------

    ocr_data = {

        "raw_text":
            "NAME JOHN DOE DOB 01/01/2000",

        "structured_fields": {

            "name":
                "John Doe",

            "dob":
                "01/01/2000",

            "place":
                None,

            "date":
                "01/01/2000",

            "authority":
                None,

            "id_number":
                "ABC123456",

            "dates_found": [
                "01/01/2000"
            ]
        },

        "mean_confidence":
            0.94,

        "is_valid_format":
            True,

        "token_count":
            8,

        "flags":
            []
    }

    # ---------------------------------------------------------
    # ACTUAL FACE MODULE STYLE
    # ---------------------------------------------------------

    face_data = {

        "matched":
            False,

        "distance":
            0.62,

        "threshold":
            0.40,

        "similarity":
            22.5,

        "message":
            "Face does not match document",

        "status":
            "MISMATCH",

        "risk_score":
            80.0,

        "risk_level":
            "CRITICAL",

        "risk_flags": [
            "BIOMETRIC_MISMATCH_SEVERE"
        ],

        "facial_areas": {

            "document":
                None,

            "selfie":
                None
        }
    }

    # ---------------------------------------------------------
    # ACTUAL TAMPER MODULE STYLE
    # ---------------------------------------------------------

    tamper_data = {

        "tamper_score":
            60,

        "reasons": [

            "High ELA residual detected",

            "Abnormal noise pattern detected"
        ],

        "heatmap":
            None,

        "ela_image":
            None
    }

    # =========================================================
    # RUN RISK ENGINE
    # =========================================================

    result = assess_document_risk(

        ocr_result=ocr_data,

        face_result=face_data,

        tamper_result=tamper_data
    )

    # =========================================================
    # DISPLAY RESULT
    # =========================================================

    print()
    print("===================================")
    print("        XAI-DOCGUARD REPORT")
    print("===================================")

    print(
        f"Risk Score : "
        f"{result['risk_score']}/100"
    )

    print(
        f"Risk Level : "
        f"{result['risk_level']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence']}%"
    )

    print()
    print("Risk Contributions:")

    for factor, value in result[
        "risk_contributions"
    ].items():

        print(
            f"  {factor.capitalize():10} : "
            f"{value}"
        )

    print()
    print("Reasons:")

    for reason in result[
        "reasons"
    ]:

        print(
            f"  - {reason}"
        )

    print()
    print("Recommendation:")

    print(
        f"  {result['recommendation']}"
    )

    print()
    print(
        "XAI Ready:",
        result["xai_ready"]
    )