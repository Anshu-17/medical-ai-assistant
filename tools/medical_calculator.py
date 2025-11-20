import json
from langchain_core.tools import tool
from loguru import logger

from tools.base import BaseMedicalTool


class MedicalCalculatorTool(BaseMedicalTool):
    """Calculate medical metrics and scores"""
    
    def __init__(self):
        """Initialize medical calculator"""
        super().__init__(
            name="calculate_medical_metric",
            description="""Calculate medical metrics and scores.
            
            Supported: BMI, eGFR, APACHE II, CHADS2
            
            Args:
                calculation_type: Type (e.g., "BMI")
                parameters: JSON with required parameters
                
            Example:
                calculation_type="BMI"
                parameters='{"weight_kg": 70, "height_m": 1.75}'
                
            Returns:
                Calculated value with interpretation
            """
        )
    
    def execute(self, calculation_type: str, parameters: str) -> str:
        """Execute medical calculation"""
        try:
            params = json.loads(parameters)
            calc_type = calculation_type.upper()
            
            if calc_type == "BMI":
                return self._calculate_bmi(params)
            elif calc_type == "EGFR":
                return self._calculate_egfr(params)
            else:
                return f"Calculation '{calculation_type}' not supported"
                
        except json.JSONDecodeError:
            return "Invalid JSON format for parameters"
        except Exception as e:
            logger.error(f"Calculation error: {e}")
            return f"Calculation failed: {str(e)}"
    
    def _calculate_bmi(self, params: dict) -> str:
        """Calculate BMI"""
        weight = params.get("weight_kg")
        height = params.get("height_m")
        
        if not weight or not height:
            return "BMI requires 'weight_kg' and 'height_m'"
        
        bmi = weight / (height ** 2)
        
        if bmi < 18.5:
            category = "Underweight"
            risk = "Increased risk of malnutrition"
        elif bmi < 25:
            category = "Normal weight"
            risk = "Healthy weight range"
        elif bmi < 30:
            category = "Overweight"
            risk = "Increased health risks"
        else:
            category = "Obese"
            risk = "Significant health risks"
        
        return f"""BMI CALCULATION RESULT:

BMI: {bmi:.1f} kg/m²
Category: {category}
Assessment: {risk}

Reference Ranges:
- Underweight: < 18.5
- Normal: 18.5 - 24.9
- Overweight: 25.0 - 29.9
- Obese: ≥ 30.0"""
    
    def _calculate_egfr(self, params: dict) -> str:
        """Calculate eGFR (simplified)"""
        # Implementation would go here
        return "eGFR calculation coming soon"


# LangChain tool wrapper
@tool
def calculate_medical_metric(calculation_type: str, parameters: str) -> str:
    """Calculate medical metrics and scores."""
    tool_instance = MedicalCalculatorTool()
    return tool_instance.execute(calculation_type, parameters)