from symbolic.java_calculator import JavaComplexityCalculator
from symbolic.python_calculator import TimeComplexityCalculator

def process_code(code, language):
    if language == 'corcod': language = 'java'
    if language == 'java':
        calculator = JavaComplexityCalculator(code, language)
        time_complexity = calculator.calculate_time_complexity_with_function_calls()
        prediction = calculator.classify_time_complexity(time_complexity)
    elif language == 'python':
        calculator = TimeComplexityCalculator(code, language)
        if "def " in code:
            time_complexity = calculator.calculate_time_complexity_with_function_calls()
        else:
            time_complexity = calculator.calculate_time_complexity()
        prediction = calculator.classify_time_complexity(time_complexity)
    # return time_complexity, prediction, calculator.error_method, calculator.success_method
    return prediction
