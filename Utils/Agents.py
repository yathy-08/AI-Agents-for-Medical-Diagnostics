from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama


class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info or {}

        # Prompt
        self.prompt_template = self.create_prompt_template()

        # Local LLaMA model via Ollama
        self.model = Ollama(
            model="llama3",
            temperature=0
        )

    def create_prompt_template(self):
        if self.role == "MultidisciplinaryTeam":
            template = """
            Act like a multidisciplinary team of healthcare professionals.

            Task:
            You will receive reports from a Cardiologist, Psychologist, and Pulmonologist.
            Analyze all three and provide exactly 3 possible health issues.
            For each issue, clearly explain the reasoning.

            Cardiologist Report:
            {cardiologist_report}

            Psychologist Report:
            {psychologist_report}

            Pulmonologist Report:
            {pulmonologist_report}

            Return only bullet points.
            """
            return PromptTemplate(
                input_variables=[
                    "cardiologist_report",
                    "psychologist_report",
                    "pulmonologist_report"
                ],
                template=template
            )

        templates = {
            "Cardiologist": """
            Act like a cardiologist.

            Task:
            Review the patient's cardiac workup including ECG, blood tests,
            Holter monitoring, and echocardiogram.

            Focus:
            Detect subtle cardiac issues such as arrhythmias or structural abnormalities.

            Recommendation:
            Suggest further cardiac tests or management strategies.

            Return only:
            - Possible causes
            - Recommended next steps

            Medical Report:
            {medical_report}
            """,

            "Psychologist": """
            Act like a psychologist.

            Task:
            Review the patient's report and identify possible psychological conditions.

            Focus:
            Anxiety, panic disorder, depression, trauma.

            Recommendation:
            Therapy, counseling, stress management.

            Return only:
            - Possible mental health issues
            - Recommended next steps

            Patient Report:
            {medical_report}
            """,

            "Pulmonologist": """
            Act like a pulmonologist.

            Task:
            Review the patient's respiratory symptoms.

            Focus:
            Asthma, COPD, lung infections, breathing disorders.

            Recommendation:
            Pulmonary tests or respiratory treatments.

            Return only:
            - Possible respiratory issues
            - Recommended next steps

            Patient Report:
            {medical_report}
            """
        }

        return PromptTemplate(
            input_variables=["medical_report"],
            template=templates[self.role]
        )

    def run(self):
        print(f"{self.role} is running...")

        if self.role == "MultidisciplinaryTeam":
            prompt = self.prompt_template.format(
                cardiologist_report=self.extra_info.get("cardiologist_report", ""),
                psychologist_report=self.extra_info.get("psychologist_report", ""),
                pulmonologist_report=self.extra_info.get("pulmonologist_report", "")
            )
        else:
            prompt = self.prompt_template.format(
                medical_report=self.medical_report
            )

        try:
            return self.model.invoke(prompt)
        except Exception as e:
            print("Error occurred:", e)
            return "Analysis unavailable due to system error."


# Specialized agents
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")


class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")


class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")


class MultidisciplinaryTeam(Agent):
    def __init__(self, cardiologist_report, psychologist_report, pulmonologist_report):
        super().__init__(
            role="MultidisciplinaryTeam",
            extra_info={
                "cardiologist_report": cardiologist_report,
                "psychologist_report": psychologist_report,
                "pulmonologist_report": pulmonologist_report
            }
        )
