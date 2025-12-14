import sys
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agents import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam

# Load API key
load_dotenv(dotenv_path="apikey.env")

# Check if a report name is provided
if len(sys.argv) < 2:
    print("Usage: python3 Main.py '<Report File Name>'")
    print("Example: python3 Main.py 'Medical Report - Anna Thompson - IBS.txt'")
    sys.exit(1)

report_name = sys.argv[1]  # Get report name from command line

# Build file path
file_path = os.path.join("Medical Reports", report_name)

if not os.path.exists(file_path):
    print(f"Error: Report '{file_path}' not found!")
    sys.exit(1)

# Read the report
with open(file_path, "r") as file:
    medical_report = file.read()

# Initialize agents
agents = {
    "Cardiologist": Cardiologist(medical_report),
    "Psychologist": Psychologist(medical_report),
    "Pulmonologist": Pulmonologist(medical_report)
}

# Function to run each agent
def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

# Run agents concurrently
responses = {}
with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(get_response, name, agent): name
        for name, agent in agents.items()
    }

    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response

# Multidisciplinary aggregation
team_agent = MultidisciplinaryTeam(
    cardiologist_report=responses["Cardiologist"],
    psychologist_report=responses["Psychologist"],
    pulmonologist_report=responses["Pulmonologist"]
)

final_diagnosis = team_agent.run()

if not final_diagnosis:
    final_diagnosis = (
        "Unable to generate diagnosis due to API quota limitations.\n\n"
        "This system architecture supports multi-agent parallel reasoning, "
        "but requires a valid LLM quota for live execution."
    )

final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis

# Save output
output_path = os.path.join("Results", f"final_diagnosis_{report_name.replace(' ', '_')}.txt")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as txt_file:
    txt_file.write(final_diagnosis_text)

print(f"Final diagnosis has been saved to {output_path}")
