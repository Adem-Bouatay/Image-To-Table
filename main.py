from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core import SimpleDirectoryReader
from json_extractor import extract, save_json_to_file
import time
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")

PROMPT = """return the time table in json format
this is an example of how the output should look like:
{
  "Monday": {
    "08:30-10:00": {
      "IA1.1": {
        "course": "Reseaux ",
        "instructor": "BEN FARAH Ahmed",
        "location": "I12"
      },
      "IA1.2": {
        "course": "Architecture",
        "instructor": "KHADRAOUI Imen",
        "location": "I13"
      }
    },
    "10:15-11:45": {
      "IA1.1": {
        "course": "Reseaux ",
        "instructor": "BEN FARAH Ahmed",
        "location": "I12"
      },
      "IA1.2": {
        "course": "Elements",
        "instructor": "BEL HADJ IBRAHIM Anis",
        "location": "E14"
      }
    },
    "12:00-13:00": {},
    "13:00-14:30": {
      "IA1.1": {
        "course": "Reseaux de Petri",
        "instructor": "ABDELLAOUI Mehrez",
        "location": "B02"
      }
    },
    "14:45-16:15": {
      "IA1.1": {
        "course": "Elements",
        "instructor": "BEL HADJ IBRAHIM Anis",
        "location": "R03"
      }
    },
    "16:30-18:00": {}
  },
  "Tuesday": {
    "08:30-10:00": {
      "IA1.1": {
        "course": "Automatique ",
        "instructor": "BEMBILI Sana",
        "location": "E13"
      }
    },
    "10:15-11:45": {
      "IA1.1": {
        "course": "Automatique ",
        "instructor": "BEMBILI Sana",
        "location": "E13"
      }
    },
    "12:00-13:00": {},
    "13:00-14:30": {
      "IA1.2": {
        "course": "Programmation Orientee Objet",
        "instructor": "ABDELATTIF Takoua",
        "location": "B04"
      }
    },
    "14:45-16:15": {
      "IA1.2": {
        "course": "Elements",
        "instructor": "BEL HADJ IBRAHIM Anis",
        "location": "B04"
      }
    },
    "16:30-18:00": {}
  },
  "Wednesday": {
    "08:30-10:00": {
      "IA1.1": {
        "course": "Reseaux",
        "instructor": "BEN ARBIA Anis",
        "location": ""
      }
    },
    "10:15-11:45": {
      "IA1.1": {
        "course": "Algorithmique ",
        "instructor": "CHAINBI Walid",
        "location": "B04"
      }
    },
    "12:00-13:00": {},
    "13:00-14:30": {},
    "14:45-16:15": {},
    "16:30-18:00": {}
  },
  "Thursday": {
    "08:30-10:00": {
      "IA1.1": {
        "course": "Algorithmique ",
        "instructor": "CHAINBI Walid",
        "location": "B04"
      }
    },
    "10:15-11:45": {
      "IA1.1": {
        "course": "Reseaux informatiques",
        "instructor": "BEN ARBIA Anis",
        "location": "R03"
      }
    },
    "12:00-13:00": {},
    "13:00-14:30": {
      "IA1.1": {
        "course": "Circuits Programm

"""

# load image documents from urls
#image_documents = load_image_urls(image_urls)

# load image documents from local directory
image_documents = SimpleDirectoryReader("images").load_data()

# non-streaming
mm_llm = GeminiMultiModal(model_name="models/gemini-1.5-pro", api_key=API_KEY)
start = time.time()
response = mm_llm.complete(
    prompt=PROMPT, image_documents=image_documents
).text
end = time.time()

print("Time taken: ", int(end-start),"sec \n" , "Table genereated!!")

save_json_to_file(extract(response), "table.json")
print("\n-------------------------\nOutput saved to table.json!!")