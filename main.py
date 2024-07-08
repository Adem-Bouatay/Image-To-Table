from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core import SimpleDirectoryReader
from json_extractor import extract, save_json_to_file
import time
from threading import Thread
import animation
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
    ...
    ...
    ...
  },
  ...
  ...
  ...
}

"""

def generate_timetable():
    """
    Generates a timetable in JSON format based on the provided prompt and image documents.

    Returns:
        None
    """
    
    Thread(target=animation.animate, daemon=True).start()
    
    # load image documents from local directory
    image_documents = SimpleDirectoryReader("images").load_data()

    mm_llm = GeminiMultiModal(model_name="models/gemini-1.5-pro", api_key=API_KEY)
    start = time.time()
    response = mm_llm.complete(
        prompt=PROMPT, image_documents=image_documents
    ).text
    end = time.time()

    print("Time taken: ", int(end-start),"sec \n" , "Table generated!!")

    try:
        extracted_json = extract(response)
        save_json_to_file(extracted_json, "output/table.json")
        print("\n-------------------------\nOutput saved to table.json!!")
    except Exception as e:
        print(f"Error extracting JSON content: {e}")

generate_timetable()
    