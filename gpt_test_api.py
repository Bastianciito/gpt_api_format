
import os 
from dotenv import load_dotenv
from openai import OpenAI

if __name__ == "__main__":
  load_dotenv()
  key = os.getenv("OPENAI_API_KEY")
  client = OpenAI(api_key=key)

  relato = """accidente sucedió cuando: CONTENIENDO A PACIENTE PSIQUIÁTRICOS
              descripcion del accidente : PACIENTE TENS REFIERE QUE EL DIA DE HOY MIENTRAS CONTENÍA A PACIENTE ESTE LO GOLPEA DE UN CABEZAZO EN LA CIEN IZQUIERDA Y TAMBIÉN LO RASGUÑA EN EL ANTEBRAZO DERECHO , ES ENVIADO POR LA ENFERMERA A CARGO LUCIA MIRANDA PARA EVALUACIÓN IST//EED PACIENTE SE PRESENTA SIN DIAT
              lugar en que se desarrollo el accidente: SALA DE ADOLECENTES HOSPITAL SALVADOR
              profesion del afectado :  0
              zonas del cuerpo afectadas por el accidente: CABEZA, SITIO NO ESPECIFICADO"""

  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
      {"role": "system", "content": """Eres una asistente que me ayuda a estructurar relatos asociados a accidentes de trabajo, tienes que estructurar la información 
                                      para que esta sea coherente, el relato se compone de cinco campos con información los cuales debes de usar para crear
                                      en un solo parrafo una descripción del suceso, teniendo en cuenta el cómo fue y que partes del curpo fueron afectadas, esto hazlo 
                                      sin añadir informacion erronea. Solo utiliza palabras en español dentro del relato."""},
      {"role": "user", "content": f"""Este es el relato: {relato}"""}
    ]
  )

  print(completion.choices[0].message.content)

